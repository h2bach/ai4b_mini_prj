"""Khung thực nghiệm công khai; model, mã hóa chuỗi và train step do người học truyền vào."""
from pathlib import Path
from time import perf_counter
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from shared.data import normalized_images
from shared.metrics import classification_metrics, save_predictions
from shared.utils import load_checkpoint, save_checkpoint, seed_everything, write_json


def make_loader(data, split, config, vocab=None, encode=None, shuffle=False, generator=None):
    part = data[split]
    if data['project'] == 'cnn':
        xs = (torch.from_numpy(normalized_images(part['x'], data['preprocessing'])),)
    else:
        if encode is None or vocab is None:
            raise ValueError('Hoàn thiện encode_text và build_vocab trước khi tạo chuỗi.')
        encoded = [encode(t, vocab, config['max_length']) for t in part['x']]
        xs = (torch.tensor([a[0] for a in encoded], dtype=torch.long),
              torch.tensor([a[1] for a in encoded], dtype=torch.long))
    ds = TensorDataset(*xs, torch.as_tensor(part['y'], dtype=torch.long))
    return DataLoader(ds, batch_size=config['batch_size'], shuffle=shuffle,
                      generator=generator, num_workers=0)


def unpack_batch(batch, device):
    return [x.to(device) for x in batch[:-1]], batch[-1].to(device)


@torch.no_grad()
def evaluate(model, loader, device, labels, ids):
    model.eval()
    all_y, probs, total_loss = [], [], 0.
    for batch in loader:
        xs, y = unpack_batch(batch, device)
        logits = model(*xs)
        if logits.shape != (len(y), len(labels)) or not torch.isfinite(logits).all():
            raise ValueError('Logits phải hữu hạn, có shape (batch, số lớp).')
        total_loss += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        all_y.extend(y.cpu().tolist())
        probs.append(logits.softmax(-1).cpu().numpy())
    scores = np.concatenate(probs)
    pred = scores.argmax(1)
    return {'metrics': classification_metrics(all_y, pred, labels), 'loss': total_loss / len(all_y),
            'ids': list(ids), 'y': np.array(all_y), 'pred': pred, 'scores': scores}


def fit(data, config, out, build_model, train_one_epoch, build_vocab=None, encode=None,
        device=None, resume=True, stop_after=None):
    """stop_after chỉ phục vụ kiểm resume, không đổi budget đã ghi trong config."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if config['project'] != data['project']:
        raise ValueError('Sai project của config')
    seed_everything(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    vocab = build_vocab(data['train']['x'], config['min_freq']) if data['project'] == 'nlp' else None
    model_config = {**config, 'vocab_size': len(vocab) if vocab else None}
    model = build_model(model_config).to(device)
    expected = 20490 if config['variant'] == 'flatten' else 5130 if data['project'] == 'cnn' else 64 * len(vocab) + 33475
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == expected, f'Số parameter {n_params} khác kỳ vọng {expected}'
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.parameters()}
    weights = None
    if config['variant'] == 'weighted':
        counts = np.bincount(data['train']['y'], minlength=len(data['labels']))
        if (counts == 0).any():
            raise ValueError('Subset thiếu lớp; cần đủ ba lớp trước khi tính trọng số.')
        weights = torch.tensor(len(data['train']['y']) / (len(counts) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = make_loader(data, 'train', config, vocab, encode, shuffle=True, generator=generator)
    val_loader = make_loader(data, 'validation', config, vocab, encode)
    history, best_score, start_epoch, elapsed = [], -1., 0, 0.
    if resume and (out / 'last.pt').exists():
        ck = load_checkpoint(out / 'last.pt', device)
        assert ck['fingerprint'] == data['fingerprint'] and ck['config'] == model_config, 'Resume khác dữ liệu/cấu hình'
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        history, best_score, start_epoch, elapsed = ck['history'], ck['best_score'], ck['epoch'], ck['elapsed_seconds']
        generator.set_state(ck['loader_rng'].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu())
        random.setstate(ck['python_rng'])
        np.random.set_state(ck['numpy_rng'])
        if device.startswith('cuda') and ck['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, min(config['epochs'], stop_after or config['epochs'])):
        start = perf_counter()
        before = [p.detach().clone() for p in model.parameters()]
        train_loss = float(train_one_epoch(model, train_loader, optimizer, criterion, device, config['clip']))
        if not np.isfinite(train_loss):
            raise ValueError('Train loss không hữu hạn')
        if not any(not torch.equal(a, b) for a, b in zip(before, model.parameters())):
            raise ValueError('Weight chưa thay đổi. Kiểm backward và optimizer.step().')
        result = evaluate(model, val_loader, device, data['labels'], data['validation']['ids'])
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        seconds = perf_counter() - start
        elapsed += seconds
        score = result['metrics']['macro_f1']
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'validation_loss': result['loss'],
                        'macro_f1': score, 'seconds': seconds})
        improved = score > best_score  # Hòa: giữ epoch sớm hơn.
        best_score = max(best_score, score)
        ck = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
              'config': model_config, 'fingerprint': data['fingerprint'], 'subset': data['subset'],
              'preprocessing': data['preprocessing'], 'vocab': vocab, 'labels': data['labels'],
              'history': history, 'best_score': best_score, 'elapsed_seconds': elapsed,
              'loader_rng': generator.get_state(), 'torch_rng': torch.get_rng_state(),
              'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
              'cuda_rng': torch.cuda.get_rng_state_all() if device.startswith('cuda') else None}
        if improved:
            save_checkpoint(out / 'best.pt', ck)
        save_checkpoint(out / 'last.pt', ck)
        write_json(out / 'history.json', history)
        print(f"{data['project']} {config['variant']} seed={config['seed']} epoch={epoch+1} loss={train_loss:.4f} val_f1={score:.4f}", flush=True)
    ck = load_checkpoint(out / 'best.pt', device)
    model.load_state_dict(ck['model'])
    result = evaluate(model, val_loader, device, data['labels'], data['validation']['ids'])
    save_predictions(out / 'validation.csv', result['ids'], result['y'], result['pred'], result['scores'])
    summary = {'project': data['project'], 'config': model_config, 'fingerprint': data['fingerprint'],
               'subset': data['subset'], 'n_params': n_params, 'best_epoch': ck['epoch'],
               'completed_epochs': len(history), 'validation': result['metrics'], 'train_seconds': elapsed,
               'peak_gpu_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
               'device': device, 'class_weights': None if weights is None else weights.cpu().tolist()}
    if vocab:
        validation_tokens = [t for sentence in data['validation']['x'] for t in sentence.split()]
        summary['vocab_size'] = len(vocab)
        summary['validation_unk_fraction'] = sum(t not in vocab for t in validation_tokens) / len(validation_tokens)
        write_json(out / 'vocab.json', vocab)
    write_json(out / 'summary.json', summary)
    return summary


def reload_model(checkpoint, build_model, device='cpu'):
    ck = load_checkpoint(checkpoint, device)
    model = build_model(ck['config']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, ck


@torch.no_grad()
def predict_image(model, image, preprocessing, labels, device='cpu'):
    x = normalized_images(np.asarray(image)[None], preprocessing)
    model.eval()
    scores = model(torch.from_numpy(x).to(device)).softmax(-1)[0].cpu().tolist()
    return {'label': labels[int(np.argmax(scores))], 'scores': dict(zip(labels, scores))}


@torch.no_grad()
def predict_text(model, text, vocab, encode, labels, max_length=40, device='cpu'):
    ids, length = encode(text, vocab, max_length)
    model.eval()
    scores = model(torch.tensor([ids], dtype=torch.long, device=device),
                   torch.tensor([length], dtype=torch.long, device=device)).softmax(-1)[0].cpu().tolist()
    return {'label': labels[int(np.argmax(scores))], 'scores': dict(zip(labels, scores))}
