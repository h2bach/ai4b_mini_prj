"""Tải nguồn được pin revision, kiểm checksum và giữ manifest split cố định."""
from collections import Counter
import gzip
import json
from pathlib import Path
import struct
import unicodedata
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from shared.utils import fingerprint, read_json, sha256, write_json

VSFC_PARQUET_REV = '2d76906157232f87e9883fa37ebfc91ffdee1f83'
FASHION_REV = 'b2617bb6d3ffa2e429640350f613e3291e10b141'
FASHION_MD5 = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
LABELS = {
    'cnn': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    'nlp': ['negative', 'neutral', 'positive'],
}


def normalize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Văn bản rỗng hoặc không hợp lệ; hãy nhập ít nhất một token.')
    return unicodedata.normalize('NFC', text).lower().strip()


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + '.part')
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with tmp.open('wb') as f:
                for b in r.iter_content(1024 * 1024):
                    f.write(b)
        tmp.replace(path)
    return path


def _idx(path):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        if magic == 2051:
            h, w = struct.unpack('>II', f.read(8))
            assert (h, w) == (28, 28)
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h, w).copy()
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n).astype(np.int64)


def prepare_fashion(root):
    import hashlib
    root = Path(root) / 'cnn'
    checksums = {}
    for name, md5 in FASHION_MD5.items():
        url = f'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{FASHION_REV}/data/fashion/{name}'
        p = download(url, root / name)
        if hashlib.md5(p.read_bytes()).hexdigest() != md5:
            raise ValueError(f'Checksum nguồn Fashion-MNIST không khớp: {p}')
        checksums[name] = sha256(p)
    y = _idx(root / 'train-labels-idx1-ubyte.gz')
    fixed = read_json(Path(__file__).parent / 'manifests' / 'cnn.json')
    train = np.array([int(i.split(':')[1]) for i in fixed['train']])
    val = np.array([int(i.split(':')[1]) for i in fixed['validation']])
    assert len(train) == 50000 and len(val) == 10000
    assert np.all(np.bincount(y[train]) == 5000) and np.all(np.bincount(y[val]) == 1000)
    manifest = {'source': 'https://github.com/zalandoresearch/fashion-mnist', 'revision': FASHION_REV,
                'checksums': checksums, 'split_seed': 42, 'labels': LABELS['cnn'],
                'train': [f'train:{i}' for i in train.tolist()],
                'validation': [f'train:{i}' for i in val.tolist()],
                'test': [f'test:{i}' for i in range(10000)]}
    _persist_manifest(root, manifest)
    return manifest


def prepare_vsfc(root):
    root = Path(root) / 'nlp'
    source_path = root / 'source.json'
    if source_path.exists():
        source = read_json(source_path)
    else:
        # Pin bản chuyển parquet chính thức, không chạy remote dataset script.
        revision = VSFC_PARQUET_REV
        source = {'source': 'https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback',
                  'revision': revision, 'format': 'official HF parquet conversion', 'license': 'unknown'}
        write_json(source_path, source)
    checksums, splits, texts = {}, {}, {}
    for split, count in [('train', 11426), ('validation', 1583), ('test', 3166)]:
        url = f"{source['source']}/resolve/{source['revision']}/default/{split}/0000.parquet"
        p = download(url, root / f'{split}.parquet')
        checksums[p.name] = sha256(p)
        frame = pd.read_parquet(p)
        assert len(frame) == count, f'Số mẫu {split} không đúng benchmark'
        assert set(frame.sentiment.unique()) == {0, 1, 2}, 'Sai mapping sentiment'
        # Lỗi chỉ rõ split và dòng, không tự bỏ mẫu rỗng.
        texts[split] = []
        for i, text in enumerate(frame.sentence):
            try:
                texts[split].append(normalize(text))
            except ValueError as e:
                raise ValueError(f'{split}:{i}: {e}') from e
        splits[split] = [f'{split}:{i}' for i in range(count)]
    dup = {}
    for a in texts:
        dup[a + '_within'] = len(texts[a]) - len(set(texts[a]))
        for b in texts:
            if a < b:
                dup[a + '__' + b] = len(set(texts[a]) & set(texts[b]))
    manifest = {**source, 'checksums': checksums, 'labels': LABELS['nlp'],
                'duplicates_normalized_unique_texts': dup, **splits}
    _persist_manifest(root, manifest)
    return manifest


def _persist_manifest(root, manifest):
    manifest['fingerprint'] = fingerprint(manifest)
    packaged = Path(__file__).parent / 'manifests' / (root.name + '.json')
    if packaged.exists() and read_json(packaged) != manifest:
        raise ValueError('Dữ liệu không khớp manifest đã phát hành; không tự sửa split/checksum.')
    path = root / 'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError(f'Dữ liệu/manifest đã thay đổi: {path}. Dùng thư mục cache mới để điều tra.')
    write_json(path, manifest)


def load_data(project, root='data', include_test=False, subset=None):
    root = Path(root) / project
    if not (root / 'manifest.json').exists():
        (prepare_fashion if project == 'cnn' else prepare_vsfc)(root.parent)
    manifest = read_json(root / 'manifest.json')
    core = {k: v for k, v in manifest.items() if k != 'fingerprint'}
    assert fingerprint(core) == manifest['fingerprint'], 'Manifest bị thay đổi'
    for name, expected in manifest['checksums'].items():
        assert sha256(root / name) == expected, f'Checksum không khớp: {name}'
    result = {'project': project, 'manifest': manifest, 'labels': LABELS[project]}
    split_names = ['train', 'validation'] + (['test'] if include_test else [])
    if project == 'cnn':
        images = _idx(root / 'train-images-idx3-ubyte.gz')
        targets = _idx(root / 'train-labels-idx1-ubyte.gz')
        for split in split_names:
            ids = manifest[split]
            indices = np.array([int(i.split(':')[1]) for i in ids])
            if split == 'test':
                x = _idx(root / 't10k-images-idx3-ubyte.gz')
                y = _idx(root / 't10k-labels-idx1-ubyte.gz')
            else:
                x, y = images[indices], targets[indices]
            result[split] = {'x': x, 'y': y, 'ids': ids}
    else:
        for split in split_names:
            frame = pd.read_parquet(root / f'{split}.parquet')
            result[split] = {'x': frame.sentence.map(normalize).tolist(),
                             'y': frame.sentiment.to_numpy(dtype=np.int64), 'ids': manifest[split]}
    # Subset kỹ thuật có định danh riêng; tuyệt đối không thay manifest benchmark.
    if subset:
        for split in split_names:
            item = result[split]
            size = min(subset, len(item['y']))
            idx = np.arange(len(item['y']))
            if size < len(idx):
                idx, _ = train_test_split(idx, train_size=size, random_state=42, stratify=item['y'])
            item['x'] = np.asarray(item['x'])[idx] if project == 'cnn' else [item['x'][i] for i in idx]
            item['y'] = item['y'][idx]
            item['ids'] = [item['ids'][i] for i in idx]
    split_ids = {s: result[s]['ids'] for s in split_names}
    result['fingerprint'] = fingerprint({'source': manifest['fingerprint'], 'subset': subset,
                                         'train': split_ids['train'], 'validation': split_ids['validation']})
    result['subset'] = subset
    if project == 'cnn':
        x = result['train']['x'].astype(np.float32) / 255
        result['preprocessing'] = {'mean': float(x.mean(dtype=np.float64)),
                                   'std': float(x.std(dtype=np.float64)), 'pixel_scale': 255}
    else:
        result['preprocessing'] = {'unicode': 'NFC', 'lower': True, 'tokenizer': 'whitespace', 'max_length': 40}
    for a in split_names:
        assert len(set(result[a]['ids'])) == len(result[a]['ids'])
        for b in split_names:
            if a != b:
                assert not set(result[a]['ids']) & set(result[b]['ids'])
    return result


def normalized_images(x, preprocessing):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (28, 28) or x.dtype != np.uint8:
        raise ValueError('Cần batch ảnh uint8 (N,28,28), pixel 0–255.')
    return ((x.astype(np.float32) / 255 - preprocessing['mean']) / preprocessing['std'])[:, None]


def describe(data):
    report = {'fingerprint': data['fingerprint'], 'subset': data['subset'], 'preprocessing': data['preprocessing']}
    for split in ['train', 'validation']:
        part = data[split]
        r = {'n': len(part['y']), 'label_counts': dict(sorted(Counter(map(int, part['y'])).items()))}
        if data['project'] == 'nlp':
            lengths = np.array([len(t.split()) for t in part['x']])
            r.update(empty=int((lengths == 0).sum()), length_median=float(np.median(lengths)),
                     length_p95=float(np.quantile(lengths, .95)), truncated_fraction=float((lengths > 40).mean()))
        report[split] = r
    return report
