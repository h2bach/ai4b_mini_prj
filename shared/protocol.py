"""Chọn theo validation, khóa checksum trước khi mở test; dùng cho cả hai gói."""
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from shared.baselines import baseline_inputs
from shared.data import load_data
from shared.engine import evaluate, make_loader, reload_model
from shared.metrics import classification_metrics, save_predictions
from shared.utils import fingerprint, read_json, sha256, write_json

RULE = 'Mean validation macro-F1 qua mọi seed; hòa: ít parameter, rồi thứ tự variant. Best epoch: F1 lớn nhất, hòa lấy epoch sớm.'


def register_protocol(root, project, config, seeds):
    value = {'project': project, 'rule': RULE, 'variants': config['variants'], 'seeds': list(seeds),
             'budget': {k: v for k, v in config.items() if k != 'train_seeds'}}
    p = Path(root) / project / 'protocol.json'
    if p.exists() and read_json(p) != value:
        raise ValueError('Quy tắc/budget đã đăng ký khác. Dùng thư mục thí nghiệm mới.')
    write_json(p, value)


def lock_selection(root, project, seeds=None):
    base = Path(root) / project
    protocol = read_json(base / 'protocol.json')
    seeds = list(protocol['seeds'] if seeds is None else seeds)
    assert seeds == protocol['seeds'], 'Phải dùng toàn bộ seed đã đăng ký'
    dest = base / 'lua_chon.json'
    if dest.exists():
        verify_selection(root, project)
        return read_json(dest)
    candidates, common_fingerprint = [], None
    for order, variant in enumerate(protocol['variants']):
        runs = []
        for seed in seeds:
            path = base / variant / f'seed_{seed}'
            summary = read_json(path / 'summary.json')
            assert not summary['subset'], 'Không chọn bằng kết quả subset kỹ thuật'
            assert summary['completed_epochs'] == protocol['budget']['epochs'], 'Run chưa đủ epoch'
            assert summary['config']['seed'] == seed and summary['config']['variant'] == variant
            assert all(summary['config'][k] == v for k, v in protocol['budget'].items()), 'Run khác budget đã đăng ký'
            common_fingerprint = common_fingerprint or summary['fingerprint']
            assert summary['fingerprint'] == common_fingerprint, 'Run khác dữ liệu/split'
            rel = str((path / 'best.pt').relative_to(base))
            runs.append({'seed': seed, 'validation_macro_f1': summary['validation']['macro_f1'],
                         'best_epoch': summary['best_epoch'], 'checkpoint': rel, 'sha256': sha256(base / rel)})
        scores = [r['validation_macro_f1'] for r in runs]
        candidates.append({'variant': variant, 'mean_macro_f1': float(np.mean(scores)),
                           'std_macro_f1': float(np.std(scores)), 'n_params': summary['n_params'],
                           'order': order, 'runs': runs})
    chosen = sorted(candidates, key=lambda c: (-c['mean_macro_f1'], c['n_params'], c['order']))[0]
    baseline = read_json(base / 'baselines' / 'baselines.json')
    assert baseline['fingerprint'] == common_fingerprint and not baseline['subset']
    artifacts = {}
    for filename in ['baselines.json', 'majority.joblib', 'logreg.joblib']:
        rel = 'baselines/' + filename
        artifacts[rel] = sha256(base / rel)
    lock = {'project': project, 'locked_at_utc': datetime.now(timezone.utc).isoformat(),
            'rule': RULE, 'protocol_sha256': sha256(base / 'protocol.json'),
            'fingerprint': common_fingerprint, 'chosen': chosen, 'candidates': candidates, 'baseline_artifacts': artifacts}
    lock['lock_fingerprint'] = fingerprint(lock)
    write_json(dest, lock)
    return lock


def verify_selection(root, project):
    base = Path(root) / project
    lock = read_json(base / 'lua_chon.json')
    assert fingerprint({k: v for k, v in lock.items() if k != 'lock_fingerprint'}) == lock['lock_fingerprint'], 'Khóa bị sửa'
    assert sha256(base / 'protocol.json') == lock['protocol_sha256'], 'Protocol bị sửa sau khi khóa'
    for run in lock['chosen']['runs']:
        p = (base / run['checkpoint']).resolve()
        assert p.is_relative_to(base.resolve()), 'Checkpoint nằm ngoài experiment'
        assert sha256(p) == run['sha256'], 'Checkpoint thay đổi sau khi khóa'
    for rel, expected in lock['baseline_artifacts'].items():
        assert sha256(base / rel) == expected, f'Baseline thay đổi sau khi khóa: {rel}'
    return lock


def final_test(root, project, data_root, build_model, encode=None, open_test=False, device='cpu'):
    if not open_test:
        raise ValueError('Test đang đóng. Khóa lua_chon.json rồi chủ động đặt open_test=True ở ngày 13.')
    base = Path(root) / project
    lock = verify_selection(root, project)
    dest = base / 'test_report.json'
    if dest.exists():
        report = read_json(dest)
        assert report['lock_fingerprint'] == lock['lock_fingerprint']
        return report
    data = load_data(project, data_root, include_test=True)
    assert data['fingerprint'] == lock['fingerprint'], 'Test khác dữ liệu đã khóa'
    report = {'project': project, 'lock_fingerprint': lock['lock_fingerprint'], 'results': {}}
    for name in ['majority', 'logreg']:
        model = joblib.load(base / 'baselines' / f'{name}.joblib')
        x = np.zeros((len(data['test']['y']), 1)) if name == 'majority' else baseline_inputs(data, 'test')
        pred = model.predict(x)
        report['results'][name] = classification_metrics(data['test']['y'], pred, data['labels'])
        save_predictions(base / f'{name}_test.csv', data['test']['ids'], data['test']['y'], pred)
    for run in lock['chosen']['runs']:
        model, ck = reload_model(base / run['checkpoint'], build_model, device)
        assert ck['fingerprint'] == data['fingerprint']
        loader = make_loader(data, 'test', ck['config'], ck['vocab'], encode)
        r = evaluate(model, loader, device, data['labels'], data['test']['ids'])
        name = f"{lock['chosen']['variant']}_seed_{run['seed']}"
        report['results'][name] = r['metrics']
        save_predictions(base / f'{name}_test.csv', r['ids'], r['y'], r['pred'], r['scores'])
    dl = [v['macro_f1'] for k, v in report['results'].items() if '_seed_' in k]
    report['selected_dl_macro_f1_mean'] = float(np.mean(dl))
    report['selected_dl_macro_f1_std'] = float(np.std(dl))
    write_json(dest, report)
    return report
