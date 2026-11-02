import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from shared.utils import write_json


def classification_metrics(y, pred, labels):
    y, pred = np.asarray(y), np.asarray(pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(labels))), zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)), 'correct': int((y == pred).sum()), 'total': len(y),
            'macro_f1': float(f1.mean()),
            'per_class': {name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                 'f1': float(f1[i]), 'support': int(support[i])} for i, name in enumerate(labels)},
            'confusion_matrix': confusion_matrix(y, pred, labels=list(range(len(labels)))).tolist()}


def save_predictions(path, ids, y, pred, scores=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert len(ids) == len(y) == len(pred)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'true', 'pred'] + ([] if scores is None else [f'prob_{i}' for i in range(scores.shape[1])]))
        for i, sid in enumerate(ids):
            writer.writerow([sid, int(y[i]), int(pred[i])] + ([] if scores is None else scores[i].tolist()))


def plot_confusion(metrics, labels):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    m = np.array(metrics['confusion_matrix'])
    im = ax.imshow(m, cmap='Blues')
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel='Dự đoán', ylabel='Nhãn thật')
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(m[i, j]), ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_history(history):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot([h['epoch'] for h in history], [h['train_loss'] for h in history], label='train')
    axes[0].plot([h['epoch'] for h in history], [h['validation_loss'] for h in history], label='validation')
    axes[0].set(xlabel='Epoch', ylabel='Loss'); axes[0].legend()
    axes[1].plot([h['epoch'] for h in history], [h['macro_f1'] for h in history])
    axes[1].set(xlabel='Epoch', ylabel='Validation macro-F1')
    fig.tight_layout()
    return fig


def show_errors(data, ids, predictions, n=5):
    """Luôn phân tích validation; không lấy mẫu test để điều chỉnh mô hình."""
    part = data['validation']
    assert list(ids) == list(part['ids']), 'Prediction phải cùng thứ tự sample ID validation'
    wrong = np.flatnonzero(part['y'] != np.asarray(predictions))
    if data['project'] == 'cnn':
        # Ưu tiên lỗi liên quan Shirt nếu model có lỗi kiểu này.
        wrong = sorted(wrong, key=lambda i: not (part['y'][i] == 6 or predictions[i] == 6))[:n]
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, max(1, len(wrong)), figsize=(3 * max(1, len(wrong)), 3), squeeze=False)
        for ax, i in zip(axes[0], wrong):
            ax.imshow(part['x'][i], cmap='gray')
            ax.set_title(f"{part['ids'][i]}\n{data['labels'][part['y'][i]]} → {data['labels'][predictions[i]]}")
        return fig
    return [{'sample_id': part['ids'][i], 'text': part['x'][i], 'true': data['labels'][part['y'][i]],
             'pred': data['labels'][predictions[i]]} for i in wrong[:n]]
