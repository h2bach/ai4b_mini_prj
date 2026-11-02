"""Hai baseline hoàn chỉnh, chỉ fit train; không đọc test tại bước này."""
from pathlib import Path
from time import perf_counter
import warnings
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from shared.data import normalize, normalized_images
from shared.metrics import classification_metrics, save_predictions
from shared.utils import read_json, write_json


def baseline_inputs(data, split):
    x = data[split]['x']
    if data['project'] == 'cnn':
        return normalized_images(x, data['preprocessing']).reshape(len(x), -1)
    return x


def run_baselines(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    existing = out / 'baselines.json'
    if existing.exists():
        result = read_json(existing)
        assert result['fingerprint'] == data['fingerprint']
        return result
    train_x, val_x = [baseline_inputs(data, s) for s in ['train', 'validation']]
    y, val_y = data['train']['y'], data['validation']['y']
    dummy_x = np.zeros((len(y), 1))
    models = {'majority': DummyClassifier(strategy='most_frequent')}
    if data['project'] == 'cnn':
        models['logreg'] = LogisticRegression(C=.1, solver='lbfgs', max_iter=1000, random_state=0)
    else:
        models['logreg'] = make_pipeline(
            TfidfVectorizer(preprocessor=normalize, lowercase=False, ngram_range=(1, 2), min_df=2),
            LogisticRegression(C=1, solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=0))
    result = {'project': data['project'], 'fingerprint': data['fingerprint'], 'subset': data['subset'],
              'preprocessing': data['preprocessing'], 'models': {}}
    for name, model in models.items():
        start = perf_counter()
        with warnings.catch_warnings(), threadpool_limits(limits=4):
            # Không công bố kết quả từ baseline chưa hội tụ.
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(dummy_x if name == 'majority' else train_x, y)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        pred = model.predict(np.zeros((len(val_y), 1)) if name == 'majority' else val_x)
        predict_seconds = perf_counter() - start
        metrics = classification_metrics(val_y, pred, data['labels'])
        result['models'][name] = {'validation': metrics, 'fit_seconds': fit_seconds,
                                  'predict_seconds': predict_seconds, 'n_iter': str(getattr(model, 'n_iter_', 'pipeline'))}
        save_predictions(out / f'{name}_validation.csv', data['validation']['ids'], val_y, pred)
        joblib.dump(model, out / f'{name}.joblib')
        print(name, 'validation macro-F1:', round(metrics['macro_f1'], 5), 'fit seconds:', round(fit_seconds, 2), flush=True)
    write_json(existing, result)
    return result
