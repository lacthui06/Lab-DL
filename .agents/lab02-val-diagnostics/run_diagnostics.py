"""Read-only validation inference; all generated files stay in this task folder."""
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / 'labs/lab02'
sys.path.insert(0, str(LAB))
from src.datasets.classification import image_loader  # noqa: E402
from src.datasets.eda import CLASSES  # noqa: E402
from src.training.trainer import predict, seed_all  # noqa: E402
from src.models.backbones import build_model  # noqa: E402

RUN = LAB / 'outputs/runs/20261027T083044Z_cifar10_macintel_cpu_control64_db231b28'
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_indices(val, train, pool_size=50000, expected=5000):
    val, train = np.asarray(val), np.asarray(train)
    for ids in (val, train):
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Indices must be a one-dimensional integer array')
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= pool_size):
            raise ValueError('Duplicate or out-of-range indices')
    if len(val) != expected or np.intersect1d(val, train).size:
        raise ValueError('Wrong validation size or overlap with train')


def summarize(probabilities, labels):
    p, y = np.asarray(probabilities), np.asarray(labels)
    if y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('Labels must be nonempty integer vector')
    if np.any(y < 0) or np.any(y > 9) or p.shape != (len(y), 10):
        raise ValueError('Invalid labels or probability shape')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Invalid probability values')
    if not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError('Probability rows must sum to one')
    pred = p.argmax(1)
    # float64 log, tiny floor only for exact softmax underflow to zero.
    losses = -np.log(np.maximum(p[np.arange(len(y)), y].astype(np.float64), np.finfo(float).tiny))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(10), zero_division=0)
    counts = confusion_matrix(y, pred, labels=np.arange(10))
    fp = counts.sum(0) - counts.diagonal()
    negatives = len(y) - counts.sum(1)
    per_class = pd.DataFrame(dict(class_name=CLASSES, precision=precision,
                                 recall_tpr=recall, f1=f1, support=support,
                                 fpr=np.divide(fp, negatives, out=np.zeros(10), where=negatives != 0)))
    metrics = dict(samples=len(y), accuracy=float(np.mean(pred == y)), loss=float(losses.mean()),
                   macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                   macro_f1=float(f1.mean()))
    return metrics, per_class, counts, pred, losses


def check_history(metrics, history, epoch):
    row = history.loc[history.epoch == epoch]
    if len(row) != 1:
        raise ValueError('Checkpoint epoch missing or duplicated in history')
    row = row.iloc[0]
    differences = dict(accuracy=metrics['accuracy'] - float(row.val_accuracy),
                       loss=metrics['loss'] - float(row.val_loss))
    if abs(differences['accuracy']) > 1e-7 or abs(differences['loss']) > 1e-5:
        raise ValueError(f'Checkpoint/history mismatch: {differences}')
    return differences


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figures(folder, name, counts, pairs, records, raw, history, epoch):
    percent = counts / np.maximum(counts.sum(1, keepdims=True), 1) * 100
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, values, title, maximum in zip(axes, (counts, percent), ('Counts', 'Within true class (%)'), (None, 100)):
        im = ax.imshow(values, cmap='Blues', vmin=0, vmax=maximum)
        for i in range(10):
            for j in range(10):
                value = f'{values[i,j]:.1f}' if maximum else str(values[i,j])
                ax.text(j, i, value, ha='center', va='center', fontsize=7,
                        color='white' if values[i,j] > (maximum or counts.max()) * .55 else 'black')
        ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted label', ylabel='True label', title=f'{name}: validation / {title}')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.colorbar(im, ax=ax, fraction=.046)
    save(fig, folder / 'confusion_matrix.png')
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f'{r.true_class} -> {r.predicted_class}' for r in pairs.itertuples()]
    ax.barh(labels[::-1], pairs['count'].to_numpy()[::-1])
    for i, count in enumerate(pairs['count'].to_numpy()[::-1]):
        ax.text(count + .5, i, str(count), va='center')
    ax.set(xlabel='Misclassified validation images', title=f'{name}: top directed confusions')
    ax.margins(x=.15)
    save(fig, folder / 'top_confusions.png')
    wrong = records.loc[records.true_id != records.predicted_id].sort_values(['loss', 'index'], ascending=[False, True])
    groups = [('highest_loss_errors', wrong.head(20))]
    pair_rows = []
    for pair in pairs.head(5).itertuples():
        pair_rows.append(wrong.loc[(wrong.true_class == pair.true_class) & (wrong.predicted_class == pair.predicted_class)].head(4))
    groups.append(('top_pair_examples', pd.concat(pair_rows) if pair_rows else wrong.head(0)))
    for title, chosen in groups:
        chosen.to_csv(folder / f'{title}.csv', index=False)
        fig, axes = plt.subplots(4, 5, figsize=(16, 13))
        for ax in axes.flat:
            ax.axis('off')
        for ax, r in zip(axes.flat, chosen.itertuples()):
            ax.imshow(raw.data[r.index], interpolation='nearest')
            ax.set_title(f'#{r.index}: {r.true_class} -> {r.predicted_class}\n'
                         f'p(true)={r.p_true:.3g}; p(pred)={r.p_predicted:.4f}\n'
                         f'loss={r.loss:.2f}; margin={r.top2_margin:.3f}', fontsize=8)
        fig.suptitle(f'{name}: validation / {title}', fontsize=14)
        save(fig, folder / f'{title}.png')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    best_loss_epoch = int(history.loc[history.val_loss.idxmin(), 'epoch'])
    for ax, metric in zip(axes, ('loss', 'accuracy')):
        ax.plot(history.epoch, history[f'train_{metric}'], marker='o', label='Train (online)')
        ax.plot(history.epoch, history[f'val_{metric}'], marker='o', label='Validation')
        ax.axvline(epoch, color='black', linestyle='--', label=f'Selected epoch {epoch}')
        ax.axvline(best_loss_epoch, color='green', linestyle=':', label=f'Min val loss epoch {best_loss_epoch}')
        ax.set(xlabel='Epoch', ylabel=metric, title=f'{name}: {metric}')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    save(fig, folder / 'learning_curves.png')


def main():
    start = time.monotonic()
    cfg = json.loads((RUN / 'config.json').read_text())
    with np.load(RUN / 'selected_indices.npz', allow_pickle=False) as selected:
        val, train = selected['val'], selected['train']
    validate_indices(val, train)
    raw = CIFAR10(str(LAB / 'data'), train=True, download=False)
    protected = [RUN / 'config.json', RUN / 'selected_indices.npz', LAB / 'outputs/selected_run.json']
    for name in ('resnet18', 'densenet121'):
        protected.extend([RUN / name / 'finetune' / f for f in ('best.pt', 'history.csv', 'config.json')])
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    execution = dict(run=str(RUN), split='validation', device='cpu', before_sha256=before, models={})
    torch.set_num_threads(cfg['threads'])
    seed_all(cfg['seed'])
    for name in ('resnet18', 'densenet121'):
        print(f'{name}: infer 5000 validation images', flush=True)
        stage = RUN / name / 'finetune'
        checkpoint = torch.load(stage / 'best.pt', map_location='cpu', weights_only=True)
        metadata = checkpoint['metadata']
        if metadata['model'] != name or metadata['image_size'] != cfg['image_size']:
            raise ValueError('Checkpoint metadata differs from run')
        if metadata['data_sha256'] != digest(RUN / 'selected_indices.npz'):
            raise ValueError('Checkpoint split hash mismatch')
        model = build_model(name, pretrained=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        loader = image_loader(LAB / 'data', 'val', val, name, metadata['batch_size'],
                              cfg['seed'], image_size=cfg['image_size'])
        p, y, indices = predict(model, loader, torch.device('cpu'))
        if not np.array_equal(indices, val) or not np.array_equal(y, np.asarray(raw.targets)[val]):
            raise ValueError('Prediction indices/labels differ from validation pool')
        metrics, per_class, counts, pred, losses = summarize(p, y)
        history = pd.read_csv(stage / 'history.csv')
        differences = check_history(metrics, history, checkpoint['best_epoch'])
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / 'validation_predictions.npz', probabilities=p, labels=y, indices=indices)
        sorted_p = np.sort(p, axis=1)
        records = pd.DataFrame(dict(index=indices, true_id=y, predicted_id=pred,
                                    true_class=np.asarray(CLASSES)[y], predicted_class=np.asarray(CLASSES)[pred],
                                    p_true=p[np.arange(len(y)), y], p_predicted=p.max(1),
                                    top2_margin=sorted_p[:, -1] - sorted_p[:, -2], loss=losses))
        records.to_csv(folder / 'per_image.csv', index=False)
        per_class.to_csv(folder / 'per_class.csv', index=False)
        pd.DataFrame(counts, index=CLASSES, columns=CLASSES).to_csv(folder / 'confusion_counts.csv')
        pd.DataFrame(counts / counts.sum(1, keepdims=True) * 100, index=CLASSES, columns=CLASSES).to_csv(folder / 'confusion_row_percent.csv')
        pairs = [(i, j, int(counts[i,j])) for i in range(10) for j in range(10) if i != j and counts[i,j]]
        pairs.sort(key=lambda r: (-r[2], r[0], r[1]))
        top = pd.DataFrame([dict(true_class=CLASSES[i], predicted_class=CLASSES[j], count=c,
                                 percent_true_class=100*c/counts[i].sum()) for i,j,c in pairs[:10]])
        top.to_csv(folder / 'top_confusions.csv', index=False)
        wrong = records.loc[y != pred]
        metrics.update(selected_epoch=int(checkpoint['best_epoch']), history_difference=differences,
                       wrong_images=len(wrong), wrong_mean_predicted_probability=float(wrong.p_predicted.mean()),
                       wrong_median_predicted_probability=float(wrong.p_predicted.median()),
                       wrong_median_top2_margin=float(wrong.top2_margin.median()))
        (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        figures(folder, name, counts, top, records, raw, history, checkpoint['best_epoch'])
        execution['models'][name] = metrics
        print(f'{name}: accuracy={metrics["accuracy"]:.4f}, loss={metrics["loss"]:.6f}; history matched', flush=True)
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    if before != after:
        raise RuntimeError('Protected input changed during diagnosis')
    execution.update(protected_inputs_unchanged=True, elapsed_seconds=time.monotonic()-start,
                     torch_version=torch.__version__, python_version=sys.version)
    (OUT / 'execution.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
