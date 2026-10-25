"""Package the completed real run and independently verify exported evidence."""
from pathlib import Path
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import markdown
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.data import file_hash
from src.experiment import write_json

root = Path(__file__).resolve().parents[1]
run = Path(json.loads((root / "outputs" / "selected_run.json").read_text(encoding="utf-8"))["run"])
if not run.is_absolute():
    run = root / run
assert json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] == "completed"
cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
assert cfg.get('dataset') == 'CIFAR10', 'Refuse to postprocess a legacy dataset with CIFAR10 code'
assets = {}
for weights_path in (root / 'data' / 'pretrained' / 'checkpoints').glob('*.pth'):
    digest = file_hash(weights_path)
    expected = weights_path.stem.rsplit('-', 1)[-1]
    assert digest.startswith(expected), f'Official pretrained hash prefix mismatch: {weights_path}'
    assets[weights_path.name] = {'sha256': digest, 'official_filename_prefix_verified': True}
archive = root / 'data' / 'cifar-10-python.tar.gz'
if archive.exists():
    with archive.open('rb') as stream:
        md5 = hashlib.file_digest(stream, 'md5').hexdigest()
    assert md5 == 'c58f30108f718f92721af3b95e74349a'
    assets[archive.name] = {'md5': md5, 'official_checksum_verified': True}
write_json(run / 'asset_verification.json', assets)
comparison = pd.read_csv(run / "comparison.csv")
assert len(comparison) == len(cfg["models"]) * 2
selected = dict(np.load(run / "selected_indices.npz"))
full_split = dict(np.load(run / "split_indices.npz"))
assert not np.intersect1d(full_split["train"], full_split["val"]).size
assert len(np.union1d(full_split["train"], full_split["val"])) == 50000
assert not np.intersect1d(selected["train"], selected["val"]).size
notes = ['### Dữ liệu RGB và phân bố lớp',
         '![Ảnh train theo lớp](eda/training_examples.png)']
class_counts = pd.read_csv(run / 'eda' / 'class_counts.csv', index_col=0)
count_table = ['| Class | Train | Validation | Test |', '|---|---:|---:|---:|']
for label, counts_row in class_counts.iterrows():
    count_table.append(f"| {label} | {int(counts_row['train'])} | {int(counts_row['val'])} | {int(counts_row['test'])} |")
notes.append('\n'.join(count_table))
from torchvision.datasets import CIFAR10
official_labels = np.asarray(CIFAR10(str(root / 'data'), train=False).targets)
for name in cfg["models"]:
    for stage in ("head_selected", "finetune"):
        folder = run / name / stage
        prediction = np.load(folder / "test_predictions.npz")
        labels, probs = prediction["labels"], prediction["probabilities"]
        np.testing.assert_array_equal(prediction["indices"], selected["test"])
        np.testing.assert_array_equal(labels, official_labels[selected['test']])
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
        assert abs(accuracy_score(labels, probs.argmax(1)) - metrics["accuracy"]) < 1e-10
        assert abs(f1_score(labels, probs.argmax(1), average="macro") - metrics["macro_f1"]) < 1e-10
        assert json.loads((folder / "roundtrip.json").read_text(encoding="utf-8"))["passed"]
        counts = pd.read_csv(folder / "confusion_counts.csv", index_col=0)
        assert counts.to_numpy().sum() == len(labels)
        offdiag = counts.to_numpy().copy()
        np.fill_diagonal(offdiag, 0)
        a, b = np.unravel_index(offdiag.argmax(), offdiag.shape)
        notes.append(f"- {name}/{stage}: cặp lỗi nhiều nhất {counts.index[a]} → {counts.columns[b]} ({offdiag[a,b]} ảnh trong tập test đã chọn).")
        per_class = pd.read_csv(folder / 'classification_report.csv', index_col=0).loc[counts.index]
        weakest = per_class.sort_values('f1-score').head(3)
        weak_table = [f'**Ba lớp F1 thấp nhất — {name}/{stage}:**', '',
                      '| Class | Precision | Recall | F1 |', '|---|---:|---:|---:|']
        for label, scores in weakest.iterrows():
            weak_table.append(f"| {label} | {scores['precision']:.4f} | {scores['recall']:.4f} | {scores['f1-score']:.4f} |")
        notes.append('\n'.join(weak_table))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        curve_metrics = []
        for c, label in enumerate(counts.index):
            target = labels == c
            fpr, tpr, _ = roc_curve(target, probs[:, c])
            precision, recall, _ = precision_recall_curve(target, probs[:, c])
            area = auc(fpr, tpr)
            ap = average_precision_score(target, probs[:, c])
            axes[0].plot(fpr, tpr, label=f'{label}: {area:.3f}')
            axes[1].plot(recall, precision, label=f'{label}: {ap:.3f}')
            curve_metrics.append({'class': label, 'roc_auc_ovr': area, 'average_precision_ovr': ap})
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=.4)
        axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC one-vs-rest')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision-recall one-vs-rest')
        for ax in axes:
            ax.legend(fontsize=7, loc='lower left')
            ax.grid(alpha=.2)
        fig.suptitle(f'CIFAR-10 / {name} / {stage}')
        fig.tight_layout()
        fig.savefig(folder / 'roc_pr.png', dpi=150)
        plt.close(fig)
        pd.DataFrame(curve_metrics).to_csv(folder / 'roc_pr_metrics.csv', index=False)
    search = pd.read_csv(run / name / "search_results.csv")
    chosen = json.loads((run / name / "selection.json").read_text(encoding="utf-8"))["head"]
    assert chosen["trial"] == int(search.loc[search.best_val_accuracy.idxmax(), "trial"])
    pivot = search.pivot(index="lr", columns="batch_size", values="best_val_accuracy")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns,
           yticks=range(len(pivot.index)), yticklabels=[f"{lr:g}" for lr in pivot.index],
           xlabel="Batch size", ylabel="Learning rate", title=f"{name}: head search (best validation accuracy)")
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i,j]:.2%}", ha="center", va="center", color="white" if pivot.iloc[i,j] > .55 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(run / name / "hyperparameter_search.png", dpi=150)
    plt.close(fig)
    notes.extend([f"\n### {name}", f"![Learning curves]({name}/learning_curves.png)",
                  f"![Hyperparameter search]({name}/hyperparameter_search.png)",
                  f"![Confusion matrix]({name}/finetune/confusion_matrix.png)",
                  f"![Prediction examples]({name}/finetune/prediction_examples.png)",
                  f"![ROC và PR]({name}/finetune/roc_pr.png)"])
scalars = []
for event_dir in sorted({p.parent for p in run.rglob("events.out.tfevents.*")}):
    accumulator = EventAccumulator(str(event_dir)).Reload()
    if "train_loss" in accumulator.Tags()["scalars"]:
        assert len(accumulator.Scalars("train_loss")) > 0
        scalars.append(str(event_dir.relative_to(run)))
assert len(scalars) >= len(cfg["models"]) * (len(cfg["learning_rates"])*len(cfg["batch_sizes"])+1)
# Reconstruct visible monitoring evidence from the actual TensorBoard event stream.
fig, axes = plt.subplots(2, len(cfg['models']), figsize=(12, 7), squeeze=False)
tensorboard_records = []
costs = []
for column, name in enumerate(cfg['models']):
    chosen = json.loads((run / name / 'selection.json').read_text(encoding='utf-8'))['head']
    for stage, directory in [('head', f"head_trial_{chosen['trial']:02d}"), ('finetune', 'finetune')]:
        accumulator = EventAccumulator(str(run / name / directory / 'tensorboard')).Reload()
        for row, metric in enumerate(['loss', 'accuracy']):
            for split in ['train', 'val']:
                tag = f'{split}_{metric}'
                entries = accumulator.Scalars(tag)
                axes[row, column].plot([e.step for e in entries], [e.value for e in entries], marker='o',
                                       label=f'{stage}/{split}')
                for entry in entries:
                    tensorboard_records.append({'model': name, 'stage': stage, 'tag': tag,
                                                'epoch': entry.step, 'value': entry.value})
            axes[row, column].set(title=f'{name}: {metric}', xlabel='Epoch within stage')
            axes[row, column].legend(fontsize=8)
            axes[row, column].grid(alpha=.2)
    summary = json.loads((run / name / 'head_selected' / 'training_summary.json').read_text(encoding='utf-8'))
    shared = summary['feature_extraction_train_val_seconds'] + summary['all_search_training_seconds']
    for stage in ['head', 'finetune']:
        record = comparison[(comparison.model == name) & (comparison.stage == stage)].iloc[0]
        costs.append({'model': name, 'stage': stage,
                      'execution': ('pooled_feature_cache' if stage == 'head' else
                                    ('cached_frozen_prefix' if (run / name / 'prefix_equivalence.json').exists() else 'full_image_forward')),
                      'feature_extraction_seconds': summary['feature_extraction_train_val_seconds'],
                      'all_head_trials_fit_seconds': summary['all_search_training_seconds'],
                      'selected_stage_fit_seconds': record.train_seconds,
                      'pipeline_training_seconds': shared + (record.train_seconds if stage == 'finetune' else 0),
                      'test_seconds': record.test_seconds, 'trainable_parameters': int(record.trainable)})
fig.suptitle('Monitoring evidence reconstructed from TensorBoard events (not a UI screenshot)')
fig.tight_layout()
fig.savefig(run / 'tensorboard_evidence.png', dpi=150)
plt.close(fig)
pd.DataFrame(tensorboard_records).to_csv(run / 'tensorboard_scalars.csv', index=False)
pd.DataFrame(costs).to_csv(run / 'cost_comparison.csv', index=False)
winner = comparison.loc[comparison.best_val_accuracy.idxmax()]
write_json(run / 'validation_recommendation.json', {'model': winner.model, 'stage': winner.stage,
    'criterion': 'maximum validation accuracy; stable first row on tie; no test-based selection',
    'validation_accuracy': winner.best_val_accuracy})
notes.extend(['### TensorBoard và chi phí', '![TensorBoard scalars](tensorboard_evidence.png)',
    'Hình này dựng từ TensorBoard events thật; không phải ảnh chụp giao diện. Mở TensorBoard theo README để xem tương tác.',
    f'Điều kiện được đề xuất bằng validation: **{winner.model}/{winner.stage}**, validation accuracy **{winner.best_val_accuracy:.2%}**.',
    'Chi phí pipeline gồm feature extraction, toàn bộ head search và fine-tune nếu có; chưa gồm EDA, ghi báo cáo và inference test. Xem cost_comparison.csv.'])
cost_table = ['| Model | Stage | Feature extraction (s) | All head search (s) | Pipeline train (s) | Test (s) |',
              '|---|---|---:|---:|---:|---:|']
for cost in costs:
    cost_table.append(f"| {cost['model']} | {cost['stage']} | {cost['feature_extraction_seconds']:.1f} | {cost['all_head_trials_fit_seconds']:.1f} | {cost['pipeline_training_seconds']:.1f} | {cost['test_seconds']:.1f} |")
notes.append('\n'.join(cost_table))
notes.append('### Cấu hình head được chọn bằng validation')
head_table = ['| Model | LR | Batch | Best epoch | Validation accuracy |', '|---|---:|---:|---:|---:|']
for name in cfg['models']:
    choice = json.loads((run / name / 'selection.json').read_text(encoding='utf-8'))['head']
    head_table.append(f"| {name} | {choice['lr']:g} | {choice['batch_size']} | {choice['best_epoch']} | {choice['best_val_accuracy']:.2%} |")
notes.append('\n'.join(head_table))
for name in cfg['models']:
    proof_path = run / name / 'prefix_equivalence.json'
    if proof_path.exists():
        proof = json.loads(proof_path.read_text(encoding='utf-8'))
        assert proof['passed']
        notes.append(f"{name} fine-tune sử dụng cache đầu ra của prefix đóng băng; đã kiểm chứng tương đương trên {proof['samples']} ảnh thật, max logit error={proof['max_logit_error']:.3g}. Test vẫn forward toàn bộ model trên ảnh RGB. Kiểm thử riêng đối chiếu cả logits, gradients và schema checkpoint.")
if any((run / name / 'prefix_equivalence.json').exists() for name in cfg['models']):
    notes.append('Chi phí giữa các kiến trúc chịu ảnh hưởng của cách thực thi: xem cột execution trong cost_comparison.csv. Trong run này, nếu một model dùng cache prefix còn model khác forward ảnh khi fine-tune, không dùng thời gian thô để kết luận riêng về tốc độ kiến trúc. Cache giữ nguyên bài toán tối ưu và số ảnh/epoch; mã hiện hành hỗ trợ cache prefix cho cả hai model ở profile CPU.')
notes.extend(['### Diễn giải chỉ số và thiết kế',
    'Accuracy là số ảnh dự đoán đúng chia tổng số ảnh. Macro F1 tính F1 riêng cho từng lớp rồi lấy trung bình không trọng số; vì vậy mỗi lớp đóng góp ngang nhau. Macro Precision/Recall và cross-entropy test được lưu trong metrics.json. ROC-AUC và average precision tính one-vs-rest từng lớp, không phải cùng một đại lượng.',
    'ResNet dùng residual connections để cộng tín hiệu qua các block; DenseNet nối các feature maps từ nhiều lớp trong dense block. Thực nghiệm thay head và giữ cấu trúc backbone gốc, không thiết kế lại CNN từ đầu.',
    'CPU profile chỉ có 2 epoch fine-tune sau head search: kết quả chứng minh pipeline và hành vi ở ngân sách đã chọn, không chứng minh đã hội tụ hoặc đạt chất lượng tối ưu. Chưa chạy nhiều seed, augmentation ablation hay baseline khởi tạo ngẫu nhiên.'])
report_path = run / "report_vi.md"
text = report_path.read_text(encoding="utf-8")
text = text.split("\n## 8. Khảo sát lỗi và hình bổ sung")[0]
text += "\n## 8. Khảo sát lỗi và hình bổ sung\n\n" + "\n\n".join(notes)
report_path.write_text(text, encoding="utf-8")

def embed_image(match):
    relative = match.group(2)
    encoded = base64.b64encode((run / relative).read_bytes()).decode("ascii")
    return f'<img alt="{match.group(1)}" src="data:image/png;base64,{encoded}">'

embedded = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", embed_image, text)
body = markdown.markdown(embedded, extensions=["tables", "fenced_code"])
html = """<!DOCTYPE html><html lang="vi"><meta charset="utf-8"><title>Lab 02 — Báo cáo</title>
<style>body{font:16px/1.65 system-ui,sans-serif;color:#172033;max-width:1120px;margin:40px auto;padding:0 24px}h1,h2,h3{line-height:1.3}h1{color:#174ea6}h2{border-bottom:1px solid #dbe2ef;padding-bottom:8px;margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #dbe2ef;text-align:left;padding:9px}th{background:#edf3fc}img{max-width:100%;height:auto}code{background:#edf2f7;padding:2px 4px}a{color:#174ea6}@media print{body{margin:0;font-size:11pt}h2{break-after:avoid}img,table{break-inside:avoid}}</style><body>""" + body + "</body></html>"
(run / "report_vi.html").write_text(html, encoding="utf-8")
lock = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(root / "requirements-lock-windows.txt").write_text(lock, encoding="utf-8")
snapshot = run / "source_snapshot"
for section in ("src", "config", "scripts", "tests"):
    for source in (root / section).glob("*"):
        if source.is_file() and source.suffix in (".py", ".json"):
            dest = snapshot / source.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(source, dest)
write_json(run / "verification.json", {"status": "passed", "metric_recomputation": True,
    "confusion_totals": True, "validation_selection": True, "roundtrips": len(cfg["models"]) * 2,
    "tensorboard_scalar_runs": scalars,
    "tests": "See pytest_validation.log for actual test execution"})
original_hashes = json.loads((run / 'source_hashes.json').read_text(encoding='utf-8'))
assert all(file_hash(run / 'source_snapshot' / p) == digest for p, digest in original_hashes.items())
resume_hash_path = run / 'resume_source_hashes.json'
if resume_hash_path.exists():
    resume_hashes = json.loads(resume_hash_path.read_text(encoding='utf-8'))
    assert all(file_hash(run / 'resume_source_snapshot' / p) == digest for p, digest in resume_hashes.items())
delivery_hashes = {str(p.relative_to(root)): file_hash(p)
    for section in ['src', 'scripts', 'tests', 'config'] for p in (root / section).glob('*') if p.is_file()}
write_json(run / 'delivery_source_hashes.json', delivery_hashes)
for relative in delivery_hashes:
    destination = run / 'delivery_source_snapshot' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / relative, destination)
manifest = {str(p.relative_to(run)): file_hash(p) for p in run.rglob("*")
            if p.is_file() and p.name != "manifest.json"}
write_json(run / "manifest.json", manifest)
assert all(file_hash(run / p) == digest for p, digest in manifest.items())
summary_rows = ['| Model | Stage | Test Accuracy | Macro F1 |', '|---|---|---:|---:|']
for _, result_row in comparison.iterrows():
    summary_rows.append(f'| {result_row.model} | {result_row.stage} | {result_row.accuracy:.2%} | {result_row.macro_f1:.4f} |')
(root / "RESULTS.md").write_text(
    "# Lab 02 — Kết quả đã chạy\n\n"
    f"Run: `{run.name}`. Profile: **{cfg['profile']}**. Số mẫu xem trong data_summary.json và báo cáo.\n\n"
    f"- [Notebook EDA và kiến trúc](notebooks/01_cifar10_eda_architecture.ipynb)\n"
    f"- [Notebook huấn luyện và đánh giá](notebooks/lab_02_transfer_learning.ipynb)\n"
    f"- [Báo cáo có hình](outputs/runs/{run.name}/report_vi.html)\n"
    f"- [Báo cáo Markdown](outputs/runs/{run.name}/report_vi.md)\n"
    f"- [So sánh kết quả](outputs/runs/{run.name}/comparison.csv)\n"
    f"- [Kiểm chứng](outputs/runs/{run.name}/verification.json)\n\n"
    f"Dữ liệu: CIFAR-10; đầu vào {cfg['image_size']}×{cfg['image_size']}. Kết quả chỉ áp dụng cho cấu hình đã ghi. Đọc README để tái lập.\n\n"
    + '\n'.join(summary_rows) + '\n', encoding="utf-8")
print(comparison[["model", "stage", "accuracy", "macro_f1"]].to_string(index=False))
print("Verified manifest:", len(manifest), "files")
print("HTML report:", run / "report_vi.html")
