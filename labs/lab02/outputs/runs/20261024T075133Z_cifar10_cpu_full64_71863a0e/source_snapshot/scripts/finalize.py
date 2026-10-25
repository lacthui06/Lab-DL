"""Package the completed real run and independently verify exported evidence."""
from pathlib import Path
import base64
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
from sklearn.metrics import accuracy_score, f1_score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.data import file_hash
from src.experiment import write_json

root = Path(__file__).resolve().parents[1]
run = Path(json.loads((root / "outputs" / "selected_run.json").read_text(encoding="utf-8"))["run"])
assert json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] == "completed"
cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
comparison = pd.read_csv(run / "comparison.csv")
assert len(comparison) == len(cfg["models"]) * 2
selected = dict(np.load(run / "selected_indices.npz"))
full_split = dict(np.load(run / "split_indices.npz"))
assert not np.intersect1d(full_split["train"], full_split["val"]).size
assert len(np.union1d(full_split["train"], full_split["val"])) == 50000
assert not np.intersect1d(selected["train"], selected["val"]).size
notes = []
for name in cfg["models"]:
    for stage in ("head_selected", "finetune"):
        folder = run / name / stage
        prediction = np.load(folder / "test_predictions.npz")
        labels, probs = prediction["labels"], prediction["probabilities"]
        np.testing.assert_array_equal(prediction["indices"], selected["test"])
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
                  f"![Prediction examples]({name}/finetune/prediction_examples.png)"])
scalars = []
for event_dir in sorted({p.parent for p in run.rglob("events.out.tfevents.*")}):
    accumulator = EventAccumulator(str(event_dir)).Reload()
    if "train_loss" in accumulator.Tags()["scalars"]:
        assert len(accumulator.Scalars("train_loss")) > 0
        scalars.append(str(event_dir.relative_to(run)))
assert len(scalars) >= len(cfg["models"]) * (len(cfg["learning_rates"])*len(cfg["batch_sizes"])+1)
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
manifest = {str(p.relative_to(run)): file_hash(p) for p in run.rglob("*")
            if p.is_file() and p.name != "manifest.json"}
write_json(run / "manifest.json", manifest)
assert all(file_hash(run / p) == digest for p, digest in manifest.items())
(root / "RESULTS.md").write_text(
    "# Lab 02 — Kết quả đã chạy\n\n"
    f"Run: `{run.name}`. Profile: **{cfg['profile']}**. Số mẫu xem trong data_summary.json và báo cáo.\n\n"
    f"- [Notebook](notebooks/lab_02_transfer_learning.ipynb)\n"
    f"- [Báo cáo có hình](outputs/runs/{run.name}/report_vi.html)\n"
    f"- [Báo cáo Markdown](outputs/runs/{run.name}/report_vi.md)\n"
    f"- [So sánh kết quả](outputs/runs/{run.name}/comparison.csv)\n"
    f"- [Kiểm chứng](outputs/runs/{run.name}/verification.json)\n\n"
    f"Dữ liệu: CIFAR-10; đầu vào {cfg['image_size']}×{cfg['image_size']}. Kết quả chỉ áp dụng cho cấu hình đã ghi. Đọc README để tái lập.\n", encoding="utf-8")
print(comparison[["model", "stage", "accuracy", "macro_f1"]].to_string(index=False))
print("Verified manifest:", len(manifest), "files")
print("HTML report:", run / "report_vi.html")
