"""Trực quan Exercise1 và reliability bins dùng chung với modeling."""
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .evidence import read_json, audit_exercise1, verify_manifest
from .utils.runs import ROOT, hash_file, run_context, save_json
from .utils.plots import finish


def _table_image(rows, columns, path, title, widths=None):
    """Bảng PNG có nguồn JSON song hành; không chỉ hiển thị trong notebook."""
    widths = widths or [1 / len(columns)] * len(columns)
    wrap_widths = [max(12, int(82 * width)) for width in widths]
    body = []; row_lines = []
    for row in rows:
        cells = []
        line_count = 1
        for key, wrap_width in zip(columns, wrap_widths):
            value = row.get(key, "")
            if isinstance(value, float): value = f"{value:.5f}"
            rendered = textwrap.fill(str(value), wrap_width, break_long_words=True, break_on_hyphens=False)
            cells.append(rendered)
            line_count = max(line_count, rendered.count("\n") + 1)
        body.append(cells)
        row_lines.append(line_count)
    fig_height = max(3.2, .30 * (1.25 + sum(row_lines)) + 1.4)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=columns, loc="center", cellLoc="left", colWidths=widths)
    table.auto_set_font_size(False); table.set_fontsize(8.5)
    weights = [1.25, *[max(1, count) for count in row_lines]]
    scale = .88 / sum(weights)
    for (i, _), cell in table.get_celld().items():
        cell.set_height(scale * weights[i])
        cell.PAD = .025
        if i == 0: cell.set_facecolor("#cfe2f3")
        elif i % 2: cell.set_facecolor("#f3f6fa")
    ax.set_title(title, pad=18)
    fig.subplots_adjust(left=.025, right=.975, top=.88, bottom=.06)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def _cache_key(source_run):
    """Cache phụ thuộc manifest nguồn và code phân tích, không chỉ tên run."""
    files = [Path(__file__), Path(__file__).parent / "training/thresholds.py"]
    return {"source_run": Path(source_run).name,
            "source_manifest_sha256": hash_file(Path(source_run) / "manifest.json"),
            "analysis_code": {p.name: hash_file(p) for p in files}}


def _cached(source_run, kind, root):
    registry = Path(root) / "config/workflow.json"
    if not registry.exists(): return None
    state = read_json(registry)
    expected = _cache_key(source_run)
    for row in reversed(state.get("runs", [])):
        if row["kind"] != kind or row["status"] != "completed": continue
        path = Path(root) / "outputs/runs" / row["id"]
        if not (path / "analysis_source.json").exists(): continue
        if read_json(path / "analysis_source.json") == expected:
            verify_manifest(path)
            return path
    return None


def ensure_exercise1_analysis(source_run, root=ROOT):
    """Đọc cache hợp lệ hoặc tạo mới bảng token và hình inference từ run thật."""
    audit = audit_exercise1(source_run)
    cached = _cached(source_run, "exercise1_analysis", root)
    if cached: return cached
    source = Path(source_run)
    with run_context("exercise1_analysis", {"source_run": source.name}, root=root) as out:
        save_json(out / "analysis_source.json", _cache_key(source))
        raw = read_json(source / "tokenizer_breakdown.json")
        tokens = [dict(position=i, tokens=t, input_ids=idx, attention_mask=mask)
                  for i, (t, idx, mask) in enumerate(zip(raw["tokens"], raw["input_ids"], raw["attention_mask"]))]
        predictions = read_json(source / "predictions.json")
        comparison = [dict(text=b["text"], pipeline_label=a["label"], manual_label=b["label"],
                     pipeline_score=a["score"], manual_score=b["score"], difference=abs(a["score"]-b["score"]))
                     for a, b in zip(predictions["pipeline"], predictions["manual"])]
        summary = {"source_run": source.name, "sentence": raw["text"], "tokenization": tokens,
                   "predictions": predictions["manual"], "comparison": comparison, "audit": audit}
        save_json(out / "summary.json", summary)
        _table_image(tokens, ["position", "tokens", "input_ids", "attention_mask"],
                     out / "tokenizer_table.png", "WordPiece tokenizer breakdown")
        _table_image(predictions["manual"], ["text", "label", "score"],
                     out / "sentiment_predictions.png", "Pretrained sentiment predictions", [.74, .13, .13])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        positions = np.arange(len(comparison))
        axes[0].bar(positions-.16, [r["pipeline_score"] for r in comparison], .32, label="Pipeline")
        axes[0].bar(positions+.16, [r["manual_score"] for r in comparison], .32, label="Manual")
        axes[0].set(xticks=positions, xticklabels=[f"Sentence {i+1}" for i in range(len(comparison))],
                    ylabel="Predicted class probability", ylim=(0, 1.1)); axes[0].legend()
        axes[1].bar(positions, [r["difference"] for r in comparison])
        axes[1].axhline(1e-5, color="red", linestyle="--", label="Tolerance 1e-5")
        axes[1].set(xlabel="Sentence index", ylabel="Absolute score difference", ylim=(0, 1.2e-5)); axes[1].legend()
        finish(fig, out / "pipeline_manual_comparison.png")
        print("Saved Exercise 1 visual artifacts from:", source.name)
    return out


def _reliability(labels, probs):
    predictions = probs.argmax(1); confidence = probs.max(1)
    correct = (predictions == labels).astype(float)
    indices = np.minimum((confidence * 10).astype(int), 9)
    bins = []; ece = 0.
    for i in range(10):
        mask = indices == i; count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean = float(confidence[mask].mean()) if count else None
        if count: ece += count / len(labels) * abs(accuracy - mean)
        bins.append({"lower": i/10, "upper": (i+1)/10, "count": count,
                     "mean_confidence": mean, "accuracy": accuracy})
    return bins, ece
