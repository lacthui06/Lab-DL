"""Hiển thị artifacts ngay trong kernel notebook, không chạy qua subprocess."""
from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown


def show_results(lab_dir=None):
    """Đọc các run được registry chọn; ảnh hiển thị chính là file trong outputs/."""
    lab = Path(lab_dir) if lab_dir else Path(__file__).resolve().parents[1]
    state = json.loads((lab / "config/workflow.json").read_text(encoding="utf-8"))
    runs_root = (lab / "outputs/runs").resolve()
    training_id = state.get("active_runs", {}).get("finetuning")
    inline_policy = bool(training_id and (runs_root / training_id / "decision_threshold.json").is_file())
    shown = []
    for kind in ["finetuning", "threshold_selection", "evaluation", "inference"]:
        if kind == "threshold_selection" and inline_policy:
            continue
        run_id = state.get("active_runs", {}).get(kind)
        if not run_id:
            continue
        run = (runs_root / run_id).resolve()
        if not run.is_relative_to(runs_root) or run.parent != runs_root:
            raise ValueError("Invalid run path in registry")
        if not run.is_dir():
            display(Markdown(f"Run `{run_id}` chưa có trong thư mục outputs."))
            continue
        status = run / "status.json"
        if not status.is_file() or json.loads(status.read_text(encoding="utf-8")).get("status") != "completed":
            display(Markdown(f"Run `{run_id}` chưa hoàn thành; không hiển thị kết quả."))
            continue
        display(Markdown(f"### {kind}\nNguồn: `{run_id}`"))
        policy_path = run / "decision_threshold.json"
        if kind in {"finetuning", "threshold_selection"} and policy_path.is_file():
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            display(Markdown(f"Ngưỡng chọn trên **validation**: **{policy['selected_threshold']}**"))
            display(pd.DataFrame([
                {"policy": "Reference 0.5", **policy["baseline_metrics"]},
                {"policy": "Validation-selected", **policy["selected_metrics"]},
            ]))
        for name in ["test_metrics.json", "test_threshold_metrics.json", "test_runtime_metrics.json", "predictions.json"]:
            path = run / name
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            # Exercise 1 lưu pipeline/manual dưới dict; chỉ bảng hóa records phù hợp.
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and all(not isinstance(v, (dict, list)) for v in data.values()):
                rows = [data]
            else:
                continue
            display(Markdown(f"**{name}**"))
            display(pd.DataFrame(rows))
        for path in sorted(run.rglob("*.png")):
            if any(part in {"best_model", "checkpoints"} for part in path.relative_to(run).parts):
                continue
            display(Markdown(f"**{path.relative_to(run).as_posix()}**"))
            display(Image(filename=str(path)))
            shown.append(path)
    if not shown:
        raise FileNotFoundError("No saved PNGs found in selected runs; check outputs/ and config/workflow.json")
    print(f"Đã hiển thị {len(shown)} hình từ outputs/. Không huấn luyện lại model.")
    return shown


if __name__ == "__main__":
    show_results()
