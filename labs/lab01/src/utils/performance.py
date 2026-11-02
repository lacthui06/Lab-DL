"""Model descriptions and bounded training benchmarks on disposable checkpoints.

RAM is sampled process RSS, not Python-only tracemalloc memory. CUDA peak memory
counts allocated tensors, not reserved memory or total GPU usage. References:
https://psutil.readthedocs.io/stable/#psutil.Process.memory_info
https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil
import torch

from ..models.mlp import FashionMLP
from ..training.trainer import build_optimizer
from .helpers import SOURCE_DIR, set_seed
from .runs import Run, write_json


def show_model_architecture(config: dict, name: str, root: Path | None = None) -> Path:
    """Print and persist the configured model without consuming the caller's RNG."""
    from IPython.display import display
    cfg = config["models"][name]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        model = FashionMLP(in_features=cfg["in_features"], hidden_dims=cfg["hidden_dims"])
    rows = []
    previous = cfg["in_features"]
    for index, layer in enumerate(model.net):
        following = layer.out_features if isinstance(layer, torch.nn.Linear) else previous
        rows.append(dict(layer=index, type=type(layer).__name__, input_width=previous,
                         output_width=following, parameters=sum(p.numel() for p in layer.parameters())))
        previous = following
    with Run("architecture_" + name, {"name": name, **cfg}, root=root) as run:
        print(name)
        print(model)
        print("Total parameters:", sum(p.numel() for p in model.parameters()))
        (run.path / "model_architecture.txt").write_text(str(model), encoding="utf-8")
        pd.DataFrame(rows).to_csv(run.path / "layers.csv", index=False)
        display(pd.DataFrame(rows))
    return run.path


def _measure_worker(model_run: Path, data_path: Path, settings: dict) -> dict:
    """Measure one training batch in a disposable process; never save modified weights.

    Batch tensors are already on device. Sampling memory and timing use separate
    passes so the RSS polling thread does not perturb the timed repeats.
    """
    model_run = Path(model_run)
    cfg = json.loads((model_run / "config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(settings["threads"])
    set_seed(cfg["seed"])
    device = torch.device(settings["device"])
    batch_size = settings["batch_size"]
    with np.load(data_path) as arrays:
        x = arrays["train_pixels"][:batch_size].copy()
        y = arrays["train_labels"][:batch_size].copy()
        if cfg["name"] == "MainMLP_FE":
            x = np.concatenate((x, arrays["train_features"][:batch_size]), axis=1)
    if len(x) != batch_size:
        raise ValueError("Prepared training split is smaller than the benchmark batch")
    state = torch.load(model_run / "best.pt", map_location=device, weights_only=True)
    # Older runs omit architecture fields; recover exact widths from their weights.
    linear = [value for key, value in state["model_state_dict"].items()
              if key.endswith(".weight") and value.ndim == 2]
    model = FashionMLP(in_features=cfg.get("in_features", linear[0].shape[1]),
                       hidden_dims=cfg.get("hidden_dims", [w.shape[0] for w in linear[:-1]])).to(device)
    model.load_state_dict(state["model_state_dict"])
    del state
    images = torch.from_numpy(x).float().to(device)
    targets = torch.from_numpy(y).long().to(device)
    optimizer = build_optimizer(cfg["optimizer"], model, learning_rate=cfg["learning_rate"],
                                weight_decay=cfg.get("weight_decay", 0.), momentum=cfg.get("momentum", .9))
    model.train()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(images), targets)
        loss.backward()
        optimizer.step()

    gc.collect()
    synchronize()
    process = psutil.Process()
    baseline = process.memory_info().rss
    sampled = [baseline]
    stop = threading.Event()

    def sample_rss():
        while not stop.wait(.001):
            sampled.append(process.memory_info().rss)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        step()  # Include the first allocation of optimizer state in the memory pass.
        synchronize()
        sampled.append(process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    ram_peak = max(sampled)
    vram_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    for _ in range(settings["warmup"]):
        step()
    synchronize()
    durations = []
    for _ in range(settings["repeats"]):
        synchronize()
        start = time.perf_counter()
        step()
        synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    median = statistics.median(durations)
    mib = 1024 ** 2
    return dict(name=cfg["name"], phase="training", device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                batch_size=batch_size, measured_samples=batch_size * settings["repeats"],
                median_batch_ms=median, ms_per_sample=median / batch_size,
                samples_per_second=1000 * batch_size / median, batch_times_ms=durations,
                ram_baseline_rss_mib=baseline / mib, ram_peak_rss_mib=ram_peak / mib,
                ram_peak_rss_mib_per_sample=ram_peak / mib / batch_size,
                ram_increment_rss_mib_per_sample=max(0, ram_peak - baseline) / mib / batch_size,
                vram_peak_allocated_mib=None if vram_peak is None else vram_peak / mib,
                vram_peak_allocated_mib_per_sample=None if vram_peak is None else vram_peak / mib / batch_size,
                pid=os.getpid(), torch_threads=torch.get_num_threads())


def benchmark_models(model_runs: list[Path], data_path: Path, settings: dict,
                     root: Path | None = None) -> Path:
    """Profile saved models in independent processes and persist comparison evidence."""
    for key in ("batch_size", "warmup", "repeats", "threads"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not model_runs:
        raise ValueError("At least one saved model run is required")
    settings = {**settings, "device": settings.get("device", "cuda" if torch.cuda.is_available() else "cpu")}
    data_path = Path(data_path).resolve()
    config = dict(settings=settings, source_runs=[str(Path(p).resolve()) for p in model_runs],
                  data_path=str(data_path), phase="training", memory_pass="one first training step",
                  time_scope="forward + CrossEntropyLoss + backward + optimizer step; preloaded batch",
                  ram_scope="sampled process RSS at 1ms; peak / batch, includes interpreter and shared allocations",
                  vram_scope="peak PyTorch allocated tensors / batch; CPU is unavailable, not zero",
                  limitations="steady-state batch-normalized timing is not single-image latency; excludes IO, transfer, validation and FE")
    with Run("training_performance", config, root=root) as run:
        rows = []
        for index, source in enumerate(model_runs):
            result_path = run.path / f"measurement_{index}.json"
            args = [sys.executable, "-X", "utf8", "-m", "src.utils.performance", "--worker",
                    str(Path(source).resolve()), "--data", str(data_path),
                    "--settings", json.dumps(settings), "--output", str(result_path.resolve())]
            result = subprocess.run(args, cwd=SOURCE_DIR, capture_output=True, text=True,
                                    encoding="utf-8", timeout=180)
            print(result.stdout)
            if result.returncode:
                raise RuntimeError(result.stderr or "Training benchmark worker failed")
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
        table = pd.DataFrame([{k: v for k, v in row.items() if k != "batch_times_ms"} for row in rows])
        table.to_csv(run.path / "training_performance.csv", index=False)
        write_json(run.path / "methodology.json", config)
        print("Training-only batch benchmark; original checkpoints remain unchanged.")
        print("RAM: sampled process RSS / batch. CPU VRAM: unavailable. See methodology.json.")
        show_training_performance(run.path, save=True)
    return run.path


def show_training_performance(run_path: Path, save: bool = False) -> None:
    """Display all resource columns without truncation, plus labeled resource plots."""
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    from io import BytesIO
    path = Path(run_path)
    table = pd.read_csv(path / "training_performance.csv")
    columns = {
        "name": "Model", "device": "Thiết bị", "batch_size": "Batch size",
        "ms_per_sample": "Training (ms/sample)",
        "ram_peak_rss_mib_per_sample": "RAM RSS (MiB/sample)",
        "ram_increment_rss_mib_per_sample": "RAM tăng (MiB/sample)",
        "vram_peak_allocated_mib_per_sample": "VRAM allocated (MiB/sample)",
    }
    compact = table[list(columns)].rename(columns=columns)
    display(compact.style.hide(axis="index").format(precision=6, na_rep="Không khả dụng"))
    fields = [("ms_per_sample", "Training time", "ms/sample"),
              ("ram_peak_rss_mib_per_sample", "Sampled process peak RSS", "MiB/sample"),
              ("ram_increment_rss_mib_per_sample", "Sampled RSS increase over baseline", "MiB/sample")]
    if table["vram_peak_allocated_mib_per_sample"].notna().any():
        fields.append(("vram_peak_allocated_mib_per_sample", "Peak allocated VRAM", "MiB/sample"))
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4))
    for ax, (field, title, unit) in zip(axes, fields):
        bars = ax.bar(table["name"], table[field], color=["#4c78a8", "#59a14f", "#f28e2b"])
        ax.bar_label(bars, fmt="%.4f", fontsize=8)
        ax.set(title=title, ylabel=unit)
        ax.set_ylim(0, max(float(table[field].max()) * 1.2, .001))
        ax.tick_params(axis="x", rotation=15)
    batch_sizes = ", ".join(str(v) for v in table.batch_size.unique())
    fig.suptitle("Training only | batch size = " + batch_sizes +
                 " | RSS includes Python/runtime; CPU VRAM is unavailable")
    fig.tight_layout()
    if save:
        fig.savefig(path / "training_resources.png", dpi=130, bbox_inches="tight")
    # Embed real PNG output even with a non-interactive Agg notebook backend.
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    display(Image(data=buffer.getvalue()))
    plt.close(fig)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, _measure_worker(args.worker, args.data, json.loads(args.settings)))


if __name__ == "__main__":
    _main()
