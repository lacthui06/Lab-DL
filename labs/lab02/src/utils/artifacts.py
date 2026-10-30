"""Artifact IO and delivery workflow, called by scripts/finalize.py [--all].

Historical source snapshots stay immutable. Viewing notebooks only requires
saved artifacts; complete_delivery additionally requires original data/checkpoints.
"""
import hashlib
import json
import os
from pathlib import Path


def write_json(path, value) -> None:
    """Write UTF-8 artifacts; shared by experiment, EDA and delivery workflows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def file_hash(path) -> str:
    """Stream SHA256 without importing the training/data-loader stack."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_data_dir(root: Path, explicit=None) -> Path:
    """One path policy for every CLI: explicit argument, LAB02_DATA, root/data."""
    return Path(explicit or os.getenv('LAB02_DATA') or root/'data').expanduser().resolve()


def resolve_run(root: Path, run_path=None) -> Path:
    """Resolve an explicit run or the registry independently of the caller's cwd."""
    if run_path is None:
        run_path = json.loads((root/'outputs/selected_run.json').read_text(encoding='utf-8'))['run']
    run = Path(run_path)
    return (run if run.is_absolute() else root/run).resolve()


def audit_snapshot(snapshot: Path, hashes: dict[str, str]) -> dict:
    """Distinguish exact bytes from Git LF/CRLF conversion; reject other edits.

    A newline-only match is explicitly NOT reported as byte-exact SHA256 proof.
    """
    exact = 0
    newline_only = []
    for relative, expected in hashes.items():
        content = (snapshot / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() == expected:
            exact += 1
            continue
        lf = content.replace(b'\r\n', b'\n')
        variants = [lf, lf.replace(b'\n', b'\r\n')]
        if expected not in [hashlib.sha256(value).hexdigest() for value in variants]:
            raise ValueError(f'Snapshot content differs beyond LF/CRLF: {relative}')
        newline_only.append(relative)
    return {'byte_exact_files': exact, 'newline_only_files': newline_only}


def finalize_run(run_path: str | None = None) -> None:
    """Validate raw run evidence and write run-local report artifacts."""
    import shutil
    import subprocess
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (accuracy_score, f1_score, roc_curve, auc,
                                 precision_recall_curve, average_precision_score)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    root = Path(__file__).resolve().parents[2]
    run = resolve_run(root,run_path)
    work = run.parents[2]
    data_dir = resolve_data_dir(work)
    assert json.loads((run / "status.json").read_text(encoding="utf-8"))["status"] == "completed"
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert cfg.get('dataset') == 'CIFAR10', 'Refuse to postprocess a legacy dataset with CIFAR10 code'
    assets = {}
    for weights_path in (work / 'data' / 'pretrained' / 'checkpoints').glob('*.pth'):
        digest = file_hash(weights_path)
        expected = weights_path.stem.rsplit('-', 1)[-1]
        assert digest.startswith(expected), f'Official pretrained hash prefix mismatch: {weights_path}'
        assets[weights_path.name] = {'sha256': digest, 'official_filename_prefix_verified': True}
    archive = data_dir / 'cifar-10-python.tar.gz'
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
    official_labels = np.asarray(CIFAR10(str(data_dir), train=False).targets)
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
        if 'best_val_loss' in search:
            from ..training.tuning import select_trial
            assert chosen['trial'] == select_trial(search.to_dict('records'),cfg.get('selection_policy'))['trial']
        else:
            assert chosen['trial'] == int(search.loc[search.best_val_accuracy.idxmax(),'trial'])
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
    for column, name in enumerate(cfg['models']):
        chosen = json.loads((run / name / 'selection.json').read_text(encoding='utf-8'))['head']
        selected_fine = json.loads((run/name/'selection.json').read_text(encoding='utf-8')).get('finetune',{})
        fine_dir = f"finetune_trial_{selected_fine.get('trial',0):02d}"
        if not (run/name/fine_dir).exists():
            fine_dir = 'finetune'
        for stage, directory in [('head', f"head_trial_{chosen['trial']:02d}"), ('finetune', fine_dir)]:
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
    from ..utils.costs import cost_table, validation_recommendation
    costs = cost_table(run,comparison).to_dict('records')
    fig.suptitle('Monitoring evidence reconstructed from TensorBoard events (not a UI screenshot)')
    fig.tight_layout()
    fig.savefig(run / 'tensorboard_evidence.png', dpi=150)
    plt.close(fig)
    pd.DataFrame(tensorboard_records).to_csv(run / 'tensorboard_scalars.csv', index=False)
    recommendation = validation_recommendation(run,comparison,cfg)
    winner = comparison[(comparison.model==recommendation['model']) & (comparison.stage==recommendation['stage'])].iloc[0]
    notes.extend(['### TensorBoard và chi phí', '![TensorBoard scalars](tensorboard_evidence.png)',
        'Hình này dựng từ TensorBoard events thật; không phải ảnh chụp giao diện. Mở TensorBoard theo README để xem tương tác.',
        f'Điều kiện được đề xuất bằng validation: **{winner.model}/{winner.stage}**, validation accuracy **{winner.best_val_accuracy:.2%}**.',
        'Chi phí pipeline gồm feature extraction, toàn bộ head search và toàn bộ fine-tune search nếu có; chưa gồm EDA, ghi báo cáo và inference test. Xem cost_comparison.csv.'])
    cost_table = ['| Model | Stage | Feature extraction (s) | All head search (s) | Pipeline search train (s) | Test (s) |',
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
    from ..utils.plots import write_final_report
    write_final_report(run,cfg,selected,notes)
    lock = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run / "requirements-freeze.txt").write_text(lock, encoding="utf-8")
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
    audits = {'source_snapshot': audit_snapshot(run / 'source_snapshot', original_hashes)}
    resume_hash_path = run / 'resume_source_hashes.json'
    if resume_hash_path.exists():
        resume_hashes = json.loads(resume_hash_path.read_text(encoding='utf-8'))
        audits['resume_source_snapshot'] = audit_snapshot(run / 'resume_source_snapshot', resume_hashes)
    write_json(run / 'snapshot_checkout_audit.json', audits)
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
    (run / "results_index.md").write_text(
        "# Lab 02 — Kết quả đã chạy\n\n"
        "[Tài liệu sống](../../../walkthrough_lab_02.md)\n\n"
        f"Run: `{run.name}`. Profile: **{cfg['profile']}**. Số mẫu xem trong data_summary.json và báo cáo.\n\n"
        f"- [Notebook EDA và tiền xử lý](../../../notebooks/01_lab02_eda.ipynb)\n"
        f"- [Notebook huấn luyện và đánh giá](../../../notebooks/02_lab02_modeling.ipynb)\n"
        f"- [Báo cáo có hình](report_vi.html)\n"
        f"- [Báo cáo Markdown](report_vi.md)\n"
        f"- [So sánh kết quả](comparison.csv)\n"
        f"- [Kiểm chứng](verification.json)\n\n"
        f"Dữ liệu: CIFAR-10; đầu vào {cfg['image_size']}×{cfg['image_size']}. Kết quả chỉ áp dụng cho cấu hình đã ghi. Đọc README để tái lập.\n\n"
        + '\n'.join(summary_rows) + '\n', encoding="utf-8")
    print(comparison[["model", "stage", "accuracy", "macro_f1"]].to_string(index=False))
    print("Verified manifest:", len(manifest), "files")
    print("HTML report:", run / "report_vi.html")



def complete_delivery(run_path: str | None = None) -> None:
    """Wait for completion, validate, execute both views and verify inference."""
    import subprocess
    import sys
    import time
    root = Path(__file__).resolve().parents[2]

    run = resolve_run(root,run_path)
    deadline = time.monotonic() + 14400
    while True:
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        if status["status"] == "completed":
            break
        if status["status"] == "failed":
            raise RuntimeError(status)
        if time.monotonic() > deadline:
            raise TimeoutError("Training not completed within four hours")
        time.sleep(5)
    environment = os.environ.copy()
    environment.update(LAB02_ROOT=str(root),LAB02_RUN=str(run),LAB02_DATA=str(resolve_data_dir(run.parents[2])))
    with (root / "outputs" / "delivery_validation.log").open("w", encoding="utf-8") as log:
        test = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'pytest', 'tests', '-q'], cwd=root,
                              text=True, encoding='utf-8', capture_output=True)
        (run / 'pytest_validation.log').write_text(test.stdout + test.stderr, encoding='utf-8')
        if test.returncode:
            raise RuntimeError(test.stdout + test.stderr)
        finalize_run(str(run))
        for script, arguments in [('execute_notebook.py', []),
                                  ('execute_notebook.py', ['01_lab02_eda.ipynb'])]:
            result = subprocess.run([sys.executable, '-X', 'utf8', str(root / "scripts" / script), *arguments], cwd=root,
                                    text=True, encoding="utf-8", capture_output=True,env=environment)
            log.write(result.stdout + result.stderr)
            log.flush()
            print(result.stdout, flush=True)
            if result.returncode:
                print(result.stderr, flush=True)
                raise RuntimeError(f"{script} failed")
        recommendation=json.loads((run/'validation_recommendation.json').read_text(encoding='utf-8'))
        stage='head_selected' if recommendation['stage']=='head' else 'finetune'
        checkpoint=run/recommendation['model']/stage/'best.pt'
        result = subprocess.run([sys.executable,'-X','utf8',str(root/'scripts/infer.py'),
            '--checkpoint',str(checkpoint),'--sample-index','0'],cwd=root,env=environment,
            text=True,encoding='utf-8',capture_output=True,check=True)
        log.write(result.stdout + result.stderr)
        write_json(run / "cli_inference_example.json", json.loads(result.stdout))
    notebook = root / "notebooks" / "02_lab02_modeling.ipynb"
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c["execution_count"] is not None for c in code_cells)
    assert not any(o.get("output_type") == "error" for c in code_cells for o in c["outputs"])
    verification = json.loads((run / "verification.json").read_text(encoding="utf-8"))
    eda_notebook = root / 'notebooks' / '01_lab02_eda.ipynb'
    eda = json.loads(eda_notebook.read_text(encoding='utf-8'))
    eda_cells = [c for c in eda['cells'] if c['cell_type'] == 'code']
    assert all(c['execution_count'] is not None for c in eda_cells)
    assert not any(o.get('output_type') == 'error' for c in eda_cells for o in c['outputs'])
    verification.update({"notebook_execution": "passed", "executed_code_cells": len(code_cells),
                         "notebook_sha256": file_hash(notebook), "cli_inference": "passed",
                         'eda_notebook_execution': 'passed', 'eda_executed_code_cells': len(eda_cells),
                         'eda_notebook_sha256': file_hash(eda_notebook),
                         'tests': test.stdout.strip()})
    write_json(run / "verification.json", verification)
    write_json(run / "manifest.json", {str(p.relative_to(run)): file_hash(p)
        for p in run.rglob("*") if p.is_file() and p.name != "manifest.json"})
    print("DELIVERY VERIFIED", flush=True)
    print(run / "report_vi.html", flush=True)
