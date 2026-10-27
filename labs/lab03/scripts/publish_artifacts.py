"""Xuất bộ hình/bảng đã chọn cho report, không train hoặc dựng lại đồ thị."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.evidence import read_json, verify_manifest
from labs.lab03.src.utils.runs import ROOT, hash_file, save_json

JSON_FILES = {
    "artifact_availability.json", "config.json", "test_metrics.json", "test_threshold_metrics.json", "test_runtime_metrics.json",
    "train_metrics.json", "trainer_history.json", "data_statistics.json", "provenance.json",
    "reload_check.json", "decision_threshold.json", "validation_metrics.json", "validation_runtime_metrics.json",
    "summary.json", "analysis_summary.json", "tables.json", "threshold_scan.json", "selected_threshold.json",
    "predictions.json", "sample_predictions.json", "all_predictions.json", "validation_predictions.json",
    "roc_pr_coordinates.json", "reliability_bins.json", "tokenizer_breakdown.json",
    "augmentation.json", "epoch_metrics.json", "epoch_history.json", "presentation_source.json",
    "study_summary.json", "candidate_comparison.json", "frozen_selection.json", "selection_summary.json",
    "test_comparison.json", "probability_diagnostics.json", "threshold_errors.json", "split_overlap_audit.json",
    "inference_predictions.json", "baseline_validation_candidates.json", "baseline_validation_policy.json", "checkpoint_reference.json",
}


def publish_artifacts(settings=None, output_dir=None, source_root=ROOT):
    """Chọn run tường minh qua report.json; giữ đường dẫn hình ổn định cho report."""
    source_root = Path(source_root)
    settings = Path(settings or source_root / "config/report.json")
    output = Path(output_dir or source_root / "outputs/published")
    configuration = read_json(settings)
    selected = configuration["runs"]
    paths = {}
    for role, run_id in selected.items():
        if role in {".", ".."} or run_id in {".", ".."} or Path(role).name != role or Path(run_id).name != run_id:
            raise ValueError("Invalid report role or run ID")
        path = source_root / "outputs/runs" / run_id
        verify_manifest(path)
        paths[role] = path
    if "threshold_selection" in paths and "evaluation" in paths:
        if read_json(paths['threshold_selection']/'decision_threshold.json') != read_json(paths['evaluation']/'decision_threshold.json'):
            raise ValueError("Report selection and evaluation use different threshold policies")
    provenance = {"runs": selected, "source_manifests": {k: hash_file(p/'manifest.json') for k,p in paths.items()},
                  "exporter_sha256": hash_file(Path(__file__)), "primary_result": configuration.get("primary_result", "historical"),
                  "weights_included": False, "raw_logs_included": False}
    if (output/'source_runs.json').is_file() and read_json(output/'source_runs.json') == provenance:
        manifest = read_json(output/'manifest.json')
        if all((output/name).is_file() and hash_file(output/name) == value for name,value in manifest.items()):
            return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.parent / ('.published-' + uuid.uuid4().hex)
    staged.mkdir()
    try:
        catalog = []
        for role, run in paths.items():
            source_manifest=read_json(run/'manifest.json')
            for path in sorted(run.rglob('*')):
                if not path.is_file() or any(p in {'best_model','checkpoints'} for p in path.relative_to(run).parts):
                    continue
                if path.relative_to(run).as_posix() not in source_manifest:
                    continue
                if path.suffix != '.png' and path.name not in JSON_FILES:
                    continue
                relative = Path(role) / path.relative_to(run)
                target = staged/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target)
                catalog.append({'role':role,'run':run.name,'file':relative.as_posix(),'sha256':hash_file(target)})
        evaluation = paths.get('evaluation',paths.get('finetuning'))
        rows=[];loss=None;policy=None
        if evaluation:
            baseline=read_json(evaluation/'test_metrics.json')
            runtime=evaluation/'test_runtime_metrics.json'
            loss=read_json(runtime).get('test_loss') if runtime.is_file() else None
            rows.append({'policy':'Argmax','accuracy':baseline['accuracy'],'precision':baseline['binary_precision'],
                'recall':baseline['binary_recall'],'binary_f1':baseline['binary_f1'],'macro_f1':baseline['macro_f1'],'test_loss':loss})
            if (evaluation/'test_threshold_metrics.json').is_file():
                selected_metrics=read_json(evaluation/'test_threshold_metrics.json')
                rows.append({'policy':f"Validation threshold {selected_metrics['threshold']:g}",
                    **{k:selected_metrics[k] for k in ['accuracy','precision','recall','binary_f1','macro_f1']},'test_loss':loss})
                policy=read_json(evaluation/'decision_threshold.json')
        historical_rows = list(rows)
        primary = configuration.get('primary_result', 'historical')
        if primary == 'study':
            if 'study' not in paths:
                raise ValueError('Primary study is missing from selected report runs')
            from labs.lab03.src.study_evidence import audit_study
            audit_study(paths['study'], root=source_root)
            study = paths['study']
            loss = read_json(study/'test_runtime_metrics.json')['test_loss']
            policy = read_json(study/'decision_threshold.json')
            rows = [{'policy':r['model'], **{k:r[k] for k in ['accuracy','precision','recall','binary_f1','macro_f1']},
                     'test_loss':loss if r['model'].startswith('DistilBERT') else None}
                    for r in read_json(study/'test_comparison.json')]
        save_json(staged/'source_runs.json',provenance)
        save_json(staged/'artifact_catalog.json',catalog)
        save_json(staged/'metrics_summary.json',{'primary_result':primary, 'source_run':paths['study'].name if primary == 'study' else evaluation.name,
            'test_comparison':rows, 'historical_three_epoch_test_comparison':historical_rows, 'measured_test_loss':loss,
            'selection_split':policy.get('split') if policy else None,
            'selected_threshold':policy.get('selected_threshold') if policy else None})
        lines=['# Lab03 — Artifacts cho report','','Nguồn run được chọn tường minh trong `config/report.json`. Không huấn luyện hoặc chọn lại threshold khi xuất.','',
            '| Policy | Accuracy | Precision | Recall | Binary F1 | Macro F1 | Test loss |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            values=' | '.join(f"{row[k]:.4%}" for k in ['accuracy','precision','recall','binary_f1','macro_f1'])
            row_loss = row.get('test_loss')
            loss_text=f"{row_loss:.6f}" if row_loss is not None else 'Not logged'
            lines.append(f"| {row['policy']} | {values} | {loss_text} |")
        if primary == 'study':
            lines += ['', 'Bảng chính thuộc study 10 epoch; các role finetuning/evaluation/analysis cũ giữ bằng chứng lịch sử 3 epoch. Không gán số liệu hai protocol cho cùng một run.']
        lines += ['','## Nguồn thí nghiệm','','| Vai trò | Run |','|---|---|']
        lines += [f'| {role} | `{run_id}` |' for role,run_id in selected.items()]
        lines += ['','## Hình và bảng','','Các PNG dưới đây là bản sao của hình đã lưu trong run nguồn. JSON giữ số liệu để tính lại.','']
        for item in catalog:
            if item['file'].endswith('.png'):
                lines += [f"### {item['file']}",'',f"![{item['role']}]({item['file']})",'']
        (staged/'index.md').write_text('\n'.join(lines),encoding='utf-8')
        save_json(staged/'manifest.json',{p.relative_to(staged).as_posix():hash_file(p) for p in sorted(staged.rglob('*')) if p.is_file()})
        if output.exists():
            history=output.parent/'publication_history';history.mkdir(exist_ok=True)
            backup=history/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
            shutil.move(str(output),str(backup))
        os.replace(staged,output)
    finally:
        if staged.exists():shutil.rmtree(staged)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'config/report.json')
    args=parser.parse_args()
    print('Report artifacts:',publish_artifacts(args.config))
