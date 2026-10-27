"""Đóng gói evidence nhẹ và checkpoint riêng; xác minh SHA256 trước khi nén."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.utils.runs import ROOT, hash_file, save_json
from labs.lab03.src.evidence import verify_manifest


def package_study(study, include_checkpoint=True, root=ROOT):
    root = Path(root).resolve()
    study = Path(study).resolve()
    runs = (root / 'outputs/runs').resolve()
    if study.parent != runs:
        raise ValueError('Study must be a direct child of outputs/runs')
    verify_manifest(study)
    if not (study / 'study_summary.json').is_file():
        raise ValueError('Only a completed study can be packaged')
    candidates = json.loads((study / 'candidate_comparison.json').read_text())
    selected = json.loads((study / 'checkpoint_reference.json').read_text())
    ids = [study.name] + [r['run_id'] for r in candidates]
    state = json.loads((root / 'config/workflow.json').read_text())
    for kind in ['exercise1', 'exercise1_analysis', 'study_audit', 'augmentation_preview']:
        if state.get('active_runs', {}).get(kind):
            run_id = state['active_runs'][kind]
            if kind == 'augmentation_preview':
                if not (study/'augmentation.json').is_file(): continue
                if json.loads((runs/run_id/'augmentation.json').read_text())['records_sha256'] != json.loads((study/'augmentation.json').read_text())['records_sha256']: continue
            if kind != 'study_audit' or json.loads((runs / run_id / 'config.json').read_text()).get('source_run') == study.name:
                ids.append(run_id)
    paths = []
    for run_id in dict.fromkeys(ids):
        if Path(run_id).name != run_id or run_id in {'.', '..'}:
            raise ValueError('Invalid source run ID')
        run = runs / run_id
        verify_manifest(run)
        for path in sorted(run.rglob('*')):
            relative = path.relative_to(run)
            if not path.is_file() or any(p in {'best_model', 'checkpoints', 'temporary'} for p in relative.parts): continue
            if path.suffix in {'.safetensors', '.bin', '.pt', '.pth', '.pkl', '.zip'}: continue
            paths.append(path)
    paths += sorted((root / 'config').glob('*.json'))
    paths += [p for p in sorted((root / 'outputs/preflight').rglob('*')) if p.is_file() and p.suffix in {'.xml', '.log'}]
    dest = root / 'outputs/bundles'; dest.mkdir(exist_ok=True)
    review = dest / f'lab03_review_{study.name}.zip'
    manifest = {p.relative_to(root).as_posix(): hash_file(p) for p in paths}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError): commit = None
    source_hashes = {p.relative_to(root).as_posix(): hash_file(p)
                     for folder in ['src', 'scripts'] for p in sorted((root / folder).rglob('*.py'))}
    info = {'source_code_sha256': source_hashes, 'study_run': study.name, 'source_commit': commit, 'run_ids': list(dict.fromkeys(ids)),
            'weights_included': False, 'raw_logs_included': True,
            'note': 'Original manifests retain checkpoint hashes; omitted weights are in the separate checkpoint archive.'}
    with ZipFile(review, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths: archive.write(path, path.relative_to(root).as_posix())
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
        archive.writestr('bundle_info.json', json.dumps(info, indent=2))
    # Kiểm tra byte của file đã đóng gói, không chỉ tin vào tên và kích thước.
    import hashlib
    with ZipFile(review) as archive:
        if archive.testzip() is not None: raise ValueError('Review ZIP CRC validation failed')
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f'ZIP checksum mismatch: {name}')
    checkpoint_zip = None
    if include_checkpoint:
        checkpoint = runs / selected['run_id'] / 'best_model'
        if checkpoint.parent.parent != runs or not checkpoint.is_dir():
            raise ValueError('Selected checkpoint is unavailable')
        checkpoint_zip = dest / f'lab03_checkpoint_{selected["run_id"]}.zip'
        files = [p for p in checkpoint.rglob('*') if p.is_file()]
        # Kèm config run để eval dùng đúng data/model/tokenizer.
        files += [checkpoint.parent / 'config.json', checkpoint.parent / 'decision_threshold.json']
        cm = {p.relative_to(root).as_posix(): hash_file(p) for p in files}
        with ZipFile(checkpoint_zip, 'w', ZIP_DEFLATED, compresslevel=1) as archive:
            for p in files: archive.write(p, p.relative_to(root).as_posix())
            archive.writestr('bundle_manifest.json', json.dumps(cm, indent=2))
        with ZipFile(checkpoint_zip) as archive:
            for name, digest in cm.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError(f'Checkpoint ZIP checksum mismatch: {name}')
    save_json(dest / 'latest_bundle.json', {'review': str(review), 'checkpoint': str(checkpoint_zip) if checkpoint_zip else None,
                                         'review_sha256': hash_file(review), 'review_file_count': len(manifest)})
    return review, checkpoint_zip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-run', type=Path)
    parser.add_argument('--no-checkpoint', action='store_true')
    args = parser.parse_args()
    study = args.study_run
    if study is None:
        state = json.loads((ROOT / 'config/workflow.json').read_text())
        study = ROOT / 'outputs/runs' / state['active_runs']['study']
    for path in package_study(study, not args.no_checkpoint):
        if path: print('ZIP:', path, 'MB:', round(path.stat().st_size/1024**2, 2))
