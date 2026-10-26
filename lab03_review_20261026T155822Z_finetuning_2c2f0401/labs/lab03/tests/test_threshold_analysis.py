"""Artifact so sánh phải có đủ hình, cache và bảo toàn ba run Kaggle nguồn."""
import shutil
import pytest
from labs.lab03.src.evidence import read_json, verify_manifest
from labs.lab03.src.threshold_analysis import ensure_threshold_analysis
from labs.lab03.src.utils.runs import ROOT, hash_file, save_json


def test_actual_threshold_visuals_cache_and_keep_source_evidence(tmp_path):
    kinds = ['threshold_selection', 'evaluation', 'inference']
    state = read_json(ROOT / 'config/workflow.json')
    copied = []
    for kind in kinds:
        source = ROOT / 'outputs/runs' / state['active_runs'][kind]
        target = tmp_path / 'outputs/runs' / source.name
        shutil.copytree(source, target)
        copied.append(target)
    save_json(tmp_path / 'config/workflow.json', {
        'active_runs': {kind: path.name for kind, path in zip(kinds, copied)}, 'runs': []})
    before = {str(p): hash_file(p) for run in copied for p in run.rglob('*') if p.is_file()}
    assert ensure_threshold_analysis('unrelated_model_run', root=tmp_path) is None
    out = ensure_threshold_analysis(root=tmp_path)
    assert len(list(out.glob('*.png'))) == 9
    summary = read_json(out / 'summary.json')
    assert summary['selected_threshold'] == .8
    assert summary['corrected_errors'] == 22
    assert summary['introduced_errors'] == 21
    assert summary['test_selected']['accuracy'] == pytest.approx(896 / 1066)
    assert summary['test_selection_performed'] is False
    assert summary['local_model_execution'] is False
    tables = read_json(out / 'tables.json')
    assert len(tables['validation_grid']) == 19
    assert len(tables['per_class_comparison']) == 4
    assert {r['status'] for r in tables['changed_prediction_examples']} == {
        'corrected_error', 'introduced_error', 'unchanged'}
    verify_manifest(out)
    assert ensure_threshold_analysis(root=tmp_path) == out
    assert {str(p): hash_file(p) for run in copied for p in run.rglob('*') if p.is_file()} == before


def test_threshold_visuals_ignore_unrelated_latest_inference(tmp_path):
    """Inference SST-2 mới không được làm hỏng bộ evidence của checkpoint cũ."""
    state = read_json(ROOT / 'config/workflow.json')
    kinds = ['threshold_selection', 'evaluation', 'inference']
    rows=[]
    active={}
    for kind in kinds:
        run_id=state['active_runs'][kind]
        shutil.copytree(ROOT/'outputs/runs'/run_id,tmp_path/'outputs/runs'/run_id)
        rows.append({'id':run_id,'kind':kind,'status':'completed'})
        active[kind]=run_id
    other=tmp_path/'outputs/runs/unrelated_inference'
    save_json(other/'config.json',{'inference':{'model':'pretrained SST-2'}})
    rows.append({'id':other.name,'kind':'inference','status':'completed'})
    active['inference']=other.name
    save_json(tmp_path/'config/workflow.json',{'active_runs':active,'runs':rows})
    out=ensure_threshold_analysis(root=tmp_path)
    summary=read_json(out/'summary.json')
    assert summary['source_runs']['inference']==state['active_runs']['inference']
    assert summary['selected_threshold']==.8
