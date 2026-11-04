"""Bộ report tái dùng hình và log thật, chọn run tường minh, không mang weights."""
import importlib.util
import json
import pytest
from src.utils.runs import ROOT, hash_file


def _module():
    spec=importlib.util.spec_from_file_location('lab03_publish',ROOT/'scripts/publish_artifacts.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_report_export_rejects_unsafe_selection(tmp_path):
    config=tmp_path/'report.json';config.write_text(json.dumps({'runs':{'../other':'x'}}))
    with pytest.raises(ValueError,match='Invalid report'):
        _module().publish_artifacts(config,output_dir=tmp_path/'published')
    assert not (tmp_path/'published').exists()


def test_primary_frozen_trial_uses_its_measured_metrics_and_saved_figures(tmp_path, audit_trial):
    from src.utils.runs import save_json
    save_json(tmp_path/'selection.json',{'primary_result':'frozen_trial','runs':{'frozen_trial':audit_trial.name}})
    import shutil
    shutil.copytree(audit_trial,tmp_path/'outputs/runs'/audit_trial.name)
    out=_module().publish_artifacts(tmp_path/'selection.json',output_dir=tmp_path/'published',source_root=tmp_path)
    summary=json.loads((out/'metrics_summary.json').read_text())
    assert summary['primary_result']=='frozen_trial'
    assert 0 < summary['selected_threshold'] < 1
    assert summary['test_comparison'][-1]['accuracy']==pytest.approx(json.loads((audit_trial/'test_comparison.json').read_text())[-1]['accuracy'])
    assert summary['measured_test_loss']==pytest.approx(json.loads((audit_trial/'test_runtime_metrics.json').read_text())['test_loss'])
    assert (out/'frozen_trial/learning_curves.png').is_file()
    assert (out/'frozen_trial/validation_threshold_curves.png').is_file()
    assert (out/'logs/frozen_trial/console.log').read_bytes()==(audit_trial/'console.log').read_bytes()
    assert summary['performance']['training_ram_complete'] is True


def test_training_autopublish_accepts_one_completed_run_and_keeps_log_bytes(tmp_path, audit_trial):
    import shutil
    from scripts.run_frozen_trial import publish_completed_trial
    source=audit_trial
    destination=tmp_path/'outputs/runs'/source.name
    shutil.copytree(source,destination)
    before={p.relative_to(destination).as_posix():hash_file(p) for p in destination.rglob('*') if p.is_file()}
    out=publish_completed_trial(destination,tmp_path)
    assert (out/'frozen_trial/learning_curves.png').is_file()
    assert (out/'logs/frozen_trial/console.log').read_bytes()==(source/'console.log').read_bytes()
    selection=json.loads((tmp_path/'outputs/latest_report.json').read_text())
    assert selection['runs']=={'frozen_trial':source.name}
    assert not (tmp_path/'config/report.json').exists()
    assert {p.relative_to(destination).as_posix():hash_file(p) for p in destination.rglob('*') if p.is_file()}==before
    manifest=json.loads((out/'manifest.json').read_text())
    assert all(hash_file(out/name)==digest for name,digest in manifest.items())


def test_autopublish_includes_only_exercise_runs_from_same_session(tmp_path,monkeypatch, audit_trial):
    import shutil
    from src.utils.runs import run_context,save_json
    from scripts.run_frozen_trial import publish_completed_trial
    source=audit_trial
    destination=tmp_path/'outputs/runs'/source.name
    shutil.copytree(source,destination)
    cfg=json.loads((destination/'config.json').read_text());cfg['session_id']='current-session'
    save_json(destination/'config.json',cfg)
    manifest=json.loads((destination/'manifest.json').read_text());manifest['config.json']=hash_file(destination/'config.json')
    save_json(destination/'manifest.json',manifest)
    monkeypatch.setenv('LAB03_SESSION_ID','different-session')
    with run_context('exercise1',{'session_id':'different-session'},root=tmp_path) as unrelated:
        save_json(unrelated/'predictions.json',[])
    monkeypatch.setenv('LAB03_SESSION_ID','current-session')
    with run_context('exercise1_analysis',{'session_id':'current-session'},root=tmp_path) as current:
        save_json(current/'analysis_summary.json',{'fixture':True})
    out=publish_completed_trial(destination,tmp_path)
    selected=json.loads((out/'source_runs.json').read_text())['runs']
    assert selected=={'frozen_trial':destination.name,'exercise1_analysis':current.name}
    assert not (out/'exercise1').exists()
    assert (out/'logs/exercise1_analysis/console.log').read_bytes()==(current/'console.log').read_bytes()


def test_default_publication_requires_completed_training_selection(tmp_path):
    (tmp_path/'config').mkdir()
    (tmp_path/'config/report.json').write_text(json.dumps({'runs':{'frozen_trial':'unrelated'}}))
    with pytest.raises(FileNotFoundError,match='latest_report'):
        _module().publish_artifacts(source_root=tmp_path)
    assert not (tmp_path/'outputs/published').exists()


def test_architecture_text_is_published_and_hash_checked_before_cache(tmp_path, audit_trial):
    import shutil
    from src.utils.runs import save_json
    run=audit_trial.name
    shutil.copytree(audit_trial,tmp_path/'outputs/runs'/run)
    save_json(tmp_path/'outputs/latest_report.json',{'primary_result':'frozen_trial','runs':{'frozen_trial':run}})
    folder=tmp_path/'outputs/architecture';folder.mkdir()
    (folder/'model_architectures.txt').write_text('Architecture unit fixture')
    save_json(folder/'summary.json',{'fixture':True})
    save_json(folder/'manifest.json',{n:hash_file(folder/n) for n in ['model_architectures.txt','summary.json']})
    out=_module().publish_artifacts(source_root=tmp_path)
    assert (out/'architecture/model_architectures.txt').read_bytes()==(folder/'model_architectures.txt').read_bytes()
    assert (out/'result_analysis/per_class_metrics.png').is_file()
    (folder/'model_architectures.txt').write_text('tampered')
    with pytest.raises(ValueError,match='Architecture SHA256'):
        _module().publish_artifacts(source_root=tmp_path)
