"""Bộ report tái dùng hình và log thật, chọn run tường minh, không mang weights."""
import importlib.util
import json
import pytest
from labs.lab03.src.utils.runs import ROOT, hash_file


def _module():
    spec=importlib.util.spec_from_file_location('lab03_publish',ROOT/'scripts/publish_artifacts.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_report_export_uses_saved_visuals_and_preserves_scientific_sources(tmp_path):
    before={str(p):hash_file(p) for p in (ROOT/'outputs/runs').rglob('*') if p.is_file()}
    configuration=json.loads((ROOT/'config/report.json').read_text())
    configuration.pop('primary_result',None)
    roles={'exercise1','exercise1_analysis','finetuning','analysis','evaluation','threshold_analysis','inference'}
    configuration['runs']={k:v for k,v in configuration['runs'].items() if k in roles}
    historical_settings=tmp_path/'historical.json';historical_settings.write_text(json.dumps(configuration))
    module=_module();out=module.publish_artifacts(historical_settings, output_dir=tmp_path/'published')
    summary=json.loads((out/'metrics_summary.json').read_text())
    assert summary['selected_threshold']==.8
    assert summary['test_comparison'][1]['accuracy']==pytest.approx(896/1066)
    assert summary['test_comparison'][0]['binary_f1']>summary['test_comparison'][1]['binary_f1']
    assert len(list(out.rglob('*.png')))>=25
    manifest=json.loads((out/'manifest.json').read_text())
    assert all(hash_file(out/name)==digest for name,digest in manifest.items())
    assert not any(p.suffix in {'.safetensors','.bin','.pt','.zip'} for p in out.rglob('*') if p.is_file())
    log=out/'logs/finetuning/console.log'
    source=ROOT/'outputs/runs'/configuration['runs']['finetuning']/'console.log'
    if source.is_file():
        assert log.read_bytes()==source.read_bytes()
    else:
        assert not log.exists()
        assert 'console.log' in json.loads((out/'finetuning/artifact_availability.json').read_text())['omitted_raw_logs']
    original_index=hash_file(out/'index.md')
    assert module.publish_artifacts(historical_settings, output_dir=out)==out
    assert hash_file(out/'index.md')==original_index
    assert not (tmp_path/'publication_history').exists()
    assert {str(p):hash_file(p) for p in (ROOT/'outputs/runs').rglob('*') if p.is_file()}==before


def test_report_export_rejects_unsafe_selection(tmp_path):
    config=tmp_path/'report.json';config.write_text(json.dumps({'runs':{'../other':'x'}}))
    with pytest.raises(ValueError,match='Invalid report'):
        _module().publish_artifacts(config,output_dir=tmp_path/'published')
    assert not (tmp_path/'published').exists()



def test_report_primary_study_keeps_actual_metrics_and_historical_results_separate(tmp_path):
    configuration=json.loads((ROOT/'config/report.json').read_text())
    configuration['primary_result']='study'
    settings=tmp_path/'study.json';settings.write_text(json.dumps(configuration))
    module=_module();out=module.publish_artifacts(settings,output_dir=tmp_path/'published')
    summary=json.loads((out/'metrics_summary.json').read_text())
    assert summary['primary_result']=='study' and summary['selected_threshold']==.5
    assert summary['test_comparison'][-1]['accuracy']==pytest.approx(889/1066)
    assert summary['test_comparison'][0]['test_loss'] is None
    assert summary['reference_test_comparison'][0]['accuracy']==pytest.approx(895/1066)
    assert (out/'study/candidate_validation_comparison.png').is_file()
    assert (out/'candidate_lr_3e5/step_learning_curves.png').is_file()
    manifest=json.loads((out/'manifest.json').read_text())
    assert all(hash_file(out/name)==digest for name,digest in manifest.items())


def test_primary_frozen_trial_uses_its_measured_metrics_and_saved_figures(tmp_path):
    out=_module().publish_artifacts(ROOT/'config/report.json',output_dir=tmp_path/'published')
    summary=json.loads((out/'metrics_summary.json').read_text())
    assert summary['primary_result']=='frozen_trial'
    assert summary['selected_threshold']==.45
    assert summary['test_comparison'][-1]['accuracy']==pytest.approx(883/1066)
    assert summary['measured_test_loss']==pytest.approx(.41256004571914673)
    assert (out/'frozen_trial/learning_curves.png').is_file()
    assert (out/'frozen_trial/validation_threshold_curves.png').is_file()
    assert (out/'logs/frozen_trial/console.log').read_bytes()==(ROOT/'outputs/runs'/summary['source_run']/'console.log').read_bytes()
    assert summary['performance']['training_ram_complete'] is False


def test_training_autopublish_accepts_one_completed_run_and_keeps_log_bytes(tmp_path):
    import shutil
    from labs.lab03.scripts.run_frozen_trial import publish_completed_trial
    source=ROOT/'outputs/runs/20261027T095655Z_frozen_trial_d8d09116'
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


def test_autopublish_includes_only_exercise_runs_from_same_session(tmp_path):
    import shutil
    from labs.lab03.src.utils.runs import run_context,save_json
    from labs.lab03.scripts.run_frozen_trial import publish_completed_trial
    source=ROOT/'outputs/runs/20261027T095655Z_frozen_trial_d8d09116'
    destination=tmp_path/'outputs/runs'/source.name
    shutil.copytree(source,destination)
    cfg=json.loads((destination/'config.json').read_text());cfg['session_id']='current-session'
    save_json(destination/'config.json',cfg)
    manifest=json.loads((destination/'manifest.json').read_text());manifest['config.json']=hash_file(destination/'config.json')
    save_json(destination/'manifest.json',manifest)
    with run_context('exercise1',{'session_id':'different-session'},root=tmp_path) as unrelated:
        save_json(unrelated/'predictions.json',[])
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
