"""Bộ report tái dùng hình thật, chọn run tường minh và không mang weights/log."""
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
    assert not any(p.suffix in {'.log','.safetensors','.bin','.pt','.zip'} for p in out.rglob('*') if p.is_file())
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
    assert summary['historical_three_epoch_test_comparison'][0]['accuracy']==pytest.approx(895/1066)
    assert (out/'study/candidate_validation_comparison.png').is_file()
    assert (out/'candidate_lr_3e5/step_learning_curves.png').is_file()
    manifest=json.loads((out/'manifest.json').read_text())
    assert all(hash_file(out/name)==digest for name,digest in manifest.items())


def test_primary_frozen_trial_uses_its_measured_metrics_and_saved_figures(tmp_path):
    out=_module().publish_artifacts(output_dir=tmp_path/'published')
    summary=json.loads((out/'metrics_summary.json').read_text())
    assert summary['primary_result']=='frozen_trial'
    assert summary['selected_threshold']==.45
    assert summary['test_comparison'][-1]['accuracy']==pytest.approx(883/1066)
    assert summary['measured_test_loss']==pytest.approx(.41256004571914673)
    assert (out/'frozen_trial/learning_curves.png').is_file()
    assert (out/'frozen_trial/validation_threshold_curves.png').is_file()
