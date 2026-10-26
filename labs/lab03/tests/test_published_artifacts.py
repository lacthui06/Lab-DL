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
    module=_module();out=module.publish_artifacts(output_dir=tmp_path/'published')
    summary=json.loads((out/'metrics_summary.json').read_text())
    assert summary['selected_threshold']==.8
    assert summary['test_comparison'][1]['accuracy']==pytest.approx(896/1066)
    assert summary['test_comparison'][0]['binary_f1']>summary['test_comparison'][1]['binary_f1']
    assert len(list(out.rglob('*.png')))>=25
    manifest=json.loads((out/'manifest.json').read_text())
    assert all(hash_file(out/name)==digest for name,digest in manifest.items())
    assert not any(p.suffix in {'.log','.safetensors','.bin','.pt','.zip'} for p in out.rglob('*') if p.is_file())
    original_index=hash_file(out/'index.md')
    assert module.publish_artifacts(output_dir=out)==out
    assert hash_file(out/'index.md')==original_index
    assert not (tmp_path/'publication_history').exists()
    assert {str(p):hash_file(p) for p in (ROOT/'outputs/runs').rglob('*') if p.is_file()}==before


def test_report_export_rejects_unsafe_selection(tmp_path):
    config=tmp_path/'report.json';config.write_text(json.dumps({'runs':{'../other':'x'}}))
    with pytest.raises(ValueError,match='Invalid report'):
        _module().publish_artifacts(config,output_dir=tmp_path/'published')
    assert not (tmp_path/'published').exists()
