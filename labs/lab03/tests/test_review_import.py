"""Artifact công bố thiếu log phải khai báo rõ; dữ liệu khoa học vẫn bắt buộc."""
import json
import shutil
import pytest
from labs.lab03.src.utils.runs import ROOT,hash_file,save_json
from labs.lab03.src.evidence import verify_manifest,selected_run
from labs.lab03.src.threshold_analysis import matching_threshold_runs
from labs.lab03.src.threshold_evidence import audit_threshold_experiment


def _fixture(tmp_path):
    source=ROOT/'outputs/runs/20261025T181532Z_exercise1_71da6f40'
    run=tmp_path/'run';shutil.copytree(source,run)
    (run/'console.log').unlink()
    return run


def _availability(run, files):
    manifest=json.loads((run/'manifest.json').read_text())
    save_json(run/'artifact_availability.json',{
        'source_manifest_sha256':hash_file(run/'manifest.json'),
        'omitted_raw_logs':{n:{'sha256':manifest[n],'reason':'not in published artifact set'} for n in files}})


def test_missing_log_requires_explicit_availability(tmp_path):
    run=_fixture(tmp_path)
    with pytest.raises(FileNotFoundError,match='console.log'):verify_manifest(run)
    _availability(run,['console.log'])
    audit=verify_manifest(run)
    assert audit['omitted_raw_logs']==['console.log']
    assert audit['verified_files']==5


def test_scientific_data_cannot_be_declared_as_omitted_log(tmp_path):
    run=_fixture(tmp_path);_availability(run,['console.log','predictions.json'])
    with pytest.raises(ValueError,match='Only raw logs'):verify_manifest(run)


def test_availability_is_bound_to_original_manifest(tmp_path):
    run=_fixture(tmp_path);_availability(run,['console.log'])
    path=run/'artifact_availability.json';data=json.loads(path.read_text());data['source_manifest_sha256']='0'*64
    save_json(path,data)
    with pytest.raises(ValueError,match='different manifest'):verify_manifest(run)


def test_present_corrupted_file_is_never_accepted_as_omitted(tmp_path):
    run=_fixture(tmp_path);_availability(run,['console.log'])
    (run/'console.log').write_text('changed')
    with pytest.raises(ValueError,match='SHA256 mismatch'):verify_manifest(run)


def test_latest_training_inline_policy_matches_full_test_reload_and_inference():
    paths=matching_threshold_runs(selected_run('finetuning'))
    audit=audit_threshold_experiment(paths['threshold_selection'],paths['evaluation'],paths['inference'])
    assert audit['selection_location']=='finetuning.validation_threshold'
    assert audit['verified_files']==46
    assert audit['policy']['selected_threshold']==.8
    assert audit['test_argmax']['accuracy']==pytest.approx(895/1066)
    assert audit['test_selected']['accuracy']==pytest.approx(896/1066)
    assert audit['test_loss']==pytest.approx(.5442677736282349)
    assert audit['selection_performed_on_test'] is False
    assert audit['manifest_audits']['threshold_selection']['omitted_raw_logs']==['console.log']
    cfg=json.loads((selected_run('finetuning')/'config.json').read_text())
    assert cfg['exercise2']['dataset_revision']=='aa13bc287fa6fcab6daf52f0dfb9994269ffea28'
