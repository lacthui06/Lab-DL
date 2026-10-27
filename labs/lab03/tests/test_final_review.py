import json
import shutil
import pytest
from labs.lab03.src.utils.runs import ROOT, hash_file
from labs.lab03.src.final_review import audit_frozen_trial

RUN = '20261027T095655Z_frozen_trial_d8d09116'


def test_actual_frozen_trial_reconstructs_selection_and_test_metrics():
    result = audit_frozen_trial(ROOT/'outputs/runs'/RUN)
    assert result['status'] == 'passed'
    assert result['selected_epoch'] == 9 and result['threshold'] == .45
    assert not result['weights_loaded_locally'] and result['test_samples'] == 1066


def test_audit_rejects_changed_selection_even_with_consistent_file_checksum(tmp_path):
    source = ROOT/'outputs/runs'/RUN
    target = tmp_path/'run';shutil.copytree(source,target)
    summary = json.loads((target/'selection_summary.json').read_text())
    summary['selected_epoch'] = 10
    (target/'selection_summary.json').write_text(json.dumps(summary))
    manifest = json.loads((target/'manifest.json').read_text())
    manifest['selection_summary.json'] = hash_file(target/'selection_summary.json')
    (target/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='minimum validation loss'):
        audit_frozen_trial(target)


def test_new_bundle_imports_preserve_original_bytes():
    ids = ['20261027T083010Z_study_99bce559', RUN]
    for run_id in ids:
        folder = ROOT/'outputs/imports'/run_id
        manifest = json.loads((folder/'bundle_manifest.json').read_text())
        for name,digest in manifest.items():
            path = folder/name if name.startswith('config/') else ROOT/name
            assert hash_file(path) == digest, name


def test_prepare_checkpoint_binds_policy_without_modifying_source(tmp_path, monkeypatch):
    from labs.lab03.scripts import prepare_frozen_checkpoint as module
    from labs.lab03.src.decision import load_decision_policy
    source=tmp_path/'source';source.mkdir()
    checkpoint=tmp_path/'extracted/best_model';checkpoint.mkdir(parents=True)
    (checkpoint/'model.safetensors').write_bytes(b'checkpoint fixture')
    (source/'frozen_selection.json').write_text(json.dumps({'weights_sha256':hash_file(checkpoint/'model.safetensors')}))
    (source/'decision_threshold.json').write_text(json.dumps({'split':'validation','selected_threshold':.45}))
    (source/'config.json').write_text(json.dumps({'seed':42}))
    monkeypatch.setattr(module,'verify_manifest',lambda path:None)
    before=(source/'decision_threshold.json').read_bytes()
    module.prepare_checkpoint(checkpoint,source)
    assert load_decision_policy(checkpoint)['selected_threshold']==.45
    assert (source/'decision_threshold.json').read_bytes()==before
    assert json.loads((checkpoint/'training_config.json').read_text())['seed']==42
    (checkpoint/'model.safetensors').write_bytes(b'different checkpoint')
    with pytest.raises(ValueError,match='SHA256'):
        module.prepare_checkpoint(checkpoint,source)
