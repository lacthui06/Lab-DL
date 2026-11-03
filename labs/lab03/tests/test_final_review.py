import json
import shutil
import pytest
from labs.lab03.src.utils.runs import ROOT, hash_file
from labs.lab03.src.final_review import audit_frozen_trial

RUN = '20261103T110437Z_frozen_trial_b021d6eb'


def test_actual_frozen_trial_reconstructs_selection_and_test_metrics():
    result = audit_frozen_trial(ROOT/'outputs/runs'/RUN)
    assert result['status'] == 'passed'
    assert result['selected_epoch'] == 5 and result['threshold'] == .45
    assert not result['weights_loaded_locally'] and result['test_samples'] == 1066


def test_audit_rejects_changed_selection_even_with_consistent_file_checksum(tmp_path):
    source = ROOT/'outputs/runs'/RUN
    target = tmp_path/'run';shutil.copytree(source,target)
    summary = json.loads((target/'selection_summary.json').read_text())
    summary['selected_epoch'] = 4
    (target/'selection_summary.json').write_text(json.dumps(summary))
    manifest = json.loads((target/'manifest.json').read_text())
    manifest['selection_summary.json'] = hash_file(target/'selection_summary.json')
    (target/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='minimum validation loss'):
        audit_frozen_trial(target)
