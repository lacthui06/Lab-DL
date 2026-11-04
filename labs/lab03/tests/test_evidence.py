import json
import shutil
import pytest
from labs.lab03.src.evidence import selected_run, audit_exercise1, verify_manifest
from labs.lab03.src.final_review import audit_frozen_trial
from labs.lab03.src.utils.runs import ROOT, hash_file


def test_current_bundle_recomputes_metrics_and_detects_tampering(tmp_path, audit_trial, exercise_run):
    source = audit_trial
    assert audit_exercise1(exercise_run)['compared_sentences'] == 3
    audit = audit_frozen_trial(source)
    assert audit['test_samples'] == 1066
    assert audit['selected_epoch'] == 3 and 0 < audit['threshold'] < 1
    target = tmp_path/'run'
    shutil.copytree(source, target)
    path = target/'test_comparison.json'
    rows = json.loads(path.read_text()); rows[-1]['accuracy'] = 1.
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='SHA256'): verify_manifest(target)
    manifest = json.loads((target/'manifest.json').read_text())
    manifest[path.name] = hash_file(path)
    (target/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): audit_frozen_trial(target)
