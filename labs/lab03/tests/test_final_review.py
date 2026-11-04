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


def test_derived_report_checks_per_class_rates_loss_and_patience_without_mutation(tmp_path):
    from labs.lab03.src.final_review import write_results_analysis
    source=ROOT/'outputs/runs'/RUN
    before={p.relative_to(source).as_posix():hash_file(p) for p in source.rglob('*') if p.is_file()}
    summary=write_results_analysis(source,tmp_path)
    rows=json.loads((tmp_path/'per_class_metrics.json').read_text())
    assert rows[0]['recall']==pytest.approx(437/533)
    assert rows[0]['false_positive_rate']==pytest.approx(84/533)
    assert rows[1]['recall']==pytest.approx(449/533)
    assert rows[1]['false_positive_rate']==pytest.approx(96/533)
    assert summary['baseline_log_loss']>0 and summary['test_loss']==pytest.approx(.409902423620224)
    stopping=json.loads((tmp_path/'early_stopping_replay.json').read_text())
    assert [r['patience_counter'] for r in stopping['epochs']]==[0,0,0,1,2]
    assert stopping['patience_reached_at_budget'] is True
    assert stopping['counter_was_logged'] is False
    assert (tmp_path/'per_class_metrics.png').is_file()
    assert {p.relative_to(source).as_posix():hash_file(p) for p in source.rglob('*') if p.is_file()}==before
