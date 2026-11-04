import json
import shutil
import pytest
from src.utils.runs import ROOT, hash_file
from src.final_review import audit_frozen_trial




def test_contract_fixture_reconstructs_selection_and_test_metrics(audit_trial):
    result = audit_frozen_trial(audit_trial)
    assert result['status'] == 'passed'
    assert result['selected_epoch'] == 3 and 0 < result['threshold'] < 1
    assert not result['weights_loaded_locally'] and result['test_samples'] == 1066


def test_audit_rejects_changed_selection_even_with_consistent_file_checksum(tmp_path, audit_trial):
    source = audit_trial
    target = tmp_path/'run';shutil.copytree(source,target)
    summary = json.loads((target/'selection_summary.json').read_text())
    summary['selected_epoch'] = 4
    (target/'selection_summary.json').write_text(json.dumps(summary))
    manifest = json.loads((target/'manifest.json').read_text())
    manifest['selection_summary.json'] = hash_file(target/'selection_summary.json')
    (target/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='minimum validation loss'):
        audit_frozen_trial(target)


def test_derived_report_checks_per_class_rates_loss_and_patience_without_mutation(tmp_path, audit_trial):
    from src.final_review import write_results_analysis
    source=audit_trial
    before={p.relative_to(source).as_posix():hash_file(p) for p in source.rglob('*') if p.is_file()}
    summary=write_results_analysis(source,tmp_path)
    rows=json.loads((tmp_path/'per_class_metrics.json').read_text())
    assert rows[0]['support']==rows[1]['support']==533
    assert all(0 <= row[key] <= 1 for row in rows for key in ('recall','false_positive_rate','f1'))
    assert rows[0]['TP']+rows[0]['FN']==533
    assert summary['baseline_log_loss']>0 and summary['test_loss']==pytest.approx(json.loads((source/'test_runtime_metrics.json').read_text())['test_loss'])
    stopping=json.loads((tmp_path/'early_stopping_replay.json').read_text())
    assert [r['patience_counter'] for r in stopping['epochs']]==[0,0,0]
    assert stopping['patience_reached_at_budget'] is False
    assert stopping['counter_was_logged'] is False
    assert (tmp_path/'per_class_metrics.png').is_file()
    detail=json.loads((tmp_path/'learning_detail.json').read_text())
    history=json.loads((source/'epoch_history.json').read_text())
    assert detail['source_run']==source.name
    assert detail['epochs'][-1]['accuracy_gap_pp']==pytest.approx(
        100*(history[-1]['train_accuracy']-history[-1]['validation_accuracy']))
    assert (tmp_path/'learning_detail.png').is_file()
    assert {p.relative_to(source).as_posix():hash_file(p) for p in source.rglob('*') if p.is_file()}==before
