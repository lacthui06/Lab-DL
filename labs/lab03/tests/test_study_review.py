"""Review kết quả thật phải giữ đúng cohort, raw bytes và phạm vi suy luận."""
import json
import shutil
from labs.lab03.src.utils.runs import ROOT, hash_file, save_json
from labs.lab03.src.evidence import verify_manifest
from labs.lab03.src.study_review import ensure_study_review


def test_imported_bundle_preserves_every_original_file_checksum():
    imports=ROOT/'outputs/imports/20261026T184219Z_study_94f2ba7b'
    manifest=json.loads((imports/'bundle_manifest.json').read_text())
    assert len(manifest)==128
    for name,digest in manifest.items():
        path=imports/'original_config'/name.removeprefix('config/') if name.startswith('config/') else ROOT/name
        assert hash_file(path)==digest,name


def test_review_fixed_predictions_reports_overfitting_and_paired_uncertainty_without_mutation(tmp_path):
    ids=['20261026T184219Z_study_94f2ba7b','20261026T184223Z_candidate_4b1da858',
         '20261026T185017Z_candidate_202c0434','20261026T185804Z_candidate_8308037d',
         '20261026T155822Z_finetuning_2c2f0401']
    for run_id in ids:shutil.copytree(ROOT/'outputs/runs'/run_id,tmp_path/'outputs/runs'/run_id)
    save_json(tmp_path/'config/workflow.json',{'runs':[],'active_runs':{}})
    before={str(p):hash_file(p) for run_id in ids for p in (tmp_path/'outputs/runs'/run_id).rglob('*') if p.is_file()}
    out=ensure_study_review(tmp_path/'outputs/runs'/ids[0],root=tmp_path)
    result=json.loads((out/'review_summary.json').read_text());paired=result['paired_comparison']
    assert paired['study_correct']==889 and paired['historical_correct']==895
    assert paired['corrected_by_study']==42 and paired['new_errors_in_study']==48
    lo,hi=paired['paired_stratified_bootstrap_95_ci'];assert lo<0<hi
    candidates=json.loads((out/'tables.json').read_text())['candidates']
    winner=next(r for r in candidates if r['learning_rate']==3e-5)
    assert winner['final_train_accuracy']==1 and winner['completed_epochs']==10
    assert winner['best_step']==1700 and winner['final_validation_loss']>winner['loss']
    assert result['checkpoint_weights_loaded_locally'] is False
    assert result['threshold_reselected_on_test'] is False
    verify_manifest(out)
    assert ensure_study_review(tmp_path/'outputs/runs'/ids[0],root=tmp_path)==out
    assert all(hash_file(p)==digest for p,digest in before.items())
