"""Kiểm toán và publish artifact của training selection, không tải model hoặc train."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.final_review import audit_frozen_trial
from labs.lab03.src.evidence import read_json
from labs.lab03.src.utils.runs import ROOT,save_json
from labs.lab03.scripts.publish_artifacts import publish_artifacts


def finalize():
    selection=ROOT/'outputs/latest_report.json'
    if not selection.is_file():
        raise FileNotFoundError('Chạy notebook modeling để tạo outputs/latest_report.json trước.')
    configuration=read_json(selection)
    run_id=configuration['runs']['frozen_trial']
    if Path(run_id).name != run_id or run_id in {'.','..'}:raise ValueError('Unsafe run ID')
    run=ROOT/'outputs/runs'/run_id
    audit=audit_frozen_trial(run)
    summary={'run_id':run_id,'status':'verified','selection':read_json(run/'selection_summary.json'),
        'threshold_policy':read_json(run/'decision_threshold.json'),
        'test_comparison':read_json(run/'test_comparison.json'),
        'test_runtime_metrics':read_json(run/'test_runtime_metrics.json'),
        'performance':read_json(run/'performance_summary.json'),
        'probability_diagnostics':read_json(run/'probability_diagnostics.json')}
    save_json(ROOT/'outputs/experiment_results.json',summary)
    save_json(ROOT/'validation/final_review_audit.json',audit)
    published=publish_artifacts(selection)
    print('Verified run:',run)
    print('Published report:',published)
    return run


if __name__=='__main__':finalize()
