"""Tính lại selection/metrics từ artifacts, lưu audit riêng dưới outputs/."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.evidence import selected_run
from labs.lab03.src.study_evidence import audit_study
from labs.lab03.src.utils.runs import run_context, save_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-run', type=Path)
    args = parser.parse_args()
    source = args.study_run or selected_run('study')
    result = audit_study(source)
    with run_context('study_audit', {'source_run':source.name}) as out:
        save_json(out / 'audit.json', result)
        print('Study audit PASSED:', source.name)
        print('Candidate epochs:', result['epochs_per_candidate'])
        print('Test records:', result['test_samples'])
