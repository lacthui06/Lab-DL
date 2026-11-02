"""Tái kiểm toán và xuất artifact report từ kết quả đã lưu; không train."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from labs.lab03.src.final_review import ensure_final_review
from labs.lab03.src.evidence import read_json
from labs.lab03.src.utils.runs import ROOT, save_json
from labs.lab03.scripts.publish_artifacts import publish_artifacts


def finalize():
    out = ensure_final_review()
    configuration = read_json(ROOT/'config/report.json')
    configuration['primary_result'] = 'frozen_trial'
    configuration['runs']['final_review'] = out.name
    save_json(ROOT/'config/report.json', configuration)
    trial = ROOT/'outputs/runs'/configuration['runs']['frozen_trial']
    summary = read_json(out/'review_summary.json')
    summary.update(test_threshold_metrics=read_json(trial/'test_comparison.json')[-1],
        test_metrics=read_json(trial/'test_runtime_metrics.json'),
        measured_test_loss_after_checkpoint_reload=read_json(trial/'test_runtime_metrics.json')['test_loss'],
        probability_diagnostics=read_json(trial/'probability_diagnostics.json'),
        per_class_metrics=read_json(out/'per_class_metrics.json'),
        generalization_gaps=read_json(out/'generalization_gaps.json'),
        comparison_is_descriptive=True)
    save_json(ROOT/'outputs/experiment_results.json', summary)
    save_json(ROOT/'validation/final_review_audit.json', {'review_run':out.name, **read_json(out/'audit.json')})
    published = publish_artifacts()
    print('Final review:', out)
    print('Published artifacts:', published)
    return out


if __name__ == '__main__':
    finalize()
