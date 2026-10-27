"""Hiển thị study ngay trong notebook kernel; không import PyTorch từ venv khác."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]

def display_study(study, root=ROOT, show_step_diagnostics=False):
    """Render chính file PNG vào kernel notebook, không gọi qua subprocess."""
    import pandas as pd
    from IPython.display import display, Image, Markdown
    study = Path(study).resolve(); root = Path(root).resolve()
    if study.parent != root / 'outputs/runs': raise ValueError('Invalid study path')
    if json.loads((study / 'status.json').read_text())['status'] != 'completed':
        raise ValueError('Study did not complete')
    display(Markdown('## So sánh validation và quyết định đã khóa'))
    rows = json.loads((study / 'candidate_comparison.json').read_text())
    display(pd.DataFrame(rows))
    baseline_candidates = json.loads((study / 'baseline_validation_candidates.json').read_text())
    display(Markdown('### Baseline: C chọn bằng validation; vocabulary fit chỉ trên train'))
    display(pd.DataFrame([{k:v for k,v in r.items() if k not in {'validation_probabilities','validation_labels'}} for r in baseline_candidates]))
    display(Markdown('### Audit overlap: không thay đổi official splits'))
    display(json.loads((study / 'split_overlap_audit.json').read_text()))
    display(json.loads((study / 'frozen_selection.json').read_text()))
    shown = []
    for row in rows:
        run = root / 'outputs/runs' / row['run_id']
        display(Markdown(f"### Candidate LR={row['learning_rate']:g} — đủ 10 epoch"))
        display(json.loads((run / 'selection_summary.json').read_text()))
        history = json.loads((run / 'trainer_history.json').read_text())
        epoch_file = run / 'epoch_history.json'
        display(pd.DataFrame(json.loads(epoch_file.read_text()) if epoch_file.is_file() else [r for r in history if 'eval_macro_f1' in r]))
        for name in ['learning_curves.png', 'step_learning_curves.png', 'token_length_distribution.png', 'validation_threshold/threshold_curves.png']:
            if name == 'step_learning_curves.png' and not show_step_diagnostics:
                continue
            path = run / name
            display(Markdown(f'**{name}**')); display(Image(filename=str(path))); shown.append(path)
    display(Markdown('## Test sau khi chốt cấu hình và threshold'))
    display(pd.DataFrame(json.loads((study / 'test_comparison.json').read_text())))
    display(json.loads((study / 'probability_diagnostics.json').read_text()))
    for path in sorted(study.rglob('*.png')):
        display(Markdown(f'**{path.relative_to(study)}**')); display(Image(filename=str(path))); shown.append(path)
    display(Markdown('### Lỗi dưới policy threshold đã chốt (tối đa 20 dòng hiển thị)'))
    display(pd.DataFrame(json.loads((study / 'threshold_errors.json').read_text())).head(20))
    display(pd.DataFrame(json.loads((study / 'inference_predictions.json').read_text())))
    display(Markdown('Confidence calibration ở trên đo argmax; threshold errors đo policy đã chọn. CI chỉ mô tả bất định lấy mẫu test, chưa đo bất định do seed huấn luyện.'))
    print('Hình đã hiển thị trực tiếp:', len(shown))
    return shown
