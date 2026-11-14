"""Artifact cost views must run without importing the model or training engine."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cost_views_and_validation_selection_do_not_import_training_engine(tmp_path):
    rows = []
    for name, seconds, test_accuracy in [('resnet18', 7, .2), ('densenet121', 9, .99)]:
        folder = tmp_path / name / 'head_selected'
        folder.mkdir(parents=True)
        (folder / 'training_summary.json').write_text(json.dumps({
            'feature_extraction_train_val_seconds': 10,
            'all_search_training_seconds': 20, 'train_seconds': 5}))
        rows.append(dict(model=name, stage='finetune', train_seconds=seconds,
            all_finetune_search_seconds=30, test_seconds=3, trainable=100,
            execution='full_image_forward', best_val_accuracy=.8, best_val_loss=.4,
            loss=.5, accuracy=test_accuracy, macro_precision=.7, macro_recall=.7, macro_f1=.7))
    (tmp_path / 'config.json').write_text('{}')
    code = '''
import importlib.abc
import json
from pathlib import Path
import sys

class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ('torch', 'torchvision', 'src.models', 'src.training.experiment')
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('Unexpected training import: ' + fullname)

sys.meta_path.insert(0, BlockTraining())
import pandas as pd
from src.utils.costs import cost_table, validation_recommendation
import matplotlib
matplotlib.use('Agg')
from src.utils import notebook_views as view
import matplotlib.pyplot as plt
run = Path(sys.argv[1])
table = pd.DataFrame(json.loads(sys.argv[2]))
table.to_csv(run / 'comparison.csv', index=False)
costs = cost_table(run, table)
assert costs.pipeline_training_seconds.tolist() == [60, 60]
assert costs.selected_recipe_training_seconds.tolist() == [22, 24]
choice = validation_recommendation(run, table, {})
assert choice['model'] == 'resnet18'  # Test accuracy must not select the model.
assert choice['time_comparable'] is True
view._save = lambda figure, *args: plt.close(figure)
view.display = lambda *args: None
view.quality_cost(run, run)
view.efficiency(run, run)
assert not any(name in sys.modules for name in ('torch', 'torchvision', 'src.models', 'src.training.experiment'))
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path), json.dumps(rows)],
                            cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('package', ['src.training', 'src.utils'])
def test_package_exports_are_lazy_and_keep_public_names(package):
    code = '''
import importlib
import sys
package = importlib.import_module(sys.argv[1])
assert not any(name in sys.modules for name in ('torch', 'matplotlib', 'pandas', 'src.training.experiment'))
from src.utils.costs import cost_table
from src.training.tuning import select_trial
assert package.cost_table is cost_table
if sys.argv[1] == 'src.training':
    assert package.select_trial is select_trial
else:
    from src.utils.delivery import iter_source_files
    assert package.iter_source_files is iter_source_files
try:
    package.unknown_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export must raise AttributeError')
assert 'torch' not in sys.modules and 'src.training.experiment' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, package], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
