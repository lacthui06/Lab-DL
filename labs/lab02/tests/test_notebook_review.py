"""Check saved metric evidence, view boundaries and links without torch.

Run: python -X utf8 -m unittest discover -s tests -p test_notebook_review.py -v
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.notebook_views import class_metrics
from src.utils.delivery import audit_snapshot


def test_single_image_view_uses_verified_artifacts_and_separate_units(tmp_path, monkeypatch):
    import shutil
    from src.utils import notebook_views as view
    evidence = ROOT/'outputs/published/single_image_benchmark/20261102T184034Z_kaggle'
    bundle = tmp_path/'benchmark'
    shutil.copytree(evidence, bundle)
    run = tmp_path/json.loads((bundle/'provenance.json').read_text())['training_run']
    run.mkdir()
    displayed = []
    monkeypatch.setattr(view, 'display', displayed.append)
    monkeypatch.setattr(view, '_save', lambda fig, output, name: view.plt.close(fig))
    tables = view.single_image_benchmark(run, bundle, tmp_path)
    assert tables['latency'].set_index('Model').loc['resnet18', 'Median (ms/image)'] == 3.761409
    assert tables['ram'].set_index('Model').loc['resnet18', 'RSS peak (MiB)'] == 1347350528/2**20
    assert tables['vram'].set_index('Model').loc['densenet121', 'Allocated peak (MiB)'] == 44929024/2**20
    assert len(tables['stages']) == 12
    assert 'torch' not in view.__dict__


def test_single_image_view_rejects_wrong_run_missing_and_corrupt_evidence(tmp_path, monkeypatch):
    import shutil
    import pytest
    from src.utils import notebook_views as view
    evidence = ROOT/'outputs/published/single_image_benchmark/20261102T184034Z_kaggle'
    bundle = tmp_path/'benchmark'
    shutil.copytree(evidence, bundle)
    identity = json.loads((bundle/'provenance.json').read_text())['training_run']
    monkeypatch.setattr(view, 'display', lambda *args: pytest.fail('Unverified evidence was displayed'))
    with pytest.raises(ValueError, match='training run'):
        view.single_image_benchmark(tmp_path/'other_run', bundle, tmp_path)
    with pytest.raises(FileNotFoundError):
        view.single_image_benchmark(tmp_path/identity, tmp_path/'missing', tmp_path)
    (bundle/'resnet18/summary.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        view.single_image_benchmark(tmp_path/identity, bundle, tmp_path)


def test_modeling_generator_has_roadmap_save_load_and_integrated_benchmark():
    import importlib.util
    import nbformat
    spec = importlib.util.spec_from_file_location('execute_notebook_review', ROOT/'scripts/execute_notebook.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    notebook = cli.scheduled_notebook(nbformat)
    markdown = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'markdown')
    code = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    assert 'Pipeline Execution Roadmap' in markdown
    assert 'RAM/VRAM' in markdown
    assert 'view.single_image_benchmark(RUN, BENCHMARK, OUT)' in code
    assert 'view.roundtrip(RUN)' in code
    assert 'subprocess.run' not in code and 'benchmark_loaded_model(' not in code


class NotebookReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir = ROOT / json.loads((ROOT / 'outputs/selected_run.json').read_text())['run']

    def test_saved_metrics_match_confusion_counts(self):
        """Independent confusion evidence must reproduce all non-loss aggregates."""
        per_class = class_metrics(self.run_dir)
        for row in pd.read_csv(self.run_dir / 'comparison.csv').itertuples():
            folder = 'head_selected' if row.stage == 'head' else 'finetune'
            counts = pd.read_csv(self.run_dir / row.model / folder / 'confusion_counts.csv', index_col=0)
            values = counts.to_numpy()
            self.assertEqual(int(values.sum()), row.test_samples)
            self.assertAlmostEqual(np.trace(values)/values.sum(), row.accuracy, places=12)
            part = per_class[(per_class.model == row.model) & (per_class.stage == row.stage)]
            self.assertAlmostEqual(part.precision.mean(), row.macro_precision, places=12)
            self.assertAlmostEqual(part.recall_tpr.mean(), row.macro_recall, places=12)
            self.assertAlmostEqual(part.f1.mean(), row.macro_f1, places=12)
            # FPR is the actual fraction of negative samples predicted positive.
            for i, actual in enumerate(part.itertuples()):
                false_positive = values[:, i].sum() - values[i, i]
                negatives = values.sum() - values[i].sum()
                self.assertAlmostEqual(actual.fpr, false_positive/negatives, places=12)

    def test_pipeline_time_has_no_double_counted_selected_head(self):
        costs = pd.read_csv(self.run_dir / 'cost_comparison.csv')
        for row in costs.itertuples():
            total = row.feature_extraction_seconds + row.all_head_trials_fit_seconds
            if row.stage == 'finetune':
                total += row.all_finetune_trials_fit_seconds
            self.assertAlmostEqual(row.pipeline_training_seconds, total, places=8)
        fine = costs[costs.stage == 'finetune']
        recorded = pd.read_csv(self.run_dir/'comparison.csv').set_index(['model','stage'])
        for row in fine.itertuples():
            self.assertEqual(row.execution, recorded.loc[(row.model,row.stage),'execution'])
            self.assertIn(row.execution, {'full_image_forward', 'cached_frozen_prefix'})

    def test_views_import_without_training_dependencies(self):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c',
            "import sys; from src.utils import notebook_views; "
            "assert 'torch' not in sys.modules; assert 'src.models' not in sys.modules; "
            "assert 'src.experiment' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_notebooks_are_executed_compact_and_visual(self):
        for name in ('01_lab02_eda.ipynb', '02_lab02_modeling.ipynb', '02_lab02_modeling-kaggle.ipynb'):
            path = ROOT / 'notebooks' / name
            notebook = json.loads(path.read_text(encoding='utf-8'))
            cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
            self.assertTrue(cells)
            image_count = 0
            for index, cell in enumerate(cells):
                source = ''.join(cell['source'])
                inline_limits = {'lab02-tuning-05': 24, 'lab02-roc-pr-output': 50}
                self.assertLessEqual(len(source.splitlines()), 24 if index==0 else inline_limits.get(cell.get('id'), 8), (path.name, source))
                self.assertIsNotNone(cell['execution_count'], path.name)
                self.assertTrue(cell['outputs'], (path.name, source))
                for output in cell['outputs']:
                    self.assertNotEqual(output.get('output_type'), 'error', path.name)
                    image_count += 'image/png' in output.get('data', {})
            self.assertGreaterEqual(image_count, 4, path.name)
        eda = json.loads((ROOT / 'notebooks/01_lab02_eda.ipynb').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in eda['cells'] if c['cell_type'] == 'code')
        for forbidden in ['src.models', 'src.experiment', 'torch', 'build_model', 'fit(', 'architecture_summary']:
            self.assertNotIn(forbidden, code)

    def test_current_document_and_notebook_links_resolve(self):
        files = [ROOT/'README.md', ROOT/'reports/walkthrough_lab_02.md', *(ROOT/'notebooks').glob('*.ipynb')]
        for path in files:
            if path.suffix == '.ipynb':
                notebook = json.loads(path.read_text(encoding='utf-8'))
                text = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'markdown')
            else:
                text = path.read_text(encoding='utf-8')
            for target in re.findall(r'\]\(([^)]+)\)', text):
                target = unquote(target.split('#', 1)[0])
                if not target or '://' in target:
                    continue
                self.assertTrue((path.parent/target).exists(), f'{path.name}: {target}')

    def test_historical_snapshots_have_no_changes_beyond_git_newlines(self):
        for prefix in ['', 'resume_', 'delivery_']:
            hash_path = self.run_dir / f'{prefix}source_hashes.json'
            if prefix == 'resume_' and not hash_path.exists():
                self.assertFalse((self.run_dir/'resume_source_snapshot').exists())
                continue
            hashes = json.loads(hash_path.read_text())
            audit = audit_snapshot(self.run_dir / f'{prefix}source_snapshot', hashes)
            self.assertEqual(audit['byte_exact_files'] + len(audit['newline_only_files']), len(hashes))

    def test_snapshot_audit_marks_newline_conversion_and_rejects_content_changes(self):
        original = b'learning_rate = 0.001\n'
        hashes = {'config.py': hashlib.sha256(original).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            path = snapshot / 'config.py'
            path.write_bytes(original.replace(b'\n', b'\r\n'))
            audit = audit_snapshot(snapshot, hashes)
            self.assertEqual(audit['byte_exact_files'], 0)
            self.assertEqual(audit['newline_only_files'], ['config.py'])
            (snapshot/'src').mkdir()
            (snapshot/'src/config.py').write_bytes(original)
            windows = audit_snapshot(snapshot, {'src\\config.py': hashes['config.py']})
            self.assertEqual(windows['byte_exact_files'], 1)
            path.write_bytes(b'learning_rate = 0.1\n')
            with self.assertRaises(ValueError):
                audit_snapshot(snapshot, hashes)


if __name__ == '__main__':
    unittest.main()


def test_context_and_notebook_bootstrap_with_missing_cwd(tmp_path, monkeypatch):
    import os
    from src.utils import notebook_views as view
    lab = tmp_path / 'lab02'
    (lab / 'src/utils').mkdir(parents=True)
    (lab / 'src/utils/notebook_views.py').touch()
    (lab / 'outputs').mkdir()
    (lab / 'outputs/selected_run.json').write_text(json.dumps({'run': 'outputs/runs/example'}))
    notebook = json.loads((ROOT / 'notebooks/02_lab02_modeling.ipynb').read_text())
    bootstrap = ''.join(next(c for c in notebook['cells'] if c['cell_type']=='code')['source']).split('sys.path.insert')[0]

    def missing_cwd():
        raise FileNotFoundError('Working directory was deleted')

    monkeypatch.setattr(Path, 'cwd', staticmethod(missing_cwd))
    changes = []
    monkeypatch.setattr(os, 'chdir', changes.append)
    monkeypatch.delenv('LAB02_RUN', raising=False)
    monkeypatch.setenv('LAB02_ROOT', str(lab))
    namespace = {}
    exec(bootstrap, namespace)
    assert namespace['ROOT'] == lab
    assert changes == [lab]
    assert view.context()[0] == lab
    monkeypatch.setenv('LAB02_ROOT', str(tmp_path / 'missing'))
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'Invalid Lab02 directory'):
        view.context()
    monkeypatch.delenv('LAB02_ROOT')
    monkeypatch.setattr(view, '__file__', str(lab / 'src/utils/notebook_views.py'))
    assert view.context()[0] == lab
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'absolute Lab02 directory'):
        exec(bootstrap, {})
    monkeypatch.setenv('LAB02_ROOT', 'relative/lab02')
    with unittest.TestCase().assertRaisesRegex(RuntimeError, 'absolute Lab02 directory'):
        exec(bootstrap, {})


def test_all_notebook_bootstraps_resolve_nested_and_explicit_roots(tmp_path, monkeypatch):
    import os
    import nbformat
    import importlib.util
    spec=importlib.util.spec_from_file_location('execute_notebook',ROOT/'scripts/execute_notebook.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    # The archived GPU runner resolves Kaggle input, not the local Lab02 source root.
    names=('01_lab02_eda.ipynb','02_lab02_modeling.ipynb','02_lab02_modeling-kaggle.ipynb')
    sources=[''.join(next(c for c in json.loads((ROOT/'notebooks'/name).read_text())['cells']
                         if c['cell_type']=='code')['source']) for name in names]
    sources.append(next(c for c in cli.scheduled_notebook(nbformat).cells if c.cell_type=='code').source)
    changes=[]
    monkeypatch.setattr(os,'chdir',changes.append)
    for source in sources:
        bootstrap=source.split('sys.path.insert')[0]
        monkeypatch.delenv('LAB02_ROOT',raising=False)
        monkeypatch.setattr(Path,'cwd',staticmethod(lambda: ROOT/'notebooks'))
        namespace={};exec(bootstrap,namespace)
        assert namespace['ROOT']==ROOT and changes[-1]==ROOT
        monkeypatch.setenv('LAB02_ROOT',str(ROOT))
        monkeypatch.setattr(Path,'cwd',staticmethod(lambda: (_ for _ in ()).throw(FileNotFoundError())))
        namespace={};exec(bootstrap,namespace)
        assert namespace['ROOT']==ROOT
        monkeypatch.setenv('LAB02_ROOT',str(tmp_path))
        monkeypatch.setattr(Path,'cwd',staticmethod(lambda: ROOT))
        with unittest.TestCase().assertRaisesRegex(RuntimeError,'absolute Lab02 directory'):
            exec(bootstrap,{})
        monkeypatch.setenv('LAB02_ROOT','relative/lab02')
        with unittest.TestCase().assertRaisesRegex(RuntimeError,'absolute Lab02 directory'):
            exec(bootstrap,{})
        monkeypatch.delenv('LAB02_ROOT')
        monkeypatch.setattr(Path,'cwd',staticmethod(lambda: tmp_path))
        with unittest.TestCase().assertRaisesRegex(RuntimeError,'absolute Lab02 directory'):
            exec(bootstrap,{})


def test_architecture_labels_match_each_saved_unfreeze_depth(tmp_path, monkeypatch):
    from src.utils import notebook_views as view
    expected = {
        'last_units': ('layer4.1 + fc', 'denseblock4.denselayer13–16 + norm5 + classifier'),
        'last_block': ('layer4 + fc', 'denseblock4 + norm5 + classifier'),
        'last_two_blocks': ('layer3 + layer4 + fc',
                            'denseblock3 + transition3 + denseblock4 + norm5 + classifier'),
    }
    displayed = []
    monkeypatch.setattr(view, 'display', displayed.append)
    for depth, labels in expected.items():
        pd.DataFrame([
            dict(model=model, stage=stage, total=100, trainable=10, unfreeze=depth)
            for model in ('resnet18', 'densenet121') for stage in ('head', 'finetune')
        ]).to_csv(tmp_path / 'comparison.csv', index=False)
        view.architecture_summary(tmp_path)
        table = displayed.pop()
        assert table.loc[table.stage == 'head', 'unfrozen_blocks'].tolist() == ['classifier'] * 2
        assert table.loc[table.stage == 'finetune', 'unfrozen_blocks'].tolist() == list(labels)


def test_unexecuted_fine_grid_is_explanatory_table_not_results(tmp_path, monkeypatch):
    from src.utils import notebook_views as view
    tables = []
    monkeypatch.setattr(view, 'display', tables.append)
    monkeypatch.setattr(view, '_save', lambda *a: (_ for _ in ()).throw(AssertionError('No measured plot without results')))
    view.fine_search(tmp_path, tmp_path, ROOT)
    assert len(tables) == 2
    assert set(tables[0]['Status']) == {'NOT RUN'}
    assert not {'accuracy', 'loss', 'train_seconds'} & set(tables[0].columns)
    assert tables[1]['Head search batches'].iloc[0] == '[64, 128]'


def test_training_benchmark_view_and_template_never_start_training(tmp_path):
    import importlib.util
    import nbformat
    spec = importlib.util.spec_from_file_location('execute_training_view', ROOT/'scripts/execute_notebook.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for notebook in (cli.scheduled_notebook(nbformat), json.loads(
            (ROOT/'notebooks/02_lab02_modeling.ipynb').read_text())):
        sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
        source = next(s for s in sources if 'view.training_benchmark(' in s)
        assert 'subprocess' not in source and 'benchmark_steps' not in source
        assert 'benchmark_training.py' in source
    from src.utils import notebook_views as view
    with unittest.TestCase().assertRaises(FileNotFoundError):
        view.training_benchmark(tmp_path, tmp_path/'missing', tmp_path)


def test_learning_plot_separates_stages_and_preserves_original_histories(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from src.utils.reporting import plot_learning_curves
    run = ROOT/json.loads((ROOT/'outputs/selected_run.json').read_text())['run']
    histories = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*/*/history.csv')}
    inspected = []
    original_save = Figure.savefig

    def save(fig, *args, **kwargs):
        inspected.append([(ax.get_title(), [line.get_label() for line in ax.lines]) for ax in fig.axes])
        return original_save(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', save)
    paths = plot_learning_curves(run, tmp_path)
    assert len(paths) == 2 and all(p.exists() for p in paths)
    config = json.loads((run/'config.json').read_text())
    models = pd.read_csv(run/'comparison.csv').model.unique()
    for model, axes in zip(models, inspected, strict=True):
        head_epochs = len(pd.read_csv(run/model/'head_selected/history.csv'))
        fine_epochs = len(pd.read_csv(run/model/'finetune/history.csv'))
        assert len(axes) == 4
        assert all('HEAD ONLY' in title and f'{head_epochs}/{config["head_epochs"]} epochs' in title for title, _ in axes[:2])
        assert all('FINE-TUNE' in title and f'{fine_epochs}/{config["finetune_epochs"]} epochs' in title for title, _ in axes[2:])
        assert all(labels == ['Train', 'Validation'] for _, labels in axes)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in histories.items())
