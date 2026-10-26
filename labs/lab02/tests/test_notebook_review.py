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
from src.notebook_views import class_metrics
from src.delivery import audit_snapshot


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
                total += row.selected_stage_fit_seconds
            self.assertAlmostEqual(row.pipeline_training_seconds, total, places=8)
        fine = costs[costs.stage == 'finetune']
        self.assertEqual(set(fine.execution), {'full_image_forward', 'cached_frozen_prefix'})

    def test_views_import_without_training_dependencies(self):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c',
            "import sys; from src import notebook_views; "
            "assert 'torch' not in sys.modules; assert 'src.models' not in sys.modules; "
            "assert 'src.experiment' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_notebooks_are_executed_compact_and_visual(self):
        for path in (ROOT / 'notebooks').glob('*.ipynb'):
            notebook = json.loads(path.read_text(encoding='utf-8'))
            cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
            self.assertTrue(cells)
            image_count = 0
            for cell in cells:
                source = ''.join(cell['source'])
                self.assertLessEqual(len(source.splitlines()), 8, (path.name, source))
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
        files = [ROOT/'README.md', ROOT/'walkthrough_lab_02.md', *(ROOT/'notebooks').glob('*.ipynb')]
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
            hashes = json.loads((self.run_dir / f'{prefix}source_hashes.json').read_text())
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
            path.write_bytes(b'learning_rate = 0.1\n')
            with self.assertRaises(ValueError):
                audit_snapshot(snapshot, hashes)


if __name__ == '__main__':
    unittest.main()
