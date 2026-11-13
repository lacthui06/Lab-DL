"""HTML phải có nguồn đúng run, code được escape và đủ liên kết nội bộ."""
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from src.utils.html_report import generate_report, notebook_sections, verify_notebook_run

ROOT = Path(__file__).resolve().parents[1]


def test_report_has_saved_figures_metrics_navigation_and_no_lab01_claims(tmp_path):
    path = generate_report(ROOT, tmp_path / 'report.html')
    html = path.read_text()
    assert '0.8414' in html and '0.918' in html
    assert html.count('data:image/png;base64,') >= 10
    assert '<details class="code-tab-drawer">' in html
    assert 'openModal(this)' in html
    assert '49/49' not in html and 'FashionMNIST' not in html
    ids = set(re.findall(r'id="([^"]+)"', html))
    for anchor in re.findall(r'href="#([^"]+)"', html):
        assert anchor in ids


def test_report_rejects_mixed_run_outputs():
    notebook = {'cells': [{'outputs': [{'output_type': 'stream', 'text': 'Run: old\n'}]}]}
    with pytest.raises(ValueError, match='không khớp'):
        verify_notebook_run(notebook, Path('new'))


def test_code_is_escaped_in_drawer():
    nb = {'cells': [{'cell_type': 'code', 'source': 'print("<script>alert(1)</script>")',
                     'outputs': [{'output_type': 'stream', 'text': '<unsafe>'}]}]}
    html, _ = notebook_sections(nb, 'test')
    assert '<script>alert(1)</script>' not in html
    assert '&lt;script&gt;' in html and '&lt;unsafe&gt;' in html


def test_notebook_links_rebase_to_report_destination(tmp_path):
    nb = {'cells': [{'cell_type': 'markdown', 'source':
                    '# Handoff\n[Modeling](02_lab02_modeling.ipynb)\n[Source](../src/utils/notebook_views.py)'}]}
    output = tmp_path / 'report.html'
    html, _ = notebook_sections(nb, 'eda', ROOT, output)
    for href in re.findall(r'href="([^"]+)"', html):
        assert (output.parent / unquote(urlsplit(href).path)).is_file()
