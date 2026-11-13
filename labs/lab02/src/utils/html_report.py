"""Báo cáo HTML đọc output notebook và metric của một run được chọn rõ ràng."""
import csv
import json
import re
import os
from html import escape
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from .delivery import resolve_run
from .report_theme import HTML_PAGE_TEMPLATE, render_code_drawer, render_hero_card


def text(value):
    return ''.join(value) if isinstance(value, list) else str(value)


def read_notebook(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            if cell.get('execution_count') is None or not cell.get('outputs'):
                raise ValueError(f'Notebook chưa có output đầy đủ: {path.name}')
            if any(o.get('output_type') == 'error' for o in cell['outputs']):
                raise ValueError(f'Notebook chứa output lỗi: {path.name}')
    return notebook


def verify_notebook_run(notebook, run):
    streams = ''.join(text(o.get('text', '')) for c in notebook['cells']
                      for o in c.get('outputs', []) if o.get('output_type') == 'stream')
    recorded = re.findall(r'^Run: (\S+)', streams, re.MULTILINE)
    if recorded != [run.name]:
        raise ValueError('Run của output notebook không khớp. Chạy lại notebook ở MODE="view" cho run đã chọn.')


def metrics_table(run):
    with (run / 'comparison.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    columns = ['model', 'stage', 'test_samples', 'accuracy', 'macro_precision',
               'macro_recall', 'macro_f1', 'loss', 'best_val_accuracy']
    header = ''.join(f'<th>{escape(c)}</th>' for c in columns)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(row[c] if c in columns[:3] else format(float(row[c]), ".4f"))}</td>' for c in columns) + '</tr>' for row in rows)
    return f'<div class="table-scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def notebook_sections(notebook, prefix, root=None, destination=None):
    import markdown
    sections, navigation = [], []
    heading = prefix
    body = []
    index = 0

    def flush():
        nonlocal index
        if body:
            anchor = f'{prefix}-{index}'
            sections.append(f'<section class="card" id="{anchor}">' + ''.join(body) + '</section>')
            navigation.append((anchor, heading))
            index += 1
            body.clear()

    for cell in notebook['cells']:
        source = text(cell['source'])
        if cell['cell_type'] == 'markdown':
            first = source.splitlines()[0]
            if first.startswith('#'):
                flush()
                heading = first.lstrip('# ').strip()
            rendered = markdown.markdown(source, extensions=['tables', 'fenced_code'])
            if root is not None:
                def rebase(match):
                    link = urlsplit(match.group(2))
                    if link.scheme or link.netloc or not link.path:
                        return match.group(0)
                    target = (root / 'notebooks' / unquote(link.path)).resolve()
                    relative = quote(os.path.relpath(target, destination.parent))
                    return f'{match.group(1)}="{relative}' + (f'#{link.fragment}' if link.fragment else '') + '"'
                rendered = re.sub(r'(href|src)="([^"]+)"', rebase, rendered)
            body.append('<div class="notebook-text">' + rendered + '</div>')
            continue
        code = render_code_drawer('Code notebook tạo output này', escape(source))
        for output in cell.get('outputs', []):
            data = output.get('data', {})
            if 'image/png' in data:
                uri = 'data:image/png;base64,' + text(data['image/png']).replace('\n', '')
                body.append(render_hero_card(uri, escape(heading), 'Notebook output',
                            'Ảnh từ lần thực thi đã lưu; nhấn để xem kích thước lớn.', []))
                body.append(code)
            elif 'text/html' in data:
                body.append('<div class="table-scroll">' + text(data['text/html']) + '</div>')
            elif 'text/plain' in data or output.get('output_type') == 'stream':
                value = text(data.get('text/plain', output.get('text', '')))
                if root is not None:
                    value = value.replace(str(root) + '/', '')
                if value.strip():
                    body.append('<pre class="output-text">' + escape(value) + '</pre>')
        if not any('image/png' in o.get('data', {}) for o in cell.get('outputs', [])):
            body.append(code)
    flush()
    return ''.join(sections), navigation


def generate_report(root: Path, output: Path, run_path=None) -> Path:
    """Xác minh nguồn trước khi ghi HTML; giữ nguyên notebook và artifact gốc."""
    root, output = Path(root).resolve(), Path(output).resolve()
    run = resolve_run(root, run_path)
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    if status.get('status') != 'completed':
        raise ValueError('Chỉ xuất báo cáo cho run completed.')
    eda = read_notebook(root / 'notebooks/01_lab02_eda.ipynb')
    modeling = read_notebook(root / 'notebooks/02_lab02_modeling.ipynb')
    verify_notebook_run(modeling, run)
    environment = json.loads((run / 'environment_snapshot.json').read_text(encoding='utf-8'))
    from .notebook_views import class_metrics
    per_class = class_metrics(run).to_html(index=False, float_format=lambda value: f'{value:.4f}')
    eda_html, eda_nav = notebook_sections(eda, 'eda', root, output)
    model_html, model_nav = notebook_sections(modeling, 'modeling', root, output)
    labels = {'eda-0': '01 · Khám phá dữ liệu', 'eda-4': '02 · Tiền xử lý và đặc trưng',
              'modeling-2': '03 · Kiến trúc và freeze', 'modeling-3': '04 · Huấn luyện',
              'modeling-4': '05 · Đánh giá mô hình', 'modeling-8': '06 · Lựa chọn bằng validation',
              'modeling-9': '07 · Chi phí thực nghiệm', 'modeling-10': '08 · Save / load',
              'modeling-11': '09 · Benchmark một ảnh'}
    navigation = [(anchor, labels[anchor]) for anchor, _ in eda_nav + model_nav if anchor in labels]
    colors = ['#0284c7', '#7c3aed', '#059669', '#d97706', '#dc2626', '#0d9488']
    steps = ''.join(f'<a class="pipeline-card" href="#{anchor}" style="border:2px solid {colors[i % len(colors)]};color:{colors[i % len(colors)]}">'
                    f'<span class="pipeline-title">{escape(title)}</span></a>'
                    for i, (anchor, title) in enumerate(navigation))
    provenance = f'''<section class="card" id="provenance"><h2>Nguồn dữ liệu và phạm vi</h2>
    <p>Run: <code>{escape(run.name)}</code>. Metric dưới đây đọc trực tiếp từ comparison.csv;
    Accuracy, Precision, Recall và F1 dùng fraction 0–1, Recall là TPR. Validation dùng chọn checkpoint;
    test đánh giá sau lựa chọn. Không gọi best_val_accuracy là test accuracy.</p>
    <p>Ảnh và bảng trong các mục tiếp theo lấy từ output đã lưu của hai notebook chính.
    EDA là bộ artifact riêng; benchmark giữ điều kiện CPU/T4 và phạm vi đo đã ghi trong notebook.</p>
    <p><a href="walkthrough_lab_02.md">Walkthrough: đặc tả, phương pháp, kết quả và giới hạn</a> ·
    <a href="../README.md">Hướng dẫn Lab02</a></p>{metrics_table(run)}</section>'''
    page = HTML_PAGE_TEMPLATE.format(
        title='Báo cáo Lab02 — Transfer Learning trên CIFAR-10',
        subtitle='EDA · ResNet18 · DenseNet121 · Head-only · Fine-tuning · Evaluation',
        author='Nhóm thực hiện Lab02',
        python_ver=escape('Python ' + environment['python'].split()[0]),
        framework_ver=escape('PyTorch ' + environment['torch']),
        dataset_info='CIFAR-10 · RGB · 10 lớp',
        test_passed='Artifact của run completed', primary_metric='Metric có nguồn truy vết',
        stepper_html=steps, sections_html=provenance + eda_html + model_html +
        '<section class="card" id="class-metrics"><h2>Phụ lục: metric đầy đủ theo lớp</h2>'
        '<p>Precision, Recall (TPR), F1 và FPR trên thang 0–1, đọc từ bằng chứng của run đã chọn.</p>'
        '<div class="table-scroll">' + per_class + '</div></section>')
    extra = '''<style>
    .pipeline-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .notebook-text { overflow-x: auto; }
    .table-scroll { overflow-x: auto; margin: 1rem 0; }
    table { border-collapse: collapse; width: 100%; font-size: .88rem; }
    th, td { padding: .6rem; border: 1px solid #e2e8f0; text-align: left; }
    th { background: #eff6ff; } .card h2, .card h3 { margin: 1rem 0; color: #1e3a8a; }
    p, ul { margin: .8rem 0; } ul { padding-left: 1.5rem; }
    .output-text, pre { overflow-x: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
    .card { overflow-wrap: anywhere; } img { max-width: 100%; }
    .image-card-img-wrap img { max-height: none; object-fit: contain; }
    @media(max-width:600px) { .card { padding: 1rem; } header { padding: 1.5rem 1rem; } .pipeline-grid { grid-template-columns: 1fr; } }
    @media(prefers-reduced-motion:reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
    </style>'''
    page = page.replace('</head>', extra + '</head>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    return output
