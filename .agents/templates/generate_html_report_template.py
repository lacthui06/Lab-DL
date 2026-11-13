r"""Template script to generate standardized, visual-first HTML reports for Lab DL.
Based on the UTH Computer Vision template style.

Best Practices:
1. Visual-first & Minimal Text: Tap trung bieu do, anh truc quan va bang so lieu. Han che viet chu dai dong.
2. Code di lien voi anh: Moi anh visualize (Hero, Duo, Trio) truyen kem `code_snippet` (2-4 dong tu notebook)
   de tu dong tao Code Drawer nam NGAY DUOI BUC ANH DANG VISUALIZE.
3. Khong dung Raw LaTeX: Tuyet doi khong dung \frac, \sqrt, \beta, $...$ vi trinh duyet khong co MathJax
   se render tho bi loi chu. Hay dung text unicode va ky hieu toan hoc pho thong.

Usage for a new lab:
1. Copy this file to labs/lab0X/scripts/generate_html_report.py
2. Update LAB_CONFIG (title, author, badges, etc.)
3. Update PIPELINE_STEPS for the interactive colored stepper
4. Configure your Sections with helper functions:
   - hero_card(image_path, title, tag, takeaway, footer_items, code_snippet=...)
   - duo_cards(card1_dict, card2_dict)
   - trio_cards(card1_dict, card2_dict, card3_dict)
   - code_drawer(summary_text, code_str, language)
   - integrity_table(...)
5. Run: python labs/lab0X/scripts/generate_html_report.py
"""

from pathlib import Path

# ==============================================================================
# 1. LAB CONFIGURATION (Thay doi thong tin lab o day)
# ==============================================================================
LAB_CONFIG = {
    "lab_num": 1,
    "lab_id": "lab01",
    "title": "Bao Cao Nghiem Thu Ky Thuat Lab 1",
    "subtitle": "Phan Loai FashionMNIST voi PyTorch & Handcrafted Features",
    "author": "Thai Anh Lac",
    "python_ver": "Python 3.13.15",
    "framework_ver": "PyTorch 2.14.0",
    "dataset_info": "FashionMNIST (70.000 anh)",
    "test_passed": "49/49 Tests Passed (100%)",
    "primary_metric": "Best Test Acc: 89.2%",
}

# 6 mau gradient chuan cho 6 buoc pipeline (tuong ung UTH modern style)
STEP_GRADIENTS = [
    {"bg": "#f0f9ff", "border": "#38bdf8", "text": "#0369a1", "badge": "linear-gradient(135deg, #0284c7, #0369a1)"},
    {"bg": "#f5f3ff", "border": "#a78bfa", "text": "#6d28d9", "badge": "linear-gradient(135deg, #7c3aed, #6d28d9)"},
    {"bg": "#eff6ff", "border": "#60a5fa", "text": "#1d4ed8", "badge": "linear-gradient(135deg, #2563eb, #1d4ed8)"},
    {"bg": "#fffbeb", "border": "#fbbf24", "text": "#b45309", "badge": "linear-gradient(135deg, #d97706, #b45309)"},
    {"bg": "#ecfdf5", "border": "#34d399", "text": "#047857", "badge": "linear-gradient(135deg, #059669, #047857)"},
    {"bg": "#f0fdfa", "border": "#2dd4bf", "text": "#0f766e", "badge": "linear-gradient(135deg, #0d9488, #0f766e)"},
]

# Danh sach cac buoc tren pipeline stepper
PIPELINE_STEPS = [
    {
        "num": "01",
        "title": "1. Kham Pha EDA",
        "desc": "Kiem Toan 0 NaN &bull; 0 Duplicate &bull; Phau Thuat Anh",
        "anchor": "#section-eda",
    },
    {
        "num": "02",
        "title": "2. Feature Engineering",
        "desc": "Bounding Box &bull; Centroid &bull; StandardScaler",
        "anchor": "#section-features",
    },
    {
        "num": "03",
        "title": "3. Kien Truc Mo Hinh",
        "desc": "SmallMLP &bull; MainMLP &bull; Model_FE",
        "anchor": "#section-models",
    },
    {
        "num": "04",
        "title": "4. Toi Uu & Tuning",
        "desc": "SGD &bull; RMSprop &bull; Adam (30 Trials)",
        "anchor": "#section-tuning",
    },
    {
        "num": "05",
        "title": "5. Danh Gia Test",
        "desc": "Acc &bull; Confusion Matrix &bull; OvR Curves",
        "anchor": "#section-evaluation",
    },
    {
        "num": "06",
        "title": "6. Kiem Toan QA",
        "desc": "Test Suite Passed &bull; Checkpoint Verification",
        "anchor": "#section-qa",
    },
]


# ==============================================================================
# 2. HELPER FUNCTIONS TO RENDER COMPONENTS
# ==============================================================================
def _render_card_drawer(code_snippet: str | None) -> str:
    if not code_snippet:
        return ""
    return f"""
        <details class="code-tab-drawer">
            <summary class="code-tab-summary">
                <div class="code-summary-left">
                    <span class="code-pill">NOTEBOOK CELL</span>
                    <span>Cell truc quan hoa anh nay</span>
                </div>
                <span class="code-badge">Xem code cell &dtrif;</span>
            </summary>
            <pre><code class="language-python">{code_snippet.strip()}</code></pre>
        </details>"""


def render_hero_card(img_path: str, title: str, tag: str, takeaway: str, footer_items: list[str], code_snippet: str | None = None) -> str:
    """Render full-width hero card for primary matrices or overview charts."""
    footer_html = "".join(f"<span>{item}</span>" for item in footer_items)
    drawer_html = _render_card_drawer(code_snippet)
    return f"""
    <figure class="image-card">
        <div class="image-card-img-wrap" onclick="openModal(this)">
            <img src="{img_path}" alt="{title}" loading="lazy">
            <span class="zoom-hint">Phong to [Zoom] &nearr;</span>
        </div>
        <figcaption class="image-card-caption">
            <div class="image-card-header">
                <span class="image-card-title">{title}</span>
                <span class="code-pill">{tag}</span>
            </div>
            <p class="image-card-takeaway">{takeaway}</p>
            <div class="image-card-footer">
                {footer_html}
            </div>
        </figcaption>
{drawer_html}
    </figure>
    """


def render_duo_cards(card1: dict, card2: dict) -> str:
    """Render 2-column comparative layout."""
    def _card(c):
        footer_html = "".join(f"<span>{item}</span>" for item in c.get("footer_items", []))
        drawer_html = _render_card_drawer(c.get("code_snippet"))
        return f"""
        <figure class="image-card">
            <div class="image-card-img-wrap" onclick="openModal(this)">
                <img src="{c['img_path']}" alt="{c['title']}" loading="lazy">
                <span class="zoom-hint">Phong to [Zoom] &nearr;</span>
            </div>
            <figcaption class="image-card-caption">
                <div class="image-card-header">
                    <span class="image-card-title">{c['title']}</span>
                    <span class="code-pill">{c['tag']}</span>
                </div>
                <p class="image-card-takeaway">{c['takeaway']}</p>
                <div class="image-card-footer">
                    {footer_html}
                </div>
            </figcaption>
{drawer_html}
        </figure>
        """
    return f"""
    <div class="grid-2col">
        {_card(card1)}
        {_card(card2)}
    </div>
    """


def render_trio_cards(card1: dict, card2: dict, card3: dict) -> str:
    """Render 3-column layout for sequences or small comparative figures."""
    def _card(c):
        footer_html = "".join(f"<span>{item}</span>" for item in c.get("footer_items", []))
        drawer_html = _render_card_drawer(c.get("code_snippet"))
        return f"""
        <figure class="image-card">
            <div class="image-card-img-wrap" onclick="openModal(this)">
                <img src="{c['img_path']}" alt="{c['title']}" loading="lazy">
                <span class="zoom-hint">Phong to [Zoom] &nearr;</span>
            </div>
            <figcaption class="image-card-caption">
                <div class="image-card-header">
                    <span class="image-card-title">{c['title']}</span>
                    <span class="code-pill">{c['tag']}</span>
                </div>
                <p class="image-card-takeaway">{c['takeaway']}</p>
                <div class="image-card-footer">
                    {footer_html}
                </div>
            </figcaption>
{drawer_html}
        </figure>
        """
    return f"""
    <div class="grid-3col">
        {_card(card1)}
        {_card(card2)}
        {_card(card3)}
    </div>
    """


def render_code_drawer(title: str, code_str: str, language: str = "python") -> str:
    """Render collapsible code drawer (closed by default)."""
    return f"""
    <details class="code-tab-drawer">
        <summary class="code-tab-summary">
            <div class="code-summary-left">
                <span class="code-pill">{language.upper()}</span>
                <span>{title}</span>
            </div>
            <span class="code-badge">Nhan de mo code &dtrif;</span>
        </summary>
        <pre><code class="language-{language}">{code_str.strip()}</code></pre>
    </details>
    """


def render_data_integrity_section(audit_rows: list[dict], struct_rows: list[dict], code_example: str) -> str:
    """Render standard Core EDA data integrity, missing values, and duplicate audit."""
    audit_tbody = ""
    for row in audit_rows:
        audit_tbody += f"""
        <tr>
            <td><strong>{row['item']}</strong></td>
            <td>{row['expected']}</td>
            <td>{row['actual']}</td>
            <td style="text-align: center;"><span class="badge badge-success">{row['status']}</span></td>
        </tr>
        """

    struct_tbody = ""
    for r in struct_rows:
        struct_tbody += f"""
        <tr>
            <td><strong>{r['pool']}</strong></td>
            <td>{r['count']}</td>
            <td>{r['dim']}</td>
            <td>{r['channel']}</td>
            <td><code>{r['dtype']}</code></td>
            <td>{r['pixel_range']}</td>
            <td>{r['label_range']}</td>
            <td>{r['classes']}</td>
            <td>{r['ram']}</td>
        </tr>
        """

    return f"""
    <div class="subsection-title">
        <span>Kiem Toan Tinh Toan Ven Du Lieu, Gia Tri Thieu & Trung Lap (Data Integrity & Duplicate Audit)</span>
        <span class="badge badge-info">CORE EDA &bull; 0 NaN &bull; 0 DUPLICATES</span>
    </div>

    <p style="font-size: 0.92rem; color: var(--text-dark); margin-bottom: 1rem;">
        Theo quy chuan Core EDA, toan bo du lieu da duoc kiem toan toan ven ve mat cau truc, quet gia tri thieu/loi (NaN/Inf)
        va quet ma bam mat ma hoc (MD5 byte hashing) tren tung ma tran anh de phat hien trung lap va ro ri du lieu:
    </p>

    <div class="table-responsive" style="margin-bottom: 1.5rem;">
        <table>
            <thead>
                <tr>
                    <th style="width: 280px;">Hang Muc Kiem Tra</th>
                    <th style="width: 180px;">Ket Qua Mong Doi</th>
                    <th>Thuc Te Do Dac</th>
                    <th style="width: 130px; text-align: center;">Trang Thai</th>
                </tr>
            </thead>
            <tbody>
                {audit_tbody}
            </tbody>
        </table>
    </div>

    <div class="table-responsive" style="margin-bottom: 1.5rem;">
        <table>
            <thead>
                <tr>
                    <th>Tap Du Lieu</th>
                    <th>So Luong Anh</th>
                    <th>Kich Thuoc (HxW)</th>
                    <th>Kenh Mau</th>
                    <th>Kieu Du Lieu (dtype)</th>
                    <th>Dai Diem Anh</th>
                    <th>Dai Nhan</th>
                    <th>So Lop</th>
                    <th>Bo Nho RAM</th>
                </tr>
            </thead>
            <tbody>
                {struct_tbody}
            </tbody>
        </table>
    </div>

    {render_code_drawer("Ma nguon kiem toan tinh toan ven & quet trung lap (MD5 Hashing)", code_example)}
    """


# ==============================================================================
# 3. BASE HTML WRAPPER TEMPLATE
# ==============================================================================
HTML_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {author}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #0284c7;
            --primary-bg: #f8fafc;
            --secondary: #475569;
            --accent: #2563eb;
            --success: #16a34a;
            --success-bg: #ecfdf5;
            --warning: #f59e0b;
            --card-bg: #ffffff;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
            --font-sans: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', Consolas, 'Courier New', Courier, monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        html {{
            scroll-behavior: smooth;
        }}

        body {{
            font-family: var(--font-sans);
            background-color: var(--primary-bg);
            color: var(--text-dark);
            line-height: 1.6;
            padding: 2.5rem 1rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #0369a1 100%);
            color: #ffffff;
            padding: 2.75rem 2.25rem;
            border-radius: 16px;
            margin-bottom: 2.5rem;
            box-shadow: var(--shadow-lg);
            border-bottom: 4px solid var(--accent);
        }}

        header h1 {{
            font-size: 2.15rem;
            font-weight: 800;
            margin-bottom: 0.65rem;
            letter-spacing: -0.025em;
        }}

        header h2 {{
            font-size: 1.2rem;
            font-weight: 400;
            opacity: 0.92;
            margin-bottom: 1.75rem;
        }}

        .meta-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.18);
            backdrop-filter: blur(8px);
            padding: 0.4rem 0.95rem;
            border-radius: 9999px;
            font-size: 0.84rem;
            font-weight: 500;
            border: 1px solid rgba(255, 255, 255, 0.28);
        }}

        .badge-success {{
            background: #dcfce7;
            color: #15803d;
            border: 1px solid #86efac;
            font-weight: 600;
        }}

        .badge-info {{
            background: #dbeafe;
            color: #1e40af;
            border: 1px solid #bfdbfe;
            font-weight: 600;
        }}

        .card {{
            background: var(--card-bg);
            border-radius: 14px;
            border: 1px solid var(--border);
            padding: 2rem 2.25rem;
            margin-bottom: 2.5rem;
            box-shadow: var(--shadow);
            scroll-margin-top: 2rem;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }}

        .card-title {{
            font-size: 1.4rem;
            font-weight: 800;
            color: var(--primary);
            letter-spacing: -0.015em;
        }}

        /* Pipeline Stepper */
        .pipeline-container {{
            margin: 1.5rem 0 2rem;
        }}

        .pipeline-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
            gap: 0.85rem;
        }}

        .pipeline-card {{
            display: block;
            text-decoration: none;
            border-radius: 12px;
            padding: 1.15rem 1rem;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }}

        .pipeline-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 10px 18px -4px rgba(0, 0, 0, 0.12);
        }}

        .pipeline-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.65rem;
        }}

        .pipeline-number {{
            width: 28px;
            height: 28px;
            color: #ffffff;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 0.78rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
        }}

        .pipeline-title {{
            font-size: 0.95rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            line-height: 1.35;
        }}

        .pipeline-desc {{
            font-size: 0.78rem;
            line-height: 1.45;
            font-weight: 500;
        }}

        /* Subsections */
        .subsection-title {{
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--primary);
            margin: 2.25rem 0 1rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px dashed var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        /* Academic Figure Cards */
        .image-card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin: 1.5rem 0 2rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }}

        .image-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.1);
            border-color: #cbd5e1;
        }}

        .image-card-img-wrap {{
            position: relative;
            background: #fbfcfe;
            border-bottom: 1px solid var(--border);
            overflow: hidden;
            cursor: zoom-in;
            text-align: center;
            padding: 1.25rem;
        }}

        .image-card img {{
            width: 100%;
            max-height: 480px;
            object-fit: contain;
            display: block;
            margin: 0 auto;
            transition: transform 0.25s ease;
        }}

        .image-card:hover img {{
            transform: scale(1.015);
        }}

        .zoom-hint {{
            position: absolute;
            bottom: 12px;
            right: 12px;
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(4px);
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.3rem 0.75rem;
            border-radius: 6px;
            opacity: 0;
            transition: opacity 0.2s ease, transform 0.2s ease;
            pointer-events: none;
        }}

        .image-card:hover .zoom-hint {{
            opacity: 1;
        }}

        .image-card-caption {{
            padding: 1.25rem 1.5rem;
            background: #ffffff;
        }}

        .image-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.65rem;
        }}

        .image-card-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-dark);
        }}

        .image-card-takeaway {{
            font-size: 0.88rem;
            color: var(--secondary);
            line-height: 1.55;
            margin-bottom: 0.85rem;
        }}

        .image-card-footer {{
            display: flex;
            flex-wrap: wrap;
            gap: 1.25rem;
            padding-top: 0.65rem;
            border-top: 1px solid #f1f5f9;
            font-size: 0.78rem;
            color: var(--text-muted);
            font-weight: 500;
        }}

        .grid-2col {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin: 1.5rem 0 2rem;
        }}

        .grid-3col {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.25rem;
            margin: 1.5rem 0 2rem;
        }}

        @media (max-width: 900px) {{
            .grid-2col, .grid-3col {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Table styles */
        .table-responsive {{
            overflow-x: auto;
            border-radius: 10px;
            border: 1px solid var(--border);
            margin: 1.5rem 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            text-align: left;
        }}

        th {{
            background: #f8fafc;
            color: var(--text-dark);
            font-weight: 700;
            padding: 0.85rem 1.15rem;
            border-bottom: 2px solid var(--border);
            white-space: nowrap;
        }}

        td {{
            padding: 0.85rem 1.15rem;
            border-bottom: 1px solid var(--border);
            color: var(--text-dark);
            vertical-align: middle;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background: #f8fafc;
        }}

        .code-pill {{
            font-family: var(--font-mono);
            background: #f1f5f9;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            color: #0f172a;
        }}

        /* Collapsible Code Tabs */
        .code-tab-drawer {{
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #ffffff;
            margin: 1.5rem 0;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }}

        .code-tab-summary {{
            padding: 0.85rem 1.15rem;
            background: #f8fafc;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            user-select: none;
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-dark);
            transition: background 0.15s ease;
        }}

        .code-tab-drawer[open] .code-tab-summary {{
            border-bottom: 1px solid var(--border);
            background: #f1f5f9;
        }}

        .code-summary-left {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .code-badge {{
            font-size: 0.75rem;
            color: var(--primary);
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            padding: 0.2rem 0.65rem;
            border-radius: 12px;
            font-weight: 600;
        }}

        pre[class*="language-"] {{
            margin: 0 !important;
            border-radius: 0 !important;
            font-family: var(--font-mono) !important;
            font-size: 0.84rem !important;
            padding: 1.25rem 1.5rem !important;
        }}

        /* Integrated Code Drawer inside Image Cards */
        .image-card .code-tab-drawer {{
            border: none;
            border-top: 1px solid var(--border);
            border-radius: 0 0 12px 12px;
            margin: 0;
            background: #f8fafc;
            box-shadow: none;
        }}

        .image-card .code-tab-drawer .code-tab-summary {{
            padding: 0.5rem 1.25rem;
            font-size: 0.8rem;
            background: #f8fafc;
            cursor: pointer;
        }}

        .image-card .code-tab-drawer[open] .code-tab-summary {{
            background: #f1f5f9;
            border-bottom: 1px solid var(--border);
        }}

        .image-card .code-tab-drawer pre {{
            margin: 0;
            border-radius: 0 0 12px 12px;
            padding: 0.85rem 1.25rem;
            font-size: 0.8rem;
            background: #0f172a;
            color: #f8fafc;
        }}

        /* Lightbox Modal */
        .image-modal {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(15, 23, 42, 0.88);
            backdrop-filter: blur(6px);
            z-index: 9999;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            cursor: zoom-out;
            opacity: 0;
            transition: opacity 0.25s ease;
        }}

        .image-modal.active {{
            display: flex;
            opacity: 1;
        }}

        .modal-content-wrap {{
            max-width: 95vw;
            max-height: 92vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            cursor: default;
        }}

        .modal-content-wrap img {{
            max-width: 100%;
            max-height: 80vh;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            background: #ffffff;
        }}

        .modal-caption {{
            color: #ffffff;
            margin-top: 1rem;
            font-size: 0.95rem;
            font-weight: 600;
            text-align: center;
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .modal-close-btn {{
            position: absolute;
            top: 24px;
            right: 28px;
            color: #ffffff;
            font-size: 2rem;
            font-weight: 700;
            cursor: pointer;
            background: rgba(255, 255, 255, 0.1);
            width: 44px;
            height: 44px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.15s ease;
        }}

        .modal-close-btn:hover {{
            background: rgba(255, 255, 255, 0.25);
        }}

        footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 3.5rem;
            padding-top: 1.75rem;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>

<div class="container">
    <!-- Header -->
    <header>
        <h1>{title}</h1>
        <h2>{subtitle}</h2>
        <div class="meta-badges">
            <span class="badge">Tac gia: {author}</span>
            <span class="badge">{python_ver}</span>
            <span class="badge">{framework_ver}</span>
            <span class="badge">{dataset_info}</span>
            <span class="badge badge-success">{test_passed}</span>
            <span class="badge badge-info">{primary_metric}</span>
        </div>
    </header>

    <!-- Pipeline Stepper -->
    <div class="card" id="section-roadmap">
        <div class="card-header">
            <h3 class="card-title">Lo Trinh Thuc Thi & So Do Tien Trinh</h3>
            <span class="badge badge-success">Clean Architecture</span>
        </div>
        <p style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 0.75rem;">
            Nhan vao tung the de chuyen nhanh den noi dung tuong ung:
        </p>
        <div class="pipeline-container">
            <div class="pipeline-grid">
                {stepper_html}
            </div>
        </div>
    </div>

    <!-- Main Content Sections -->
    {sections_html}

    <!-- Footer -->
    <footer>
        <p>Bao cao nghiem thu ky thuat mon Deep Learning - Truong Dai hoc Giao thong van tai TP.HCM (UTH)</p>
        <p style="margin-top: 0.35rem; font-size: 0.8rem; color: var(--text-muted);">
            Bao cao duoc tao tu dong tu artifact goc trong outputs/published/ - Thiet ke theo tieu chuan UTH-CV Template
        </p>
    </footer>
</div>

<!-- Modal Zoom Lightbox -->
<div id="imageModal" class="image-modal" onclick="closeModal(event)">
    <div class="modal-close-btn" onclick="closeModalForce()">&times;</div>
    <div class="modal-content-wrap">
        <img id="modalImg" src="" alt="Zoomed artifact">
        <div class="modal-caption">
            <span id="modalCaptionText"></span>
        </div>
    </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<script>
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    const modalCaption = document.getElementById('modalCaptionText');

    function openModal(imgWrap) {{
        const img = imgWrap.querySelector('img');
        if (!img) return;
        modalImg.src = img.src;
        modalCaption.textContent = img.alt || 'Chi tiet artifact';
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }}

    function closeModal(e) {{
        if (e.target === modal || e.target.classList.contains('image-modal')) {{
            closeModalForce();
        }}
    }}

    function closeModalForce() {{
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }}

    document.addEventListener('keydown', function(e) {{
        if (e.key === 'Escape' && modal.classList.contains('active')) {{
            closeModalForce();
        }}
    }});

    // Dynamic toggle text for code tab drawers
    document.querySelectorAll('.code-tab-drawer').forEach(drawer => {{
        drawer.addEventListener('toggle', () => {{
            const badge = drawer.querySelector('.code-badge');
            if (badge) {{
                badge.innerHTML = drawer.open ? 'Dong code &utrif;' : 'Xem code cell &dtrif;';
            }}
        }});
    }});
</script>
</body>
</html>
"""


# ==============================================================================
# 4. MAIN BUILD FUNCTION (Chay sinh file HTML)
# ==============================================================================
def build_html_report(output_path: Path, sections_html: str):
    """Gop tat ca cac thanh phan va xuat file HTML hoan chinh."""
    # Build stepper cards
    stepper_cards = []
    for i, step in enumerate(PIPELINE_STEPS):
        grad = STEP_GRADIENTS[i % len(STEP_GRADIENTS)]
        card = f"""
        <a href="{step['anchor']}" class="pipeline-card" style="background: {grad['bg']}; border: 2px solid {grad['border']}; color: {grad['text']};">
            <div class="pipeline-card-header">
                <span class="pipeline-number" style="background: {grad['badge']};">{step['num']}</span>
                <span style="font-weight: 700; font-size: 0.9rem;">&darr;</span>
            </div>
            <div class="pipeline-title">{step['title']}</div>
            <div class="pipeline-desc">{step['desc']}</div>
        </a>
        """
        stepper_cards.append(card)

    stepper_html = "\n".join(stepper_cards)

    full_html = HTML_PAGE_TEMPLATE.format(
        title=LAB_CONFIG["title"],
        subtitle=LAB_CONFIG["subtitle"],
        author=LAB_CONFIG["author"],
        python_ver=LAB_CONFIG["python_ver"],
        framework_ver=LAB_CONFIG["framework_ver"],
        dataset_info=LAB_CONFIG["dataset_info"],
        test_passed=LAB_CONFIG["test_passed"],
        primary_metric=LAB_CONFIG["primary_metric"],
        stepper_html=stepper_html,
        sections_html=sections_html,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_html, encoding="utf-8")
    print(f"Bao cao HTML da duoc tao thanh cong tai: {output_path.resolve()}")


if __name__ == "__main__":
    # Vi du chay sinh bao cao mau
    sample_audit = [
        {"item": "Gia tri thieu (NaN / Null / Inf)", "expected": "Khong co (0 NaN)", "actual": "0 mau thieu (100% nguyen ven)", "status": "PASSED"},
        {"item": "Trung lap noi bo tap Train", "expected": "0 mau trung", "actual": "0 mau trung lap (MD5 Hashing)", "status": "PASSED"},
    ]
    sample_struct = [
        {"pool": "Train Pool", "count": "60.000", "dim": "28 x 28", "channel": "1 (Grayscale)", "dtype": "uint8", "pixel_range": "[0, 255]", "label_range": "[0, 9]", "classes": "10", "ram": "44.86 MB"},
    ]
    sample_code = """
import numpy as np
import hashlib
assert not np.isnan(X_train).any()
# MD5 hashing logic...
"""
    eda_section = f"""
    <div class="card" id="section-eda">
        <div class="card-header">
            <h3 class="card-title">Kham Pha & Kiem Toan Du Lieu (EDA)</h3>
            <span class="badge badge-info">STAGE 1</span>
        </div>
        {render_data_integrity_section(sample_audit, sample_struct, sample_code)}
    </div>
    """

    out_file = Path("labs/lab01/reports/sample_report_test.html")
    build_html_report(out_file, eda_section)
    if out_file.exists():
        out_file.unlink()  # don dep file test mau
