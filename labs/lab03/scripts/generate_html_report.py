"""Script to generate standardized, visual-first HTML report for Lab 03.
Based on the UTH Deep Learning & Computer Vision template style.

Generates: labs/lab03/reports/walkthrough_lab_03.html
Author: Thai Anh Lac
"""

from pathlib import Path

# ==============================================================================
# 1. LAB CONFIGURATION
# ==============================================================================
LAB_CONFIG = {
    "lab_num": 3,
    "lab_id": "lab03",
    "title": "Báo Cáo Nghiệm Thu Kỹ Thuật Lab 3",
    "subtitle": "Binary Sentiment Classification với Hugging Face & DistilBERT",
    "author": "Thái Anh Lạc",
    "python_ver": "Python 3.12 / 3.13",
    "framework_ver": "PyTorch 2.14.0 &bull; Transformers 4.35+",
    "dataset_info": "Rotten Tomatoes (10.662 mẫu: 8.530 train / 1.066 val / 1.066 test)",
    "test_passed": "95 / 95 (100% Passed)",
    "primary_metric": "Test Acc: 83.11% &bull; Binary F1: 83.30% (Optimal Threshold: 0.45)",
}

PIPELINE_STEPS = [
    {
        "num": "01",
        "title": "Pretrained & Tokenizer",
        "desc": "Exercise 1 &bull; Pipeline vs Manual Inference",
        "anchor": "#section-pretrained",
    },
    {
        "num": "02",
        "title": "Core NLP EDA",
        "desc": "Kiểm toán 0 NaN &bull; Phân bố câu & Token Length",
        "anchor": "#section-eda",
    },
    {
        "num": "03",
        "title": "Kỹ Nghệ & Augmentation",
        "desc": "TF-IDF &bull; Chi2 &bull; Ngôn ngữ học & Tăng cường",
        "anchor": "#section-features",
    },
    {
        "num": "04",
        "title": "Huấn Luyện DistilBERT",
        "desc": "Động học 5 Epochs &bull; Đo đạc tài nguyên",
        "anchor": "#section-training",
    },
    {
        "num": "05",
        "title": "Ngưỡng & Đánh Giá",
        "desc": "Threshold Sweep 0.45 &bull; Confusion & ROC/PR",
        "anchor": "#section-evaluation",
    },
    {
        "num": "06",
        "title": "Kiểm Toán QA & DoD",
        "desc": "95/95 Pytest Passed &bull; Checkpoint Verification",
        "anchor": "#section-qa",
    },
]

STEP_GRADIENTS = [
    {"bg": "#f0fdf4", "border": "#86efac", "text": "#166534", "badge": "#22c55e"},
    {"bg": "#eff6ff", "border": "#93c5fd", "text": "#1e40af", "badge": "#3b82f6"},
    {"bg": "#faf5ff", "border": "#d8b4fe", "text": "#6b21a8", "badge": "#a855f7"},
    {"bg": "#fffbeb", "border": "#fde68a", "text": "#92400e", "badge": "#f59e0b"},
    {"bg": "#ecfdf5", "border": "#a7f3d0", "text": "#065f46", "badge": "#10b981"},
    {"bg": "#f8fafc", "border": "#cbd5e1", "text": "#334155", "badge": "#64748b"},
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
                    <span>Cell Notebook: Trực quan hóa ảnh này</span>
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
            <span class="zoom-hint">Phóng to [Zoom] &nearr;</span>
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
                <span class="zoom-hint">Phóng to [Zoom] &nearr;</span>
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
                <span class="zoom-hint">Phóng to [Zoom] &nearr;</span>
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
            <span class="code-badge">Xem code &dtrif;</span>
        </summary>
        <pre><code class="language-{language}">{code_str.strip()}</code></pre>
    </details>
    """


# ==============================================================================
# 3. HTML PAGE MASTER TEMPLATE
# ==============================================================================
HTML_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — {author}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">

    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --primary-light: #eff6ff;
            --text-main: #0f172a;
            --text-dark: #334155;
            --text-muted: #64748b;
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --border: #e2e8f0;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07);
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', Consolas, Monaco, monospace;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: var(--font-sans);
            background-color: var(--bg-page);
            color: var(--text-main);
            line-height: 1.6;
            padding-bottom: 5rem;
        }}

        /* Header Banner */
        header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
            color: #ffffff;
            padding: 3rem 2rem 2.5rem 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            position: relative;
        }}
        .header-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}
        .uth-brand {{
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #93c5fd;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .author-box {{
            font-size: 0.9rem;
            color: #e2e8f0;
            background: rgba(255, 255, 255, 0.1);
            padding: 0.35rem 0.85rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }}
        h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0.5rem;
            line-height: 1.25;
        }}
        .header-subtitle {{
            font-size: 1.15rem;
            color: #cbd5e1;
            margin-bottom: 1.75rem;
            font-weight: 400;
        }}
        .header-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.75rem;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
        }}
        .badge-success {{ background: #059669; color: #ffffff; }}
        .badge-info {{ background: #0284c7; color: #ffffff; }}
        .badge-dark {{ background: #334155; color: #f8fafc; }}

        /* Main Container */
        .container {{
            max-width: 1200px;
            margin: -1.5rem auto 0 auto;
            padding: 0 1.5rem;
            position: relative;
            z-index: 10;
        }}

        /* Pipeline Stepper */
        .pipeline-stepper {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.85rem;
            margin-bottom: 2.25rem;
        }}
        .pipeline-card {{
            border-radius: var(--radius-md);
            padding: 1.1rem 1rem;
            text-decoration: none;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}
        .pipeline-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.1);
        }}
        .pipeline-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .pipeline-number {{
            color: #ffffff;
            font-weight: 700;
            font-size: 0.8rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
        }}
        .pipeline-title {{
            font-weight: 700;
            font-size: 0.95rem;
            line-height: 1.3;
        }}
        .pipeline-desc {{
            font-size: 0.78rem;
            opacity: 0.85;
            line-height: 1.35;
        }}

        /* Section Cards */
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 2.25rem 2.5rem;
            margin-bottom: 2.25rem;
            box-shadow: var(--shadow-sm);
        }}
        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 1.25rem;
            margin-bottom: 1.75rem;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}
        .card-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #0f172a;
        }}

        /* Subsection Title */
        .subsection-title {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 2rem 0 1rem 0;
            font-size: 1.1rem;
            font-weight: 700;
            color: #1e293b;
        }}

        /* Grid Layouts */
        .grid-2col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 1.5rem;
            margin: 1.25rem 0;
        }}
        .grid-3col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.25rem;
            margin: 1.25rem 0;
        }}

        /* Image Cards */
        .image-card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: var(--shadow-sm);
            margin: 1rem 0;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .image-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.08);
        }}
        .image-card-img-wrap {{
            position: relative;
            background: #fbfcfe;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            cursor: pointer;
        }}
        .image-card-img-wrap img {{
            width: 100%;
            max-height: 440px;
            object-fit: contain;
            display: block;
        }}
        .zoom-hint {{
            position: absolute;
            bottom: 0.75rem;
            right: 0.75rem;
            background: rgba(15, 23, 42, 0.85);
            color: #ffffff;
            font-size: 0.75rem;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            pointer-events: none;
            opacity: 0.85;
            transition: opacity 0.2s ease;
        }}
        .image-card-img-wrap:hover .zoom-hint {{
            opacity: 1;
        }}
        .image-card-caption {{
            padding: 1.25rem;
            background: #ffffff;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }}
        .image-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            gap: 0.5rem;
        }}
        .image-card-title {{
            font-weight: 700;
            font-size: 0.95rem;
            color: #0f172a;
        }}
        .image-card-takeaway {{
            font-size: 0.88rem;
            color: var(--text-dark);
            line-height: 1.5;
            margin-bottom: 0.85rem;
            flex-grow: 1;
        }}
        .image-card-footer {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.78rem;
            color: var(--text-muted);
            border-top: 1px dashed var(--border);
            padding-top: 0.65rem;
        }}
        .code-pill {{
            font-family: var(--font-mono);
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
            font-size: 0.75rem;
            color: #0f172a;
            white-space: nowrap;
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
            display: flex;
            align-items: center;
            justify-content: space-between;
            user-select: none;
            font-weight: 600;
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

        /* Standalone Code Drawer */
        .code-tab-drawer {{
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #ffffff;
            margin: 1.25rem 0;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }}
        .code-tab-summary {{
            padding: 0.75rem 1.15rem;
            background: #f8fafc;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            user-select: none;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-dark);
        }}
        .code-tab-drawer[open] .code-tab-summary {{
            border-bottom: 1px solid var(--border);
            background: #f1f5f9;
        }}
        .code-summary-left {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }}
        .code-badge {{
            font-size: 0.75rem;
            color: var(--primary);
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-weight: 600;
        }}
        pre[class*="language-"] {{
            margin: 0 !important;
            border-radius: 0 !important;
            font-family: var(--font-mono) !important;
            font-size: 0.82rem !important;
            padding: 1rem 1.25rem !important;
        }}

        /* Tables */
        .table-responsive {{
            overflow-x: auto;
            margin: 1rem 0;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            background: #ffffff;
        }}
        th {{
            background: #f8fafc;
            font-weight: 700;
            color: #1e293b;
            text-align: left;
            padding: 0.75rem 1rem;
            border-bottom: 2px solid var(--border);
        }}
        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #f1f5f9;
            color: var(--text-dark);
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover td {{
            background-color: #f8fafc;
        }}

        /* Callouts */
        .callout {{
            border-left: 4px solid var(--primary);
            background: var(--primary-light);
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            margin: 1.25rem 0;
        }}

        /* Lightbox Modal */
        .image-modal {{
            display: none;
            position: fixed;
            z-index: 9999;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(15, 23, 42, 0.9);
            align-items: center;
            justify-content: center;
            padding: 2rem;
            cursor: zoom-out;
        }}
        .image-modal.active {{ display: flex; }}
        .modal-content {{
            max-width: 90vw;
            max-height: 85vh;
            background: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            cursor: default;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }}
        .modal-img-wrap {{
            background: #0f172a;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            max-height: 70vh;
        }}
        .modal-img-wrap img {{
            max-width: 100%;
            max-height: 68vh;
            object-fit: contain;
        }}
        .modal-caption {{
            padding: 1rem 1.5rem;
            background: #ffffff;
            border-top: 1px solid var(--border);
        }}
        .modal-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }}
        .modal-desc {{
            font-size: 0.88rem;
            color: var(--text-dark);
        }}

        @media (max-width: 768px) {{
            .grid-2col, .grid-3col {{ grid-template-columns: 1fr; }}
            .card {{ padding: 1.5rem 1rem; }}
            h1 {{ font-size: 1.6rem; }}
        }}
    </style>
</head>
<body>

<header>
    <div class="header-container">
        <div class="header-top">
            <div class="uth-brand">
                <span>&diams;</span> TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH) &bull; KHOA CNTT
            </div>
            <div class="author-box">Tác giả thực hiện: <strong>{author}</strong></div>
        </div>
        <h1>{title}</h1>
        <div class="header-subtitle">{subtitle}</div>
        <div class="header-badges">
            <span class="badge badge-dark">{python_ver}</span>
            <span class="badge badge-info">{framework_ver}</span>
            <span class="badge badge-dark">{dataset_info}</span>
            <span class="badge badge-success">{test_passed}</span>
            <span class="badge badge-success">{primary_metric}</span>
        </div>
    </div>
</header>

<main class="container">
    <!-- Pipeline Stepper Navigation -->
    <nav class="pipeline-stepper" aria-label="Pipeline Roadmap">
        {stepper_html}
    </nav>

    <!-- Sections Content -->
    {sections_html}
</main>

<!-- Lightbox Modal -->
<div class="image-modal" id="imageModal" onclick="closeModal(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
        <div class="modal-img-wrap">
            <img src="" alt="" id="modalImg">
        </div>
        <div class="modal-caption">
            <div class="modal-title" id="modalTitle"></div>
            <div class="modal-desc" id="modalDesc"></div>
        </div>
    </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<script>
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    const modalTitle = document.getElementById('modalTitle');
    const modalDesc = document.getElementById('modalDesc');

    function openModal(wrapEl) {{
        const img = wrapEl.querySelector('img');
        const figure = wrapEl.closest('.image-card');
        const titleEl = figure ? figure.querySelector('.image-card-title') : null;
        const descEl = figure ? figure.querySelector('.image-card-takeaway') : null;

        modalImg.src = img.src;
        modalImg.alt = img.alt;
        modalTitle.textContent = titleEl ? titleEl.textContent : img.alt;
        modalDesc.textContent = descEl ? descEl.textContent : '';
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

    // Toggle button label
    document.querySelectorAll('.code-tab-drawer').forEach(drawer => {{
        drawer.addEventListener('toggle', () => {{
            const badge = drawer.querySelector('.code-badge');
            if (badge) {{
                badge.innerHTML = drawer.open ? 'Đóng code &utrif;' : 'Xem code cell &dtrif;';
            }}
        }});
    }});
</script>
</body>
</html>
"""


# ==============================================================================
# 4. SECTION BUILDERS
# ==============================================================================
def build_sections() -> str:
    sections = []

    # --------------------------------------------------------------------------
    # SECTION 1: Pretrained Sentiment & Tokenizer Mechanics (Exercise 1)
    # --------------------------------------------------------------------------
    sec1_html = f"""
    <section class="card" id="section-pretrained">
        <div class="card-header">
            <h2 class="card-title">1. Phân Tích Cơ Chế Pretrained Model & Tokenizer (Exercise 1)</h2>
            <span class="badge badge-info">TRANSFORMERS PIPELINE &bull; EXERCISE 1</span>
        </div>

        <p style="font-size: 0.92rem; color: var(--text-dark); margin-bottom: 1.25rem;">
            Khảo sát cơ chế hoạt động của mô hình cảm xúc tiền huấn luyện <code>distilbert-base-uncased-finetuned-sst-2-english</code>: 
            so sánh tính nhất quán giữa Pipeline cấp cao (High-level API) và quá trình suy luận thủ công qua PyTorch tensor, 
            đồng thời phân tích trực quan hóa token IDs, attention masks và vector xác suất Softmax.
        </p>

        {render_hero_card(
            img_path="../outputs/published/exercise1_analysis/tokenizer_table.png",
            title="Hình 1: Phân Rã Tokenizer Mechanics — Token IDs & Attention Masks Trên Các Câu Mẫu",
            tag="Tokenizer Breakdown",
            takeaway="Bảng phân tích chi tiết WordPiece tokenizer: câu được chia nhỏ thành subwords, đệm thêm token đặc biệt [CLS] (ID 101) và [SEP] (ID 102). Attention Mask gán giá trị 1 cho token hợp lệ và 0 cho vùng đệm padding.",
            footer_items=["DistilBERT Tokenizer", "Vocab Size: 30.522", "Artifact: tokenizer_table.png"],
            code_snippet='''# Notebook 03: Cell 7 - Phân tích cơ chế Tokenizer và Attention Masks
session.exercise1()
session.show("exercise1", ["tokenizer_table.png"])'''
        )}

        {render_duo_cards(
            card1={
                "img_path": "../outputs/published/exercise1_analysis/pipeline_manual_comparison.png",
                "title": "2. Đối Chiếu Pipeline vs Manual PyTorch Inference",
                "tag": "Verification &bull; 0 Delta",
                "takeaway": "So sánh xác suất dự đoán giữa transformers.pipeline và tính toán thủ công (logits -> softmax): độ lệch xác suất tuyệt đối đạt max_delta = 0.000000 (khớp 100%), xác nhận tính đúng đắn của logic suy luận.",
                "footer_items": ["Max Delta: 0.000000", "Artifact: pipeline_manual_comparison.png"],
                "code_snippet": '''# Notebook 03: Cell 7 - So sánh Pipeline cấp cao vs Manual PyTorch
session.show("exercise1", ["pipeline_manual_comparison.png"])'''
            },
            card2={
                "img_path": "../outputs/published/exercise1_analysis/sentiment_predictions.png",
                "title": "3. Phân Bố Xác Suất Dự Đoán Trên Các Câu Đánh Giá",
                "tag": "Sentiment Predictions",
                "takeaway": "Dự đoán cảm xúc trên 3 câu văn bản mẫu: các câu thể hiện cảm xúc rõ ràng đạt độ tin cậy vượt 99.8%; câu chứa cấu trúc tương phản ('but') được mô hình nhận diện chính xác ý nghĩa chốt hạ.",
                "footer_items": ["3 Samples Verified", "Artifact: sentiment_predictions.png"],
                "code_snippet": '''# Notebook 03: Cell 7 - Dự đoán cảm xúc trên câu văn bản mẫu
session.show("exercise1", ["sentiment_predictions.png"])'''
            }
        )}
    </section>
    """
    sections.append(sec1_html)

    # --------------------------------------------------------------------------
    # SECTION 2: Core NLP EDA & Data Integrity Audit
    # --------------------------------------------------------------------------
    sec2_html = f"""
    <section class="card" id="section-eda">
        <div class="card-header">
            <h2 class="card-title">2. Khám Phá Dữ Liệu Văn Bản & Kiểm Toán Toàn Vẹn (Core NLP EDA)</h2>
            <span class="badge badge-info">ROTTEN TOMATOES DATASET &bull; 10.662 MẪU</span>
        </div>

        <p style="font-size: 0.92rem; color: var(--text-dark); margin-bottom: 1.25rem;">
            Thực hiện quy trình Core EDA chuẩn 6 bước theo <code>docs/eda.md</code>: kiểm toán dữ liệu thiếu (Missing), phát hiện mẫu trùng lặp 
            qua MD5 hashing, phân tích phân bố nhãn cân bằng tuyệt đối và đo đạc độ dài câu để xác định ngưỡng cắt <code>max_length</code> tối ưu.
        </p>

        {render_duo_cards(
            card1={
                "img_path": "../outputs/published/frozen_trial/data_quality.png",
                "title": "Hình 4: Kiểm Toán Chất Lượng & Tính Toàn Vẹn Dữ Liệu",
                "tag": "Data Integrity: 100%",
                "takeaway": "Kiểm tra toàn bộ 10.662 mẫu: 0 mẫu thiếu (0 NaN/Null/Inf), độ dài văn bản từ 4 đến 266 ký tự (trung bình 115 ký tự). Không có văn bản rỗng hoặc nhãn sai domain.",
                "footer_items": ["0 Missing Values", "100% Valid Samples", "Artifact: data_quality.png"],
                "code_snippet": '''# Notebook 01: Cell 4 & 6 - Kiểm toán tính toàn vẹn dữ liệu
quality = session.read(session.paths["data"] / "data_quality.json")
session.show("data", ["data_quality.png"])'''
            },
            card2={
                "img_path": "../outputs/published/frozen_trial/class_distribution.png",
                "title": "Hình 5: Phân Bố Số Lượng Mẫu Trên 3 Tập Dữ Liệu",
                "tag": "Balanced Split 1:1",
                "takeaway": "Tập dữ liệu cân bằng hoàn hảo 50/50 giữa Positive và Negative trên cả 3 tập: Train có 8.530 mẫu (4.265/lớp), Validation có 1.066 mẫu (533/lớp), Test có 1.066 mẫu (533/lớp).",
                "footer_items": ["Train: 8.530 &bull; Val: 1.066 &bull; Test: 1.066", "Artifact: class_distribution.png"],
                "code_snippet": '''# Notebook 01: Cell 8 - Phân bố số lượng mẫu 3 tập
session.show("data", ["class_distribution.png"])'''
            }
        )}

        {render_trio_cards(
            card1={
                "img_path": "../outputs/published/frozen_trial/lengths_boxplot.png",
                "title": "6. Phân Bố Số Lượng Từ (Word Counts)",
                "tag": "Boxplot",
                "takeaway": "Độ dài từ trung bình đạt ~21 từ/câu; phân vị 95th đạt 42 từ; mẫu dài nhất ghi nhận 56 từ. Phân bố giữa 2 lớp cân xứng cao.",
                "footer_items": ["Median: 20 words", "Artifact: lengths_boxplot.png"],
                "code_snippet": '''# Notebook 01: Cell 10 - Phân bố độ dài từ
session.show("data", ["lengths_boxplot.png"])'''
            },
            card2={
                "img_path": "../outputs/published/frozen_trial/lengths_by_class.png",
                "title": "7. Histogram Độ Dài Câu Theo Nhãn",
                "tag": "Length Distribution",
                "takeaway": "Đường cong mật độ của nhãn Positive và Negative gần như trùng khít hoàn hảo, loại bỏ nguy cơ độ dài câu trở thành đặc trưng rò rỉ giả tạo.",
                "footer_items": ["Symmetric Density", "Artifact: lengths_by_class.png"],
                "code_snippet": '''# Notebook 01: Cell 10 - Histogram độ dài câu
session.show("data", ["lengths_by_class.png"])'''
            },
            card3={
                "img_path": "../outputs/published/frozen_trial/token_length_distribution.png",
                "title": "8. Phân Bố Token Length (DistilBERT)",
                "tag": "Max Length Selection",
                "takeaway": "Phân bố token sau WordPiece tokenizer: 99.4% số câu có độ dài dưới 64 tokens. Quyết định chọn max_length = 64 giúp tối ưu 100% ngữ cảnh mà không lãng phí đệm padding.",
                "footer_items": ["Cutoff: max_length=64 (99.4%)", "Artifact: token_length_distribution.png"],
                "code_snippet": '''# Notebook 01: Cell 8 - Phân tích Token Length & Cutoff
session.show("data", ["token_length_distribution.png"])'''
            }
        )}
    </section>
    """
    sections.append(sec2_html)

    # --------------------------------------------------------------------------
    # SECTION 3: Feature Engineering & Data Augmentation
    # --------------------------------------------------------------------------
    sec3_html = f"""
    <section class="card" id="section-features">
        <div class="card-header">
            <h2 class="card-title">3. Kỹ Nghệ Đặc Trưng Ngôn Ngữ Học & Tăng Cường Dữ Liệu (Feature Engineering)</h2>
            <span class="badge badge-info">NLP FEATURE ENGINEERING &bull; AUGMENTATION</span>
        </div>

        <p style="font-size: 0.92rem; color: var(--text-dark); margin-bottom: 1.25rem;">
            Phân tích mức độ phân tách cảm xúc của các đặc trưng truyền thống (TF-IDF, Chi-squared, chỉ số ngôn ngữ học về sự phủ định/tương phản) 
            để làm mốc tham chiếu so sánh (Baseline) trước khi đưa vào mô hình Deep Learning Transformer, kết hợp kiểm toán chất lượng tăng cường dữ liệu.
        </p>

        {render_duo_cards(
            card1={
                "img_path": "../outputs/published/frozen_trial/tfidf_features.png",
                "title": "Hình 9: Top 20 Từ Khóa TF-IDF Phân Tách Hai Lớp Cảm Xúc",
                "tag": "TF-IDF Salience",
                "takeaway": "Từ khóa tích cực nổi bật: 'best', 'funny', 'love', 'charming', 'solid'. Từ khóa tiêu cực đặc trưng: 'bad', 'dull', 'boring', 'flat', 'cliche', phản ánh rõ nét từ vựng đánh giá phim.",
                "footer_items": ["Vocabulary: 5.000 unigrams", "Artifact: tfidf_features.png"],
                "code_snippet": '''# Notebook 02: Cell 10 - Trực quan hóa đặc trưng TF-IDF
session.show("features", ["tfidf_features.png"])'''
            },
            card2={
                "img_path": "../outputs/published/frozen_trial/tfidf_class_delta.png",
                "title": "Hình 10: Chênh Lệch Trọng Số TF-IDF Giữa Hai Lớp",
                "tag": "Delta TF-IDF Spectrum",
                "takeaway": "Biểu đồ phổ chênh lệch trọng số (Delta TF-IDF) làm lộ diện các từ ngữ mang tính phân định cực kỳ cao: từ 'bad' lệch âm mạnh nhất (-0.18), trong khi từ 'best' lệch dương cao nhất (+0.16).",
                "footer_items": ["Delta Span: [-0.18, +0.16]", "Artifact: tfidf_class_delta.png"],
                "code_snippet": '''# Notebook 02: Cell 10 - Biểu đồ chênh lệch Delta TF-IDF
session.show("features", ["tfidf_class_delta.png"])'''
            }
        )}

        {render_duo_cards(
            card1={
                "img_path": "../outputs/published/frozen_trial/chi2_features.png",
                "title": "Hình 11: Xếp Hạng Đặc Trưng Theo Kiểm Định Chi-squared",
                "tag": "Chi-Square Independence",
                "takeaway": "Kiểm định Chi-squared khẳng định ý nghĩa thống kê của các từ khóa chỉ thị cảm xúc. Các từ 'bad', 'movie', 'film', 'dull', 'funny' đạt giá trị Chi2 vượt trội so với các từ dừng.",
                "footer_items": ["p-value < 0.001", "Artifact: chi2_features.png"],
                "code_snippet": '''# Notebook 02: Cell 10 - Xếp hạng đặc trưng Chi-squared
session.show("features", ["chi2_features.png"])'''
            },
            card2={
                "img_path": "../outputs/published/frozen_trial/linguistic_features.png",
                "title": "Hình 12: Đặc Trưng Ngôn Ngữ Học (Phủ Định & Tương Phản)",
                "tag": "Linguistic Indicators",
                "takeaway": "Các câu chứa từ tương phản ('but', 'however', 'although') xuất hiện nhiều hơn ở đánh giá tiêu cực (14.2% so với 8.9% ở tích cực), phản ánh cấu trúc khen trước chê sau của các nhà phê bình.",
                "footer_items": ["Contrast Lexicon Checked", "Artifact: linguistic_features.png"],
                "code_snippet": '''# Notebook 02: Cell 10 - Phân tích đặc trưng ngôn ngữ học
session.show("features", ["linguistic_features.png"])'''
            }
        )}

        {render_duo_cards(
            card1={
                "img_path": "../outputs/published/frozen_trial/augmentation_counts.png",
                "title": "Hình 13: Thống Kê Số Lượng Mẫu Tăng Cường (Augmentation)",
                "tag": "Data Augmentation",
                "takeaway": "Áp dụng kỹ thuật thay thế từ đồng nghĩa theo ngữ cảnh (Synonym Replacement) có kiểm soát: tạo thêm các biến thể văn bản giúp mô hình tăng cường khả năng chống chịu nhiễu từ vựng.",
                "footer_items": ["Controlled Augmentation", "Artifact: augmentation_counts.png"],
                "code_snippet": '''# Notebook 02: Cell 14 - Thống kê mẫu sinh tăng cường
session.show("augmentation", ["augmentation_counts.png"])'''
            },
            card2={
                "img_path": "../outputs/published/frozen_trial/augmentation_diagnostics.png",
                "title": "Hình 14: Kiểm Toán Chất Lượng & Đa Dạng Của Mẫu Tăng Cường",
                "tag": "Semantic Drift Audit",
                "takeaway": "Đo lường độ dịch chuyển ngữ nghĩa (Semantic Drift): 96.8% mẫu tăng cường giữ nguyên nhãn gốc cảm xúc, đảm bảo không gây ô nhiễm nhãn trong quá trình huấn luyện.",
                "footer_items": ["Retained Sentiment: 96.8%", "Artifact: augmentation_diagnostics.png"],
                "code_snippet": '''# Notebook 02: Cell 14 - Kiểm toán chất lượng tăng cường
session.show("augmentation", ["augmentation_diagnostics.png"])'''
            }
        )}
    </section>
    """
    sections.append(sec3_html)

    # --------------------------------------------------------------------------
    # SECTION 4: DistilBERT Training & Resource Benchmark
    # --------------------------------------------------------------------------
    sec4_html = f"""
    <section class="card" id="section-training">
        <div class="card-header">
            <h2 class="card-title">4. Huấn Luyện Fine-tuning DistilBERT & Động Học Học Tập</h2>
            <span class="badge badge-success">DISTILBERT BASE &bull; 66M PARAMS</span>
        </div>

        <p style="font-size: 0.92rem; color: var(--text-dark); margin-bottom: 1.25rem;">
            Triển khai quy trình fine-tuning mô hình DistilBERT với chiến lược đóng băng/giải phóng backbone (Frozen vs Full-tuning), 
            sử dụng bộ tối ưu AdamW kết hợp Linear Warmup và Early Stopping (patience=2) để ngăn chặn hiện tượng quá khớp (Overfitting).
        </p>

        {render_duo_cards(
            card1={
                "img_path": "../outputs/published/frozen_trial/model_architecture.png",
                "title": "Hình 15: Sơ Đồ Kiến Trúc DistilBERT & Classification Head",
                "tag": "Architecture Diagram",
                "takeaway": "Backbone gồm 6 Transformer Encoder layers (768 hidden dimension, 12 attention heads), nối vào Classification Head gồm Linear(768, 768) -> Pre-classifier -> Dropout(0.2) -> Linear(768, 2) cho 2 nhãn cảm xúc.",
                "footer_items": ["Total: 66.955.010 params", "Artifact: model_architecture.png"],
                "code_snippet": '''# Notebook 03: Cell 5 - In sơ đồ kiến trúc mô hình
session.architectures()
session.show("model", ["model_architecture.png"])'''
            },
            card2={
                "img_path": "../outputs/published/frozen_trial/attention_mask.png",
                "title": "Hình 16: Ma Trận Tự Chú Ý (Self-Attention Visualizer)",
                "tag": "Attention Map",
                "takeaway": "Trực quan hóa ma trận trọng số chú ý: các đầu chú ý tập trung mật độ cao vào token [CLS] và các từ mang sắc thái tình cảm rõ rệt ('charming', 'dull'), phản ánh cơ chế biểu diễn ngữ cảnh sâu sắc.",
                "footer_items": ["Layer 6 Attention", "Artifact: attention_mask.png"],
                "code_snippet": '''# Notebook 03: Cell 11 - Trực quan hóa ma trận Attention
session.show("model", ["attention_mask.png"])'''
            }
        )}

        {render_hero_card(
            img_path="../outputs/published/frozen_trial/learning_curves.png",
            title="Hình 17: Động Học Huấn Luyện 5 Epochs — Quỹ Đạo Hội Tụ Của Loss & Accuracy",
            tag="5 Epochs Tracking",
            takeaway="Theo dõi 5 epochs huấn luyện: Train Loss giảm liên tục từ 0.68 xuống 0.22; Validation Loss đạt cực tiểu 0.40 tại epoch 3; Validation Accuracy đạt đỉnh 83.58% trước khi kích hoạt Early Stopping tại epoch 5, bảo vệ mô hình khỏi hiện tượng học vẹt.",
            footer_items=["Best Epoch: 3/5", "Best Val Acc: 83.58%", "Early Stopping: patience=2", "Artifact: learning_curves.png"],
            code_snippet='''# Notebook 03: Cell 11 - Huấn luyện mô hình và hiển thị learning curves
session.train()
session.show("model", ["learning_curves.png"])'''
        )}

        {render_hero_card(
            img_path="../outputs/published/frozen_trial/performance.png",
            title="Hình 18: Đo Đạc Chi Phí Tài Nguyên Tính Toán & Bộ Nhớ RAM/VRAM",
            tag="Compute Benchmark",
            takeaway="Đo đạc tốc độ huấn luyện thực tế trên GPU Kaggle P100: thời gian trung bình 34.2 giây/epoch (thông lượng ~250 mẫu/s), bộ nhớ GPU VRAM cấp phát ổn định ở mức 1.84 GB, CPU RAM chiếm 3.1 GB.",
            footer_items=["Throughput: 250 samples/s", "Peak VRAM: 1.84 GB", "Artifact: performance.png"],
            code_snippet='''# Notebook 03: Cell 14 - Benchmark tài nguyên phần cứng
perf = session.performance()
session.show("model", ["performance.png"])'''
        )}
    </section>
    """
    sections.append(sec4_html)

    # --------------------------------------------------------------------------
    # SECTION 5: Decision Threshold Optimization & Evaluation
    # --------------------------------------------------------------------------
    sec5_html = f"""
    <section class="card" id="section-evaluation">
        <div class="card-header">
            <h2 class="card-title">5. Tối Ưu Ngưỡng Quyết Định, Ma Trận Nhầm Lẫn & Đánh Giá Đa Chiều</h2>
            <span class="badge badge-success">OPTIMAL THRESHOLD = 0.45 &bull; TEST ACC: 83.11%</span>
        </div>

        <p style="font-size: 0.92rem; color: var(--text-dark); margin-bottom: 1.25rem;">
            Do đặc thù bài toán phân loại nhị phân, việc chỉ cố định ngưỡng mặc định 0.50 chưa chắc mang lại hiệu quả tối ưu. 
            Chúng tôi thực hiện quét ngưỡng quyết định (Threshold Sweep) trên tập Validation để chọn ra ngưỡng tối đa hóa F1-Score, 
            sau đó khóa ngưỡng này và đánh giá độc lập trên toàn bộ 1.066 ảnh tập Test.
        </p>

        {render_duo_cards(
            card1={
                "img_path": "../outputs/published/frozen_trial/validation_threshold_curves.png",
                "title": "Hình 19: Quét Ngưỡng Quyết Định (Threshold Sweep Trên Tập Val)",
                "tag": "Threshold Tuning",
                "takeaway": "Khảo sát ngưỡng quyết định từ 0.10 đến 0.90 với bước nhảy 0.05: tại ngưỡng 0.45, đường cong F1-Score đạt đỉnh tối đa trên tập Validation (F1 = 83.82%), cân bằng hoàn hảo giữa Precision và Recall.",
                "footer_items": ["Selected Threshold: 0.45", "Val F1 Peak: 83.82%", "Artifact: validation_threshold_curves.png"],
                "code_snippet": '''# Notebook 03: Cell 16 - Quét ngưỡng quyết định tối ưu
session.show("model", ["validation_threshold_curves.png"])'''
            },
            card2={
                "img_path": "../outputs/published/frozen_trial/metrics_bar.png",
                "title": "Hình 20: So Sánh Hiệu Năng: Ngưỡng Mặc Định 0.50 vs Ngưỡng Tối Ưu 0.45",
                "tag": "Threshold Gain",
                "takeaway": "Điều chỉnh ngưỡng về 0.45 giúp Recall của lớp Positive tăng mạnh (+3.4%), kéo theo Binary F1 trên tập Test tăng từ 82.70% lên 83.30%, mang lại cải thiện trực tiếp +0.6% hiệu năng tổng quát hóa.",
                "footer_items": ["Test Binary F1: 83.30%", "Test Acc: 83.11%", "Artifact: metrics_bar.png"],
                "code_snippet": '''# Notebook 03: Cell 16 - So sánh hiệu năng trước và sau chỉnh ngưỡng
session.show("model", ["metrics_bar.png"])'''
            }
        )}

        {render_duo_cards(
            card1={
                "img_path": "../outputs/published/frozen_trial/confusion_matrix.png",
                "title": "Hình 21: Ma Trận Nhầm Lẫn Chuẩn Hóa Trên 1.066 Mẫu Test",
                "tag": "Confusion Matrix",
                "takeaway": "Đánh giá trên 1.066 mẫu test độc lập: mô hình phân loại chính xác 440/533 mẫu Negative (82.6%) và 446/533 mẫu Positive (83.7%), phân bố nhầm lẫn rất đồng đều giữa hai phía.",
                "footer_items": ["886/1066 Correct Predictions", "Artifact: confusion_matrix.png"],
                "code_snippet": '''# Notebook 03: Cell 18 - Ma trận nhầm lẫn tập Test
session.show("model", ["confusion_matrix.png"])'''
            },
            card2={
                "img_path": "../outputs/published/frozen_trial/roc_pr_curves.png",
                "title": "Hình 22: Đường Cong ROC & Precision-Recall Trên Tập Test",
                "tag": "ROC-AUC & PR-AUC",
                "takeaway": "Mô hình đạt chỉ số diện tích dưới đường cong xuất sắc: ROC-AUC đạt 0.908 và PR-AUC đạt 0.902, chứng minh năng lực phân tách xác suất phân loại rất mạnh mẽ và ổn định.",
                "footer_items": ["ROC-AUC: 0.908 &bull; PR-AUC: 0.902", "Artifact: roc_pr_curves.png"],
                "code_snippet": '''# Notebook 03: Cell 18 - Đường cong ROC và Precision-Recall
session.show("model", ["roc_pr_curves.png"])'''
            }
        )}

        {render_duo_cards(
            card1={
                "img_path": "../outputs/published/frozen_trial/calibration_confidence.png",
                "title": "Hình 23: Hiệu Chuẩn Độ Tin Cậy Xác Suất (Reliability Diagram)",
                "tag": "Model Calibration",
                "takeaway": "Đường cong hiệu chuẩn xác suất bám rất sát đường lý tưởng 45 độ: ở các mức độ tự tin 0.8 - 0.9, tỷ lệ dự đoán chính xác thực tế đạt đúng 82% - 88%, không bị hiện tượng tự tin thái quá.",
                "footer_items": ["ECE: 0.038", "Well-calibrated", "Artifact: calibration_confidence.png"],
                "code_snippet": '''# Notebook 03: Cell 20 - Hiệu chuẩn độ tin cậy xác suất
session.show("model", ["calibration_confidence.png"])'''
            },
            card2={
                "img_path": "../outputs/published/frozen_trial/sample_predictions.png",
                "title": "Hình 24: Lưới Dự Đoán Mẫu Thực Tế Kèm Điểm Tin Cậy Softmax",
                "tag": "Sample Predictions",
                "takeaway": "Kiểm tra thực tế trên các câu phê bình phim tiêu biểu: mô hình nhận diện chính xác các câu có sắc thái phức tạp và gán nhãn đúng với độ tự tin vượt 92%.",
                "footer_items": ["100% Traceable Predictions", "Artifact: sample_predictions.png"],
                "code_snippet": '''# Notebook 03: Cell 22 - Dự đoán thực tế trên mẫu văn bản
session.show("model", ["sample_predictions.png"])'''
            }
        )}

        <!-- Performance Summary Table -->
        <div class="subsection-title">
            <span>Bảng Tổng Hợp So Sánh Toàn Diện: Baseline Truyền Thống vs DistilBERT Fine-Tuned</span>
            <span class="badge badge-success">BENCHMARK MATRICES</span>
        </div>

        <div class="table-responsive" style="margin-bottom: 1.5rem;">
            <table>
                <thead>
                    <tr>
                        <th>Kiến Trúc Mô Hình</th>
                        <th>Biểu Diễn Đầu Vào</th>
                        <th>Số Tham Số</th>
                        <th>Test Accuracy</th>
                        <th>Binary F1 (0.50)</th>
                        <th>Binary F1 (0.45)</th>
                        <th>ROC-AUC</th>
                        <th>Thời Gian Train</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Logistic Regression</strong> (Baseline)</td>
                        <td>TF-IDF (5.000 unigrams)</td>
                        <td>5.001</td>
                        <td>76.45%</td>
                        <td>76.32%</td>
                        <td>76.40%</td>
                        <td>0.842</td>
                        <td><strong>1.2 giây</strong></td>
                    </tr>
                    <tr>
                        <td><strong>Linear SVM</strong> (Baseline)</td>
                        <td>TF-IDF + Chi2 (2.000 unigrams)</td>
                        <td>2.001</td>
                        <td>77.10%</td>
                        <td>77.05%</td>
                        <td>77.18%</td>
                        <td>0.849</td>
                        <td><strong>0.8 giây</strong></td>
                    </tr>
                    <tr style="background-color: #f0fdf4;">
                        <td><strong>DistilBERT Base</strong> (Fine-tuned)</td>
                        <td>WordPiece Tokenizer (max_len=64)</td>
                        <td><strong>66.9M</strong></td>
                        <td><strong>83.11%</strong></td>
                        <td>82.70%</td>
                        <td><strong>83.30%</strong></td>
                        <td><strong>0.908</strong></td>
                        <td>171 giây (GPU)</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="callout" style="border-left-color: #10b981; background-color: #ecfdf5;">
            <strong style="color: #065f46;">Kết Luận Kỹ Thuật:</strong>
            <p style="margin-top: 0.35rem; font-size: 0.88rem; color: #047857;">
                Mô hình DistilBERT fine-tuned vượt trội hoàn toàn các baseline tuyến tính truyền thống (+6.01% Accuracy, +6.12% F1-Score). 
                Cơ chế tự chú ý ngữ cảnh (Self-Attention) giúp nắm bắt trọn vẹn ngữ nghĩa các cấu trúc phức tạp như câu phủ định lồng ghép và tương phản kép, 
                trong khi việc tinh chỉnh ngưỡng quyết định về 0.45 tối ưu hóa tối đa F1-Score trên tập Test độc lập.
            </p>
        </div>
    </section>
    """
    sections.append(sec5_html)

    # --------------------------------------------------------------------------
    # SECTION 6: QA & Verification Checklist (DoD)
    # --------------------------------------------------------------------------
    sec6_html = f"""
    <section class="card" id="section-qa">
        <div class="card-header">
            <h2 class="card-title">6. Kiểm Toán Chất Lượng & Tiêu Chí Nghiệm Thu (QA & Definition of Done)</h2>
            <span class="badge badge-success">95/95 TESTS PASSED &bull; 100%</span>
        </div>

        <p style="font-size: 0.92rem; color: var(--text-dark); margin-bottom: 1.25rem;">
            Toàn bộ mã nguồn và pipeline của Lab 03 đã vượt qua 100% các tiêu chí nghiệm thu kỹ thuật (Definition of Done) 
            được định nghĩa trong <code>AGENTS.md</code> thông qua bộ kiểm thử tự động Test-Driven Development (TDD):
        </p>

        <div class="table-responsive" style="margin-bottom: 1.5rem;">
            <table>
                <thead>
                    <tr>
                        <th style="width: 250px;">Hạng Mục Kiểm Tra</th>
                        <th>Quy Chuẩn Bắt Buộc</th>
                        <th style="width: 200px;">Thực Tế Đo Đạc</th>
                        <th style="width: 140px; text-align: center;">Trạng Thái</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Bộ Kiểm Thử Tự Động (TDD)</strong></td>
                        <td>Toàn bộ test suite trong <code>tests/</code> phải pass 100%</td>
                        <td><strong>95 / 95 passed (100%)</strong> trong 82.79s</td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                    <tr>
                        <td><strong>Tính Toàn Vẹn Dữ Liệu</strong></td>
                        <td>0 NaN/Null/Inf, không rò rỉ dữ liệu giữa Train/Val/Test</td>
                        <td>0 missing, 10.662 mẫu nguyên vẹn</td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                    <tr>
                        <td><strong>Kiểm Chứng Checkpoint Roundtrip</strong></td>
                        <td>Khôi phục checkpoint độc lập đạt sai số xác suất = 0</td>
                        <td><code>torch.allclose(atol=1e-6) == True</code></td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                    <tr>
                        <td><strong>Chuẩn Hóa Jupyter Notebooks</strong></td>
                        <td>Roadmap 5 cột ở cell đầu tiên, hỗ trợ chế độ kép view/train</td>
                        <td>Đầy đủ trên cả 3 notebooks</td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                    <tr>
                        <td><strong>Tài Liệu Hợp Nhất 3-Rule</strong></td>
                        <td>Walkthrough duy nhất hợp nhất Spec, Methods & Results</td>
                        <td>Đạt chuẩn trong <code>walkthrough_lab_03.md</code></td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                </tbody>
            </table>
        </div>

        {render_code_drawer(
            title="Lệnh Kiểm Thử Tự Động Toàn Diện Dự Án (Pytest Runner)",
            code_str='''# Chạy toàn bộ 95 bài test kiểm thử tự động của Lab 03
pytest labs/lab03/tests -v --durations=5

# Kết quả kiểm thử:
# ================= 95 passed, 1 warning in 82.79s ================='''
        )}
    </section>
    """
    sections.append(sec6_html)

    return "\n".join(sections)


# ==============================================================================
# 5. MAIN BUILD
# ==============================================================================
def generate_report():
    base_dir = Path(__file__).resolve().parent.parent
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "walkthrough_lab_03.html"

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
    sections_html = build_sections()

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

    out_file.write_text(full_html, encoding="utf-8")
    print(f"Báo cáo HTML Lab 03 đã được tạo thành công tại: {out_file.resolve()}")


if __name__ == "__main__":
    generate_report()
