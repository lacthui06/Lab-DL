r"""Template script to generate standardized, visual-first HTML reports for Lab DL.
Based on the official UTH Computer Vision & Deep Learning template standards.

Best Practices:
1. Visual-first & Minimal Text: Tap trung bieu do, anh truc quan va bang so lieu. Han che viet chu dai dong.
2. Code di lien voi anh: Moi the anh visualize (Hero, Duo, Trio) co the truyen kem `code_snippet` (2-4 dong tu notebook)
   de tu dong tao Code Drawer nam NGAY DUOI BUC ANH DANG VISUALIZE.
3. Khong dung Raw LaTeX: Tuyet doi khong dung \frac, \sqrt, \beta, $...$ vi trinh duyet khong co MathJax
   se render tho bi loi chu. Hay dung text unicode va ky hieu toan hoc pho thong.
4. Tuyen bo tac gia duy nhat: Ghi ro Thong tin tac gia (Thai Anh Lac) va Don vi dao tao (UTH).

Usage for a new lab (e.g., Lab 0X):
1. Copy this file to labs/lab0X/scripts/generate_html_report.py
2. Update LAB_CONFIG (lab_num, lab_id, title, subtitle, metrics, etc.)
3. Update PIPELINE_STEPS (6 buoc phu hop voi workflow cua bai toan)
4. Implement build_sections() to assemble report sections using:
   - render_hero_card(img_path, title, tag, takeaway, footer_items, code_snippet=...)
   - render_duo_cards(card1_dict, card2_dict)
   - render_trio_cards(card1_dict, card2_dict, card3_dict)
   - render_code_drawer(title, code_str, language="python")
   - render_data_integrity_section(audit_rows, struct_rows, code_example)
5. Run: python labs/lab0X/scripts/generate_html_report.py
"""

from pathlib import Path

# ==============================================================================
# 1. LAB CONFIGURATION (Thay doi thong tin lab cu the tai day)
# ==============================================================================
LAB_CONFIG = {
    "lab_num": "0X",
    "lab_id": "lab0X",
    "title": "Báo Cáo Nghiệm Thu Kỹ Thuật Lab 0X",
    "subtitle": "Tên Đề Tài & Bài Toán Nghiên Cứu (Kiến Trúc Mô Hình & Dataset)",
    "author": "Thái Anh Lạc",
    "author_role": "Học viên / Kỹ sư AI",
    "school_name": "TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM",
    "sub_dept": "VIỆN ĐÀO TẠO KỸ THUẬT & CÔNG NGHỆ CHẤT LƯỢNG CAO",
    "python_ver": "Python 3.13.x",
    "framework_ver": "PyTorch 2.x",
    "dataset_info": "Dataset Name (Số lượng mẫu & Đặc trưng)",
    "test_passed": "XX/XX Tests Passed (100%)",
    "primary_metric": "Best Metric (e.g. Test Acc: XX.X% / Macro F1: XX.X%)",
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

# Danh sach 6 buoc chuan tren pipeline stepper (anh xa theo docs/dl_flow.md)
PIPELINE_STEPS = [
    {
        "num": "01",
        "title": "1. Khám Phá EDA",
        "desc": "Kiểm Toán Dữ Liệu &bull; 0 NaN &bull; Phân Bố Mẫu",
        "anchor": "#section-eda",
    },
    {
        "num": "02",
        "title": "2. Tiền Xử Lý Dữ Liệu",
        "desc": "Data Pipeline &bull; Augmentation &bull; Feature Transforms",
        "anchor": "#section-features",
    },
    {
        "num": "03",
        "title": "3. Kiến Trúc Mô Hình",
        "desc": "Network Design &bull; Baseline &bull; Parameter Budget",
        "anchor": "#section-models",
    },
    {
        "num": "04",
        "title": "4. Huấn Luyện & Tối Ưu",
        "desc": "Optimizer Dynamics &bull; Tuning &bull; Learning Curves",
        "anchor": "#section-training",
    },
    {
        "num": "05",
        "title": "5. Đánh Giá & Chẩn Đoán",
        "desc": "Test Benchmark &bull; Confusion Matrix &bull; ROC/PR",
        "anchor": "#section-evaluation",
    },
    {
        "num": "06",
        "title": "6. Nghiệm Thu Kỹ Thuật",
        "desc": "Test Suite Passed &bull; Artifacts &bull; Model Handoff",
        "anchor": "#section-qa",
    },
]


# ==============================================================================
# 2. HELPER FUNCTIONS TO RENDER COMPONENTS
# ==============================================================================
def _render_card_drawer(code_snippet: str | None) -> str:
    """Tao collapsible code drawer tich hop ben trong card chan anh."""
    if not code_snippet:
        return ""
    return f"""
        <details class="code-tab-drawer">
            <summary class="code-tab-summary">
                <div class="code-summary-left">
                    <span class="code-pill">NOTEBOOK CELL</span>
                    <span>Cell trực quan hóa ảnh này</span>
                </div>
                <span class="code-badge">Xem code cell &dtrif;</span>
            </summary>
            <pre><code class="language-python">{code_snippet.strip()}</code></pre>
        </details>"""


def render_hero_card(
    img_path: str,
    title: str,
    tag: str,
    takeaway: str,
    footer_items: list[str],
    code_snippet: str | None = None,
) -> str:
    """Render full-width hero card cho bieu do tong quan hoac ma tran trung tam."""
    footer_html = "".join(f"<span>{item}</span>" for item in footer_items)
    drawer_html = _render_card_drawer(code_snippet)
    return f"""
    <figure class="image-card">
        <div class="image-card-img-wrap" onclick="openModal(this)">
            <img src="{img_path}" alt="{title}" loading="lazy">
            <span class="zoom-hint">Phóng to [Zoom]</span>
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
            {drawer_html}
        </figcaption>
    </figure>
    """


def render_duo_cards(card1: dict, card2: dict) -> str:
    """Render layout so sanh 2 cot (Grid 2-col)."""
    def _card(c):
        footer_html = "".join(f"<span>{item}</span>" for item in c.get("footer_items", []))
        drawer_html = _render_card_drawer(c.get("code_snippet"))
        return f"""
        <figure class="image-card">
            <div class="image-card-img-wrap" onclick="openModal(this)">
                <img src="{c['img_path']}" alt="{c['title']}" loading="lazy">
                <span class="zoom-hint">Phóng to [Zoom]</span>
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
                {drawer_html}
            </figcaption>
        </figure>
        """
    return f"""
    <div class="grid-2col">
        {_card(card1)}
        {_card(card2)}
    </div>
    """


def render_trio_cards(card1: dict, card2: dict, card3: dict) -> str:
    """Render layout 3 cot (Grid 3-col) cho cac chuoi phan tich hoac bieu do nho."""
    def _card(c):
        footer_html = "".join(f"<span>{item}</span>" for item in c.get("footer_items", []))
        drawer_html = _render_card_drawer(c.get("code_snippet"))
        return f"""
        <figure class="image-card">
            <div class="image-card-img-wrap" onclick="openModal(this)">
                <img src="{c['img_path']}" alt="{c['title']}" loading="lazy">
                <span class="zoom-hint">Phóng to [Zoom]</span>
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
                {drawer_html}
            </figcaption>
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
    """Render collapsible code drawer doc lap ngoai the card."""
    return f"""
    <details class="code-tab-drawer">
        <summary class="code-tab-summary">
            <div class="code-summary-left">
                <span class="code-pill">{language.upper()}</span>
                <span>{title}</span>
            </div>
            <span class="code-badge">Nhấn để mở code &dtrif;</span>
        </summary>
        <pre><code class="language-{language}">{code_str.strip()}</code></pre>
    </details>
    """


def render_data_integrity_section(audit_rows: list[dict], struct_rows: list[dict], code_example: str | None = None) -> str:
    """Render bang kiem toan toan ven du lieu (Core EDA)."""
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

    struct_headers = "".join(f"<th>{k}</th>" for k in struct_rows[0].keys()) if struct_rows else ""
    struct_tbody = ""
    for r in struct_rows:
        cols = "".join(f"<td>{v}</td>" for v in r.values())
        struct_tbody += f"<tr>{cols}</tr>"

    code_html = render_code_drawer("Mã nguồn kiểm toán tính toàn vẹn dữ liệu", code_example) if code_example else ""

    return f"""
    <div class="subsection-title">
        <span>Bảng 1: Kiểm Toán Tính Toàn Vẹn & Không Thất Thoát Dữ Liệu</span>
        <span class="badge badge-success">DATA INTEGRITY: 100%</span>
    </div>
    <div class="table-responsive">
        <table>
            <thead>
                <tr>
                    <th>Hạng Mục Kiểm Toán</th>
                    <th>Quy Chuẩn Kỳ Vọng</th>
                    <th>Kết Quả Thực Nghiệm</th>
                    <th style="text-align: center;">Trạng Thái</th>
                </tr>
            </thead>
            <tbody>
                {audit_tbody}
            </tbody>
        </table>
    </div>

    <div class="subsection-title" style="margin-top: 1.5rem;">
        <span>Bảng 2: Cấu Trúc Kỹ Thuật Tập Dữ Liệu</span>
        <span class="badge badge-info">DATASET CONTRACT</span>
    </div>
    <div class="table-responsive">
        <table>
            <thead>
                <tr>
                    {struct_headers}
                </tr>
            </thead>
            <tbody>
                {struct_tbody}
            </tbody>
        </table>
    </div>
    {code_html}
    """


# ==============================================================================
# 3. HTML PAGE MASTER TEMPLATE
# ==============================================================================
HTML_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {author}</title>
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

        /* Header with UTH Brand & Author Box */
        header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
            color: #ffffff;
            padding: 2.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-lg);
        }}

        .header-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.15);
            padding-bottom: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .uth-brand {{
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }}

        .uth-logo-text {{
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: 2px;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .uth-brand-details {{
            border-left: 2px solid rgba(255, 255, 255, 0.2);
            padding-left: 1.25rem;
        }}

        .uth-school-name {{
            font-size: 0.95rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #f8fafc;
        }}

        .uth-sub-dept {{
            font-size: 0.8rem;
            color: #94a3b8;
            font-weight: 500;
        }}

        .author-box {{
            background: rgba(255, 255, 255, 0.08);
            padding: 0.75rem 1.25rem;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            text-align: right;
        }}

        .author-label {{
            display: block;
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .author-name {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #ffffff;
        }}

        .author-role {{
            display: block;
            font-size: 0.8rem;
            color: #38bdf8;
            font-weight: 500;
        }}

        h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header-subtitle {{
            font-size: 1.05rem;
            color: #94a3b8;
            margin-bottom: 1.5rem;
        }}

        .header-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }}

        .badge-success {{ background: #059669; color: #ffffff; }}
        .badge-info {{ background: #0284c7; color: #ffffff; }}
        .badge-dark {{ background: #334155; color: #f8fafc; }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        /* Pipeline Stepper */
        .pipeline-stepper {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .pipeline-card {{
            border-radius: 12px;
            padding: 1.15rem 1rem;
            text-decoration: none;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}

        .pipeline-card:hover {{
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }}

        .pipeline-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .pipeline-number {{
            color: #ffffff;
            font-weight: 800;
            font-size: 0.75rem;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
        }}

        .pipeline-title {{
            font-weight: 700;
            font-size: 0.95rem;
        }}

        .pipeline-desc {{
            font-size: 0.75rem;
            opacity: 0.85;
            line-height: 1.4;
        }}

        /* Cards & Sections */
        .card {{
            background: var(--card-bg);
            border-radius: 14px;
            border: 1px solid var(--border);
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow);
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
            gap: 0.75rem;
        }}

        .card-title {{
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--primary);
        }}

        .subsection-title {{
            font-size: 1rem;
            font-weight: 700;
            color: var(--secondary);
            margin: 1.5rem 0 0.85rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .grid-2col {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-top: 1.25rem;
        }}

        .grid-3col {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.25rem;
            margin-top: 1.25rem;
        }}

        /* Image Cards */
        .image-card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-bottom: 1.5rem;
        }}

        .image-card:hover {{
            box-shadow: var(--shadow-lg);
        }}

        .image-card-img-wrap {{
            position: relative;
            background: #f8fafc;
            border-bottom: 1px solid var(--border);
            overflow: hidden;
            cursor: zoom-in;
            text-align: center;
            padding: 1rem;
        }}

        .image-card-img-wrap img {{
            width: 100%;
            max-height: 480px;
            object-fit: contain;
            display: block;
            margin: 0 auto;
            transition: transform 0.25s ease;
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
            transition: opacity 0.2s ease;
            pointer-events: none;
        }}

        .image-card-img-wrap:hover .zoom-hint {{
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

        .code-pill {{
            background: #f1f5f9;
            color: #475569;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: var(--font-mono);
            letter-spacing: 0.3px;
        }}

        /* Code Tab Drawer Inside Card */
        .image-card .code-tab-drawer {{
            margin-top: 1rem;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
            background: #f8fafc;
            box-shadow: none;
        }}

        .image-card .code-tab-drawer .code-tab-summary {{
            padding: 0.55rem 0.9rem;
            background: #f1f5f9;
            font-size: 0.8rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            user-select: none;
        }}

        .image-card .code-tab-drawer[open] .code-tab-summary {{
            border-bottom: 1px solid #e2e8f0;
            background: #e2e8f0;
        }}

        .image-card .code-tab-drawer pre {{
            margin: 0;
            border-radius: 0;
            padding: 0.85rem 1rem;
            background: #1e293b !important;
            font-size: 0.78rem;
        }}

        /* Independent Code Drawer */
        .code-tab-drawer {{
            margin: 1.25rem 0;
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
            background: #ffffff;
            box-shadow: var(--shadow);
        }}

        .code-tab-summary {{
            padding: 0.8rem 1.25rem;
            background: #f8fafc;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.88rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            user-select: none;
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
            color: var(--primary-light);
            font-weight: 600;
        }}

        pre[class*="language-"] {{
            margin: 0;
            border-radius: 0;
            padding: 1.25rem;
            background: #0f172a !important;
            font-size: 0.82rem;
            font-family: var(--font-mono);
        }}

        /* Table styles */
        .table-responsive {{
            overflow-x: auto;
            border-radius: 10px;
            border: 1px solid var(--border);
            margin: 1rem 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            text-align: left;
        }}

        th {{
            background: #f8fafc;
            color: var(--secondary);
            font-weight: 700;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
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
            background: #f8fafc;
        }}

        /* Lightbox Image Modal */
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

        .image-modal.active {{ display: flex; opacity: 1; }}

        .modal-content {{
            max-width: 92vw;
            max-height: 90vh;
            background: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            display: flex;
            flex-direction: column;
            cursor: default;
        }}

        .modal-img-wrap {{
            overflow: auto;
            max-height: 80vh;
            padding: 1.5rem;
            background: #f8fafc;
            text-align: center;
        }}

        .modal-img-wrap img {{
            max-width: 100%;
            height: auto;
            object-fit: contain;
        }}

        .modal-caption {{
            padding: 1rem 1.5rem;
            background: #ffffff;
            border-top: 1px solid var(--border);
        }}

        .modal-title {{
            font-weight: 700;
            font-size: 1rem;
            color: var(--text-dark);
            margin-bottom: 0.25rem;
        }}

        .modal-desc {{
            font-size: 0.85rem;
            color: var(--text-muted);
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
                <span class="uth-logo-text">UTH</span>
                <div class="uth-brand-details">
                    <div class="uth-school-name">{school_name}</div>
                    <div class="uth-sub-dept">{sub_dept}</div>
                </div>
            </div>
            <div class="author-box">
                <span class="author-label">Báo cáo Khoa học &amp; Kỹ thuật</span>
                <span class="author-name">{author}</span>
                <span class="author-role">{author_role}</span>
            </div>
        </div>
        <h1>{title}</h1>
        <p class="header-subtitle">{subtitle}</p>
        <div class="header-badges">
            <span class="badge badge-dark">{python_ver}</span>
            <span class="badge badge-dark">{framework_ver}</span>
            <span class="badge badge-dark">{dataset_info}</span>
            <span class="badge badge-success">{test_passed}</span>
            <span class="badge badge-info">{primary_metric}</span>
        </div>
    </div>
</header>

<div class="container">
    <!-- Pipeline Stepper -->
    <div class="pipeline-stepper">
        {stepper_html}
    </div>

    <!-- Main Content Sections -->
    {sections_html}
</div>

<!-- Modal Zoom -->
<div id="imageModal" class="image-modal" onclick="closeModal(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
        <div class="modal-img-wrap">
            <img id="modalImg" src="" alt="">
        </div>
        <div class="modal-caption">
            <div id="modalTitle" class="modal-title"></div>
            <div id="modalDesc" class="modal-desc"></div>
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
        const card = wrapEl.closest('.image-card');
        const titleEl = card ? card.querySelector('.image-card-title') : null;
        const takeawayEl = card ? card.querySelector('.image-card-takeaway') : null;

        modalImg.src = img.src;
        modalTitle.textContent = titleEl ? titleEl.textContent : img.alt;
        modalDesc.textContent = takeawayEl ? takeawayEl.textContent : '';

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

    // Dynamic toggle badge text for code drawers
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
# 4. MAIN BUILD FUNCTION (Chay sinh file HTML)
# ==============================================================================
def build_html_report(output_path: Path, sections_html: str):
    """Gop tat ca cac thanh phan va xuat file HTML hoan chinh."""
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
        author_role=LAB_CONFIG["author_role"],
        school_name=LAB_CONFIG["school_name"],
        sub_dept=LAB_CONFIG["sub_dept"],
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
    print(f"Báo cáo HTML đã được tạo thành công tại: {output_path.resolve()}")


if __name__ == "__main__":
    # Vi du minh hoa cach dung template
    sample_audit = [
        {"item": "Giá trị thiếu (NaN / Null / Inf)", "expected": "0 NaN", "actual": "0 mẫu thiếu (100% toàn vẹn)", "status": "PASSED"},
        {"item": "Mẫu trùng lặp (Duplicate Samples)", "expected": "0 trùng", "actual": "0 mẫu trùng lặp", "status": "PASSED"},
    ]
    sample_struct = [
        {"Tập dữ liệu": "Train Split", "Số mẫu": "40.000", "Kích thước": "3 x 64 x 64", "Kiểu dữ liệu": "float32", "Khoảng pixel": "[0.0, 1.0]", "Số lớp": "10"},
    ]
    sample_code = """# Notebook minh hoa: Kiem toan tinh toan ven du lieu
assert not np.isnan(data).any()
print("Data integrity audit: PASSED")"""

    sample_sections = f"""
    <section class="card" id="section-eda">
        <div class="card-header">
            <h2 class="card-title">1. Khám Phá & Kiểm Toán Dữ Liệu (EDA)</h2>
            <span class="badge badge-info">STAGE 1</span>
        </div>
        {render_data_integrity_section(sample_audit, sample_struct, sample_code)}
    </section>
    """

    out_file = Path("sample_report_preview.html")
    build_html_report(out_file, sample_sections)
    if out_file.exists():
        out_file.unlink()  # Don dep file preview sau khi kiem tra
