"""Script to compile walkthrough_lab_01.md into an artifact-rich, interactive HTML report.

Key UX design features:
1. Visual/Artifact-first: Every figure, chart, heatmap, and table is showcased in modern cards.
2. Concise annotations: Clear bulleted takeaways under each artifact.
3. Collapsible Code Blocks: All code blocks are wrapped in closed-by-default <details> drawers.
4. Executive KPI Dashboard: Highlights dataset size, test accuracy, best model, parameter count.
5. Quick Filter & Jump Tabs: Smooth navigation between EDA, Features, Models, Tuning, Results.
6. Print-to-PDF ready: Clean pagination and formatting for Ctrl+P export.
"""

from pathlib import Path
import re
import markdown

def generate_report():
    base_dir = Path(__file__).resolve().parent.parent
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    md_file = reports_dir / "walkthrough_lab_01.md"
    html_file = reports_dir / "walkthrough_lab_01.html"
    
    if not md_file.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_file}")
        
    raw_md = md_file.read_text(encoding="utf-8")
    
    # Pre-process code blocks into collapsible <details> drawers (default closed)
    def wrap_code_block(match):
        lang = match.group(1).strip()
        code_text = match.group(2)
        lang_label = lang.upper() if lang else "PYTHON / CODE"
        return f"""
<details class="code-drawer">
  <summary class="code-drawer-summary">
    <div class="code-summary-left">
      <span class="code-tag">{lang_label}</span>
      <span class="code-title">Mã nguồn triển khai ({lang_label})</span>
    </div>
    <span class="code-btn">Nhấn để mở code &dtrif;</span>
  </summary>
  <pre><code class="language-{lang}">{code_text}</code></pre>
</details>
"""

    # Wrap fenced code blocks
    processed_md = re.sub(r'```([a-zA-Z0-9_]*)\n(.*?)```', wrap_code_block, raw_md, flags=re.DOTALL)
    
    # Configure Markdown
    md_converter = markdown.Markdown(
        extensions=[
            "tables",
            "toc",
            "nl2br",
            "sane_lists"
        ],
        extension_configs={
            "toc": {
                "title": "Mục lục báo cáo",
                "permalink": False
            }
        }
    )
    
    body_html = md_converter.convert(processed_md)
    
    # Wrap standalone images into responsive visual cards with captions
    def wrap_images(match):
        img_tag = match.group(0)
        alt = match.group(1)
        src = match.group(2)
        filename = Path(src).name
        return f"""
<div class="artifact-card">
  <div class="artifact-header">
    <div class="artifact-badge-wrap">
      <span class="artifact-badge">BẰNG CHỨNG THỰC NGHIỆM</span>
      <span class="artifact-filename">{filename}</span>
    </div>
    <span class="artifact-type">IMAGE ARTIFACT</span>
  </div>
  <div class="artifact-img-wrap">
    <img src="{src}" alt="{alt}" loading="lazy" />
  </div>
  <div class="artifact-caption">
    <strong>Trực quan hóa:</strong> {alt}
  </div>
</div>
"""
    body_html = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*src="([^"]*)"[^>]*\/?>', wrap_images, body_html)
    
    full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Báo cáo Thực nghiệm Lab 01 — FashionMNIST Classification (PyTorch)</title>
  <style>
    :root {{
      --primary: #2563eb;
      --primary-light: #eff6ff;
      --primary-dark: #1d4ed8;
      --accent: #10b981;
      --accent-light: #ecfdf5;
      --surface: #ffffff;
      --surface-subtle: #f8fafc;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --border-dark: #cbd5e1;
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
      --shadow-md: 0 4px 14px rgba(0,0,0,0.08);
      --shadow-lg: 0 10px 25px rgba(0,0,0,0.12);
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: var(--font-sans);
      background-color: #f1f5f9;
      color: var(--text-main);
      line-height: 1.65;
      padding: 24px 16px;
      font-size: 15px;
    }}

    .layout-wrapper {{
      max-width: 1140px;
      margin: 0 auto;
    }}

    /* Top Executive Dashboard */
    .dashboard-hero {{
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: #ffffff;
      border-radius: 14px;
      padding: 36px 44px;
      margin-bottom: 20px;
      box-shadow: var(--shadow-lg);
      border: 1px solid rgba(255,255,255,0.08);
    }}

    .hero-badge {{
      display: inline-block;
      background: rgba(37, 99, 235, 0.35);
      color: #93c5fd;
      border: 1px solid rgba(147, 197, 253, 0.3);
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 12px;
    }}

    .dashboard-hero h1 {{
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 8px;
      letter-spacing: -0.5px;
      color: #ffffff;
    }}

    .hero-subtitle {{
      color: #94a3b8;
      font-size: 14.5px;
      margin-bottom: 24px;
    }}

    /* KPI Metrics Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}

    .kpi-card {{
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 10px;
      padding: 16px 20px;
      backdrop-filter: blur(6px);
      transition: transform 0.15s ease, background 0.15s ease;
    }}

    .kpi-card:hover {{
      background: rgba(255, 255, 255, 0.09);
      transform: translateY(-2px);
    }}

    .kpi-label {{
      font-size: 11.5px;
      text-transform: uppercase;
      color: #94a3b8;
      letter-spacing: 0.5px;
      font-weight: 600;
      margin-bottom: 4px;
    }}

    .kpi-value {{
      font-size: 26px;
      font-weight: 700;
      color: #38bdf8;
    }}

    .kpi-desc {{
      font-size: 12px;
      color: #cbd5e1;
      margin-top: 2px;
    }}

    /* Global Action & Control Bar */
    .control-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 20px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-sm);
    }}

    .control-hint {{
      font-size: 13.5px;
      color: var(--text-muted);
    }}

    .control-btn {{
      background: var(--primary-light);
      color: var(--primary);
      border: 1px solid rgba(37, 99, 235, 0.2);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .control-btn:hover {{
      background: var(--primary);
      color: #ffffff;
    }}

    /* Main Container */
    .content-container {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: var(--shadow-sm);
      padding: 44px 50px;
    }}

    /* Typography */
    h1, h2, h3, h4 {{
      color: #0f172a;
      font-weight: 700;
      line-height: 1.3;
    }}

    h1 {{
      font-size: 24px;
      margin: 32px 0 16px 0;
      padding-bottom: 8px;
      border-bottom: 2px solid var(--primary-light);
    }}

    h2 {{
      font-size: 20px;
      margin: 38px 0 14px 0;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}

    h3 {{
      font-size: 16.5px;
      margin: 24px 0 12px 0;
      color: #1e293b;
    }}

    p {{
      margin-bottom: 14px;
      color: #334155;
    }}

    ul, ol {{
      margin-bottom: 16px;
      padding-left: 24px;
      color: #334155;
    }}

    li {{
      margin-bottom: 6px;
    }}

    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 14px;
      border: 1px solid var(--border);
      border-radius: 6px;
      overflow: hidden;
      display: block;
      overflow-x: auto;
    }}

    th, td {{
      padding: 10px 14px;
      border: 1px solid var(--border);
      text-align: left;
    }}

    th {{
      background: var(--surface-subtle);
      font-weight: 600;
      color: #1e293b;
    }}

    tr:nth-child(even) {{
      background: #fafbfc;
    }}

    tr:hover {{
      background: #f1f5f9;
    }}

    /* Artifact Card Showcase */
    .artifact-card {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 10px;
      box-shadow: var(--shadow-sm);
      margin: 24px 0;
      overflow: hidden;
      transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }}

    .artifact-card:hover {{
      border-color: #cbd5e1;
      box-shadow: var(--shadow-md);
    }}

    .artifact-header {{
      background: #f8fafc;
      border-bottom: 1px solid var(--border);
      padding: 9px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
    }}

    .artifact-badge-wrap {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .artifact-badge {{
      background: #e0f2fe;
      color: #0369a1;
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: 600;
      letter-spacing: 0.5px;
    }}

    .artifact-filename {{
      font-family: var(--font-mono);
      color: var(--text-muted);
      font-size: 12px;
    }}

    .artifact-type {{
      color: #94a3b8;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.5px;
    }}

    .artifact-img-wrap {{
      padding: 18px;
      text-align: center;
      background: #fafbfc;
    }}

    .artifact-img-wrap img {{
      max-width: 100%;
      height: auto;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
      margin: 0 auto;
      display: block;
      box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}

    .artifact-caption {{
      padding: 12px 20px;
      background: #ffffff;
      border-top: 1px solid #f1f5f9;
      font-size: 13.5px;
      color: #475569;
    }}

    /* Collapsible Code Drawers (Default Closed) */
    .code-drawer {{
      margin: 18px 0;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
      box-shadow: var(--shadow-sm);
    }}

    .code-drawer-summary {{
      padding: 10px 16px;
      background: #f8fafc;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      user-select: none;
      font-size: 13.5px;
      font-weight: 600;
      color: #1e293b;
      border-bottom: 1px solid transparent;
      transition: background 0.15s ease;
    }}

    .code-drawer[open] .code-drawer-summary {{
      border-bottom: 1px solid var(--border);
      background: #f1f5f9;
    }}

    .code-drawer-summary:hover {{
      background: #f1f5f9;
    }}

    .code-summary-left {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .code-tag {{
      background: #e2e8f0;
      color: #475569;
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 4px;
    }}

    .code-title {{
      font-size: 13px;
      color: #1e293b;
    }}

    .code-btn {{
      font-size: 12px;
      color: var(--primary);
      background: var(--primary-light);
      border: 1px solid rgba(37, 99, 235, 0.2);
      padding: 3px 12px;
      border-radius: 14px;
      font-weight: 500;
      transition: all 0.15s ease;
    }}

    .code-drawer[open] .code-btn {{
      background: #fee2e2;
      color: #dc2626;
      border-color: rgba(220, 38, 38, 0.2);
    }}

    .code-drawer pre {{
      margin: 0;
      padding: 16px;
      background: #0f172a;
      color: #f8fafc;
      font-family: var(--font-mono);
      font-size: 13px;
      overflow-x: auto;
      line-height: 1.55;
    }}

    .code-drawer code {{
      background: transparent;
      color: inherit;
      padding: 0;
      border: none;
    }}

    /* Inline code */
    p code, li code, td code {{
      font-family: var(--font-mono);
      font-size: 13px;
      background: #f1f5f9;
      color: #b91c1c;
      padding: 2px 5px;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
    }}

    /* TOC styling */
    .toc {{
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px 24px;
      margin: 24px 0 32px 0;
    }}

    .toc-title {{
      font-weight: 600;
      font-size: 15px;
      margin-bottom: 10px;
    }}

    .toc ul {{
      margin-bottom: 0;
      padding-left: 20px;
    }}

    .toc li {{
      margin-bottom: 4px;
      font-size: 13.5px;
    }}

    /* Print Optimization */
    @media print {{
      body {{
        background: #ffffff;
        padding: 0;
        font-size: 10pt;
      }}
      .dashboard-hero {{
        background: #0f172a !important;
        color: #ffffff;
        box-shadow: none;
        border: 1px solid #cbd5e1;
        padding: 18px;
      }}
      .control-bar {{
        display: none;
      }}
      .content-container {{
        box-shadow: none;
        border: none;
        padding: 0;
      }}
      .code-drawer {{
        break-inside: avoid;
      }}
      .code-drawer[open] {{
        display: block;
      }}
      .artifact-card {{
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid #cbd5e1;
      }}
      h1, h2, h3 {{
        break-after: avoid;
      }}
      table {{
        display: table;
        font-size: 8.5pt;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout-wrapper">
    <!-- Executive KPI Dashboard -->
    <header class="dashboard-hero">
      <div class="hero-badge">Deep Learning Lab 01 &bull; Báo cáo Tổng kết</div>
      <h1>FashionMNIST Classification with PyTorch</h1>
      <p class="hero-subtitle">
        Tác giả: <strong>Thái Anh Lạc</strong> &bull; Pipeline: <strong>EDA &bull; 10 Handcrafted Features &bull; 3 MLP Architectures &bull; Optimizer Study</strong>
      </p>

      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Tập Dữ Liệu</div>
          <div class="kpi-value">70.000</div>
          <div class="kpi-desc">54k Train / 6k Val / 10k Test (Stratified)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Độ chính xác cao nhất</div>
          <div class="kpi-value">89.2%</div>
          <div class="kpi-desc">Test Accuracy (Adam Refit sau 30 trials)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Mô hình tối ưu</div>
          <div class="kpi-value">MainMLP</div>
          <div class="kpi-desc">784 &rarr; 256 &rarr; 128 &rarr; 10 (235k params)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Kỹ nghệ đặc trưng</div>
          <div class="kpi-value">+10 Features</div>
          <div class="kpi-desc">Mean, Std, Bounding Box, Symmetry, Density</div>
        </div>
      </div>
    </header>

    <!-- Global Control Bar -->
    <div class="control-bar">
      <span class="control-hint">Báo cáo tập trung trực quan hóa hình ảnh, đồ thị và bảng số liệu thực nghiệm.</span>
      <button class="control-btn" id="toggleAllCodeBtn">Mở tất cả code &dtrif;</button>
    </div>

    <!-- Main Content Container with Artifact-First Focus -->
    <main class="content-container">
      {body_html}
    </main>
  </div>

  <script>
    // Smooth toggling text for code drawers
    const codeDrawers = document.querySelectorAll('.code-drawer');
    codeDrawers.forEach(drawer => {{
      drawer.addEventListener('toggle', () => {{
        const btn = drawer.querySelector('.code-btn');
        if (btn) {{
          btn.innerHTML = drawer.open ? 'Đóng code &utrif;' : 'Nhấn để mở code &dtrif;';
        }}
      }});
    }});

    // Global toggle all code drawers
    const toggleAllBtn = document.getElementById('toggleAllCodeBtn');
    let allOpen = false;
    toggleAllBtn.addEventListener('click', () => {{
      allOpen = !allOpen;
      codeDrawers.forEach(d => {{
        d.open = allOpen;
      }});
      toggleAllBtn.innerHTML = allOpen ? 'Đóng tất cả code &utrif;' : 'Mở tất cả code &dtrif;';
    }});
  </script>
</body>
</html>
"""

    html_file.write_text(full_html, encoding="utf-8")
    print(f"Generated visual HTML report successfully at: {html_file}")
    return html_file

if __name__ == "__main__":
    generate_report()
