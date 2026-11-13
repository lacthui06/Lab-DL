"""Script to compile walkthrough_lab_03.md into a standalone, styled HTML report.

Output: labs/lab03/reports/walkthrough_lab_03.html
Includes full CSS for browser viewing and print-to-PDF formatting.
"""

from pathlib import Path
import markdown

def generate_report():
    base_dir = Path(__file__).resolve().parent.parent
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    md_file = reports_dir / "walkthrough_lab_03.md"
    html_file = reports_dir / "walkthrough_lab_03.html"
    
    if not md_file.exists():
        raise FileNotFoundError(f"Markdown report not found at {md_file}")
        
    raw_md = md_file.read_text(encoding="utf-8")
    
    # Configure Markdown extensions
    md_converter = markdown.Markdown(
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "nl2br",
            "sane_lists"
        ],
        extension_configs={
            "toc": {
                "title": "Mục Lục Báo Cáo",
                "permalink": False
            }
        }
    )
    
    body_html = md_converter.convert(raw_md)
    toc_html = getattr(md_converter, "toc", "")
    
    # Complete HTML template with modern GitHub/Academic styling
    full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Báo cáo Walkthrough — Lab 03: Binary Sentiment Classification</title>
  <style>
    :root {{
      --color-fg: #24292f;
      --color-fg-muted: #57606a;
      --color-border: #d0d7de;
      --color-bg: #ffffff;
      --color-bg-subtle: #f6f8fa;
      --color-primary: #0969da;
      --color-primary-subtle: #ddf4ff;
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
      color: var(--color-fg);
      background-color: #f0f2f5;
      line-height: 1.65;
      padding: 24px;
      font-size: 15px;
    }}

    .container {{
      max-width: 1040px;
      margin: 0 auto;
      background: var(--color-bg);
      border: 1px solid var(--color-border);
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      padding: 48px 56px;
    }}

    /* Header banner */
    .header-tag {{
      display: inline-block;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-radius: 12px;
      background-color: var(--color-primary-subtle);
      color: var(--color-primary);
      margin-bottom: 12px;
    }}

    h1, h2, h3, h4, h5, h6 {{
      color: #1f2328;
      font-weight: 600;
      line-height: 1.35;
      margin-top: 28px;
      margin-bottom: 14px;
    }}

    h1 {{
      font-size: 28px;
      padding-bottom: 12px;
      border-bottom: 2px solid var(--color-border);
      margin-top: 0;
    }}

    h2 {{
      font-size: 22px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--color-border);
      margin-top: 36px;
    }}

    h3 {{
      font-size: 18px;
      margin-top: 24px;
    }}

    h4 {{
      font-size: 16px;
      margin-top: 20px;
    }}

    p {{
      margin-bottom: 14px;
    }}

    ul, ol {{
      margin-bottom: 14px;
      padding-left: 28px;
    }}

    li {{
      margin-bottom: 6px;
    }}

    hr {{
      border: 0;
      border-top: 1px solid var(--color-border);
      margin: 28px 0;
    }}

    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 14px;
      overflow-x: auto;
      display: block;
    }}

    th, td {{
      padding: 10px 14px;
      border: 1px solid var(--color-border);
      text-align: left;
    }}

    th {{
      background-color: var(--color-bg-subtle);
      font-weight: 600;
      color: #24292f;
    }}

    tr:nth-child(even) {{
      background-color: #fafbfc;
    }}

    tr:hover {{
      background-color: #f2f4f8;
    }}

    /* Code */
    code {{
      font-family: var(--font-mono);
      font-size: 13px;
      padding: 2px 6px;
      background-color: var(--color-bg-subtle);
      border: 1px solid rgba(175, 184, 193, 0.3);
      border-radius: 4px;
      color: #b31d28;
    }}

    pre {{
      background-color: var(--color-bg-subtle);
      border: 1px solid var(--color-border);
      border-radius: 6px;
      padding: 16px;
      overflow-x: auto;
      margin: 18px 0;
    }}

    pre code {{
      background-color: transparent;
      border: none;
      padding: 0;
      color: #24292f;
      font-size: 13.5px;
    }}

    /* Blockquotes */
    blockquote {{
      padding: 8px 16px;
      color: var(--color-fg-muted);
      border-left: 4px solid var(--color-primary);
      background-color: var(--color-bg-subtle);
      margin: 16px 0;
      border-radius: 0 4px 4px 0;
    }}

    /* Images */
    img {{
      max-width: 100%;
      height: auto;
      display: block;
      margin: 20px auto;
      border: 1px solid var(--color-border);
      border-radius: 6px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}

    /* Links */
    a {{
      color: var(--color-primary);
      text-decoration: none;
    }}

    a:hover {{
      text-decoration: underline;
    }}

    /* Table of contents styling */
    .toc {{
      background: var(--color-bg-subtle);
      border: 1px solid var(--color-border);
      border-radius: 6px;
      padding: 18px 24px;
      margin: 24px 0 32px 0;
    }}

    .toc-title {{
      font-weight: 600;
      font-size: 16px;
      margin-bottom: 10px;
      color: #1f2328;
    }}

    .toc ul {{
      margin-bottom: 0;
      padding-left: 20px;
    }}

    .toc li {{
      margin-bottom: 4px;
      font-size: 14px;
    }}

    /* Print styling */
    @media print {{
      body {{
        background: #fff;
        padding: 0;
        font-size: 12pt;
      }}
      .container {{
        box-shadow: none;
        border: none;
        max-width: 100%;
        padding: 0;
      }}
      .toc {{
        display: none;
      }}
      h1, h2, h3 {{
        page-break-after: avoid;
      }}
      table, pre, img, blockquote {{
        page-break-inside: avoid;
      }}
      table {{
        display: table;
        font-size: 10pt;
      }}
      a {{
        color: #000;
        text-decoration: underline;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header-tag">Deep Learning Lab Report &bull; Rotten Tomatoes Sentiment</div>
    {body_html}
  </div>
</body>
</html>
"""

    html_file.write_text(full_html, encoding="utf-8")
    print(f"Generated HTML report successfully at: {html_file}")
    return html_file

if __name__ == "__main__":
    generate_report()
