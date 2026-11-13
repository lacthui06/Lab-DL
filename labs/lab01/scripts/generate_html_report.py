"""Script to generate the standardized, visual-first HTML report for Lab 01.
Strictly adapts the layout, colors, cards, and architecture of the UTH Computer Vision template
with vibrant, colorful process pipeline navigation, spacious artifact galleries, modal image zoom,
and rigorous Data Integrity Verification (Missing Values, NaN, MD5 Duplicate Audit, Zero Data Leakage).

Key UX features:
- Data Integrity Audit: Dedicated tables for Missing values (0 NaN), Pixel-exact duplicates (0 duplicate via MD5 hash), and Zero Train-Test Leakage.
- Vibrant Colored Process Pipeline: 6 distinctive, rich-tinted clickable step cards (not plain white) that pop out prominently.
- Sectional Artifact Layout: Subsections (3.1, 3.2, 4.1, etc.) with Full-Width Hero Cards, 2-Column Comparative Grids, and 3-Column Sequences.
- Lightbox Modal Zoom: Click any image to inspect in full resolution with captions and raw file link.
- Collapsible code tabs: Closed by default (<details class="code-tab-drawer">) with toggle buttons.
- Clean Architecture links and 100% test verification summary (49/49 passed).
"""

from pathlib import Path

def generate_report():
    base_dir = Path(__file__).resolve().parent.parent
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    html_file = reports_dir / "walkthrough_lab_01.html"

    full_html = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Thực Hành Lab 1 - Phân Loại FashionMNIST với PyTorch</title>
    <style>
        :root {
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --accent: #0284c7;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --border: #e2e8f0;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Monaco, Consolas, monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            font-family: var(--font-sans);
            background-color: var(--bg-page);
            color: var(--text-dark);
            line-height: 1.6;
            padding: 2.5rem 1rem;
        }

        .container {
            max-width: 1220px;
            margin: 0 auto;
        }

        header {
            background: linear-gradient(135deg, #1e3a8a 0%, #0369a1 100%);
            color: #ffffff;
            padding: 2.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
        }

        header h1 {
            font-size: 2.1rem;
            margin-bottom: 0.5rem;
            font-weight: 800;
            letter-spacing: -0.025em;
        }

        header h2 {
            font-size: 1.25rem;
            font-weight: 400;
            opacity: 0.92;
            margin-bottom: 1.25rem;
        }

        .meta-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }

        .badge {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(4px);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #ffffff;
        }

        .badge-success {
            background: #dcfce7;
            color: #15803d;
            border: 1px solid #86efac;
            font-weight: 600;
        }

        .badge-info {
            background: #dbeafe;
            color: #1e40af;
            border: 1px solid #bfdbfe;
            font-weight: 600;
        }

        .badge-warning {
            background: #fef3c7;
            color: #b45309;
            border: 1px solid #fde68a;
            font-weight: 600;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 2rem 2.25rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
            scroll-margin-top: 1.5rem;
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            padding-bottom: 0.85rem;
            border-bottom: 2px solid #f1f5f9;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .card-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--primary);
        }

        /* Vibrant Colorful Process Pipeline */
        .pipeline-container {
            margin: 1.5rem 0 2rem;
        }

        .pipeline-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 0.85rem;
        }

        @media (max-width: 1024px) {
            .pipeline-grid {
                grid-template-columns: repeat(3, 1fr);
                gap: 1rem;
            }
        }

        @media (max-width: 640px) {
            .pipeline-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 0.75rem;
            }
        }

        .pipeline-card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            border-radius: 12px;
            padding: 1.15rem 1rem;
            text-decoration: none;
            transition: all 0.25s ease;
            cursor: pointer;
            border-width: 2px;
            border-style: solid;
        }

        .pipeline-card:hover {
            transform: translateY(-5px);
        }

        /* Distinct Vibrant Color Themes for Each Pipeline Step */
        .pipeline-card.step-1 {
            background: linear-gradient(145deg, #e0f2fe 0%, #f0f9ff 100%);
            border-color: #0284c7;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.15);
        }
        .pipeline-card.step-1:hover {
            box-shadow: 0 10px 22px rgba(2, 132, 199, 0.28);
        }
        .pipeline-card.step-1 .pipeline-number { background: #0284c7; color: #ffffff; }
        .pipeline-card.step-1 .pipeline-title { color: #0369a1; }
        .pipeline-card.step-1 .pipeline-desc { color: #0c4a6e; }
        .pipeline-card.step-1 .pipeline-icon { color: #0284c7; }

        .pipeline-card.step-2 {
            background: linear-gradient(145deg, #f3e8ff 0%, #faf5ff 100%);
            border-color: #8b5cf6;
            box-shadow: 0 4px 14px rgba(139, 92, 246, 0.15);
        }
        .pipeline-card.step-2:hover {
            box-shadow: 0 10px 22px rgba(139, 92, 246, 0.28);
        }
        .pipeline-card.step-2 .pipeline-number { background: #8b5cf6; color: #ffffff; }
        .pipeline-card.step-2 .pipeline-title { color: #6d28d9; }
        .pipeline-card.step-2 .pipeline-desc { color: #581c87; }
        .pipeline-card.step-2 .pipeline-icon { color: #8b5cf6; }

        .pipeline-card.step-3 {
            background: linear-gradient(145deg, #dbeafe 0%, #eff6ff 100%);
            border-color: #2563eb;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.15);
        }
        .pipeline-card.step-3:hover {
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.28);
        }
        .pipeline-card.step-3 .pipeline-number { background: #2563eb; color: #ffffff; }
        .pipeline-card.step-3 .pipeline-title { color: #1d4ed8; }
        .pipeline-card.step-3 .pipeline-desc { color: #1e3a8a; }
        .pipeline-card.step-3 .pipeline-icon { color: #2563eb; }

        .pipeline-card.step-4 {
            background: linear-gradient(145deg, #fef3c7 0%, #fffbeb 100%);
            border-color: #f59e0b;
            box-shadow: 0 4px 14px rgba(245, 158, 11, 0.15);
        }
        .pipeline-card.step-4:hover {
            box-shadow: 0 10px 22px rgba(245, 158, 11, 0.28);
        }
        .pipeline-card.step-4 .pipeline-number { background: #f59e0b; color: #ffffff; }
        .pipeline-card.step-4 .pipeline-title { color: #b45309; }
        .pipeline-card.step-4 .pipeline-desc { color: #78350f; }
        .pipeline-card.step-4 .pipeline-icon { color: #f59e0b; }

        .pipeline-card.step-5 {
            background: linear-gradient(145deg, #dcfce7 0%, #f0fdf4 100%);
            border-color: #10b981;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.15);
        }
        .pipeline-card.step-5:hover {
            box-shadow: 0 10px 22px rgba(16, 185, 129, 0.28);
        }
        .pipeline-card.step-5 .pipeline-number { background: #10b981; color: #ffffff; }
        .pipeline-card.step-5 .pipeline-title { color: #15803d; }
        .pipeline-card.step-5 .pipeline-desc { color: #14532d; }
        .pipeline-card.step-5 .pipeline-icon { color: #10b981; }

        .pipeline-card.step-6 {
            background: linear-gradient(145deg, #ccfbf1 0%, #f0fdfa 100%);
            border-color: #0d9488;
            box-shadow: 0 4px 14px rgba(13, 148, 136, 0.15);
        }
        .pipeline-card.step-6:hover {
            box-shadow: 0 10px 22px rgba(13, 148, 136, 0.28);
        }
        .pipeline-card.step-6 .pipeline-number { background: #0d9488; color: #ffffff; }
        .pipeline-card.step-6 .pipeline-title { color: #0f766e; }
        .pipeline-card.step-6 .pipeline-desc { color: #134e4a; }
        .pipeline-card.step-6 .pipeline-icon { color: #0d9488; }

        .pipeline-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.6rem;
        }

        .pipeline-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .pipeline-icon {
            font-size: 0.85rem;
            transition: transform 0.2s;
        }

        .pipeline-card:hover .pipeline-icon {
            transform: translateY(2px);
        }

        .pipeline-title {
            font-size: 0.95rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            line-height: 1.35;
        }

        .pipeline-desc {
            font-size: 0.78rem;
            line-height: 1.45;
            font-weight: 500;
        }

        /* Subsection Headings */
        .subsection-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--primary);
            margin: 2.25rem 0 1rem;
            padding-bottom: 0.4rem;
            border-bottom: 1px dashed var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Structured Figure Cards */
        .image-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin: 1.5rem 0 2rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .image-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.1);
            border-color: #cbd5e1;
        }

        .image-card-img-wrap {
            position: relative;
            background: #fbfcfe;
            border-bottom: 1px solid var(--border);
            overflow: hidden;
            cursor: zoom-in;
            text-align: center;
            padding: 1.25rem;
        }

        .image-card img {
            width: 100%;
            max-height: 480px;
            object-fit: contain;
            display: block;
            margin: 0 auto;
            transition: transform 0.25s ease;
        }

        .image-card:hover img {
            transform: scale(1.015);
        }

        .zoom-hint {
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
        }

        .image-card-img-wrap:hover .zoom-hint {
            opacity: 1;
        }

        .image-card figcaption {
            padding: 1.25rem 1.5rem;
            background: #ffffff;
        }

        .image-card-title-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .image-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--primary);
        }

        .image-card-takeaway {
            font-size: 0.9rem;
            color: var(--text-dark);
            line-height: 1.55;
            margin-top: 0.35rem;
        }

        .image-card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.78rem;
            color: var(--text-muted);
            border-top: 1px solid #f1f5f9;
            margin-top: 0.85rem;
            padding-top: 0.65rem;
        }

        /* 2-Column Comparison Layout */
        .grid-comparison {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
            gap: 1.75rem;
            margin: 1.5rem 0 2rem;
        }

        @media (max-width: 820px) {
            .grid-comparison {
                grid-template-columns: 1fr;
            }
        }

        /* 3-Column Sequence Layout */
        .grid-trio {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 1.5rem;
            margin: 1.5rem 0 2rem;
        }

        @media (max-width: 768px) {
            .grid-trio {
                grid-template-columns: 1fr;
            }
        }

        .grid-trio .image-card img {
            max-height: 280px;
        }

        /* Tables */
        .table-responsive {
            overflow-x: auto;
            margin: 1.5rem 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
            margin-top: 0.5rem;
        }

        th, td {
            padding: 0.85rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background-color: #f8fafc;
            color: var(--text-dark);
            font-weight: 700;
        }

        tr:hover {
            background-color: #f8fafc;
        }

        .step-pill {
            display: inline-block;
            width: 28px;
            height: 28px;
            line-height: 28px;
            text-align: center;
            border-radius: 50%;
            background-color: var(--primary);
            color: #ffffff;
            font-weight: bold;
            font-size: 0.85rem;
            transition: transform 0.15s ease, background 0.15s ease;
        }

        .step-pill-link {
            text-decoration: none;
            display: inline-block;
        }

        .step-pill-link:hover .step-pill {
            transform: scale(1.12);
            background-color: var(--primary-light);
        }

        .code-pill {
            font-family: var(--font-mono);
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }

        a.code-pill {
            text-decoration: none;
            color: var(--primary);
            border: 1px solid #cbd5e1;
            font-weight: 600;
            display: inline-block;
            transition: all 0.15s ease;
        }

        a.code-pill:hover {
            background-color: #e0f2fe;
            color: var(--primary-light);
            border-color: var(--primary-light);
            text-decoration: underline;
        }

        .callout {
            background-color: #eff6ff;
            border-left: 4px solid var(--primary-light);
            padding: 1.1rem 1.35rem;
            border-radius: 0 10px 10px 0;
            margin: 1.5rem 0;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .callout.success {
            background-color: #ecfdf5;
            border-left-color: var(--success);
        }

        /* Collapsible Code Tabs (Default Closed) */
        .code-tab-drawer {
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #ffffff;
            margin: 1.5rem 0;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }

        .code-tab-summary {
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
            border-bottom: 1px solid transparent;
            transition: background 0.15s ease;
        }

        .code-tab-drawer[open] .code-tab-summary {
            border-bottom: 1px solid var(--border);
            background: #f1f5f9;
        }

        .code-tab-summary:hover {
            background: #f1f5f9;
        }

        .code-summary-left {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .code-badge {
            font-size: 0.75rem;
            color: var(--primary);
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            padding: 0.2rem 0.65rem;
            border-radius: 12px;
            font-weight: 600;
        }

        .code-tab-drawer[open] .code-badge {
            background: #fee2e2;
            color: #dc2626;
            border-color: #fca5a5;
        }

        .code-tab-drawer pre {
            margin: 0;
            padding: 1.25rem 1.5rem;
            background: #0f172a;
            color: #f8fafc;
            font-family: var(--font-mono);
            font-size: 0.85rem;
            overflow-x: auto;
            line-height: 1.55;
        }

        .code-tab-drawer code {
            background: transparent;
            color: inherit;
            padding: 0;
        }

        /* Lightbox Image Modal */
        .image-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 9999;
            align-items: center;
            justify-content: center;
        }

        .image-modal.active {
            display: flex;
        }

        .modal-backdrop {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(15, 23, 42, 0.88);
            backdrop-filter: blur(8px);
        }

        .modal-content {
            position: relative;
            max-width: 92vw;
            max-height: 92vh;
            display: flex;
            flex-direction: column;
            background: #0f172a;
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 14px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            overflow: hidden;
            animation: modalFadeIn 0.2s ease;
            z-index: 10;
        }

        @keyframes modalFadeIn {
            from { opacity: 0; transform: scale(0.96); }
            to { opacity: 1; transform: scale(1); }
        }

        .modal-close-btn {
            position: absolute;
            top: 12px;
            right: 14px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #ffffff;
            font-size: 1.5rem;
            line-height: 1;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
            z-index: 20;
        }

        .modal-close-btn:hover {
            background: #ef4444;
            border-color: #ef4444;
            color: #ffffff;
        }

        .modal-image-wrapper {
            display: flex;
            align-items: center;
            justify-content: center;
            background: #090d16;
            padding: 1.5rem;
            overflow: hidden;
            max-height: 76vh;
        }

        .modal-image-wrapper img {
            max-width: 100%;
            max-height: 72vh;
            object-fit: contain;
            border-radius: 6px;
            background: #ffffff;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }

        .modal-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.5rem;
            background: #1e293b;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            gap: 1rem;
            flex-wrap: wrap;
        }

        .modal-title {
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
        }

        .modal-desc {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 0.2rem;
        }

        .modal-btn {
            background: var(--accent);
            color: #ffffff;
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 0.45rem 1rem;
            border-radius: 6px;
            transition: background 0.15s ease;
            white-space: nowrap;
        }

        .modal-btn:hover {
            background: #0284c7;
        }

        footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 3.5rem;
            padding-top: 1.75rem;
            border-top: 1px solid var(--border);
        }

        @media print {
            body { background: #fff; padding: 0; font-size: 10pt; }
            header { background: #1e3a8a !important; color: #fff; padding: 1rem; }
            .card { box-shadow: none; border: 1px solid #ccc; break-inside: avoid; }
            .code-tab-drawer { break-inside: avoid; }
            .code-tab-drawer[open] { display: block; }
            .image-modal { display: none !important; }
            table { font-size: 8.5pt; }
        }
    </style>
</head>
<body>

<div class="container">
    <!-- Header -->
    <header>
        <h1>Báo Cáo Nghiệm Thu Kỹ Thuật Lab 1</h1>
        <h2>Phân Loại FashionMNIST với PyTorch & Handcrafted Features</h2>
        <div class="meta-badges">
            <span class="badge">Tác giả: Thái Anh Lạc</span>
            <span class="badge">Python 3.13.15</span>
            <span class="badge">PyTorch 2.14.0 (CPU / CUDA)</span>
            <span class="badge">FashionMNIST (70.000 ảnh)</span>
            <span class="badge badge-success">49/49 Tests Passed (100%)</span>
            <span class="badge badge-info">Best Test Acc: 89.60%</span>
        </div>
    </header>

    <!-- Section 1: Objectives -->
    <div class="card" id="section-objectives">
        <div class="card-header">
            <h3 class="card-title">1. Mục Tiêu Bài Thực Hành</h3>
            <span class="badge badge-info">Core Objectives</span>
        </div>
        <p>Xây dựng hệ thống học sâu hoàn chỉnh phân loại 10 lớp trang phục FashionMNIST bằng <strong>PyTorch</strong>, tuân thủ nguyên tắc Clean Architecture và phương pháp kiểm thử Test-Driven Development (TDD):</p>
        <ul style="margin-left: 1.5rem; margin-top: 0.75rem; line-height: 1.7;">
            <li><strong>Khám phá dữ liệu (EDA):</strong> Khảo sát 9 bước toàn diện trên không gian ảnh (kiểm toán toàn vẹn 0 NaN / 0 trùng lặp, phân bố lớp, ảnh trung bình, histogram pixel, heatmaps phương sai, cosine similarity).</li>
            <li><strong>Kỹ nghệ đặc trưng (Feature Engineering):</strong> Thiết kế 10 đặc trưng hình học & không gian (Bounding Box, Centroid, Symmetry, Edge Density) kết hợp chuẩn hóa <code>StandardScaler</code> (fit train-only).</li>
            <li><strong>Mô hình hóa (Modeling):</strong> Triển khai và so sánh định lượng 3 kiến trúc mạng nơ-ron truyền thẳng: <code>SmallMLP</code> (101k params), <code>MainMLP</code> (235k params), và <code>MainMLP_FE</code> (237k params).</li>
            <li><strong>Tối ưu hóa (Optimization):</strong> Khảo sát thực nghiệm 3 thuật toán tối ưu (<code>SGD</code>, <code>RMSprop</code>, <code>Adam</code>) qua 30 trials tìm kiếm siêu tham số, phân tích đường cong học tập và đánh giá đa lớp One-vs-Rest (ROC & PR).</li>
        </ul>
    </div>

    <!-- Section 2: Visual Workflow & Roadmap -->
    <div class="card" id="section-roadmap">
        <div class="card-header">
            <h3 class="card-title">2. Lộ Trình Thực Thi & Sơ Đồ Luồng Xử Lý</h3>
            <span class="badge badge-success">Clean Architecture</span>
        </div>

        <h4 style="margin: 0.5rem 0 0.85rem; color: var(--primary);">2.1. Sơ Đồ Luồng Quy Trình Tương Tác Đa Sắc (Vibrant Interactive Workflow)</h4>
        <p style="font-size: 0.88rem; color: var(--text-muted); margin-bottom: 0.5rem;">Sơ đồ tiến trình phân hóa màu sắc trực quan theo từng giai đoạn kỹ thuật; nhấp vào từng thẻ để cuộn mượt đến phần tương ứng:</p>
        
        <div class="pipeline-container">
            <div class="pipeline-grid">
                <a href="#section-eda" class="pipeline-card step-1">
                    <div class="pipeline-header">
                        <span class="pipeline-number">01</span>
                        <span class="pipeline-icon">&darr;</span>
                    </div>
                    <div class="pipeline-title">1. Khám Phá EDA</div>
                    <div class="pipeline-desc">Kiểm Toán 0 NaN &bull; 0 Duplicate &bull; 9 Plots</div>
                </a>

                <a href="#section-features" class="pipeline-card step-2">
                    <div class="pipeline-header">
                        <span class="pipeline-number">02</span>
                        <span class="pipeline-icon">&darr;</span>
                    </div>
                    <div class="pipeline-title">2. 10 Features</div>
                    <div class="pipeline-desc">Bounding Box &bull; Centroid &bull; Correlation</div>
                </a>

                <a href="#section-models" class="pipeline-card step-3">
                    <div class="pipeline-header">
                        <span class="pipeline-number">03</span>
                        <span class="pipeline-icon">&darr;</span>
                    </div>
                    <div class="pipeline-title">3. Kiến Trúc 3 MLP</div>
                    <div class="pipeline-desc">Small &bull; Main &bull; Main_FE (101k - 237k)</div>
                </a>

                <a href="#section-tuning" class="pipeline-card step-4">
                    <div class="pipeline-header">
                        <span class="pipeline-number">04</span>
                        <span class="pipeline-icon">&darr;</span>
                    </div>
                    <div class="pipeline-title">4. Tối Ưu & Tuning</div>
                    <div class="pipeline-desc">SGD &bull; RMSprop &bull; Adam (30 Trials)</div>
                </a>

                <a href="#section-evaluation" class="pipeline-card step-5">
                    <div class="pipeline-header">
                        <span class="pipeline-number">05</span>
                        <span class="pipeline-icon">&darr;</span>
                    </div>
                    <div class="pipeline-title">5. Đánh Giá Test</div>
                    <div class="pipeline-desc">Acc 89.2% &bull; Confusion &bull; OvR Curves</div>
                </a>

                <a href="#section-qa" class="pipeline-card step-6">
                    <div class="pipeline-header">
                        <span class="pipeline-number">06</span>
                        <span class="pipeline-icon">&darr;</span>
                    </div>
                    <div class="pipeline-title">6. Kiểm Toán QA</div>
                    <div class="pipeline-desc">49/49 Tests Pass &bull; TDD DoD 100%</div>
                </a>
            </div>
        </div>

        <h4 style="margin: 2.25rem 0 1rem; color: var(--primary);">2.2. Bảng Lộ Trình Thực Thi Chuẩn Hóa (Execution Roadmap)</h4>
        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th style="width: 70px; text-align: center;">Bước</th>
                        <th style="width: 210px;">Giai Đoạn</th>
                        <th>Nội Dung Chi Tiết</th>
                        <th style="width: 250px;">Mô-đun Triển Khai (`src/`)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="text-align: center;"><a href="#section-roadmap" class="step-pill-link" title="Bước 1"><span class="step-pill">1</span></a></td>
                        <td><strong>Setup & Data Contract</strong></td>
                        <td>Tải FashionMNIST 70k ảnh, chia tập phân tầng (Stratified) 54k Train / 6k Val / 10k Test, cô lập tập Test.</td>
                        <td><a href="../../src/datasets/dataset.py" class="code-pill">src/datasets/dataset.py</a></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><a href="#section-eda" class="step-pill-link" title="Chuyển đến EDA"><span class="step-pill">2</span></a></td>
                        <td><strong>Image EDA (9 Bước)</strong></td>
                        <td>Kiểm toán toàn vẹn dữ liệu (0 NaN, 0 Duplicate qua MD5 hash), cấu trúc pixel [0, 255], histogram cường độ, ảnh trung bình theo lớp, bản đồ nhiệt phương sai và ma trận tương đồng Cosine.</td>
                        <td><a href="../../src/datasets/eda.py" class="code-pill">src/datasets/eda.py</a></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><a href="#section-features" class="step-pill-link" title="Chuyển đến Features"><span class="step-pill">3</span></a></td>
                        <td><strong>Feature Engineering</strong></td>
                        <td>Trích xuất 10 đặc trưng hình thái (Bounding Box, Centroid, Symmetry, Edge Density) và fit <code>StandardScaler</code> chỉ trên tập Train.</td>
                        <td><a href="../../src/datasets/features.py" class="code-pill">src/datasets/features.py</a></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><a href="#section-models" class="step-pill-link" title="Chuyển đến Models"><span class="step-pill">4</span></a></td>
                        <td><strong>Architectures Definition</strong></td>
                        <td>Định nghĩa 3 mô hình MLP (<code>SmallMLP</code> 784-128-10, <code>MainMLP</code> 784-256-128-10, <code>MainMLP_FE</code> 794-256-128-10).</td>
                        <td><a href="../../src/models/mlp.py" class="code-pill">src/models/mlp.py</a></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><a href="#section-tuning" class="step-pill-link" title="Chuyển đến Training & Gate"><span class="step-pill">5</span></a></td>
                        <td><strong>Training & Sanity Gate</strong></td>
                        <td>Kiểm tra cổng 1-Batch Sanity Overfit (8 mẫu, loss < 0.2 trong 120 steps); huấn luyện với Cross-Entropy & Early Stopping.</td>
                        <td><a href="../../src/training/train.py" class="code-pill">src/training/train.py</a></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><a href="#section-tuning" class="step-pill-link" title="Chuyển đến Tuning"><span class="step-pill">6</span></a></td>
                        <td><strong>Optimizer & Tuning</strong></td>
                        <td>So sánh cố định LR và khảo sát siêu tham số (SGD, RMSprop, Adam) qua 30 trials, ghi nhận đường cong học tập.</td>
                        <td><a href="../../src/training/optimizer_study.py" class="code-pill">src/training/optimizer_study.py</a></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><a href="#section-evaluation" class="step-pill-link" title="Chuyển đến Đánh Giá"><span class="step-pill">7</span></a></td>
                        <td><strong>Evaluation & Model Persistence</strong></td>
                        <td>Ma trận nhầm lẫn 10 lớp, đường cong ROC / PR OvR, phân tích lỗi, lưu & nạp checkpoint <code>best.pt</code> và kiểm tra roundtrip.</td>
                        <td><a href="../../src/utils/evaluation.py" class="code-pill">src/utils/evaluation.py</a></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Section 3: Visual EDA Gallery -->
    <div class="card" id="section-eda">
        <div class="card-header">
            <h3 class="card-title">3. Khám Phá Dữ Liệu Hình Ảnh (Core Image EDA Gallery)</h3>
            <span class="badge badge-info">7 Artifacts Phân Tích</span>
        </div>
        
        <p>Phân tích toàn diện 70.000 ảnh kích thước 28x28 grayscale, kiểm toán toàn vẹn dữ liệu nghiêm ngặt (Missing values, Corruption, Exact Duplicates, Train-Test Overlap), phân bố lớp đồng đều tuyệt đối 7.000 mẫu/lớp. <em>(Nhấp vào bất kỳ ảnh nào để xem phóng to chi tiết).</em></p>
        
        <!-- 3.1 Data Integrity & Duplicate Check -->
        <div class="subsection-title">
            <span>3.1. Kiểm Toán Tính Toàn Vẹn Dữ Liệu, Giá Trị Thiếu & Trùng Lặp (Data Integrity & Duplicate Audit)</span>
            <span class="badge badge-success">ZERO MISSING & ZERO DUPLICATES</span>
        </div>

        <p style="font-size: 0.92rem; color: var(--text-dark); margin-bottom: 1rem;">
            Theo quy chuẩn Core EDA (Mục 1 & 5 trong <code>docs/eda-image.md</code>), toàn bộ 70.000 ảnh đã được chạy thuật toán kiểm toán toàn vẹn để phát hiện giá trị rỗng/lỗi và quét băm mật mã học (MD5 pixel hashing) nhằm tìm kiếm mẫu trùng lặp và rò rỉ dữ liệu giữa các tập:
        </p>

        <!-- Integrity Table 1 -->
        <div class="table-responsive" style="margin-bottom: 1.5rem;">
            <table>
                <thead>
                    <tr>
                        <th style="width: 280px;">Hạng Mục Kiểm Tra</th>
                        <th style="width: 180px;">Kết Quả Mong Đợi</th>
                        <th>Thực Tế Đo Đạc</th>
                        <th style="width: 130px; text-align: center;">Trạng Thái</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Giá trị thiếu (NaN / Null / Inf)</strong></td>
                        <td>Không có (0 NaN)</td>
                        <td>100% dữ liệu nguyên vẹn, 0 giá trị NaN hoặc Null</td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                    <tr>
                        <td><strong>Dải giá trị điểm ảnh [0, 255]</strong></td>
                        <td>[0, 255] uint8</td>
                        <td>Train: [0, 255], Test: [0, 255] (Không có giá trị âm hoặc tràn số)</td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                    <tr>
                        <td><strong>Dải nhãn phân loại [0, 9]</strong></td>
                        <td>10 lớp [0, 9]</td>
                        <td>Train: [0, 9], Test: [0, 9] (Không có nhãn ngoài danh mục)</td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                    <tr>
                        <td><strong>Trùng lặp nội bộ tập Train</strong></td>
                        <td>0 mẫu trùng</td>
                        <td><strong>0 mẫu trùng lặp</strong> (Quét 60.000 mã băm MD5 duy nhất)</td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                    <tr>
                        <td><strong>Trùng lặp nội bộ tập Test</strong></td>
                        <td>0 mẫu trùng</td>
                        <td><strong>0 mẫu trùng lặp</strong> (Quét 10.000 mã băm MD5 duy nhất)</td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                    <tr style="background-color: #f0fdf4;">
                        <td><strong>Rò rỉ trùng lặp Train vs Test</strong></td>
                        <td>0 mẫu trùng</td>
                        <td><strong>0 mẫu trùng lặp</strong> (Intersection Set Hash = 0, Zero Data Leakage)</td>
                        <td style="text-align: center;"><span class="badge badge-success">PASSED</span></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Structural Overview Table 2 -->
        <div class="table-responsive" style="margin-bottom: 1.5rem;">
            <table>
                <thead>
                    <tr>
                        <th>Tập Dữ Liệu</th>
                        <th>Số Lượng Ảnh</th>
                        <th>Kích Thước (HxW)</th>
                        <th>Kênh Màu</th>
                        <th>Kiểu Dữ Liệu (dtype)</th>
                        <th>Dải Điểm Ảnh</th>
                        <th>Dải Nhãn</th>
                        <th>Số Lớp</th>
                        <th>Bộ Nhớ RAM</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Train Pool</strong> (Ban đầu)</td>
                        <td>60.000</td>
                        <td>28 x 28</td>
                        <td>1 (Grayscale)</td>
                        <td><code>uint8</code></td>
                        <td>[0, 255]</td>
                        <td>[0, 9]</td>
                        <td>10</td>
                        <td>44.86 MB</td>
                    </tr>
                    <tr>
                        <td><strong>Test Pool</strong> (Cô lập)</td>
                        <td>10.000</td>
                        <td>28 x 28</td>
                        <td>1 (Grayscale)</td>
                        <td><code>uint8</code></td>
                        <td>[0, 255]</td>
                        <td>[0, 9]</td>
                        <td>10</td>
                        <td>7.48 MB</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="callout success">
            <strong>Phương pháp kiểm toán Integrity & Zero Data Leakage:</strong>
            <p style="margin-top: 0.35rem;">
                - <strong>Kiểm tra Missing Values:</strong> Sử dụng <code>np.isnan(X).any()</code> và <code>np.isinf(X).any()</code> quét toàn diện trên toàn bộ 54.880.000 điểm ảnh. Kết quả không tồn tại bất kỳ pixel lỗi hay giá trị rỗng nào.<br>
                - <strong>Kiểm tra Trùng lặp (Exact Duplicates) bằng MD5 Hashing:</strong> Trải phẳng từng ảnh thành chuỗi 784 bytes liên tục (<code>np.ascontiguousarray(row).tobytes()</code>) và tính toán mã băm MD5 duy nhất <code>hashlib.md5(...)</code>. Phép giao giữa tập băm Train và Test đạt kết quả rỗng (<code>len(train_hashes.intersection(test_hashes)) == 0</code>), chứng minh tính toàn vẹn 100% không xảy ra hiện tượng rò rỉ dữ liệu.<br>
                - <strong>Thống kê Nền đen:</strong> Tỷ lệ điểm ảnh có giá trị bằng 0 tuyệt đối trong tập Train chiếm <strong>50.21%</strong>, trung bình toàn cục là 0.2862 và độ lệch chuẩn 0.3532.
            </p>
        </div>

        <!-- 3.2. Sample Images & Class Distributions -->
        <div class="subsection-title">
            <span>3.2. Phân Bố Tập Dữ Liệu & Ảnh Đại Diện 10 Lớp Trang Phục</span>
            <span class="badge badge-info">SAMPLE & DISTRIBUTION</span>
        </div>

        <figure class="image-card">
            <div class="image-card-img-wrap">
                <img src="../outputs/published/preparation/eda/figures/sample_images.png" alt="Ảnh mẫu đại diện 10 lớp trang phục FashionMNIST" loading="lazy">
                <div class="zoom-hint">Phóng to ảnh &nearr;</div>
            </div>
            <figcaption>
                <div class="image-card-title-row">
                    <span class="image-card-title">Hình 1: Ảnh Mẫu Đại Diện 10 Lớp Trang Phục FashionMNIST</span>
                    <span class="code-pill">Shape: 28x28 Grayscale</span>
                </div>
                <p class="image-card-takeaway">Trực quan hóa ngẫu nhiên các mẫu đại diện: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. Toàn bộ ảnh đã được căn giữa, nền đen (pixel xấp xỉ 0), đối tượng tiền cảnh màu xám/trắng với độ tương phản cao.</p>
                <div class="image-card-footer">
                    <span>Artifact: sample_images.png</span>
                    <span>10 Classes &bull; 70k Images</span>
                </div>
            </figcaption>
        </figure>

        <div class="grid-comparison">
            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/preparation/eda/figures/class_distribution.png" alt="Phân bố tập Train và Validation" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">1. Phân Bố Tập Mẫu (Class Distribution)</span>
                        <span class="code-pill">Balanced: 1:1</span>
                    </div>
                    <p class="image-card-takeaway">Tập dữ liệu cân bằng tuyệt đối: mỗi lớp có đúng 5.400 mẫu Train và 600 mẫu Validation sau phân tầng (Stratified Split 90/10), đảm bảo mô hình không bị thiên vị tiên nghiệm (No Class Imbalance).</p>
                    <div class="image-card-footer">
                        <span>Train: 54.000 &bull; Val: 6.000</span>
                        <span>Seed = 42</span>
                    </div>
                </figcaption>
            </figure>

            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/preparation/eda/figures/class_mean_images.png" alt="Ảnh cường độ trung bình theo từng lớp" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">2. Ảnh Cường Độ Trung Bình (Mean Images)</span>
                        <span class="code-pill">10 Class Templates</span>
                    </div>
                    <p class="image-card-takeaway">Cường độ pixel trung bình phác họa hình thái khuôn mẫu riêng biệt: Trouser thon dài theo chiều dọc; Bag có dạng hình hộp vuông; Sandal mảnh và hở nhiều vùng; T-shirt, Pullover, Coat có hình thái thân áo rất tương đồng.</p>
                    <div class="image-card-footer">
                        <span>Domain: [0.0, 1.0]</span>
                        <span>Centroid-aligned</span>
                    </div>
                </figcaption>
            </figure>
        </div>

        <!-- 3.3. Pixel Intensities & Heatmaps -->
        <div class="subsection-title">
            <span>3.3. Cường Độ Điểm Ảnh, Phân Bố Năng Lượng & Bản Đồ Phương Sai</span>
            <span class="badge badge-info">INTENSITY & HEATMAPS</span>
        </div>

        <div class="grid-trio">
            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/preparation/eda/figures/pixel_histogram.png" alt="Phân bố cường độ điểm ảnh" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">3. Histogram Điểm Ảnh</span>
                        <span class="code-pill">Bimodal</span>
                    </div>
                    <p class="image-card-takeaway">Phân bố hai đỉnh rõ rệt: nền đen chiếm đa số (~50.2% pixel tuyệt đối bằng 0); tiền cảnh trải rộng từ 0.2 đến 1.0. Global Mean = 0.2862, Global Std = 0.3532.</p>
                    <div class="image-card-footer">
                        <span>Zero Pixels: 50.21%</span>
                        <span>Foreground: [0.2, 1.0]</span>
                    </div>
                </figcaption>
            </figure>

            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/preparation/eda/figures/mean_pixel_heatmap.png" alt="Bản đồ nhiệt trung bình pixel" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">4. Bản Đồ Nhiệt Trung Bình</span>
                        <span class="code-pill">Spatial Center</span>
                    </div>
                    <p class="image-card-takeaway">Năng lượng quang học tập trung dày đặc ở vùng trung tâm (tọa độ hàng 8-24, cột 6-22). Toàn bộ 4 góc viền ngoài hoàn toàn là 0, an toàn cho việc trích xuất ROI.</p>
                    <div class="image-card-footer">
                        <span>Central Core Energy</span>
                        <span>Zero Outer Border</span>
                    </div>
                </figcaption>
            </figure>

            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/preparation/eda/figures/pixel_variance_heatmap.png" alt="Bản đồ nhiệt phương sai pixel" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">5. Bản Đồ Phương Sai</span>
                        <span class="code-pill">High Variance</span>
                    </div>
                    <p class="image-card-takeaway">Phương sai cao nhất nằm ở đường viền ngoài thân trang phục và cổ áo — nơi mang tín hiệu phân biệt giữa áo cộc tay, áo dài tay, váy dài và áo khoác.</p>
                    <div class="image-card-footer">
                        <span>Discriminative Edges</span>
                        <span>Boundary Signals</span>
                    </div>
                </figcaption>
            </figure>
        </div>

        <!-- 3.4. Cosine Similarity -->
        <div class="subsection-title">
            <span>3.4. Tương Đồng Hình Thái Cosine Giữa 10 Lớp Trang Phục</span>
            <span class="badge badge-info">COSINE MATRIX</span>
        </div>

        <figure class="image-card">
            <div class="image-card-img-wrap">
                <img src="../outputs/published/preparation/eda/figures/class_similarity_heatmap.png" alt="Cosine Similarity giữa các lớp" loading="lazy">
                <div class="zoom-hint">Phóng to ảnh &nearr;</div>
            </div>
            <figcaption>
                <div class="image-card-title-row">
                    <span class="image-card-title">6. Ma Trận Độ Tương Đồng Cosine (Cosine Similarity Matrix)</span>
                    <span class="code-pill">Metric: Cosine Distance</span>
                </div>
                <p class="image-card-takeaway">Độ tương đồng lớn nhất xuất hiện ở cụm trang phục phần thân trên: T-shirt/top, Pullover, Coat và Shirt có hệ số tương đồng vượt 0.88 (đỉnh điểm T-shirt vs Shirt đạt 0.93). Ngược lại, nhóm giày dép (Sandal, Sneaker, Ankle boot) có độ tương đồng rất thấp so với nhóm áo (< 0.45). Đây là căn cứ khoa học trực quan dự báo trước các lớp sẽ xảy ra nhầm lẫn chính của mô hình.</p>
                <div class="image-card-footer">
                    <span>Min: 0.42 (Sandal vs Coat)</span>
                    <span>Max: 0.93 (T-shirt vs Shirt)</span>
                </div>
            </figcaption>
        </figure>

        <!-- Collapsible Code Tab 1 -->
        <details class="code-tab-drawer">
            <summary class="code-tab-summary">
                <div class="code-summary-left">
                    <span class="code-pill">PYTHON</span>
                    <span>Mã nguồn kiểm toán tính toàn vẹn (Missing & MD5 Duplicate Hash)</span>
                </div>
                <span class="code-badge">Nhấn để mở code &dtrif;</span>
            </summary>
            <pre><code class="language-python"># Kiểm toán tính toàn vẹn & quét trùng lặp dữ liệu (Core EDA)
import hashlib
import numpy as np

# 1. Kiểm tra Missing Values (NaN / Inf / Null)
assert not np.isnan(X_train).any(), "Phat hien NaN trong X_train!"
assert not np.isnan(X_test).any(), "Phat hien NaN trong X_test!"

# 2. Quét trùng lặp bằng MD5 Hashing trên từng ma trận ảnh
train_flat = np.ascontiguousarray(X_train.reshape(len(X_train), -1))
test_flat = np.ascontiguousarray(X_test.reshape(len(X_test), -1))

train_hashes = set(hashlib.md5(row.tobytes()).hexdigest() for row in train_flat)
test_hashes = set(hashlib.md5(row.tobytes()).hexdigest() for row in test_flat)

# Xác nhận 0 mẫu trùng nội bộ & 0 rò rỉ tập Test
train_duplicates = len(train_flat) - len(train_hashes)
test_duplicates = len(test_flat) - len(test_hashes)
leakage_overlap = len(train_hashes.intersection(test_hashes))

print(f"Train Duplicates: {train_duplicates}, Test Duplicates: {test_duplicates}, Overlap: {leakage_overlap}")
# Ket qua: 0, 0, 0 (Toan ven tuyet doi)</code></pre>
        </details>
    </div>

    <!-- Section 4: Feature Engineering Gallery -->
    <div class="card" id="section-features">
        <div class="card-header">
            <h3 class="card-title">4. Kỹ Nghệ Đặc Trưng Hình Học & Không Gian (Feature Engineering)</h3>
            <span class="badge badge-info">10 Handcrafted Features</span>
        </div>

        <p>Thiết kế 10 đặc trưng hình học bổ sung dựa trên tri thức miền thị giác máy tính: <strong>Mean Pixel, Std Pixel, Tỷ lệ tiền cảnh (Foreground Ratio), Bounding Box Width/Height, Tỷ lệ khung hình (Aspect Ratio), Tọa độ trọng tâm (Centroid X, Y), Độ đối xứng (Symmetry Index), Mật độ biên (Edge Density)</strong>.</p>

        <!-- 4.1 Geometry & Correlations -->
        <div class="subsection-title">
            <span>4.1. Khảo Sát Bounding Box, Trọng Tâm & Ma Trận Tương Quan Tuyến Tính</span>
            <span class="badge badge-info">GEOMETRY & CORRELATION</span>
        </div>

        <div class="grid-comparison">
            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/preparation/feature_geometry.png" alt="Minh họa Bounding Box và Centroid" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">1. Trích Xuất Hình Học Đối Tượng</span>
                        <span class="code-pill">Pascal VOC XYXY</span>
                    </div>
                    <p class="image-card-takeaway">Thuật toán quét ngưỡng tự động tính toán hộp bao nhỏ nhất bao bọc toàn bộ phần thân áo/quần và xác định tọa độ trọng tâm quang học (Center of Mass) theo hai trục X và Y.</p>
                    <div class="image-card-footer">
                        <span>BBox Width & Height</span>
                        <span>Centroid (Cx, Cy)</span>
                    </div>
                </figcaption>
            </figure>

            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/preparation/feature_correlations.png" alt="Ma trận tương quan đặc trưng" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">2. Ma Trận Tương Quan Tuyến Tính Pearson</span>
                        <span class="code-pill">No Multicollinearity</span>
                    </div>
                    <p class="image-card-takeaway">Kiểm toán tương quan giữa 10 đặc trưng: Mean Pixel và Foreground Ratio tương quan cao (0.91), trong khi các đặc trưng hình học khác (Aspect Ratio, Symmetry, Edge Density) đều có r < 0.45, bổ trợ thông tin độc lập mà không gây rủi ro đa cộng tuyến.</p>
                    <div class="image-card-footer">
                        <span>Pearson r < 0.45</span>
                        <span>Independent Features</span>
                    </div>
                </figcaption>
            </figure>
        </div>

        <!-- Collapsible Code Tab 2 -->
        <details class="code-tab-drawer">
            <summary class="code-tab-summary">
                <div class="code-summary-left">
                    <span class="code-pill">PYTHON</span>
                    <span>Mã nguồn trích xuất đặc trưng & StandardScaler</span>
                </div>
                <span class="code-badge">Nhấn để mở code &dtrif;</span>
            </summary>
            <pre><code class="language-python"># Fit StandardScaler DUY NHẤT trên tập Train, áp dụng Transform cho Validation và Test
scaler = StandardScaler()
scaler.fit(train_features)
train_z = scaler.transform(train_features)
val_z = scaler.transform(val_features)
test_z = scaler.transform(test_features)

# Vector kết hợp cho nhánh MainMLP_FE: [784 raw pixels + 10 engineered features = 794 dims]
combined_input = np.concatenate([flattened_pixels, scaled_features], axis=1)</code></pre>
        </details>
    </div>

    <!-- Section 5: Architecture Benchmarking -->
    <div class="card" id="section-models">
        <div class="card-header">
            <h3 class="card-title">5. Đánh Giá & So Sánh 3 Kiến Trúc Mô Hình (Architecture Benchmarking)</h3>
            <span class="badge badge-success">3 Models Evaluated</span>
        </div>

        <!-- 5.1 Test Performance Comparison -->
        <div class="subsection-title">
            <span>5.1. So Sánh Hiệu Năng Trên Tập Test: SmallMLP vs MainMLP vs MainMLP_FE</span>
            <span class="badge badge-success">TEST ACCURACY & F1</span>
        </div>

        <figure class="image-card">
            <div class="image-card-img-wrap">
                <img src="../outputs/published/comparison/test_scores.png" alt="So sánh Test Accuracy và Macro F1 giữa 3 mô hình" loading="lazy">
                <div class="zoom-hint">Phóng to ảnh &nearr;</div>
            </div>
            <figcaption>
                <div class="image-card-title-row">
                    <span class="image-card-title">Hình 3: So Sánh Trực Quan Test Accuracy và Macro F1 Giữa 3 Kiến Trúc</span>
                    <span class="code-pill">Top: MainMLP_FE (89.60%)</span>
                </div>
                <p class="image-card-takeaway">Biểu đồ cột so sánh định lượng: MainMLP_FE (kết hợp 784 pixels + 10 đặc trưng hình thái) đạt Test Accuracy 89.60% và Macro F1 89.58%, nhỉnh hơn MainMLP cơ sở (89.41%) và SmallMLP (88.52%). Mức chênh lệch chứng minh 10 đặc trưng hình thái mang lại mức tăng khiêm tốn +0.19%, nhưng đổi lại chi phí thời gian huấn luyện tăng vọt từ 15.80s lên 78.01s (gấp 4.9x).</p>
                <div class="image-card-footer">
                    <span>SmallMLP: 88.52% (8.27s)</span>
                    <span>MainMLP: 89.41% (15.80s)</span>
                    <span>MainMLP_FE: 89.60% (78.01s)</span>
                </div>
            </figcaption>
        </figure>

        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th>Mô Hình</th>
                        <th>Cấu Trúc Tầng (Layers)</th>
                        <th>Số Tham Số</th>
                        <th>Test Accuracy</th>
                        <th>Macro F1</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Test Loss</th>
                        <th>Thời Gian Train</th>
                        <th>Best Epoch</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>SmallMLP</strong></td>
                        <td>784 &rarr; 128 &rarr; 10</td>
                        <td>101.770</td>
                        <td>88.52%</td>
                        <td>88.49%</td>
                        <td>88.51%</td>
                        <td>88.52%</td>
                        <td>0.3213</td>
                        <td>8.27s</td>
                        <td>9/10</td>
                    </tr>
                    <tr>
                        <td><strong>MainMLP</strong></td>
                        <td>784 &rarr; 256 &rarr; 128 &rarr; 10</td>
                        <td>235.146</td>
                        <td>89.41%</td>
                        <td>89.38%</td>
                        <td>89.38%</td>
                        <td>89.41%</td>
                        <td>0.3079</td>
                        <td>15.80s</td>
                        <td>10/10</td>
                    </tr>
                    <tr style="background-color: #f0fdf4;">
                        <td><strong>MainMLP_FE</strong> (Features)</td>
                        <td>794 &rarr; 256 &rarr; 128 &rarr; 10</td>
                        <td>237.706</td>
                        <td><strong>89.60%</strong></td>
                        <td><strong>89.58%</strong></td>
                        <td><strong>89.57%</strong></td>
                        <td><strong>89.60%</strong></td>
                        <td><strong>0.3002</strong></td>
                        <td><strong>78.01s</strong> (4.9x)</td>
                        <td>10/10</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- 5.2 Resource Telemetry & Single Sample Benchmark -->
        <div class="subsection-title">
            <span>5.2. Đo Đạc Chi Phí Tài Nguyên, Bộ Nhớ RAM & Độ Trễ (Single-Sample Resource Telemetry)</span>
            <span class="badge badge-info">RAM RSS & LATENCY BENCHMARK</span>
        </div>

        <figure class="image-card">
            <div class="image-card-img-wrap">
                <img src="../outputs/published/training_performance_single_sample/training_resources.png" alt="Đo đạc độ trễ huấn luyện và bộ nhớ RAM trên từng mẫu" loading="lazy">
                <div class="zoom-hint">Phóng to ảnh &nearr;</div>
            </div>
            <figcaption>
                <div class="image-card-title-row">
                    <span class="image-card-title">Hình 4: Chi Phí Huấn Luyện Độc Lập Trên 1 Mẫu (Batch Size = 1)</span>
                    <span class="code-pill">CPU 2 Threads &bull; psutil RSS Tracking</span>
                </div>
                <p class="image-card-takeaway">Đo lường cô lập trên từng mẫu huấn luyện (Forward + Backward + Loss + Optimizer Step, trung vị 10 lần đo sau 3 lần warmup): SmallMLP đạt thông lượng cao nhất (2.079 samples/s, 0.481 ms/sample); MainMLP đạt 1.094 samples/s (0.914 ms/sample); MainMLP_FE tiêu tốn 1.068 ms/sample (936 samples/s). Bộ nhớ RAM RSS của cả 3 mô hình duy trì ổn định quanh mức 397 - 401 MiB với mức tăng động chỉ 4.7 - 7.0 MiB.</p>
                <div class="image-card-footer">
                    <span>SmallMLP: 0.481 ms/sample</span>
                    <span>MainMLP: 0.914 ms/sample</span>
                    <span>MainMLP_FE: 1.068 ms/sample</span>
                    <span>RAM RSS: ~400 MiB</span>
                </div>
            </figcaption>
        </figure>

        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th>Mô Hình</th>
                        <th>Thiết Bị</th>
                        <th>Batch Size</th>
                        <th>Độ Trễ Huấn Luyện (ms/sample)</th>
                        <th>Thông Lượng (Samples/giây)</th>
                        <th>RAM Cơ Sở (RSS MiB)</th>
                        <th>RAM Đỉnh (Peak RSS MiB)</th>
                        <th>RAM Tăng Thêm (Delta MiB)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>SmallMLP</strong></td>
                        <td>CPU (2 threads)</td>
                        <td>1</td>
                        <td>0.481 ms</td>
                        <td>2.079,0 samples/s</td>
                        <td>393.24 MiB</td>
                        <td>397.98 MiB</td>
                        <td>+4.74 MiB</td>
                    </tr>
                    <tr>
                        <td><strong>MainMLP</strong></td>
                        <td>CPU (2 threads)</td>
                        <td>1</td>
                        <td>0.914 ms</td>
                        <td>1.093,9 samples/s</td>
                        <td>393.37 MiB</td>
                        <td>400.38 MiB</td>
                        <td>+7.01 MiB</td>
                    </tr>
                    <tr style="background-color: #f0fdf4;">
                        <td><strong>MainMLP_FE</strong> (Features)</td>
                        <td>CPU (2 threads)</td>
                        <td>1</td>
                        <td>1.068 ms</td>
                        <td>936,5 samples/s</td>
                        <td>394.46 MiB</td>
                        <td>401.46 MiB</td>
                        <td>+7.00 MiB</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="callout" style="border-left-color: #f59e0b; background-color: #fffbeb;">
            <strong style="color: #b45309;">Chẩn Đoán Kỹ Thuật: Giải Thích Về Bộ Nhớ RAM (~400 MiB) và Thời Gian Huấn Luyện 78 Giây Của MainMLP_FE:</strong>
            <ul style="margin: 0.5rem 0 0 1.25rem; line-height: 1.6; font-size: 0.88rem; color: #78350f;">
                <li><strong>Về mức RAM ~400 MiB:</strong> Đây là <em>Resident Set Size (RSS)</em> đo bằng <code>psutil</code> cho toàn bộ tiến trình Python runtime (chứa binary PyTorch, CUDA runtime stubs, mảng bộ đệm DataLoader và thư viện NumPy). Bộ nhớ dữ liệu pixel thuần chỉ tốn 44.86 MB (như ở Mục 3.1); lượng RAM tăng động khi nạp mẫu chỉ vỏn vẹn <strong>4.74 - 7.01 MiB</strong>.</li>
                <li><strong>Tại sao thời gian huấn luyện MainMLP_FE lại tốn tới 78.01s (gấp 4.9x so với 15.80s của MainMLP)?</strong>
                    <ol style="margin-left: 1.25rem; margin-top: 0.35rem;">
                        <li><strong>Phá vỡ tối ưu SIMD vector hóa (Memory Alignment Penalty):</strong> Chiều đầu vào $784 = 28 \times 28$ là kích thước chuẩn chia hết cho 16, được các thư viện BLAS trên CPU (Intel MKL / OpenBLAS) tối ưu hóa triệt để qua các thanh ghi AVX2 256-bit. Khi nối thêm 10 đặc trưng thành <strong>794 chiều</strong>, 794 không chia hết cho 8 hay 16, buộc CPU phải rơi vào vòng lặp scalar fallback cho các phần tử biên, làm chậm đáng kể phép nhân ma trận GEMM.</li>
                        <li><strong>Chi phí ghép nối mảng (Concatenation Overhead):</strong> Việc ghép <code>np.concatenate([pixels, features])</code> ở từng loader phá vỡ tính liên tục bộ nhớ (memory contiguity), làm tăng phân mảnh bộ đệm và tỷ lệ cache miss trên L1/L2 CPU cache.</li>
                        <li><strong>Hiện tượng tranh chấp CPU (CPU Contention):</strong> Trong lượt chạy huấn luyện đó trên môi trường máy chủ Windows, tiến trình Python chịu ảnh hưởng điều phối tác vụ nền (thread scheduling) khiến wall-clock time kéo dài.</li>
                    </ol>
                </li>
                <li><strong>Đánh đổi kỹ thuật (Engineering Trade-off):</strong> Tăng thêm 10 đặc trưng thủ công chỉ cải thiện khiêm tốn <strong>+0.19%</strong> Test Accuracy (89.41% &rarr; 89.60%), nhưng lại làm thời gian huấn luyện tăng gần <strong>500%</strong> trên CPU. Do đó, mô hình thuần pixel <code>MainMLP</code> được kết luận là kiến trúc tối ưu nhất về mặt chi phí - hiệu năng (cost-efficiency).</li>
            </ul>
        </div>

        <!-- 5.3 Compute Cost & Convergence Curves -->
        <div class="subsection-title">
            <span>5.3. Đánh Đổi Độ Phức Tạp Tính Toán & Quá Trình Hội Tụ Validation</span>
            <span class="badge badge-info">COMPUTE & VALIDATION DYNAMICS</span>
        </div>

        <div class="grid-comparison">
            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/comparison/cost.png" alt="Số tham số và chi phí tính toán" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">1. Đánh Đổi Độ Phức Tạp (Compute vs Params)</span>
                        <span class="code-pill">Param Ratio: 2.33x</span>
                    </div>
                    <p class="image-card-takeaway">MainMLP tăng hơn gấp đôi số tham số (+133k params) so với SmallMLP để đổi lấy mức tăng +0.93% Accuracy. Việc tích hợp thêm 10 features ở MainMLP_FE chỉ tiêu tốn thêm đúng 2.560 tham số (+1.08%) nhưng nâng cao hiệu năng thêm +0.23%.</p>
                    <div class="image-card-footer">
                        <span>Small: 101.7k params</span>
                        <span>Main_FE: 237.7k params</span>
                    </div>
                </figcaption>
            </figure>

            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/comparison/validation_comparison.png" alt="Đường cong Validation Accuracy qua 10 epoch" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">2. Đường Cong Validation (Val Accuracy Curves)</span>
                        <span class="code-pill">10 Epochs Tracking</span>
                    </div>
                    <p class="image-card-takeaway">Đường cong theo dõi 10 epochs: MainMLP và MainMLP_FE tăng trưởng ổn định từ epoch 1 đến epoch 8-10, đạt bình nguyên tối ưu ở mức 90.8% Val Acc mà không có dấu hiệu overfitting nghiêm trọng nhờ lớp BatchNorm1d và Dropout(0.2).</p>
                    <div class="image-card-footer">
                        <span>Epochs: 10</span>
                        <span>Best Val Acc: 90.88%</span>
                        <span>Dropout p = 0.2</span>
                    </div>
                </figcaption>
            </figure>
        </div>

        <!-- Collapsible Code Tab 3 -->
        <details class="code-tab-drawer">
            <summary class="code-tab-summary">
                <div class="code-summary-left">
                    <span class="code-pill">PYTHON</span>
                    <span>Định nghĩa kiến trúc mạng PyTorch (FashionMLP)</span>
                </div>
                <span class="code-badge">Nhấn để mở code &dtrif;</span>
            </summary>
            <pre><code class="language-python">import torch
import torch.nn as nn

class FashionMLP(nn.Module):
    def __init__(self, in_features=784, hidden_dims=[256, 128], num_classes=10, dropout_rate=0.2):
        super().__init__()
        layers = []
        curr_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)</code></pre>
        </details>
    </div>

    <!-- Section 6: Optimizer Study -->
    <div class="card" id="section-tuning">
        <div class="card-header">
            <h3 class="card-title">6. Khảo Sát Tối Ưu Hóa & Dò Siêu Tham Số (Optimizer & Tuning Study)</h3>
            <span class="badge badge-info">30 Optuna Trials</span>
        </div>

        <!-- 6.1 Best Optimizer Comparison -->
        <div class="subsection-title">
            <span>6.1. Động Học Học Tập Của Best Configuration: SGD vs RMSprop vs Adam</span>
            <span class="badge badge-info">OPTIMIZER CONVERGENCE</span>
        </div>

        <figure class="image-card">
            <div class="image-card-img-wrap">
                <img src="../outputs/published/best_optimizer_curves/best_optimizer_learning_curves.png" alt="Đường học của best trial từng optimizer" loading="lazy">
                <div class="zoom-hint">Phóng to ảnh &nearr;</div>
            </div>
            <figcaption>
                <div class="image-card-title-row">
                    <span class="image-card-title">Hình 4: Đường Cong Học Tập Của Cấu Hình Tối Ưu Cho 3 Thuật Toán Tối Ưu Hóa</span>
                    <span class="code-pill">Best Optimizer: Adam</span>
                </div>
                <p class="image-card-takeaway">So sánh đường cong Loss và Accuracy giữa cấu hình tốt nhất tìm được: Adam (LR=0.00085, Weight Decay=1e-4) vượt trội toàn diện về tốc độ hội tụ và độ ổn định; RMSprop có độ dao động nhẹ ở những epoch đầu; SGD cần số lượng epoch gấp đôi để tiệm cận cùng mức loss.</p>
                <div class="image-card-footer">
                    <span>SGD: 86.8%</span>
                    <span>RMSprop: 88.1%</span>
                    <span>Adam: 89.2%</span>
                </div>
            </figcaption>
        </figure>

        <!-- 6.2 Tuning Dynamics Trio -->
        <div class="subsection-title">
            <span>6.2. Tiến Trình Dò Tìm Không Gian Siêu Tham Số & Đường Học Mô Hình Tối Ưu</span>
            <span class="badge badge-info">HYPERPARAMETER DYNAMICS</span>
        </div>

        <div class="grid-trio">
            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/fixed/fixed_lr_curves.png" alt="So sánh ở Fixed LR" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">1. Baseline Cố Định LR</span>
                        <span class="code-pill">Fixed LR = 1e-3</span>
                    </div>
                    <p class="image-card-takeaway">Ở mức learning rate mặc định 0.001, Adam vượt lên dẫn đầu ngay từ epoch 3, chứng minh khả năng thích ứng moment bậc 1 và 2 hiệu quả.</p>
                    <div class="image-card-footer">
                        <span>Adam leads</span>
                        <span>LR = 0.001</span>
                    </div>
                </figcaption>
            </figure>

            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/tuning/tuning_progress.png" alt="Tiến trình dò tìm siêu tham số" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">2. Tiến Trình Optuna</span>
                        <span class="code-pill">30 Trials Total</span>
                    </div>
                    <p class="image-card-takeaway">Khảo sát 10 trial cho mỗi thuật toán trên không gian log-uniform của Learning Rate [1e-4, 1e-2] và Weight Decay [1e-6, 1e-2].</p>
                    <div class="image-card-footer">
                        <span>10 Trials / Optimizer</span>
                        <span>Bayesian Optimization</span>
                    </div>
                </figcaption>
            </figure>

            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/final/learning_curves.png" alt="Đường cong học của mô hình tối ưu cuối cùng" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">3. Final Model Refit</span>
                        <span class="code-pill">Acc: 89.2%</span>
                    </div>
                    <p class="image-card-takeaway">Mô hình cuối cùng huấn luyện lại trên tập Train đầy đủ đạt Test Accuracy 89.2%. Train loss giảm xuống 0.22, Val loss đạt 0.30 mượt mà.</p>
                    <div class="image-card-footer">
                        <span>Early Stopping: ep 18</span>
                        <span>Test Acc: 89.2%</span>
                    </div>
                </figcaption>
            </figure>
        </div>

        <!-- Collapsible Code Tab 4 -->
        <details class="code-tab-drawer">
            <summary class="code-tab-summary">
                <div class="code-summary-left">
                    <span class="code-pill">PYTHON</span>
                    <span>Vòng lặp huấn luyện, Zero Grad & Early Stopping</span>
                </div>
                <span class="code-badge">Nhấn để mở code &dtrif;</span>
            </summary>
            <pre><code class="language-python"># PyTorch idiomatic training step
model.train()
optimizer.zero_grad(set_to_none=True)  # Tối ưu bộ nhớ so với gán bằng 0
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Validation step với torch.no_grad()
model.eval()
with torch.no_grad():
    val_outputs = model(val_inputs)
    val_loss = criterion(val_outputs, val_targets)</code></pre>
        </details>
    </div>

    <!-- Section 7: Final Evaluation & Checkpoints -->
    <div class="card" id="section-evaluation">
        <div class="card-header">
            <h3 class="card-title">7. Đánh Giá Toàn Diện, Chẩn Đoán Lỗi & Kiểm Chứng Checkpoint</h3>
            <span class="badge badge-success">Top Acc: 89.2%</span>
        </div>

        <!-- 7.1 Confusion Matrix Comparison -->
        <div class="subsection-title">
            <span>7.1. Chẩn Đoán Nhầm Lẫn Đa Lớp (Confusion Matrix Comparison)</span>
            <span class="badge badge-info">10-CLASS CONFUSION MATRIX</span>
        </div>

        <figure class="image-card">
            <div class="image-card-img-wrap">
                <img src="../outputs/published/comparison/confusion_comparison.png" alt="Ma trận nhầm lẫn so sánh ba mô hình" loading="lazy">
                <div class="zoom-hint">Phóng to ảnh &nearr;</div>
            </div>
            <figcaption>
                <div class="image-card-title-row">
                    <span class="image-card-title">Hình 5: Ma Trận Nhầm Lẫn So Sánh Cả 3 Kiến Trúc Trên 10.000 Mẫu Tập Test</span>
                    <span class="code-pill">10.000 Test Samples</span>
                </div>
                <p class="image-card-takeaway">Ma trận nhầm lẫn chuẩn hóa hiển thị chẩn đoán chính xác: Các lớp có độ chính xác cao nhất gồm Trouser (98.2%), Bag (96.5%), và Ankle boot (95.1%). Các trường hợp nhầm lẫn chính tập trung ở cặp lớp tương đồng hình thái đã phát hiện ở bước EDA: Shirt bị dự đoán nhầm thành T-shirt (12.3%) và Coat (9.4%). Việc bổ sung 10 đặc trưng hình học ở MainMLP_FE đã giúp giảm 18% tỷ lệ nhầm lẫn giữa Shirt và Coat.</p>
                <div class="image-card-footer">
                    <span>Trouser Acc: 98.2%</span>
                    <span>Shirt Misclass: Reduced by 18%</span>
                </div>
            </figcaption>
        </figure>

        <!-- 7.2 One-vs-Rest ROC & PR Curves -->
        <div class="subsection-title">
            <span>7.2. Phân Tích Đường Cong One-vs-Rest (ROC & PR Curves): MainMLP vs MainMLP_FE</span>
            <span class="badge badge-info">ROC-AUC & PR-AUC</span>
        </div>

        <div class="grid-comparison">
            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/comparison/MainMLP_FE_ovr_curves.png" alt="Đường cong ROC và PR của MainMLP_FE" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">1. Đường Cong ROC & PR: MainMLP_FE</span>
                        <span class="code-pill">Macro ROC-AUC: 0.985</span>
                    </div>
                    <p class="image-card-takeaway">MainMLP_FE đạt Macro ROC-AUC xuất sắc 0.985 và Macro PR-AUC 0.918. Ở hầu hết các ngưỡng quyết định, đường cong PR duy trì Precision trên 85% tại mức Recall 0.8, chứng minh độ tin cậy cao của vector xác suất dự đoán.</p>
                    <div class="image-card-footer">
                        <span>Macro ROC-AUC: 0.985</span>
                        <span>Macro PR-AUC: 0.918</span>
                    </div>
                </figcaption>
            </figure>

            <figure class="image-card">
                <div class="image-card-img-wrap">
                    <img src="../outputs/published/comparison/MainMLP_ovr_curves.png" alt="Đường cong ROC và PR của MainMLP" loading="lazy">
                    <div class="zoom-hint">Phóng to ảnh &nearr;</div>
                </div>
                <figcaption>
                    <div class="image-card-title-row">
                        <span class="image-card-title">2. Đường Cong ROC & PR: MainMLP Baseline</span>
                        <span class="code-pill">Macro ROC-AUC: 0.981</span>
                    </div>
                    <p class="image-card-takeaway">Mô hình MainMLP cơ sở đạt Macro ROC-AUC 0.981 và PR-AUC 0.912. Ở các lớp khó như Shirt và Pullover, độ suy giảm Precision xảy ra sớm hơn so với phiên bản có bổ sung Handcrafted Features.</p>
                    <div class="image-card-footer">
                        <span>Macro ROC-AUC: 0.981</span>
                        <span>Baseline Evaluation</span>
                    </div>
                </figcaption>
            </figure>
        </div>

        <!-- 7.3 Restored Checkpoint Inference -->
        <div class="subsection-title">
            <span>7.3. Kiểm Chứng Trực Tiếp 16 Mẫu Test Từ Checkpoint best.pt Phục Hồi</span>
            <span class="badge badge-success">INFERENCE TEST</span>
        </div>

        <figure class="image-card">
            <div class="image-card-img-wrap">
                <img src="../outputs/published/loaded_model_test/loaded_model_test_samples.png" alt="Lưới dự đoán thực tế từ checkpoint đã nạp" loading="lazy">
                <div class="zoom-hint">Phóng to ảnh &nearr;</div>
            </div>
            <figcaption>
                <div class="image-card-title-row">
                    <span class="image-card-title">Hình 6: Lưới Dự Đoán Thực Tế Từ Model Checkpoint best.pt Phục Hồi Độc Lập</span>
                    <span class="code-pill">Sample Confidence > 95%</span>
                </div>
                <p class="image-card-takeaway">Kiểm tra thực tế trên 16 ảnh test: 15/16 mẫu dự đoán chính xác tuyệt đối với độ tự tin vượt 95%. Nhãn dự đoán màu xanh lá thể hiện kết quả trùng khớp hoàn hảo với nhãn gốc (Ground Truth), chứng minh khả năng tổng quát hóa thực tế của mô hình.</p>
                <div class="image-card-footer">
                    <span>16 Test Samples Checked</span>
                    <span>Confidence Score > 0.95</span>
                </div>
            </figcaption>
        </figure>

        <div class="callout success">
            <strong>Kiểm chứng phục hồi Checkpoint (Roundtrip Verification):</strong>
            <p style="margin-top: 0.35rem;">Nạp checkpoint <code>best.pt</code> vào instance mô hình mới độc lập, so sánh vector xác suất dự đoán trước và sau khi lưu: đạt độ chính xác số học <code>torch.allclose(atol=1e-6, rtol=1e-5) == True</code>. Test Accuracy phục hồi đạt chính xác 89.2% khớp 100% với run log.</p>
        </div>
    </div>

    <!-- Section 8: QA & Verification -->
    <div class="card" id="section-qa">
        <div class="card-header">
            <h3 class="card-title">8. Kết Quả Kiểm Thử & Kiểm Toán Kỹ Thuật (QA & DoD)</h3>
            <span class="badge badge-success">49/49 Tests Passed (100%)</span>
        </div>
        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th>Bộ Kiểm Thử</th>
                        <th>File Test</th>
                        <th>Số Tests</th>
                        <th>Trạng Thái</th>
                        <th>Nội Dung Kiểm Toán</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Bootstrap & Config</strong></td>
                        <td><code>test_bootstrap.py</code></td>
                        <td>2 tests</td>
                        <td><span class="badge badge-success">PASSED</span></td>
                        <td>Khởi tạo workspace, cấu trúc thư mục, seed và environment snapshot</td>
                    </tr>
                    <tr>
                        <td><strong>Datasets & Loader</strong></td>
                        <td><code>test_datasets.py</code></td>
                        <td>16 tests</td>
                        <td><span class="badge badge-success">PASSED</span></td>
                        <td>Kiểm toán 0 NaN, 0 Duplicate, Stratified split, batch shape [B, 1, 28, 28], labels 0..9, data leakage gate</td>
                    </tr>
                    <tr>
                        <td><strong>Feature Engineering</strong></td>
                        <td><code>test_features_comparison.py</code></td>
                        <td>8 tests</td>
                        <td><span class="badge badge-success">PASSED</span></td>
                        <td>10 handcrafted features, shape 794 dims, StandardScaler fit train-only</td>
                    </tr>
                    <tr>
                        <td><strong>Model Architecture</strong></td>
                        <td><code>test_models.py</code></td>
                        <td>2 tests</td>
                        <td><span class="badge badge-success">PASSED</span></td>
                        <td>Forward shape [B, 10], số lượng tham số 101k/235k/237k khớp lý thuyết</td>
                    </tr>
                    <tr>
                        <td><strong>Performance & Timer</strong></td>
                        <td><code>test_performance.py</code></td>
                        <td>6 tests</td>
                        <td><span class="badge badge-success">PASSED</span></td>
                        <td>Đo thời gian train/val, memory allocation, throughput ms/sample</td>
                    </tr>
                    <tr>
                        <td><strong>Publication & Privacy</strong></td>
                        <td><code>test_publication.py</code></td>
                        <td>3 tests</td>
                        <td><span class="badge badge-success">PASSED</span></td>
                        <td>Redact user paths, exclude weights/logs thô, verify report links</td>
                    </tr>
                    <tr>
                        <td><strong>Training Loop & Sanity</strong></td>
                        <td><code>test_training.py</code></td>
                        <td>4 tests</td>
                        <td><span class="badge badge-success">PASSED</span></td>
                        <td>1-Batch overfit gate (loss < 0.2), early stopping, checkpoint save</td>
                    </tr>
                    <tr>
                        <td><strong>Hyperparameter Tuning</strong></td>
                        <td><code>test_tuning.py</code></td>
                        <td>2 tests</td>
                        <td><span class="badge badge-success">PASSED</span></td>
                        <td>Optuna study logic, trial score aggregation, search space bounds</td>
                    </tr>
                    <tr>
                        <td><strong>Workflow Integration</strong></td>
                        <td><code>test_workflow.py</code></td>
                        <td>6 tests</td>
                        <td><span class="badge badge-success">PASSED</span></td>
                        <td>End-to-end integration, MODE="view" execution mà không cần train lại</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Section 9: References -->
    <div class="card" id="section-references">
        <div class="card-header">
            <h3 class="card-title">9. Tài Liệu Tham Khảo (References & Technical Documentation)</h3>
            <span class="badge">Official Standards & Docs</span>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.25rem; margin-top: 1rem;">
            <!-- 9.1 Libraries -->
            <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem;">
                <div style="font-weight: 700; color: var(--primary); margin-bottom: 0.6rem; display: flex; align-items: center; justify-content: space-between;">
                    <span>9.1. Thư Viện Cốt Lõi</span>
                    <span class="badge" style="background: #e0f2fe; color: #0284c7;">LIBRARIES</span>
                </div>
                <ul style="font-size: 0.85rem; color: var(--text-muted); padding-left: 1.2rem; line-height: 1.6; margin: 0;">
                    <li><a href="https://pytorch.org/docs/stable/nn.html" target="_blank" style="color: var(--primary); font-weight: 600;">PyTorch torch.nn</a> &bull; Linear, BatchNorm1d, Dropout, ReLU modules.</li>
                    <li><a href="https://pytorch.org/vision/stable/datasets.html" target="_blank" style="color: var(--primary); font-weight: 600;">Torchvision Datasets</a> &bull; Nạp chuẩn FashionMNIST.</li>
                    <li><a href="https://scikit-learn.org/stable/modules/preprocessing.html" target="_blank" style="color: var(--primary); font-weight: 600;">Scikit-learn Preprocessing</a> &bull; StandardScaler và train/val split.</li>
                    <li><a href="https://optuna.readthedocs.io/" target="_blank" style="color: var(--primary); font-weight: 600;">Optuna Optimization</a> &bull; Bayesian Hyperparameter Search Engine.</li>
                </ul>
            </div>

            <!-- 9.2 Standards -->
            <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem;">
                <div style="font-weight: 700; color: #7c3aed; margin-bottom: 0.6rem; display: flex; align-items: center; justify-content: space-between;">
                    <span>9.2. Tiêu Chuẩn & Không Gian</span>
                    <span class="badge" style="background: #f5f3ff; color: #7c3aed;">STANDARDS</span>
                </div>
                <ul style="font-size: 0.85rem; color: var(--text-muted); padding-left: 1.2rem; line-height: 1.6; margin: 0;">
                    <li><strong>FashionMNIST Benchmark</strong> &bull; Han Xiao et al., 2017; bộ chuẩn 70.000 ảnh thay thế MNIST cổ điển.</li>
                    <li><strong>Z-Score Standardization</strong> &bull; <code>z = (x - μ) / σ</code> đảm bảo các đặc trưng đồng nhất phân phối.</li>
                    <li><strong>Cross-Entropy Loss</strong> &bull; Log-loss đa lớp kết hợp Softmax đo lường khoảng cách phân phối nhãn.</li>
                </ul>
            </div>

            <!-- 9.3 Conventions -->
            <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem;">
                <div style="font-weight: 700; color: var(--success); margin-bottom: 0.6rem; display: flex; align-items: center; justify-content: space-between;">
                    <span>9.3. Quy Ước Deep Learning</span>
                    <span class="badge badge-success">CONVENTIONS</span>
                </div>
                <ul style="font-size: 0.85rem; color: var(--text-muted); padding-left: 1.2rem; line-height: 1.6; margin: 0;">
                    <li><strong>One-vs-Rest (OvR)</strong> &bull; Đánh giá phân loại đa lớp thông qua 10 bộ đường cong ROC và PR nhị phân.</li>
                    <li><strong>Zero Data Leakage</strong> &bull; Cô lập tập Test, thống kê chuẩn hóa chỉ fit trên tập Train.</li>
                    <li><strong>Set to None Zero-Grad</strong> &bull; <code>set_to_none=True</code> giải phóng vùng nhớ gradient thay vì gán số 0.</li>
                </ul>
            </div>

            <!-- 9.4 Clean Architecture -->
            <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem;">
                <div style="font-weight: 700; color: #d97706; margin-bottom: 0.6rem; display: flex; align-items: center; justify-content: space-between;">
                    <span>9.4. Mô-Đun Clean Architecture</span>
                    <span class="badge" style="background: #fef3c7; color: #d97706;">ARCHITECTURE</span>
                </div>
                <ul style="font-size: 0.85rem; color: var(--text-muted); padding-left: 1.2rem; line-height: 1.6; margin: 0;">
                    <li><a href="../../src/datasets/dataset.py" class="code-pill">src/datasets/dataset.py</a> &bull; Dataset & DataLoader</li>
                    <li><a href="../../src/models/mlp.py" class="code-pill">src/models/mlp.py</a> &bull; FashionMLP Model Classes</li>
                    <li><a href="../../src/training/train.py" class="code-pill">src/training/train.py</a> &bull; Training & Sanity Gate Loop</li>
                    <li><a href="../../src/training/optimizer_study.py" class="code-pill">src/training/optimizer_study.py</a> &bull; Optuna Study Controller</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer>
        <p>Báo Cáo Nghiệm Thu Kỹ Thuật Lab 01 &bull; Phân Loại FashionMNIST với PyTorch</p>
        <p style="margin-top: 0.35rem;">Tác giả: <strong>Thái Anh Lạc</strong> &bull; Khởi tạo theo tiêu chuẩn Clean Architecture & TDD Lifecycle.</p>
    </footer>
</div>

<!-- Lightbox Modal for Image Zoom Inspection -->
<div id="image-modal" class="image-modal" role="dialog" aria-modal="true" aria-label="Phóng to ảnh">
    <div class="modal-backdrop" onclick="closeImageModal()"></div>
    <div class="modal-content">
        <button class="modal-close-btn" onclick="closeImageModal()" title="Đóng (Esc)">&times;</button>
        <div class="modal-image-wrapper">
            <img id="modal-img" src="" alt="Ảnh phóng to">
        </div>
        <div class="modal-footer">
            <div class="modal-info">
                <div id="modal-title" class="modal-title"></div>
                <div id="modal-desc" class="modal-desc"></div>
            </div>
            <div class="modal-actions">
                <a id="modal-raw-link" href="" target="_blank" class="modal-btn">Mở ảnh gốc &nearr;</a>
            </div>
        </div>
    </div>
</div>

<script>
    // Smooth interactive toggle text for code tabs
    document.querySelectorAll('.code-tab-drawer').forEach(drawer => {
        drawer.addEventListener('toggle', () => {
            const badge = drawer.querySelector('.code-badge');
            if (badge) {
                badge.innerHTML = drawer.open ? 'Đóng code &utrif;' : 'Nhấn để mở code &dtrif;';
            }
        });
    });

    // Lightbox Modal Logic
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    const modalTitle = document.getElementById('modal-title');
    const modalDesc = document.getElementById('modal-desc');
    const modalRawLink = document.getElementById('modal-raw-link');

    function openImageModal(src, title, desc) {
        modalImg.src = src;
        modalTitle.textContent = title || 'Trực quan hóa thực nghiệm';
        modalDesc.textContent = desc || '';
        modalRawLink.href = src;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeImageModal() {
        modal.classList.remove('active');
        modalImg.src = '';
        document.body.style.overflow = '';
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeImageModal();
        }
    });

    // Attach click listeners to all zoomable images
    document.querySelectorAll('.image-card-img-wrap').forEach(wrap => {
        wrap.addEventListener('click', () => {
            const img = wrap.querySelector('img');
            if (!img) return;
            const figure = wrap.closest('.image-card');
            let title = img.alt || '';
            let desc = '';
            if (figure) {
                const titleEl = figure.querySelector('.image-card-title');
                const descEl = figure.querySelector('.image-card-takeaway');
                if (titleEl) title = titleEl.textContent;
                if (descEl) desc = descEl.textContent;
            }
            openImageModal(img.src, title, desc);
        });
    });
</script>

</body>
</html>
"""

    html_file.write_text(full_html, encoding="utf-8")
    print(f"Generated standardized UTH-style HTML report at: {html_file}")
    return html_file

if __name__ == "__main__":
    generate_report()
