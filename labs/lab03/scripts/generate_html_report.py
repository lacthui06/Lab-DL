"""Script to generate standardized, visual-first HTML report for Lab 03.
Based on the UTH Deep Learning & Computer Vision executive template style.

Generates: labs/lab03/reports/walkthrough_lab_03.html
Author: Thai Anh Lac
"""

from pathlib import Path

# ==============================================================================
# 1. LAB CONFIGURATION & METRICS
# ==============================================================================
LAB_CONFIG = {
    "lab_num": 3,
    "lab_id": "lab03",
    "title": "Báo Cáo Nghiệm Thu Kỹ Thuật Lab 3",
    "subtitle": "Binary Sentiment Classification với Hugging Face & DistilBERT",
    "school": "TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH)",
    "author": "Thái Anh Lạc",
    "python_ver": "Python 3.12 / 3.13",
    "framework_ver": "PyTorch 2.x • Transformers 4.35+",
    "dataset_info": "Rotten Tomatoes (10.662 mẫu: 8.530 train / 1.066 val / 1.066 test)",
    "test_passed": "95 / 95 Passed (100%)",
    "primary_metric": "Test Acc: 83.11% • Binary F1: 83.30% (Ngưỡng tối ưu: 0.45)",
}

KPIS = [
    {"label": "Test Accuracy", "val": "83.11%", "sub": "Baseline TF-IDF: 78.99% (+4.12%)", "color": "#2563eb"},
    {"label": "Test Binary F1", "val": "83.30%", "sub": "Tối ưu hóa tại ngưỡng 0.45", "color": "#059669"},
    {"label": "ROC-AUC Score", "val": "0.908", "sub": "PR-AUC: 0.902 • Phân tách xuất sắc", "color": "#7c3aed"},
    {"label": "Tốc Độ Huấn Luyện", "val": "34.2s / epoch", "sub": "Kaggle Tesla T4 • 1.84 GB VRAM", "color": "#d97706"},
    {"label": "Tham Số Trainable", "val": "14.77M", "sub": "22.1% (Đóng băng 4 block đầu)", "color": "#0284c7"},
]

PIPELINE_STEPS = [
    {
        "num": "01",
        "title": "Pretrained Inference",
        "desc": "Exercise 1 • Tokenizer & Zero-shot",
        "anchor": "#section-pretrained",
    },
    {
        "num": "02",
        "title": "Core NLP EDA",
        "desc": "Exercise 2 • Kiểm toán 0 Leakage & Phân bố",
        "anchor": "#section-eda",
    },
    {
        "num": "03",
        "title": "Kỹ Nghệ & Augmentation",
        "desc": "Exercise 2 • TF-IDF, Chi2 & Tăng cường",
        "anchor": "#section-features",
    },
    {
        "num": "04",
        "title": "Fine-Tuning DistilBERT",
        "desc": "Exercise 2 • Động học 5 Epochs & Baseline",
        "anchor": "#section-training",
    },
    {
        "num": "05",
        "title": "Ngưỡng & Đánh Giá",
        "desc": "Exercise 2 • Quét 19 Ngưỡng & Ma Trận",
        "anchor": "#section-evaluation",
    },
    {
        "num": "06",
        "title": "Kiểm Toán QA & DoD",
        "desc": "Exercise 2 • 95/95 Pytest & Checkpoint",
        "anchor": "#section-qa",
    },
]


# ==============================================================================
# 2. HELPER FUNCTIONS TO RENDER VISUAL CARDS
# ==============================================================================
def render_hero_card(img_path: str, title: str, tag: str, takeaway: str, footer_items: list[str]) -> str:
    """Render full-width hero card for primary charts."""
    footer_html = "".join(f"<span>{item}</span>" for item in footer_items)
    return f"""
    <div class="gallery-card hero-card">
        <div class="card-img-wrap" onclick="openModal(this)">
            <img src="{img_path}" alt="{title}" loading="lazy">
            <span class="zoom-badge">Phóng to &nearr;</span>
        </div>
        <div class="card-caption">
            <div class="card-header-row">
                <h4 class="card-title">{title}</h4>
                <span class="badge badge-pill">{tag}</span>
            </div>
            <p class="card-takeaway">{takeaway}</p>
            <div class="card-footer">
                {footer_html}
            </div>
        </div>
    </div>
    """


def render_gallery_card(img_path: str, title: str, tag: str, takeaway: str, footer_items: list[str]) -> str:
    """Render standard gallery card for 2-column or 3-column grids."""
    footer_html = "".join(f"<span>{item}</span>" for item in footer_items)
    return f"""
    <div class="gallery-card">
        <div class="card-img-wrap" onclick="openModal(this)">
            <img src="{img_path}" alt="{title}" loading="lazy">
            <span class="zoom-badge">Phóng to &nearr;</span>
        </div>
        <div class="card-caption">
            <div class="card-header-row">
                <h4 class="card-title">{title}</h4>
                <span class="badge badge-pill">{tag}</span>
            </div>
            <p class="card-takeaway">{takeaway}</p>
            <div class="card-footer">
                {footer_html}
            </div>
        </div>
    </div>
    """


def render_section_code_drawer(title: str, code_snippet: str) -> str:
    """Render a single, elegant code drawer at the end of each section."""
    return f"""
    <details class="section-code-drawer">
        <summary>
            <div class="drawer-summary-left">
                <span class="code-tag">CODE TRIỂN KHAI</span>
                <span>{title}</span>
            </div>
            <span class="drawer-icon">Xem chi tiết &dtrif;</span>
        </summary>
        <div class="drawer-content">
            <pre><code class="language-python">{code_snippet.strip()}</code></pre>
        </div>
    </details>
    """


# ==============================================================================
# 3. HTML BUILDER
# ==============================================================================
def build_html() -> str:
    # 3.1 Header KPIs HTML
    kpis_html = "".join(f"""
        <div class="kpi-card" style="border-top: 4px solid {k['color']};">
            <div class="kpi-label">{k['label']}</div>
            <div class="kpi-val" style="color: {k['color']};">{k['val']}</div>
            <div class="kpi-sub">{k['sub']}</div>
        </div>""" for k in KPIS)

    # 3.2 Stepper HTML
    stepper_html = "".join(f"""
        <a href="{s['anchor']}" class="step-card">
            <div class="step-card-top">
                <span class="step-badge">{s['num']}</span>
                <span class="step-arrow">&darr;</span>
            </div>
            <div class="step-title">{s['title']}</div>
            <div class="step-desc">{s['desc']}</div>
        </a>""" for s in PIPELINE_STEPS)

    # 3.3 Sections Content
    # --------------------------------------------------------------------------
    # Section 1: Pretrained Inference (Exercise 1)
    # --------------------------------------------------------------------------
    sec1_hero = render_hero_card(
        img_path="../outputs/published/exercise1_analysis/tokenizer_table.png",
        title="1. Cơ Chế Phân Rã Tokenizer WordPiece (Subwords, Token IDs & Attention Masks)",
        tag="Tokenizer Mechanics",
        takeaway="Phân tích chi tiết quá trình mã hóa chuỗi câu thành subwords: gán cặp token biên đặc biệt [CLS] (101) và [SEP] (102). Attention Mask gán giá trị 1 cho token thật và 0 cho vùng đệm padding.",
        footer_items=["DistilBERT Tokenizer", "Vocab Size: 30.522", "Artifact: tokenizer_table.png"],
    )
    sec1_card1 = render_gallery_card(
        img_path="../outputs/published/exercise1_analysis/pipeline_manual_comparison.png",
        title="2. Đối Chiếu Pipeline vs Manual PyTorch",
        tag="Verification • Δ = 0",
        takeaway="Độ lệch xác suất tuyệt đối giữa transformers.pipeline và tính toán PyTorch thuần (logits -> softmax) đạt Δ = 0.000000 (khớp 100%), chứng minh tính đúng đắn của logic suy luận.",
        footer_items=["Max Delta: 0.000000", "Artifact: pipeline_manual_comparison.png"],
    )
    sec1_card2 = render_gallery_card(
        img_path="../outputs/published/exercise1_analysis/sentiment_predictions.png",
        title="3. Dự Đoán Xác Suất Trên Câu Thực Tế",
        tag="Sentiment Inference",
        takeaway="Thực nghiệm suy luận trên các câu có cấu trúc tương phản: mô hình nhận diện chính xác cảm xúc chốt hạ sau liên từ 'but' với độ tin cậy vượt 99.8%.",
        footer_items=["3 Samples Verified", "Artifact: sentiment_predictions.png"],
    )
    sec1_code = """# Trích xuất từ labs/lab03/src/analysis.py & notebooks/03_modeling.ipynb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 1. Pipeline inference cấp cao
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
pipe_result = classifier("This movie was surprisingly fantastic with brilliant acting!")

# 2. Quy trình suy luận thủ công qua PyTorch Tensor đối chứng
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
inputs = tokenizer("This movie was surprisingly fantastic with brilliant acting!", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

# Kiểm chứng độ lệch tuyệt đối: Δ = 0.000000
assert abs(probs[1] - pipe_result[0]["score"]) < 1e-6"""

    sec1_html = f"""
    <section class="content-section" id="section-pretrained">
        <div class="section-header">
            <div class="section-title-wrap">
                <span class="section-num">01</span>
                <div>
                    <h2 class="section-title">Khảo Sát Pretrained Model & Tokenizer (Exercise 1)</h2>
                    <p class="section-subtitle">Mô hình SST-2 Pretrained • 0 Trainable Parameter • Xác thực đối chứng Pipeline vs Manual PyTorch (Δ = 0)</p>
                </div>
            </div>
            <span class="badge badge-blue">Exercise 1</span>
        </div>

        {sec1_hero}

        <div class="grid-2col">
            {sec1_card1}
            {sec1_card2}
        </div>

        {render_section_code_drawer("Mã nguồn Exercise 1: Tokenizer & Đối chiếu suy luận", sec1_code)}
    </section>
    """

    # --------------------------------------------------------------------------
    # Section 2: Core NLP EDA (Exercise 2)
    # --------------------------------------------------------------------------
    sec2_hero = render_hero_card(
        img_path="../outputs/published/frozen_trial/token_length_distribution.png",
        title="1. Phân Bố Độ Dài Token & Xác Lập Ngưỡng Cắt Ngắn (Truncation Cutoff 64)",
        tag="Truncation Strategy",
        takeaway="Độ dài 64 tokens bao phủ 99.4% số câu trong tập Rotten Tomatoes. Việc chọn max_length = 64 thay vì 128 giúp giảm 75% chi phí tính toán Attention O(N²) và tăng gấp đôi thông lượng huấn luyện trên GPU.",
        footer_items=["Cutoff: 64 tokens", "Context Coverage: 99.4%", "Artifact: token_length_distribution.png"],
    )
    sec2_c1 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/data_quality.png",
        title="2. Kiểm Toán Toàn Vẹn & Data Leakage",
        tag="Audit 0 NaN / 0 Leakage",
        takeaway="Kiểm toán 10.662 mẫu: 0 giá trị thiếu (0 NaN, 0 Null) và 0 mẫu trùng lặp chéo giữa Train/Val/Test qua kiểm tra MD5 hashing.",
        footer_items=["Missing: 0", "Cross-split Overlap: 0", "Artifact: data_quality.png"],
    )
    sec2_c2 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/class_distribution.png",
        title="3. Cân Bằng Nhãn Tuyệt Đối 50:50",
        tag="Class Balance",
        takeaway="Tập dữ liệu cân bằng hoàn hảo giữa hai lớp Positive (4.265) và Negative (4.265), triệt tiêu hoàn toàn nguy cơ sai lệch lớp (Class Imbalance).",
        footer_items=["Train: 4.265 / 4.265", "Val/Test: 533 / 533", "Artifact: class_distribution.png"],
    )
    sec2_c3 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/lengths_boxplot.png",
        title="4. Biểu Đồ Hộp Phân Bố Độ Dài Câu",
        tag="5-Number Summary",
        takeaway="Phân tích độ dài câu (Min=1, Q1=11, Median=20, Q3=29, Max=51 từ). Phân bố đồng dạng giữa hai cực tính với độ lệch chuẩn IQR tương đương.",
        footer_items=["Median: 20 words", "Max: 51 words", "Artifact: lengths_boxplot.png"],
    )
    sec2_c4 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/lengths_by_class.png",
        title="5. Mật Độ Chiều Dài Theo Nhãn",
        tag="Density Curve",
        takeaway="Đường cong mật độ KDE cho thấy câu Positive và Negative có dạng phân phối chồng khít, chứng minh độ dài câu không phải là biến thiên gây nhiễu.",
        footer_items=["KDE Distribution", "Positive vs Negative", "Artifact: lengths_by_class.png"],
    )
    sec2_c5 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/train_examples.png",
        title="6. Trích Xuất Mẫu Dữ Liệu Thực Tế",
        tag="Sample Audit",
        takeaway="Kiểm tra các mẫu review điển hình: cấu trúc câu ngắn, giàu tính biểu cảm và chứa nhiều liên từ ngữ cảnh phức tạp.",
        footer_items=["Split: Train Sample", "Artifact: train_examples.png"],
    )
    sec2_code = """# Trích xuất từ labs/lab03/src/data_preparation.py & notebooks/01_eda.ipynb
from datasets import load_dataset
import pandas as pd

# Nạp official dataset Rotten Tomatoes
dataset = load_dataset("rotten_tomatoes")

# Kiểm toán chất lượng dữ liệu: 0 NaN & 0 Duplicates
for split in ["train", "validation", "test"]:
    df = pd.DataFrame(dataset[split])
    assert df["text"].isnull().sum() == 0, f"Phat hien gia tri Null tai {split}"
    assert len(df) == len(df["text"].drop_duplicates()), f"Phat hien ban ghi trung lap tai {split}"

print("Kiem toan toan ven thanh cong: 10.662 mau sach 100%.")"""

    sec2_html = f"""
    <section class="content-section" id="section-eda">
        <div class="section-header">
            <div class="section-title-wrap">
                <span class="section-num">02</span>
                <div>
                    <h2 class="section-title">Khám Phá Dữ Liệu & Kiểm Toán Toàn Vẹn (Core NLP EDA)</h2>
                    <p class="section-subtitle">Tập Dữ Liệu Rotten Tomatoes • 10.662 Mẫu Cân Bằng 50:50 • Kiểm Toán 0 NaN / 0 Data Leakage</p>
                </div>
            </div>
            <span class="badge badge-blue">Exercise 2</span>
        </div>

        {sec2_hero}

        <div class="grid-2col">
            {sec2_c1}
            {sec2_c2}
            {sec2_c3}
            {sec2_c4}
        </div>
        <div style="margin-top: 1.25rem;">
            {sec2_c5}
        </div>

        {render_section_code_drawer("Mã nguồn EDA: Kiểm toán toàn vẹn và phân tích độ dài", sec2_code)}
    </section>
    """

    # --------------------------------------------------------------------------
    # Section 3: Feature Engineering & Augmentation (Exercise 2)
    # --------------------------------------------------------------------------
    sec3_hero = render_hero_card(
        img_path="../outputs/published/frozen_trial/chi2_features.png",
        title="1. Lựa Chọn Đặc Trưng Có Tính Phân Tách Cao Bằng Kiểm Định Chi-Square (χ²)",
        tag="Feature Selection",
        takeaway="Xác định top 25 từ/cụm từ có giá trị thống kê phân cực cao nhất (too, bad, dull, boring, wonderful, moving, touching). Các từ mang cảm xúc cực đoan có p-value < 0.001.",
        footer_items=["Top-25 Terms", "Ranked by χ²", "Artifact: chi2_features.png"],
    )
    sec3_c1 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/tfidf_features.png",
        title="2. Không Gian Vector TF-IDF Unigram/Bigram",
        tag="TF-IDF Baseline Setup",
        takeaway="Trích xuất 5.000 đặc trưng unigram/bigram: độ thưa ma trận đạt 99.898%, phục vụ huấn luyện mô hình Baseline Logistic Regression đối chứng.",
        footer_items=["5.000 Features", "Sparsity: 99.9%", "Artifact: tfidf_features.png"],
    )
    sec3_c2 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/tfidf_class_delta.png",
        title="3. Chênh Lệch Trọng Số TF-IDF Giữa Hai Lớp",
        tag="Class Vocabulary Delta",
        takeaway="Biểu đồ phân cực từ vựng: thể hiện các từ có chênh lệch trọng số trung bình lớn nhất giữa tập Positive và Negative.",
        footer_items=["Vocabulary Delta", "Polarity Metric", "Artifact: tfidf_class_delta.png"],
    )
    sec3_c3 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/linguistic_features.png",
        title="4. 8 Đặc Trưng Ngôn Ngữ Học (Linguistic Features)",
        tag="Linguistic Signals",
        takeaway="Trích xuất tỷ lệ viết hoa, dấu câu biểu cảm (!?), mật độ từ phủ định và từ tương phản. Nhãn Negative có xu hướng chứa nhiều dấu chấm hỏi và từ phủ định hơn.",
        footer_items=["8 Features Extracted", "Artifact: linguistic_features.png"],
    )
    sec3_c4 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/negation_contrast_by_label.png",
        title="5. Phân Bố Phủ Định & Tương Phản (Negation & Contrast)",
        tag="Semantic Polarity",
        takeaway="Khoảng 33.4% số câu trong tập dữ liệu chứa từ phủ định (not, no, never) hoặc liên từ tương phản (but, however), đòi hỏi bảo tồn tuyệt đối khi tiền xử lý.",
        footer_items=["2.848 Samples Protected", "Artifact: negation_contrast_by_label.png"],
    )
    sec3_c5 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/attention_mask.png",
        title="6. Biểu Diễn Attention Masks & Dynamic Padding",
        tag="Transformer Tensors",
        takeaway="Minh họa cơ chế Attention Mask: triệt tiêu ảnh hưởng của token [PAD] trong phép nhân Attention ma trận, tập trung trọng số vào các token nội dung thật.",
        footer_items=["Batch Tensors", "Dynamic Padding", "Artifact: attention_mask.png"],
    )
    sec3_c6 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/augmentation_counts.png",
        title="7. Tăng Cường Dữ Liệu Văn Bản Có Ràng Buộc",
        tag="Safe Text Augmentation",
        takeaway="Tạo thêm 274 mẫu tăng cường chất lượng cao cho tập Train: áp dụng quy tắc bảo vệ 100% từ phủ định và tương phản, đảm bảo 96.8% mẫu giữ nguyên vẹn nhãn gốc.",
        footer_items=["+274 Samples", "Train-only Applied", "Artifact: augmentation_counts.png"],
    )
    sec3_code = """# Trích xuất từ labs/lab03/src/datasets/augmentation.py & loader.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

# 1. Trích xuất TF-IDF 5.000 unigram/bigram
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(train_texts)

# 2. Xếp hạng đặc trưng bằng Chi-Square
scores, p_values = chi2(X_train_tfidf, train_labels)
ranked_terms = [tfidf.get_feature_names_out()[i] for i in scores.argsort()[::-1]]
print("Top 5 terms co tinh phan cuc cao nhat:", ranked_terms[:5])"""

    sec3_html = f"""
    <section class="content-section" id="section-features">
        <div class="section-header">
            <div class="section-title-wrap">
                <span class="section-num">03</span>
                <div>
                    <h2 class="section-title">Kỹ Nghệ Đặc Trưng & Tăng Cường Dữ Liệu</h2>
                    <p class="section-subtitle">TF-IDF Vectorization • Chọn Lọc Chi-Square (χ²) • 8 Đặc Trưng Ngôn Ngữ Học • Safe Text Augmentation</p>
                </div>
            </div>
            <span class="badge badge-blue">Exercise 2</span>
        </div>

        {sec3_hero}

        <div class="grid-2col">
            {sec3_c1}
            {sec3_c2}
            {sec3_c3}
            {sec3_c4}
            {sec3_c5}
            {sec3_c6}
        </div>

        {render_section_code_drawer("Mã nguồn Feature Engineering: TF-IDF & Chi-Square Selection", sec3_code)}
    </section>
    """

    # --------------------------------------------------------------------------
    # Section 4: DistilBERT Fine-Tuning & Training Dynamics (Exercise 2)
    # --------------------------------------------------------------------------
    sec4_hero = render_hero_card(
        img_path="../outputs/published/frozen_trial/learning_curves.png",
        title="1. Động Học Huấn Luyện 5 Epochs — Hội Tụ & Đánh Đổi Train/Validation Loss",
        tag="Convergence Dynamics",
        takeaway="Đường cong học tập qua 5 epochs: Validation Loss giảm ổn định từ 0.394 xuống 0.368, Validation Accuracy đạt 83.96%. Khoảng cách loss gap cuối đạt 0.019 (không có dấu hiệu overfitting nghiêm trọng). Checkpoint tối ưu được lưu tại Epoch 5.",
        footer_items=["Best Checkpoint: Epoch 5", "Min Val Loss: 0.367875", "Artifact: learning_curves.png"],
    )
    sec4_c1 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/model_architecture.png",
        title="2. Phân Bổ Tham Số Đóng Băng (Frozen vs Trainable)",
        tag="Transfer Learning",
        takeaway="Khóa cứng Embeddings và 4 Transformer Block đầu (52.2M params), chỉ mở 2 Transformer Block cuối + Classification Head (14.8M params ~ 22.1%), giảm 78% bộ nhớ gradient.",
        footer_items=["Total: 66.96M", "Trainable: 14.77M (22.1%)", "Artifact: model_architecture.png"],
    )
    sec4_c2 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/performance.png",
        title="3. Hiệu Năng Tính Toán & Mức Tiêu Thụ VRAM/RAM",
        tag="Compute Benchmark",
        takeaway="Huấn luyện tốc độ 34.2s/epoch trên GPU Kaggle Tesla T4, bộ nhớ VRAM đỉnh chỉ 1.84 GB (peak reserved 638 MiB), tốc độ suy luận siêu tốc 1.6ms/sample.",
        footer_items=["Epoch Time: 34.2s", "VRAM: 1.84 GB", "Artifact: performance.png"],
    )
    sec4_code = """# Trích xuất từ labs/lab03/src/training/trainer.py
import torch
from transformers import AutoModelForSequenceClassification

# 1. Đóng băng 4 block đầu của DistilBERT
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
for param in model.distilbert.embeddings.parameters():
    param.requires_grad = False
for layer in model.distilbert.transformer.layer[:4]:
    for param in layer.parameters():
        param.requires_grad = False

# 2. Khởi tạo Optimizer với Discriminative Learning Rates
optimizer = torch.optim.AdamW([
    {"params": model.distilbert.transformer.layer[4:].parameters(), "lr": 5e-6},
    {"params": model.pre_classifier.parameters(), "lr": 2e-5},
    {"params": model.classifier.parameters(), "lr": 2e-5},
], weight_decay=0.01)"""

    sec4_html = f"""
    <section class="content-section" id="section-training">
        <div class="section-header">
            <div class="section-title-wrap">
                <span class="section-num">04</span>
                <div>
                    <h2 class="section-title">Huấn Luyện Fine-Tuning & Động Học Mô Hình</h2>
                    <p class="section-subtitle">DistilBERT Base • Đóng Băng 4 Block Đầu • Discriminative LR (5e-6 / 2e-5) • Early Stopping Patience 2</p>
                </div>
            </div>
            <span class="badge badge-blue">Exercise 2</span>
        </div>

        {sec4_hero}

        <div class="grid-2col">
            {sec4_c1}
            {sec4_c2}
        </div>

        {render_section_code_drawer("Mã nguồn Training: Chiến lược đóng băng và cấu hình AdamW", sec4_code)}
    </section>
    """

    # --------------------------------------------------------------------------
    # Section 5: Evaluation & Threshold Optimization (Exercise 2)
    # --------------------------------------------------------------------------
    sec5_hero1 = render_hero_card(
        img_path="../outputs/published/frozen_trial/validation_threshold_curves.png",
        title="1. Tối Ưu Hóa Ngưỡng Quyết Định Phân Loại Nhị Phân (Validation Threshold Sweep)",
        tag="Operating Point: 0.45",
        takeaway="Quét 19 ngưỡng từ 0.05 đến 0.95 trên tập Validation: tại ngưỡng 0.45, F1-Score đạt đỉnh tối ưu 83.82%. Khóa policy ngưỡng 0.45 và áp dụng lên Test giúp Test F1 tăng từ 82.70% lên 83.30% (+0.60% cải thiện thực chất).",
        footer_items=["Default: 0.50", "Optimal Policy: 0.45", "Artifact: validation_threshold_curves.png"],
    )
    sec5_hero2 = render_hero_card(
        img_path="../outputs/published/frozen_trial/roc_pr_curves.png",
        title="2. Đường Cong Khả Năng Phân Tách ROC và PR (ROC-AUC: 0.908 • PR-AUC: 0.902)",
        tag="Separation Quality",
        takeaway="Đường cong ROC và Precision-Recall tiệm cận góc trên bên trái với diện tích dưới đường cong đạt ROC-AUC = 0.908 và PR-AUC = 0.902, khẳng định năng lực phân định cảm xúc cực kỳ tin cậy.",
        footer_items=["ROC-AUC: 0.908", "PR-AUC: 0.902", "Artifact: roc_pr_curves.png"],
    )
    sec5_c1 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/confusion_matrix.png",
        title="3. Ma Trận Nhầm Lẫn Chuẩn Hóa",
        tag="Test Matrix: 886 / 1.066 Đúng",
        takeaway="Trên 1.066 mẫu test độc lập, mô hình dự đoán chính xác 886 mẫu (Accuracy 83.11%). Tỷ lệ True Positive đạt 83.3% và True Negative đạt 82.9% rất cân bằng.",
        footer_items=["TP: 444 • TN: 442", "Acc: 83.11%", "Artifact: confusion_matrix.png"],
    )
    sec5_c2 = render_gallery_card(
        img_path="../outputs/published/result_analysis/per_class_metrics.png",
        title="4. Chỉ Số Chi Tiết Theo Lớp (Per-class Rates)",
        tag="Detailed Rates",
        takeaway="Phân tích Precision, Recall và F1 trên từng lớp nhãn: Lớp Positive đạt F1 83.30%, Lớp Negative đạt F1 82.92%, chứng minh mô hình không thiên vị cực tính nào.",
        footer_items=["Macro F1: 83.11%", "Artifact: per_class_metrics.png"],
    )
    sec5_c3 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/metrics_bar.png",
        title="5. So Sánh Mô Hình Đề Xuất vs Baseline TF-IDF",
        tag="Baseline Outperformance",
        takeaway="DistilBERT fine-tuned đạt 83.11% Acc vượt trội hoàn toàn so với Baseline truyền thống TF-IDF + Logistic Regression (78.99% Acc), cải thiện thực chất +4.12 điểm phần trăm.",
        footer_items=["Transformer: 83.11%", "TF-IDF: 78.99%", "Artifact: metrics_bar.png"],
    )
    sec5_c4 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/calibration_confidence.png",
        title="6. Độ Hiệu Chuẩn Xác Suất (Reliability Diagram)",
        tag="Calibration ECE: 0.038",
        takeaway="Biểu đồ hiệu chuẩn độ tin cậy: sai số hiệu chuẩn kỳ vọng đạt mức thấp lý tưởng ECE = 0.038, xác suất đầu ra phản ánh trung thực xác suất đúng trong thực tế.",
        footer_items=["ECE: 0.038", "Calibrated Output", "Artifact: calibration_confidence.png"],
    )
    sec5_c5 = render_gallery_card(
        img_path="../outputs/published/frozen_trial/sample_predictions.png",
        title="7. Lưới Dự Đoán Mẫu Kiểm Chứng & Phân Tích Lỗi",
        tag="Qualitative Error Analysis",
        takeaway="Trực quan hóa các mẫu dự đoán đúng và sai: các trường hợp phân loại sai chủ yếu rơi vào các câu đánh giá mang tính châm biếm sâu cay (sarcasm) hoặc ngữ nghĩa mơ hồ.",
        footer_items=["16 Samples Inspected", "Artifact: sample_predictions.png"],
    )
    sec5_code = """# Trích xuất từ labs/lab03/src/training/threshold.py
import numpy as np
from sklearn.metrics import f1_score

# Quét 19 ngưỡng trên tập Validation
thresholds = np.linspace(0.05, 0.95, 19)
best_f1, best_threshold = 0.0, 0.5

for th in thresholds:
    preds = (val_probs[:, 1] >= th).astype(int)
    f1 = f1_score(val_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = th

print(f"Nguong toi uu xac lap tren Validation: {best_threshold:.2f} (Val F1: {best_f1:.4f})")"""

    sec5_html = f"""
    <section class="content-section" id="section-evaluation">
        <div class="section-header">
            <div class="section-title-wrap">
                <span class="section-num">05</span>
                <div>
                    <h2 class="section-title">Đánh Giá Toàn Diện & Tối Ưu Hóa Ngưỡng</h2>
                    <p class="section-subtitle">Validation Threshold Sweep (0.45) • Ma Trận Nhầm Lẫn 886/1.066 Mẫu Đúng • ROC-AUC 0.908 • ECE 0.038</p>
                </div>
            </div>
            <span class="badge badge-blue">Exercise 2</span>
        </div>

        {sec5_hero1}
        {sec5_hero2}

        <div class="grid-2col">
            {sec5_c1}
            {sec5_c2}
            {sec5_c3}
            {sec5_c4}
        </div>
        <div style="margin-top: 1.25rem;">
            {sec5_c5}
        </div>

        {render_section_code_drawer("Mã nguồn Evaluation: Thuật toán quét ngưỡng và đánh giá", sec5_code)}
    </section>
    """

    # --------------------------------------------------------------------------
    # Section 6: QA Audit & Acceptance Verification
    # --------------------------------------------------------------------------
    sec6_html = f"""
    <section class="content-section" id="section-qa">
        <div class="section-header">
            <div class="section-title-wrap">
                <span class="section-num">06</span>
                <div>
                    <h2 class="section-title">Kiểm Toán QA, Tiêu Chí Nghiệm Thu & Tái Lập</h2>
                    <p class="section-subtitle">Bảo Toàn Bằng Chứng Thực Nghiệm • 95/95 Pytest Passed (100%) • Checkpoint Reload Consistency</p>
                </div>
            </div>
            <span class="badge badge-green">Nghiệm Thu Đạt</span>
        </div>

        <div class="table-responsive">
            <table class="report-table">
                <thead>
                    <tr>
                        <th style="width: 70px; text-align: center;">Bước</th>
                        <th>Hạng Mục Nghiệm Thu (DoD Checklist)</th>
                        <th>Tiêu Chuẩn Ràng Buộc</th>
                        <th style="width: 180px; text-align: center;">Trạng Thái Thực Tế</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="text-align: center;"><span class="step-num-pill">1</span></td>
                        <td><strong>Exercise 1: SST-2 Pretrained Inference</strong></td>
                        <td>Độ lệch Pipeline vs Manual PyTorch Δ = 0.000000</td>
                        <td style="text-align: center;"><span class="badge badge-success">✓ 100% Khớp (Δ = 0)</span></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><span class="step-num-pill">2</span></td>
                        <td><strong>Exercise 2: Sanity 1-Batch Overfit</strong></td>
                        <td>Vượt qua overfit 1-batch cục bộ trong 5s trước khi train GPU</td>
                        <td style="text-align: center;"><span class="badge badge-success">✓ Passed (Loss → 0.001)</span></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><span class="step-num-pill">3</span></td>
                        <td><strong>Kiểm Toán Dữ Liệu & Data Leakage</strong></td>
                        <td>Tuyệt đối 0 NaN, 0 Null, 0 mẫu trùng lặp chéo giữa Train/Val/Test</td>
                        <td style="text-align: center;"><span class="badge badge-success">✓ 0 Leakage Verified</span></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><span class="step-num-pill">4</span></td>
                        <td><strong>Hiệu Năng Phân Loại So Với Baseline</strong></td>
                        <td>Vượt mốc Baseline TF-IDF + Logistic Regression (78.99% Acc)</td>
                        <td style="text-align: center;"><span class="badge badge-success">✓ 83.11% Acc (+4.12%)</span></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><span class="step-num-pill">5</span></td>
                        <td><strong>Tối Ưu Hóa Ngưỡng Quyết Định Nhị Phân</strong></td>
                        <td>Quét 19 ngưỡng trên Val, policy tối ưu 0.45 cải thiện Test F1</td>
                        <td style="text-align: center;"><span class="badge badge-success">✓ F1 83.30% (+0.60%)</span></td>
                    </tr>
                    <tr>
                        <td style="text-align: center;"><span class="step-num-pill">6</span></td>
                        <td><strong>Kiểm Thử Tự Động Toàn Diện</strong></td>
                        <td>95 bài kiểm tra chuyên sâu trong labs/lab03/tests/</td>
                        <td style="text-align: center;"><span class="badge badge-success">✓ 95/95 Passed (100%)</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </section>
    """

    # 3.4 Full HTML Document Assembly
    full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{LAB_CONFIG['title']} — {LAB_CONFIG['author']}</title>
    
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">

    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --accent: #2563eb;
            --accent-subtle: #eff6ff;
            --success: #059669;
            --success-subtle: #ecfdf5;
            --warning: #d97706;
            --border: #e2e8f0;
            --border-card: #e5e7eb;
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --text-dark: #0f172a;
            --text-main: #334155;
            --text-muted: #64748b;
            --font-sans: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', Consolas, Monaco, monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        html {{
            scroll-behavior: smooth;
            scroll-padding-top: 80px;
        }}

        body {{
            font-family: var(--font-sans);
            background-color: var(--bg-page);
            color: var(--text-main);
            line-height: 1.65;
            padding-bottom: 5rem;
        }}

        /* Header Banner */
        header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0369a1 100%);
            color: #ffffff;
            padding: 3rem 1.5rem 2.5rem;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        }}

        .header-inner {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header-top-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .school-brand {{
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            color: #93c5fd;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .author-tag {{
            font-size: 0.85rem;
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(8px);
            padding: 0.35rem 0.9rem;
            border-radius: 9999px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #f8fafc;
            font-weight: 500;
        }}

        h1 {{
            font-size: clamp(1.8rem, 3.5vw, 2.5rem);
            font-weight: 800;
            letter-spacing: -0.025em;
            margin-bottom: 0.5rem;
            line-height: 1.2;
            color: #ffffff;
        }}

        .header-sub {{
            font-size: 1.1rem;
            color: #cbd5e1;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }}

        .header-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.3rem 0.75rem;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            line-height: 1.4;
        }}

        .badge-glass {{
            background: rgba(255, 255, 255, 0.12);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}

        .badge-success {{
            background: #dcfce7;
            color: #15803d;
            border: 1px solid #86efac;
        }}

        .badge-blue {{
            background: #e0f2fe;
            color: #0369a1;
            border: 1px solid #bae6fd;
        }}

        .badge-green {{
            background: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
        }}

        .badge-pill {{
            font-family: var(--font-mono);
            font-size: 0.75rem;
            padding: 0.2rem 0.55rem;
            background: #f1f5f9;
            color: #1e293b;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
        }}

        /* Main Container */
        .main-container {{
            max-width: 1200px;
            margin: -1.75rem auto 0;
            padding: 0 1rem;
            position: relative;
            z-index: 10;
        }}

        /* Executive KPI Dashboard */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 1rem;
            margin-bottom: 1.75rem;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem 1.15rem;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.07);
        }}

        .kpi-label {{
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }}

        .kpi-val {{
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            line-height: 1.15;
        }}

        .kpi-sub {{
            font-size: 0.78rem;
            color: var(--text-muted);
            line-height: 1.35;
        }}

        /* Pipeline Stepper Navigation */
        .stepper-nav {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.75rem;
            margin-bottom: 2rem;
        }}

        .step-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.9rem 0.85rem;
            text-decoration: none;
            color: var(--text-dark);
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        }}

        .step-card:hover {{
            border-color: var(--accent);
            background: #f0f7ff;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(37, 99, 235, 0.08);
        }}

        .step-card-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .step-badge {{
            width: 26px;
            height: 26px;
            line-height: 26px;
            text-align: center;
            border-radius: 50%;
            background: var(--primary);
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 700;
        }}

        .step-arrow {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: bold;
        }}

        .step-title {{
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--text-dark);
            line-height: 1.3;
        }}

        .step-desc {{
            font-size: 0.75rem;
            color: var(--text-muted);
            line-height: 1.3;
        }}

        /* Content Sections */
        .content-section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 2rem 2.25rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
        }}

        .section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 1.25rem;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #f1f5f9;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .section-title-wrap {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .section-num {{
            font-size: 1.75rem;
            font-weight: 800;
            color: #cbd5e1;
            line-height: 1;
        }}

        .section-title {{
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text-dark);
            letter-spacing: -0.3px;
        }}

        .section-subtitle {{
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}

        /* Layout Grids */
        .grid-2col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
            gap: 1.25rem;
            margin: 1.25rem 0;
        }}

        /* Visual Cards */
        .gallery-card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            flex-direction: column;
        }}

        .gallery-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.08);
            border-color: #cbd5e1;
        }}

        .card-img-wrap {{
            position: relative;
            background: #fbfcfe;
            border-bottom: 1px solid #f1f5f9;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.75rem;
            cursor: pointer;
        }}

        .card-img-wrap img {{
            width: 100%;
            max-height: 380px;
            object-fit: contain;
            display: block;
        }}

        .hero-card .card-img-wrap img {{
            max-height: 520px;
        }}

        .zoom-badge {{
            position: absolute;
            bottom: 0.65rem;
            right: 0.65rem;
            background: rgba(15, 23, 42, 0.8);
            color: #ffffff;
            font-size: 0.72rem;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            pointer-events: none;
            opacity: 0.85;
            transition: opacity 0.2s ease;
        }}

        .card-img-wrap:hover .zoom-badge {{
            opacity: 1;
        }}

        .card-caption {{
            padding: 1.15rem 1.25rem;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }}

        .card-header-row {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.45rem;
        }}

        .card-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-dark);
            line-height: 1.35;
        }}

        .card-takeaway {{
            font-size: 0.85rem;
            color: var(--text-main);
            line-height: 1.5;
            margin-bottom: 0.75rem;
            flex-grow: 1;
        }}

        .card-footer {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.75rem;
            color: var(--text-muted);
            border-top: 1px dashed var(--border);
            padding-top: 0.55rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        /* Section Code Drawer */
        .section-code-drawer {{
            margin-top: 1.75rem;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #f8fafc;
            overflow: hidden;
        }}

        .section-code-drawer summary {{
            padding: 0.85rem 1.25rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            user-select: none;
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-dark);
            background: #f8fafc;
            transition: background 0.15s ease;
        }}

        .section-code-drawer summary:hover {{
            background: #f1f5f9;
        }}

        .section-code-drawer[open] summary {{
            background: #f1f5f9;
            border-bottom: 1px solid var(--border);
        }}

        .drawer-summary-left {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .code-tag {{
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 700;
            background: #1e3a8a;
            color: #ffffff;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
        }}

        .drawer-icon {{
            font-size: 0.8rem;
            color: var(--primary-light);
        }}

        .drawer-content {{
            padding: 0;
            background: #0f172a;
        }}

        .drawer-content pre {{
            margin: 0;
            padding: 1.25rem 1.5rem;
            font-size: 0.82rem;
            line-height: 1.6;
            background: transparent !important;
        }}

        /* Report Table */
        .table-responsive {{
            overflow-x: auto;
            margin-top: 1rem;
        }}

        .report-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        .report-table th, .report-table td {{
            padding: 0.85rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        .report-table th {{
            background: #f8fafc;
            font-weight: 700;
            color: var(--text-dark);
        }}

        .report-table tr:hover {{
            background: #f8fafc;
        }}

        .step-num-pill {{
            display: inline-block;
            width: 26px;
            height: 26px;
            line-height: 26px;
            text-align: center;
            border-radius: 50%;
            background: var(--primary);
            color: #ffffff;
            font-weight: 700;
            font-size: 0.8rem;
        }}

        /* Modal Lightbox */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(15, 23, 42, 0.92);
            backdrop-filter: blur(4px);
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
        }}

        .modal.active {{
            display: flex;
        }}

        .modal-content {{
            max-width: 92vw;
            max-height: 88vh;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            background: #ffffff;
        }}

        .modal-close {{
            position: absolute;
            top: 1.5rem;
            right: 2rem;
            color: #ffffff;
            font-size: 2.5rem;
            font-weight: 700;
            cursor: pointer;
            line-height: 1;
            transition: color 0.15s ease;
        }}

        .modal-close:hover {{
            color: #93c5fd;
        }}

        /* Footer */
        footer {{
            text-align: center;
            margin-top: 3rem;
            padding: 2rem 1rem 0;
            border-top: 1px solid var(--border);
            font-size: 0.85rem;
            color: var(--text-muted);
        }}

        @media (max-width: 768px) {{
            .grid-2col {{
                grid-template-columns: 1fr;
            }}
            .content-section {{
                padding: 1.5rem 1.25rem;
            }}
            .stepper-nav {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>

<!-- Header Banner -->
<header>
    <div class="header-inner">
        <div class="header-top-bar">
            <div class="school-brand">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>
                {LAB_CONFIG['school']}
            </div>
            <div class="author-tag">Tác giả: <strong>{LAB_CONFIG['author']}</strong></div>
        </div>

        <h1>{LAB_CONFIG['title']}</h1>
        <div class="header-sub">{LAB_CONFIG['subtitle']}</div>

        <div class="header-badges">
            <span class="badge badge-glass">{LAB_CONFIG['python_ver']}</span>
            <span class="badge badge-glass">{LAB_CONFIG['framework_ver']}</span>
            <span class="badge badge-glass">{LAB_CONFIG['dataset_info']}</span>
            <span class="badge badge-success">✓ {LAB_CONFIG['test_passed']}</span>
            <span class="badge badge-blue">{LAB_CONFIG['primary_metric']}</span>
        </div>
    </div>
</header>

<!-- Main Container -->
<main class="main-container">
    <!-- Executive KPI Dashboard -->
    <div class="kpi-grid">
        {kpis_html}
    </div>

    <!-- Pipeline Stepper Navigation -->
    <nav class="stepper-nav" aria-label="Pipeline Roadmap">
        {stepper_html}
    </nav>

    <!-- Sections Content -->
    {sec1_html}
    {sec2_html}
    {sec3_html}
    {sec4_html}
    {sec5_html}
    {sec6_html}

    <!-- Footer -->
    <footer>
        <p>Báo Cáo Nghiệm Thu Kỹ Thuật Deep Learning — {LAB_CONFIG['school']}</p>
        <p style="margin-top: 0.35rem;">Phiên bản đã kiểm chứng trên GPU Kaggle Tesla T4 • Nguồn dữ liệu & artifact: <code>outputs/published/</code></p>
    </footer>
</main>

<!-- Lightbox Modal -->
<div id="imageModal" class="modal" onclick="closeModal()">
    <span class="modal-close">&times;</span>
    <img class="modal-content" id="modalImg" alt="Phóng to ảnh">
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<script>
function openModal(el) {{
    const img = el.querySelector('img');
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    modalImg.src = img.src;
    modal.classList.add('active');
}}

function closeModal() {{
    document.getElementById('imageModal').classList.remove('active');
}}

document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') {{
        closeModal();
    }}
}});
</script>
</body>
</html>
"""
    return full_html


def main():
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "walkthrough_lab_03.html"
    
    html = build_html()
    out_file.write_text(html, encoding="utf-8")
    print(f"Generated clean, executive HTML report: {out_file} ({len(html.encode('utf-8')) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
