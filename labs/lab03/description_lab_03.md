# Description — Lab 03: Getting Started with Hugging Face & Fine-Tuning DistilBERT

- **Tác giả:** Thái Anh Lạc
- **Đơn vị:** TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH)
- **Môn học:** Deep Learning (Lab Thực Hành)

---

## 1. Đề bài gốc & Phân định rạch ròi hai bài tập (Exercise 1 vs Exercise 2)

Bài thực hành Lab 03 được cấu trúc thành **hai bài tập kỹ thuật độc lập với phạm vi, mục tiêu và phương pháp luận hoàn toàn rạch ròi**:

### 1.1. Exercise 1: Sentiment Analysis with Hugging Face (Pretrained & Pipeline Inference)
- **Bối cảnh & Mục tiêu:** Làm quen với hệ sinh thái Hugging Face Transformers, hiểu sâu cơ chế Tokenizer WordPiece, biểu diễn vector ngữ cảnh và quy trình suy luận (Inference) không qua huấn luyện lại (Zero-shot / Off-the-shelf).
- **Yêu cầu kỹ thuật cốt lõi:**
  1. Cài đặt và cấu hình thư viện `transformers`, `torch`.
  2. Nạp mô hình phân loại cảm xúc pretrained chính thức từ Hugging Face Hub: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
  3. Tokenize các câu văn bản tiếng Anh mẫu: phân rã thành Subwords/Tokens, ánh xạ sang Token IDs, gán cặp token biên đặc biệt `[CLS]`, `[SEP]` và thiết lập Attention Mask.
  4. Thực hiện dự đoán cảm xúc bằng đường ống cấp cao `transformers.pipeline("sentiment-analysis")`.
  5. **Yêu cầu đối chứng chuyên sâu (Verification):** Tự triển khai quy trình suy luận thủ công bằng PyTorch thuần (`model(**inputs) -> logits -> torch.softmax`) để đối chiếu xác suất với `transformers.pipeline`, chứng minh độ lệch tuyệt đối $\Delta = 0.000000$.
- **Sản phẩm đầu ra của Exercise 1:**
  - Bảng tra cứu Tokenizer: WordPiece subwords, Token IDs, Attention Mask (`tokenizer_table.png`).
  - Biểu đồ đối chiếu Pipeline vs Manual Inference xác nhận độ lệch $\Delta = 0$ (`pipeline_manual_comparison.png`).
  - Biểu đồ phân bố xác suất dự đoán trên các câu văn bản phức tạp có cấu trúc tương phản (`sentiment_predictions.png`).

---

### 1.2. Exercise 2: Fine-Tuning DistilBERT for Binary Text Classification
- **Bối cảnh & Mục tiêu:** Nắm vững quy trình Transfer Learning và Fine-tuning mô hình Transformer từ đầu đến cuối trên một bài toán phân loại văn bản nhị phân thực tế, thiết lập baseline so sánh và tối ưu hóa ngưỡng quyết định.
- **Yêu cầu kỹ thuật cốt lõi:**
  1. Tải và kiểm toán toàn vẹn tập dữ liệu phân loại nhị phân chính thức: **Rotten Tomatoes** (10.662 đánh giá phim, cân bằng 50/50 giữa Positive và Negative).
  2. Nạp mô hình nền tảng chưa fine-tune: `distilbert/distilbert-base-uncased` và thay thế Classification Head 2 logits (`Linear(768, 768) -> Pre-classifier -> Dropout(0.2/0.4) -> Linear(768, 2)`).
  3. Tiền xử lý dữ liệu: phân tích phân bố độ dài token, xác lập ngưỡng cắt ngắn an toàn `max_length = 64` (bao phủ 99.4% ngữ cảnh) kết hợp Dynamic Padding.
  4. Xây dựng mô hình Baseline truyền thống: TF-IDF (5.000 unigrams/bigrams) kết hợp Logistic Regression để làm mốc neo so sánh hiệu năng.
  5. Thiết lập quy trình huấn luyện Fine-tuning:
     - Chiến lược đóng băng/giải phóng: Khóa Embeddings và 4 Transformer Block đầu, chỉ mở 2 Block cuối (Encoder 5 & 6) cùng Classification Head (14.767.874 tham số trainable).
     - Bộ tối ưu: AdamW với Discriminative Learning Rates ($5\times 10^{-6}$ cho Encoder, $2\times 10^{-5}$ cho Head) và Cosine Decay Scheduler.
     - Cơ chế chốt chặn: Kiểm tra Sanity Overfit 1 batch trước khi chạy full training; Early Stopping với `patience = 2` trên Validation Loss để chống Overfitting.
  6. Tối ưu hóa ngưỡng quyết định (Threshold Tuning): Quét 19 ngưỡng trên tập Validation để tìm operating point tối đa hóa F1-Score (chọn ngưỡng tối ưu `0.45`).
  7. Đánh giá toàn diện trên tập Test độc lập: Tính toán đầy đủ bộ chỉ số (Accuracy, Macro F1, Binary F1, Precision, Recall, Confusion Matrix, ROC-AUC, PR-AUC, Calibration Diagram, Per-class TPR/FPR và Phân tích lỗi ngữ nghĩa).
- **Sản phẩm đầu ra của Exercise 2:**
  - Bộ artifact EDA và Feature Engineering (14 hình ảnh và bảng thống kê).
  - Checkpoint mô hình tối ưu `best.pt` cùng báo cáo huấn luyện 5 epochs (`learning_curves.png`, `performance.png`).
  - Ma trận nhầm lẫn chuẩn hóa, đường cong ROC/PR và lưới dự đoán mẫu thực tế.

---

### 1.3. Bảng phân định ranh giới kỹ thuật rạch ròi giữa Exercise 1 và Exercise 2

| Tiêu chí phân định | Exercise 1: Sentiment Inference | Exercise 2: DistilBERT Fine-Tuning |
|---|---|---|
| **Mục đích chính** | Khảo sát cơ chế Tokenizer & Pretrained Pipeline | Huấn luyện thích nghi Transfer Learning toàn diện |
| **Mô hình sử dụng** | `distilbert-base-uncased-finetuned-sst-2-english` | `distilbert-base-uncased` (Head 2 lớp khởi tạo mới) |
| **Trạng thái trọng số** | Đóng băng 100% (Zero-shot / Pre-trained weights) | Đóng băng 4 block đầu; Fine-tune 2 block cuối + Head |
| **Số tham số học (Trainable)**| **0** (Không cập nhật gradient) | **14.767.874** / 66.955.010 tham số (~22.1%) |
| **Tập dữ liệu** | 3 câu văn bản mẫu thực tế có ngữ cảnh đa dạng | Toàn bộ 10.662 mẫu Rotten Tomatoes (Train/Val/Test) |
| **Quá trình huấn luyện** | Không huấn luyện (Chỉ chạy 1 lượt Forward Pass) | Huấn luyện 5 epochs AdamW, Early Stopping, Baseline |
| **Ngưỡng quyết định** | Cố định mặc định 0.50 | Quét 19 ngưỡng trên Val, chọn tối ưu **0.45** |
| **Mô-đun triển khai** | [`src/analysis.py`](src/analysis.py) | [`src/datasets/`](src/datasets/), [`src/training/`](src/training/), [`src/models/`](src/models/) |
| **Vị trí trong Notebooks** | [03_modeling.ipynb](notebooks/03_modeling.ipynb) (Mục 2) | [01_eda.ipynb](notebooks/01_eda.ipynb), [02_fe.ipynb](notebooks/02_feature_engineering.ipynb), [03_modeling.ipynb](notebooks/03_modeling.ipynb) (Mục 3-8) |
| **Thư mục Artifacts** | `outputs/published/exercise1_analysis/` (3 ảnh) | `outputs/published/frozen_trial/` (18 ảnh + báo cáo) |

---

## 2. Phạm vi và kiến trúc triển khai

Dự án được phân rã thành **3 Jupyter Notebooks chuyên biệt**, vận hành theo nguyên tắc Clean Architecture (Code nặng nằm trong `src/`, notebook đóng vai trò giao diện trực quan hóa và kiểm chứng):

| Notebook | Nhiệm vụ chính | Bài tập phục vụ | Đầu ra & Sản phẩm cốt lõi |
|---|---|:---:|---|
| **`01_eda.ipynb`** | Khám phá dữ liệu 6 bước (`docs/eda.md`): Schema, 5-number summary, phân bố nhãn, độ dài từ/ký tự, kiểm toán 0 NaN và 0 Data Leakage. | **Exercise 2** | `class_distribution.png`, `data_quality.png`, `lengths_boxplot.png`, `lengths_by_class.png`. |
| **`02_feature_engineering.ipynb`** | Kỹ nghệ đặc trưng 6 bước (`docs/feature.md`): Token length cutoff, Attention masks, 8 đặc trưng ngôn ngữ học, TF-IDF 5.000 unigram/bigram, Chi-Square ranking và Text Augmentation. | **Exercise 2** | `token_length_distribution.png`, `tfidf_features.png`, `chi2_features.png`, `augmentation_diagnostics.png`. |
| **`03_modeling.ipynb`** | **Phần A (Mục 2):** Pretrained Sentiment Inference (Exercise 1).<br>**Phần B (Mục 3-8):** Baseline TF-IDF, Fine-tuning DistilBERT, Benchmark phần cứng, Threshold Sweep và Test Evaluation (Exercise 2). | **Exercise 1 & Exercise 2** | `pipeline_manual_comparison.png`, `learning_curves.png`, `performance.png`, `confusion_matrix.png`, `roc_pr_curves.png`. |

Mọi Jupyter Notebook đều hỗ trợ chế độ kép: mặc định `MODE="view"` để đọc offline các artifact đã được kiểm chứng mà không làm tốn tài nguyên GPU, và `MODE="train"` (hoặc `prepare`) khi cần huấn luyện mới.

---

## 3. Data Contract & Hợp đồng dữ liệu (Áp dụng cho Exercise 2)

- **Dataset:** Rotten Tomatoes movie reviews (nguồn chính thức Hugging Face Datasets, mã revision cố định).
- **Quy mô phân vùng:**
  - `Train`: 8.530 mẫu (4.265 Positive, 4.265 Negative).
  - `Validation`: 1.066 mẫu (533 Positive, 533 Negative).
  - `Test`: 1.066 mẫu (533 Positive, 533 Negative).
- **Data Integrity Contract:**
  - Tuyệt đối 0 mẫu khuyết thiếu (0 NaN, 0 Null, 0 chuỗi rỗng).
  - Tuyệt đối 0 mẫu trùng lặp chéo giữa Train với Validation hoặc Test (0 Data Leakage).
- **Tensor Contract:**
  - `input_ids`: Kích thước `[Batch, Length]`, kiểu `torch.int64`, giá trị nằm trong khoảng `[0, 30521]`.
  - `attention_mask`: Kích thước `[Batch, Length]`, kiểu `torch.int64`, giá trị `1` (token thật) hoặc `0` (padding).
  - `labels`: Kích thước `[Batch]`, kiểu `torch.int64`, nhãn nhị phân `{0, 1}`.
  - `logits`: Kích thước `[Batch, 2]`, kiểu `torch.float32`.

---

## 4. Kiến trúc mô hình chi tiết

### 4.1. Kiến trúc Exercise 1 (Pretrained Inference)
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`.
- Gồm 6 Transformer Encoder layers đã được fine-tune hoàn chỉnh trên tập Stanford Sentiment Treebank (SST-2).
- Đầu ra gồm 2 logits tương ứng với hai lớp NEGATIVE (index 0) và POSITIVE (index 1).

### 4.2. Kiến trúc Exercise 2 (Fine-Tuning DistilBERT)
- **Backbone:** `distilbert-base-uncased` (6 Transformer Encoder layers, 768 hidden dimension, 12 attention heads, FFN 3072 dimension).
- **Classification Head (Khởi tạo mới):**
  $$\text{Input [B, 768]} \rightarrow \text{Linear}(768, 768) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow \text{Linear}(768, 2) \rightarrow \text{Logits [B, 2]}$$
- **Phân bổ tham số (Parameter Allocation):**
  - Tổng số tham số: **66.955.010** tham số.
  - Tham số đóng băng (Frozen): **52.187.136** tham số (Embeddings + Encoder 1 đến 4).
  - Tham số được huấn luyện (Trainable): **14.767.874** tham số (~22.1%, gồm Encoder 5, Encoder 6 và Classification Head).

---

## 5. Thảo luận kỹ thuật & Quyết định thực nghiệm

### 5.1. Quyết định về Tiền xử lý & Tokenization
- **Chiến lược Truncation Cutoff (`max_length = 64`):**
  Phân tích EDA cho thấy 99.4% số câu trong tập Rotten Tomatoes có độ dài dưới 64 tokens. Việc đặt `max_length = 64` thay vì 128 giúp giảm 75% khối lượng tính toán cơ chế tự chú ý (Attention complexity $O(N^2)$), tăng thông lượng huấn luyện lên gấp đôi trên GPU mà bảo toàn được 99.4% ngữ cảnh.
- **Bảo toàn từ chỉ thị ngữ nghĩa:** Không loại bỏ stopwords và dấu câu vì các từ phủ định (*not, no, never*) và từ tương phản (*but, however*) chứa tín hiệu cảm xúc quyết định.

### 5.2. Quyết định về Tăng cường dữ liệu (Text Augmentation)
- Áp dụng thay thế từ đồng nghĩa theo ngữ cảnh có ràng buộc nghiêm ngặt: bảo vệ 100% các từ thuộc tập `PROTECTED` (từ phủ định và liên từ tương phản), đảm bảo 96.8% mẫu tăng cường giữ nguyên vẹn nhãn gốc.
- Chỉ thực hiện trên tập Train; tuyệt đối không áp dụng lên Validation và Test.

### 5.3. Quyết định về Huấn luyện & Tối ưu hóa
- **Mô hình Baseline đối chứng:** Xây dựng mô hình TF-IDF + Logistic Regression (đạt 78.99% Test Acc) để làm mốc neo định lượng giá trị gia tăng của mô hình Deep Learning.
- **Tối ưu hóa:** Sử dụng AdamW với Discriminative Learning Rates ($5\times 10^{-6}$ cho Encoder, $2\times 10^{-5}$ cho Head).
- **Early Stopping:** Dừng huấn luyện khi Validation Loss không cải thiện sau `patience = 2` epochs để ngăn chặn hiện tượng quá khớp (Overfitting).

### 5.4. Quyết định về Ngưỡng quyết định nhị phân (Threshold Tuning)
- Mặc định ngưỡng phân loại nhị phân là 0.50. Chúng tôi tiến hành quét 19 ngưỡng từ 0.05 đến 0.95 trên tập Validation.
- Tại ngưỡng **0.45**, chỉ số F1-Score trên tập Validation đạt đỉnh cao nhất (83.82%). Áp dụng ngưỡng 0.45 lên tập Test giúp Test Binary F1 tăng từ 82.70% lên **83.30%** (+0.60% cải thiện thực chất).

---

## 6. Tiêu chí nghiệm thu & Kết quả bàn giao đã kiểm chứng

### 6.1. Nghiệm thu Exercise 1
- [x] Nạp thành công pipeline sentiment analysis và suy luận chính xác trên 3 câu văn bản mẫu.
- [x] Khớp xác suất 100% giữa pipeline cấp cao và suy luận thủ công PyTorch ($\text{max\_delta} = 0.000000$).
- [x] Trực quan hóa đầy đủ bảng Tokenizer và phân bố xác suất dự đoán.

### 6.2. Nghiệm thu Exercise 2
- [x] Vượt qua bài kiểm tra Sanity 1-batch overfit cục bộ trong vòng 5 giây trước khi huấn luyện từ xa.
- [x] Hoàn thành huấn luyện 5 epochs trên GPU Kaggle P100 với thời gian 34.2 giây/epoch (bộ nhớ VRAM chỉ 1.84 GB).
- [x] Mô hình DistilBERT fine-tuned đạt **83.11% Test Accuracy** và **83.30% Test Binary F1** (vượt trội hoàn toàn so với Baseline TF-IDF đạt 78.99% Acc).
- [x] Chỉ số phân tách xuất sắc: **ROC-AUC = 0.908** và **PR-AUC = 0.902**.
- [x] Hoàn thành phân tích ma trận nhầm lẫn chuẩn hóa (886/1.066 mẫu đúng), hiệu chuẩn độ tin cậy xác suất ($ECE = 0.038$) và phân tích lỗi định tính.

### 6.3. Báo cáo & Kiểm thử tự động
- [x] Toàn bộ 95 bài kiểm tra trong `labs/lab03/tests/` đạt **95/95 passed (100%)**.
- [x] Báo cáo kỹ thuật tổng hợp [reports/walkthrough_lab_03.html](reports/walkthrough_lab_03.html) tuân thủ tiêu chuẩn visual-first UTH, tích hợp 24 Code Drawers tương tác dưới chân từng ảnh.

---

## 7. Quan hệ tài liệu sống (Single Source of Truth)

- **Báo cáo kỹ thuật sống:** [labs/lab03/reports/walkthrough_lab_03.md](reports/walkthrough_lab_03.md) và [labs/lab03/reports/walkthrough_lab_03.html](reports/walkthrough_lab_03.html) là nguồn tài liệu duy nhất trình bày toàn bộ số liệu, biểu đồ và kết luận khoa học.
- **Tài liệu hướng dẫn vận hành:** [labs/lab03/README.md](README.md) hướng dẫn cài đặt, chạy notebook ở chế độ `MODE="view"` và tái lập kết quả.
- **Thảo luận kỹ thuật chuyên sâu:** Phân bổ tại các submodule README: [src/datasets/README.md](src/datasets/README.md), [src/models/README.md](src/models/README.md), [src/training/README.md](src/training/README.md), [src/utils/README.md](src/utils/README.md), [tests/README.md](tests/README.md).
