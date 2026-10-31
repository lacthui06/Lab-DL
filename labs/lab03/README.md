# Lab 03: Rotten Tomatoes Sentiment Analysis with DistilBERT & Hugging Face

Dự án phân loại văn bản cảm xúc nhị phân trên toàn bộ tập dữ liệu Rotten Tomatoes (8.530 train / 1.066 validation / 1.066 test) áp dụng mô hình Transformer tiền huấn luyện DistilBERT, kết hợp chiến lược Freeze Backbone, Discriminative Learning Rates, Regularization và tối ưu hóa ngưỡng quyết định (Operating Threshold Tuning).

Kiến trúc dự án tuân thủ nghiêm ngặt chuẩn modular hóa, tách bạch giữa cấu hình (`config/`), thư viện tái sử dụng (`src/`), công cụ thực thi dòng lệnh (`scripts/`), kiểm thử tự động (`tests/`) và sổ tay nghiên cứu trực quan (`notebooks/`).

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
labs/lab03/
├── config/                             # Cấu hình tham số và kịch bản thí nghiệm
│   ├── config.json                     # Cấu hình cơ bản (Baseline & Standard fine-tuning)
│   ├── frozen_trial.json               # Cấu hình chính (Discriminative LR, Freeze 4 layers, Dropout 0.4)
│   ├── augmented.json                  # Cấu hình thử nghiệm Text Augmentation
│   ├── report.json                     # Nguồn dữ liệu cố định cho báo cáo tổng kết
│   └── workflow.json                   # Sổ đăng ký trạng thái và active run runtime
├── notebooks/                          # Sổ tay nghiên cứu trực quan
│   ├── 01_eda_preprocessing_feature_engineering.ipynb   # Phân tích dữ liệu, độ dài token, từ điển và augmentation
│   └── 02_modeling.ipynb               # Exercise 1 Pretrained, Baseline, Frozen DistilBERT & Đánh giá
├── outputs/                            # Artifacts sinh ra từ các pipeline
│   ├── published/                      # Toàn bộ biểu đồ PNG và bảng JSON được chọn lọc cho báo cáo
│   │   ├── final_review/               # Confusion matrix, Learning curves, ROC-PR curves
│   │   ├── epoch_presentation/         # Biểu đồ diễn tiến 10 epochs
│   │   └── index.md                    # Danh mục artifacts tra cứu nhanh
│   ├── runs/                           # Dữ liệu thực nghiệm chi tiết của từng run
│   └── preflight/                      # Kết quả kiểm tra môi trường và sanity test
├── src/                                # Gói thư viện mã nguồn lõi (Core Python Package)
│   ├── datasets/                       # Quản lý nạp dữ liệu, tokenizer và augmentation
│   │   ├── loader.py                   # Nạp official splits và WordPiece dynamic padding
│   │   └── augmentation.py             # Kỹ thuật mở rộng văn bản an toàn không rò rỉ
│   ├── models/                         # Kiến trúc mô hình Transformer
│   │   ├── classifier.py               # Hugging Face AutoModel, classification head và manual inference
│   │   └── reloading.py                # Kiểm chứng tải lại checkpoint và tính toàn vẹn logits
│   ├── training/                       # Động cơ huấn luyện và tối ưu hóa
│   │   ├── trainer.py                  # PyTorch / Hugging Face Trainer, callbacks và 1-batch sanity check
│   │   ├── thresholds.py               # Quét 19 ngưỡng quyết định (Operating Thresholds) trên Validation
│   │   ├── metrics.py                  # Tính toán Accuracy, Binary F1, Macro F1, Precision, Recall
│   │   └── frozen.py                   # Đóng băng Embedding + 4/6 Encoder layers
│   └── utils/                          # Công cụ bổ trợ
│       ├── runs.py                     # Quản lý thư mục run, manifest SHA256 và snapshot môi trường
│       └── plots.py                    # Module visualization tập trung cho toàn bộ biểu đồ
├── scripts/                            # Điểm thực thi dòng lệnh (CLI Entry Points)
│   ├── run_frozen_trial.py             # Script chạy huấn luyện mô hình chính (10 epochs)
│   ├── run_exercise1.py                # Chạy minh họa Exercise 1 sentiment pretrained
│   ├── eval.py                         # Đánh giá checkpoint mô hình trên tập Test
│   └── infer.py                        # Dự đoán nhãn cảm xúc cho văn bản tùy biến
├── tests/                              # Bộ kiểm thử tự động toàn diện (112 test cases)
│   ├── test_datasets.py                # Kiểm tra phân chia dữ liệu, không rò rỉ ranh giới
│   ├── test_models.py                  # Kiểm tra cấu trúc tham số, head và logits
│   ├── test_training.py                # Kiểm tra sanity overfit và chu trình huấn luyện
│   ├── test_thresholds.py              # Kiểm tra chính sách quét ngưỡng trên validation
│   └── test_augmentation.py            # Kiểm tra an toàn dữ liệu augmentation
├── requirements.txt                    # Danh sách thư viện phụ thuộc đã kiểm chứng
├── description_lab_03.md               # Diễn giải đề bài, yêu cầu và quyết định kỹ thuật
├── walkthrough_lab_03.md               # Báo cáo kỹ thuật sống duy nhất (Spec, Changelog, Results)
└── README.md                           # Tài liệu hướng dẫn sử dụng này
```

---

## 2. Điểm Nhấn Kỹ Thuật & Quyết Định Thiết Kế

- **Bảo toàn Hợp đồng Dữ liệu (Strict Official Data Contract):**
  - Giữ nguyên toàn bộ 10.662 mẫu chính thức của Rotten Tomatoes chia theo 3 tập: 8.530 train, 1.066 validation, 1.066 test. Tuyệt đối không xáo trộn nhãn hoặc chia lại tập kiểm thử.
- **Chiến lược Đóng băng Layer & Discriminative Learning Rates:**
  - Để tránh phá vỡ các đặc trưng biểu diễn ngữ nghĩa đã học của Transformer, toàn bộ Embedding và 4 tầng Encoder đầu tiên được đóng băng (`requires_grad = False`).
  - Phân tầng tốc độ học: Áp dụng $2 \times 10^{-6}$ cho 2 tầng Encoder còn lại và $1 \times 10^{-5}$ cho Classification Head mới.
- **Tối ưu hóa Điểm Vận hành (Operating Threshold Tuning):**
  - Quét 19 ngưỡng xác suất (từ 0.05 đến 0.95, bước 0.05) **độc quyền trên tập Validation**.
  - Ngưỡng tối ưu $\tau = 0.45$ (hoặc 0.50 tùy kịch bản) được khóa cứng trước khi suy luận trên tập Test, đảm bảo đánh đổi tối ưu giữa False Positive và False Negative mà không rò rỉ dữ liệu Test.
- **Chống Overfitting & Đánh giá Đa diện:**
  - Không lựa chọn mô hình dựa trên đỉnh nhọn ngẫu nhiên của Test Accuracy. Lựa chọn checkpoint dựa trên **Validation Loss thấp nhất** (Epoch 9) kết hợp đo đạc toàn diện bộ chỉ số: Accuracy (82.83%), Macro-F1 (82.82%), Binary F1 (83.23%).

---

## 3. Hướng Dẫn Thực Thi

### Cách 1: Sử Dụng Công Cụ Dòng Lệnh (CLI)

```powershell
# 1. Chạy Exercise 1: Kiểm chứng mô hình Pretrained Sentiment Analysis
python labs/lab03/scripts/run_exercise1.py

# 2. Huấn luyện mô hình DistilBERT (10 epochs với Frozen Layers và Discriminative LR)
python labs/lab03/scripts/run_frozen_trial.py --config labs/lab03/config/frozen_trial.json --device auto

# 3. Đánh giá mô hình đã huấn luyện trên tập Test
python labs/lab03/scripts/eval.py --config labs/lab03/config/frozen_trial.json

# 4. Suy luận phân loại cảm xúc trên một câu văn bản bất kỳ
python labs/lab03/scripts/infer.py --text "This movie is an absolute masterpiece of modern cinema!"
```

### Cách 2: Sử Dụng Jupyter Notebook

Mở các sổ tay nghiên cứu trong thư mục `labs/lab03/notebooks/`:
- [`01_eda_preprocessing_feature_engineering.ipynb`](notebooks/01_eda_preprocessing_feature_engineering.ipynb): Khám phá phân bố từ vựng, độ dài token và cơ chế Text Augmentation.
- [`02_modeling.ipynb`](notebooks/02_modeling.ipynb): Toàn bộ pipeline mô hình hóa từ Pretrained Sentiment, Frozen DistilBERT, huấn luyện 10 epochs và chẩn đoán lỗi phân loại trên tập kiểm thử.

---

## 4. Kiểm Thử Tự Động (TDD Test Suite)

Chạy toàn bộ 112 bài kiểm thử tự động để xác nhận tính toàn vẹn của mã nguồn, luồng huấn luyện và các cổng kiểm soát rò rỉ:

```powershell
pytest labs/lab03/tests/ -q
```
