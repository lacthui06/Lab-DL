# Lab 01: FashionMNIST Image Classification with PyTorch

Dự án phân loại hình ảnh 10 lớp trang phục FashionMNIST áp dụng kiến trúc sạch (Clean Architecture), tách bạch rõ ràng giữa cấu hình (`config/`), mã nguồn thư viện (`src/`), công cụ thực thi dòng lệnh (`scripts/`), kiểm thử tự động (`tests/`) và bảng điều khiển trực quan (`notebooks/`).

---

## 1. Cấu Trúc Thư Mục Chuẩn Hóa

```text
labs/lab01/
├── config/
│   ├── config.yaml                     # Siêu tham số mô hình, training, optimizer
│   └── augmentations.yaml              # Quy tắc tăng cường ảnh (Flip, Rotate, Crop)
├── data/                               # Dữ liệu FashionMNIST (tự động tải)
├── outputs/                            # Sản phẩm sinh ra sau khi chạy
│   ├── checkpoints/                    # Trọng số mô hình (.pt)
│   ├── logs/                           # Nhật ký huấn luyện và lịch sử số liệu
│   ├── predictions/                    # Biểu đồ đánh giá: Confusion Matrix, ROC-AUC, Prediction Grid
│   └── eda/                            # Dữ liệu thống kê và 3 biểu đồ phân tích EDA
├── src/                                # Thư viện logic lõi tái sử dụng
│   ├── datasets/                       # Base dataset và DataLoader pipeline
│   ├── models/                         # Kiến trúc FashionMLP thuần túy
│   ├── training/                       # Class Trainer đóng gói vòng lặp và scheduler
│   └── utils/                          # Công cụ hỗ trợ: helpers, metrics, plots
├── scripts/                            # Điểm thực thi từ dòng lệnh (CLI Entry Points)
│   ├── train.py                        # Huấn luyện mô hình từ terminal
│   ├── eval.py                         # Đánh giá checkpoint trên tập test
│   └── infer.py                        # Dự đoán nhãn cho ảnh mẫu
├── tests/                              # Bộ kiểm thử tự động (TDD)
│   ├── test_datasets.py                # Kiểm tra tính toàn vẹn dữ liệu
│   ├── test_models.py                  # Kiểm tra chiều tensor và tham số
│   └── test_training.py                # Kiểm tra 1-batch sanity overfit bắt buộc
├── notebooks/
│   └── lab01.ipynb                     # Dashboard trực quan hóa và nộp bài
└── requirements.txt                    # Danh sách thư viện phụ thuộc
```

---

## 2. Hướng Dẫn Thực Thi

### Cách 1: Chạy Bằng Dòng Lệnh (CLI Scripts)
Phù hợp khi chạy trên máy chủ, server Linux hoặc chạy nền:

```bash
# 1. Huấn luyện mô hình (sử dụng cấu hình trong config/config.yaml)
python scripts/train.py --config config/config.yaml

# 2. Đánh giá mô hình trên tập test
python scripts/eval.py --config config/config.yaml --checkpoint outputs/checkpoints/fashion_mlp_best.pt

# 3. Dự đoán nhãn cho mẫu ảnh kiểm thử
python scripts/infer.py --sample_idx 0
```

### Cách 2: Chạy Bằng Jupyter Notebook
Phù hợp khi làm báo cáo, trực quan hóa đồ thị và nộp bài:
- Mở file `notebooks/lab01.ipynb` trên VS Code, Jupyter Lab hoặc Kaggle.
- Bấm **"Run All"** để chạy toàn bộ quy trình: từ EDA, Huấn luyện, So sánh 3 Optimizer bằng Optuna, đến Đóng gói file nén `all_outputs.zip`.

---

## 3. Kiểm Thử Tự Động (TDD Tests)
Chạy toàn bộ 30 bài test kiểm tra chất lượng mã nguồn:

```bash
pytest tests/ -v
```
