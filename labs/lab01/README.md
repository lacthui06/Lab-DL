# Lab 01: FashionMNIST Image Classification with PyTorch

Dự án phân loại hình ảnh 10 lớp trang phục FashionMNIST áp dụng kiến trúc sạch (Clean Computer Vision Architecture theo chuẩn Gaurav Goyal), tách bạch rõ ràng giữa cấu hình (`config/`), thư viện tái sử dụng (`src/`), công cụ thực thi dòng lệnh (`scripts/`), kiểm thử tự động (`tests/`) và sổ tay nghiên cứu trực quan (`notebooks/`).

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
labs/lab01/
├── config/
│   ├── config.yaml                     # Cấu hình chính: dataset, kiến trúc MLP, training params
│   ├── augmentations.yaml              # Quy tắc tăng cường ảnh (Flip, Rotation)
│   └── experiments.yaml                # Cấu hình không gian tìm kiếm Optuna và so sánh Optimizer
├── data/                               # Dữ liệu thô FashionMNIST (tự động tải qua torchvision)
├── outputs/                            # Artifacts sinh ra từ các pipeline
│   ├── checkpoints/                    # Trọng số mô hình tốt nhất (fashion_mlp_best.pt)
│   ├── logs/                           # training_history.json, environment_snapshot.json
│   ├── predictions/                    # Đồ thị đánh giá: learning_curves.png, confusion_matrix.png,
│   │                                   # roc_curves.png, threshold_scan.png, prediction_grid.png
│   └── eda/                            # Artifacts chuẩn hóa (split_indices.npz, preprocessing_config.json)
│       └── figures/                    # 3 biểu đồ EDA: sample_images, class_mean_images, class_distribution
├── src/                                # Gói thư viện lõi (Core Python Package)
│   ├── __init__.py                     # Re-export API hợp nhất, tương thích ngược 100%
│   ├── datasets/                       # Quản lý dữ liệu và tiền xử lý
│   │   ├── __init__.py
│   │   ├── base.py                     # TransformedSubset và phân chia Stratified Split
│   │   ├── classification.py           # Dataset, Transforms pipeline và DataLoader factory
│   │   └── eda.py                      # Đo đạc thống kê, quét dữ liệu hỏng, khóa artifact EDA
│   ├── models/                         # Thiết kế kiến trúc mạng nơ-ron
│   │   ├── __init__.py
│   │   └── mlp.py                      # Mô hình FashionMLP (3 hidden layers, Dropout, BatchNorm)
│   ├── training/                       # Động cơ huấn luyện và tối ưu hóa
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Class Trainer, train loop, evaluate, 1-batch sanity check
│   │   └── experiments.py              # Optuna study, so sánh 3 optimizer, ROC-AUC, threshold sweep
│   └── utils/                          # Công cụ bổ trợ
│       ├── __init__.py
│       ├── helpers.py                  # Định tuyến đường dẫn động, device CUDA/CPU, seed, checkpoint
│       ├── metrics.py                  # Tính TPR, FPR, F1-Score, Precision, Recall theo từng lớp
│       `── plots.py                    # Module visualization duy nhất cho toàn bộ biểu đồ
├── scripts/                            # Điểm thực thi dòng lệnh (CLI Entry Points)
│   ├── train.py                        # Huấn luyện mô hình từ terminal
│   ├── eval.py                         # Đánh giá checkpoint mô hình trên tập test
│   └── infer.py                        # Dự đoán nhãn cho ảnh mẫu cụ thể
├── tests/                              # Bộ kiểm thử đơn vị và tích hợp (TDD)
│   ├── test_datasets.py                # Kiểm tra tính toàn vẹn dữ liệu, chống rò rỉ ranh giới
│   ├── test_models.py                  # Kiểm tra số lượng tham số và tensor output shape
│   └── test_training.py                # Kiểm tra 1-batch sanity overfit, checkpoint roundtrip, metric
├── notebooks/
│   └── lab01.ipynb                     # Sổ tay tương tác hoàn chỉnh từ EDA đến xuất báo cáo
├── requirements.txt                    # Danh sách các thư viện phụ thuộc đã kiểm chứng
└── README.md                           # Tài liệu hướng dẫn sử dụng này
```

---

## 2. Điểm Nhấn Kiến Trúc & Lý Thuyết

- **Chống rò rỉ dữ liệu (Anti-leakage):** Thống kê chuẩn hóa (`mean = 0.286209`, `std = 0.353160`) được đo đạc nghiêm ngặt độc quyền trên tập Train thông qua phân chia Stratified Split cố định (`seed=42`), sau đó khóa cứng vào `outputs/eda/preprocessing_config.json`.
- **Tập trung hóa trực quan (Centralized Visualization):** Toàn bộ mã nguồn vẽ đồ họa `matplotlib` và `seaborn` (từ 3 hình ảnh EDA, learning curves, ma trận nhầm lẫn đến đường cong ROC và Threshold sweep) được gom duy nhất vào `src/utils/plots.py`.
- **Nguyên tắc thuần túy cho RMSprop:** Giữ nguyên `momentum=0.0` để bảo toàn bản chất lý thuyết tách biệt với Adam. Đường cong hội tụ được làm mượt hoàn hảo thông qua bộ lập lịch `CosineAnnealingLR`.
- **Cổng kiểm soát 1-Batch Sanity Overfit:** Mô hình bắt buộc phải overfit 1 mini-batch (8 mẫu) kéo loss về sát 0 trước khi huấn luyện dài hạn.

---

## 3. Hướng Dẫn Thực Thi

### Cách 1: Sử dụng Công Cụ Dòng Lệnh (CLI)

```powershell
# 1. Huấn luyện mô hình với cấu hình mặc định (hoặc truyền file config tùy biến)
python scripts/train.py --config config/config.yaml

# 2. Đánh giá mô hình đã huấn luyện trên tập kiểm thử Test
python scripts/eval.py --config config/config.yaml --checkpoint outputs/checkpoints/fashion_mlp_best.pt

# 3. Dự đoán phân loại trên một mẫu ảnh cụ thể
python scripts/infer.py --sample_idx 0
```

### Cách 2: Sử Dụng Jupyter Notebook

Mở và chạy toàn bộ sổ tay tương tác tại [`notebooks/lab01.ipynb`](file:///c:/Users/Win%2011/Documents/Lab%20DL/labs/lab01/notebooks/lab01.ipynb):
- Chạy 8 bước tuần tự: EDA dữ liệu thô -> Phân chia Stratified Split -> Kiểm tra 1-Batch Sanity -> Huấn luyện mô hình -> So sánh 3 Optimizer (SGD, RMSprop, Adam) với Optuna -> Đánh giá đa chiều (ROC-AUC, Threshold sweep) -> Xuất gói bàn giao `all_outputs.zip`.

---

## 4. Kiểm Thử Tự Động (TDD Test Suite)

Chạy toàn bộ 22 bài kiểm thử tự động để xác minh tính ổn định của mã nguồn:

```powershell
$env:MPLBACKEND="Agg"; pytest tests/ -v
```

Kết quả mong đợi:
```text
============================= 22 passed in ~25s =============================
```

---

## 5. Kết Quả Thực Nghiệm Mốc Chuẩn (Benchmark)

| Chỉ số đánh giá | Kết quả đạt được | Ngưỡng yêu cầu | Trạng thái |
|---|:---:|:---:|:---:|
| **Test Accuracy** | **88.62%** | >= 87.00% | ĐẠT |
| **Macro F1-Score** | **0.8851** | >= 0.8500 | ĐẠT |
| **Macro ROC-AUC** | **0.9912** | >= 0.9500 | ĐẠT |
| **1-Batch Sanity Overfit Loss** | **< 0.05** | < 0.15 | ĐẠT |
| **Số lượng tham số mô hình** | **232,714** | < 500,000 | TỐI ƯU |
