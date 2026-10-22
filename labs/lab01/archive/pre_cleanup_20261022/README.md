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
│   ├── checkpoints/                    # Trọng số mô hình tốt nhất (fashion_mlp_best.pt, fashion_mlp_final.pt)
│   ├── logs/                           # training_history.json, environment_snapshot.json, optimizer_study_results.json
│   ├── predictions/                    # Đồ thị đánh giá: learning_curves.png, confusion_matrix.png,
│   │                                   # roc_curves.png, optimizer_comparison.png, optimizer_fixed_lr_comparison.png,
│   │                                   # prediction_grid.png, test_prediction_grid.png
│   └── eda/                            # Artifacts chuẩn hóa (split_indices.npz, split_metadata.json, preprocessing_config.json)
│       ├── tables/                     # 7 bảng thống kê CSV (overview, integrity, distribution, pixel_stats, spatial, handcrafted, similarity)
│       └── figures/                    # 7 biểu đồ EDA (sample_images, class_mean_images, class_distribution, pixel_histogram,
│                                       # mean_pixel_heatmap, pixel_variance_heatmap, class_similarity_heatmap)
├── src/                                # Gói thư viện lõi (Core Python Package)
│   ├── __init__.py                     # Re-export API hợp nhất
│   ├── datasets/                       # Quản lý dữ liệu và tiền xử lý
│   │   ├── __init__.py
│   │   ├── base.py                     # TransformedSubset và phân chia Stratified Split
│   │   ├── classification.py           # Dataset, Transforms pipeline (kết nối augmentations.yaml) và DataLoader factory
│   │   └── eda.py                      # Đầy đủ 9 bước Core EDA & Feature Engineering, khóa artifact chuẩn hóa
│   ├── models/                         # Thiết kế kiến trúc mạng nơ-ron
│   │   ├── __init__.py
│   │   └── mlp.py                      # Mô hình FashionMLP thuần PyTorch (Flatten -> 256 -> 128 -> 10, 235,146 tham số)
│   ├── training/                       # Động cơ huấn luyện và tối ưu hóa
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Class Trainer, train loop, evaluate, 1-batch sanity check
│   │   └── experiments.py              # Optuna study, so sánh 3 optimizer (Fixed LR vs Optuna), đa lớp OvR ROC-AUC
│   └── utils/                          # Công cụ bổ trợ
│       ├── __init__.py
│       ├── helpers.py                  # Định tuyến đường dẫn động, device CUDA/CPU, seed, đọc chuẩn hóa, checkpoint
│       ├── metrics.py                  # Tính TPR, FPR, F1-Score, Precision, Recall theo từng lớp
│       └── plots.py                    # Module visualization duy nhất cho toàn bộ biểu đồ EDA và đánh giá
├── scripts/                            # Điểm thực thi dòng lệnh (CLI Entry Points)
│   ├── train.py                        # Huấn luyện mô hình từ terminal
│   ├── eval.py                         # Đánh giá checkpoint mô hình trên tập test
│   └── infer.py                        # Dự đoán nhãn cho ảnh mẫu cụ thể
├── tests/                              # Bộ kiểm thử đơn vị và tích hợp (TDD)
│   ├── test_datasets.py                # Kiểm tra tính toàn vẹn dữ liệu, chống rò rỉ ranh giới
│   ├── test_models.py                  # Kiểm tra số lượng tham số và tensor output shape
│   └── test_training.py                # Kiểm tra 1-batch sanity overfit, checkpoint roundtrip, metric
├── notebooks/
│   └── lab01.ipynb                     # Sổ tay tương tác hoàn chỉnh 10 bước từ EDA đến kiểm chứng Test & đóng gói
├── requirements.txt                    # Danh sách các thư viện phụ thuộc đã kiểm chứng
└── README.md                           # Tài liệu hướng dẫn sử dụng này
```

---

## 2. Điểm Nhấn Kiến Trúc & Lý Thuyết

- **Chống rò rỉ dữ liệu & Không dùng số liệu giả (Zero Fake Stats):** Thống kê chuẩn hóa (`mean = 0.286209`, `std = 0.353160`) được đo đạc nghiêm ngặt độc quyền trên tập Train (54,000 mẫu) thông qua phân chia Stratified Split cố định (`seed=42`), khóa cứng vào `outputs/eda/preprocessing_config.json` và được nạp qua một hàm hợp nhất duy nhất `resolve_normalization_stats()`.
- **Kết nối cấu hình Augmentation động:** Hàm `get_transforms(augment=True)` tự động đọc quy tắc tăng cường ảnh từ `config/augmentations.yaml` (`random_horizontal_flip`, `random_rotation`, `random_crop`) mà không cần sửa code trong `src/`.
- **Tập trung hóa trực quan (Centralized Visualization):** Toàn bộ mã nguồn vẽ đồ họa `matplotlib` và `seaborn` (từ 7 biểu đồ EDA, learning curves, ma trận nhầm lẫn, lưới dự đoán đến đường cong ROC đa lớp) được gom duy nhất vào `src/utils/plots.py`.
- **Nguyên tắc thuần túy cho RMSprop:** Giữ nguyên `momentum=0.0` để bảo toàn bản chất lý thuyết tách biệt với Adam. Đường cong hội tụ được làm mượt thông qua bộ lập lịch `CosineAnnealingLR`.
- **Cổng kiểm soát 1-Batch Sanity Overfit:** Mô hình bắt buộc phải overfit 1 mini-batch (8 mẫu) kéo loss về sát 0 (`0.0005 < 0.15`) trước khi huấn luyện chính thức.

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
- Chạy 10 bước tuần tự: Thiết lập môi trường -> Đầy đủ 9 bước EDA & Khóa cấu hình FE (7 bảng CSV & 7 biểu đồ PNG) -> Nạp DataLoaders & Guard Assert -> Khởi tạo FashionMLP & Kiểm tra 1-Batch Sanity -> Huấn luyện mô hình 10 epochs -> Đánh giá & Ma trận nhầm lẫn -> So sánh 3 Optimizer (Cùng LR 1e-3 vs Optuna) -> Khảo sát ROC-AUC đa lớp (One-vs-Rest) -> Lưu Checkpoint -> Tải lại mô hình kiểm chứng trên 10,000 ảnh Test (`test_prediction_grid.png`) & Đóng gói `all_outputs.zip`.

---

## 4. Kiểm Thử Tự Động (TDD Test Suite)

Chạy toàn bộ 22 bài kiểm thử tự động để xác minh tính ổn định của mã nguồn:

```powershell
$env:MPLBACKEND="Agg"; pytest tests/ -v
```

Kết quả mong đợi:
```text
============================= 22 passed =============================
```

---

## 5. Kết Quả Thực Nghiệm Mốc Chuẩn (Benchmark)

| Chỉ số đánh giá | Kết quả đạt được | Ngưỡng yêu cầu | Trạng thái |
|---|:---:|:---:|:---:|
| **Validation Accuracy (Best)** | **90.92%** | >= 87.00% | ĐẠT |
| **Test Accuracy (10,000 ảnh)** | **89.30%** | >= 87.00% | ĐẠT |
| **Macro F1-Score (Validation)** | **0.9085** | >= 0.8500 | ĐẠT |
| **Macro ROC-AUC (One-vs-Rest)** | **0.9943** | >= 0.9500 | ĐẠT |
| **1-Batch Sanity Overfit Loss** | **0.0005** | < 0.15 | ĐẠT |
| **Số lượng tham số mô hình** | **235,146** | < 500,000 | TỐI ƯU |


## Pipeline ba cấu hình (cập nhật 22/10/2026)

1. Mở `notebooks/01_eda_preprocessing_features.ipynb`: EDA, preprocessing, 10 feature và artifact bàn giao.
2. Mở `notebooks/02_modeling_comparison.ipynb`: đặt DATA_PATH đến prepared.npz; full training hỗ trợ cả local CPU/CUDA và Kaggle, tự chọn thiết bị. Mỗi model tạo run riêng.
3. Kết quả ở `outputs/runs/<UTC>_<name>_<uuid>/`; không xóa run cũ. `runs/` được gitignore vì có dữ liệu/weight lớn, cần lưu trữ và đồng bộ riêng khi chuyển máy.
4. Notebook `lab01.ipynb` là lịch sử. Không dùng output lịch sử thay kết quả pipeline mới. Chưa xuất PDF.


## Optimizer và tuning

Notebook 02 có các giai đoạn riêng: fixed LR, tuning ba study độc lập, refit và tổng hợp. Hoặc chạy từ repository:

```powershell
python labs/lab01/scripts/run_optimizer_experiments.py "<preparation-run>/prepared.npz" --epochs 10 --trials 10 --threads 2
```

Mỗi optimizer có 10 trial riêng (tổng 30); test không tham gia objective. Optuna là dependency thật trong requirements. SQLite và artifact từng trial được lưu; pruning là trạng thái dừng sớm hợp lệ. TPESampler mặc định có 10 startup trials nên ngân sách 10 trial/optimizer của đợt này vẫn ở pha sampling khởi động. Tăng trial budget nếu cần đánh giá pha adaptive TPE. Không dùng kết quả tuned để so sánh kiến trúc với model chưa tune.
