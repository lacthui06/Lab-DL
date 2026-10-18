# Lab 01 — FashionMNIST Image Classification with PyTorch

Bài lab xây dựng pipeline Deep Learning end-to-end để phân loại 10 nhóm trang phục trong bộ dữ liệu FashionMNIST (60.000 mẫu train, 10.000 mẫu test). Kiến trúc sử dụng mô hình `FashionMLP` thuần túy PyTorch với 3 loại layer cơ bản (`Flatten`, `Linear`, `ReLU`), kiểm tra TDD 1-batch sanity overfit, tối ưu hóa siêu tham số bằng Optuna, đo benchmark 3 Optimizer theo thời gian thực và khảo sát Multi-class ROC-AUC & Decision Threshold Scan.

---

## Structure

- `description_lab_01.md`: đề bài gốc do giảng viên giao kèm phạm vi thống nhất.
- `walkthrough_lab_01.md`: tài liệu sống duy nhất ghi nhận toàn bộ kết quả, log và phân tích kỹ thuật.
- `notebooks/`:
  - `lab01_fashionmnist_walkthrough.ipynb`: notebook duy nhất của lab, thực thi toàn bộ pipeline từ nạp dữ liệu, huấn luyện, Optuna HPO, benchmark optimizer đến ROC-AUC & Threshold Scan.
- `src/`: các module Python tái sử dụng (`config.py`, `split.py`, `eda.py`, `dataset.py`, `model.py`, `engine.py`, `experiment.py`, `utils.py`).
- `tests/`: bộ kiểm thử TDD xác thực pipeline và tính toàn vẹn (`29 passed`).
- `data/`: dataset đầu vào cục bộ (được git ignore).
- `outputs/`: checkpoint mô hình, biểu đồ kết quả (`.png`) và các artifacts sinh ra (`outputs/eda/split_indices.npz`, `outputs/eda/split_metadata.json`, `outputs/eda/preprocessing_config.json`).

---

## Pipeline versions & Guard Assert

Hai version dưới đây được khóa tự động tại bước Core EDA & Feature Engineering (`run_eda_and_fe_pipeline`) và được kiểm tra bắt buộc qua cổng `verify_pipeline_handoff` trước khi huấn luyện:

| Version | Giá trị | Ghi chú |
|---|---|---|
| `split_v2_stratified` | 54,000 train / 6,000 validation, `random_state=42`, `stratify=y` | Cho đúng 5,400 ảnh train và 600 ảnh validation mỗi lớp. Split lưu tại `outputs/eda/split_indices.npz` và `split_metadata.json`. |
| `preproc_v2_train_global_std` | `mean=0.286209`, `std=0.353160` (population, `ddof=0`) | Đo độc lập trên 54,000 ảnh train đã scale `[0,1]` (chống data leakage). Đọc từ `outputs/eda/preprocessing_config.json`. |

---

## Reproduce

**Phiên bản môi trường:** yêu cầu Python `>=3.12, !=3.14.1` (suy ra từ Requires-Python của các thư viện đã pin), ghi dạng comment trong `requirements.txt`. Bản đã kiểm chứng ở local là 3.13.15.

1. Cài đặt môi trường theo `requirements.txt` ở thư mục gốc repository.
2. Chạy test TDD cục bộ để xác thực toàn bộ 29 bài kiểm thử (bao gồm 1-batch sanity overfit và Guard Assert):
   `pytest labs/lab01/tests/ -q`
3. Mở và chạy notebook `notebooks/lab01_fashionmnist_walkthrough.ipynb` trên máy cục bộ hoặc tải lên Kaggle GPU để thực thi toàn bộ 9 bước của pipeline.
4. Tải artifact trở về `outputs/` (bao gồm checkpoints, figures) và đối chiếu báo cáo chi tiết tại `walkthrough_lab_01.md`.

---

## Pipeline Summary (9 Sections trong Walkthrough Notebook)

1. **Thiết lập môi trường & Seed:** Cố định `RANDOM_SEED = 42` trên toàn bộ thư viện (`random`, `numpy`, `torch`) và tự động nhận diện Local / Kaggle GPU.
2. **Core EDA & Feature Engineering (`run_eda_and_fe_pipeline`):** Kiểm tra cấu trúc dữ liệu thô (`uint8`, `[0, 255]`), quét tính toàn vẹn (NaN/Inf, trùng lặp nội bộ Train/Test và rò rỉ chéo Train-Test), chia Stratified Split 90/10 và khóa thông số chuẩn hóa trên tập Train.
3. **DataLoaders & Guard Assert (`verify_pipeline_handoff`):** Khởi tạo `TransformedSubset` + `DataLoader` và kiểm tra 4 tiêu chí bàn giao bắt buộc.
4. **Kiến trúc `FashionMLP` & 1-Batch Sanity Overfit:** Mạng MLP thuần túy (`Flatten -> Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)`, `235,146` tham số), vượt bài kiểm tra overfit 16 mẫu (`loss < 0.05`).
5. **Huấn luyện Baseline `FashionMLP` (10 Epochs):** Huấn luyện với `CrossEntropyLoss` và `Adam(lr=1e-3)`, lưu checkpoint tốt nhất.
6. **Đánh giá & Chẩn đoán hình ảnh:** Vẽ đường cong Loss/Accuracy, Confusion Matrix 10 lớp và lưới 16 ảnh dự đoán thực tế.
7. **Optuna HPO & Wall-Clock Optimizer Benchmark:** Dùng Bayesian TPE + MedianPruner tìm siêu tham số tối ưu cho `SGD + Momentum`, `RMSprop`, `Adam`; sau đó chạy đối đầu 10 epochs đo thời gian thực (wall-clock seconds) chạm mốc `87.0%` Val Accuracy và chọn ra `winner_model`.
8. **Multi-class ROC & ROC-AUC (One-vs-Rest):** Đánh giá năng lực phân biệt trên cả 10 lớp cùng Macro-average và Micro-average AUC của `winner_model`.
9. **Decision Threshold Scan (`t*`):** Quét 100 ngưỡng quyết định trên lớp trọng tâm (`T-shirt/top`), tối ưu hóa F1-Score và lưu checkpoint cuối cùng.

