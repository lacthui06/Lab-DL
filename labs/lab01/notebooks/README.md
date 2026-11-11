# Thư Mục Jupyter Notebooks (labs/lab01/notebooks)

## 1. Thông tin chung
- **Tác giả:** Thái Anh Lạc
- **Ngày cập nhật:** 2026-11-11
- **Vị trí:** `labs/lab01/notebooks/`

## 2. Danh mục Notebooks
- [`lab01.ipynb`](lab01.ipynb): Notebook tổng hợp hoàn chỉnh toàn bộ quy trình thực nghiệm của Lab 01 (Phân loại ảnh FashionMNIST với mạng nơ-ron đa tầng MLP PyTorch).
- Chế độ hỗ trợ: Chế độ kép (Dual Mode) — mặc định `MODE="view"` để đọc nhanh báo cáo offline và `MODE="train"` khi cần huấn luyện mới.

---

## 3. Bảng Lộ Trình Triển Khai Thực Tế (Pipeline Execution Roadmap)

Cell Markdown đầu tiên của notebook tuân thủ nghiêm ngặt chuẩn mực Roadmap 10 bước:

| Bước | Giai Đoạn (Phase) | Nội Dung Chi Tiết (Core Description) | Module Triển Khai (`src/...`) | Sản Phẩm Đầu Ra (Output / Artifact) |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Setup & Import** | Cấu hình thiết bị tính toán (CUDA/CPU), cố định seed 42 và nạp toàn bộ API từ `src/` | `src.utils.helpers` | Thông số môi trường thực thi, Device & Seed |
| **2** | **EDA & Preprocessing** | Thực hiện đủ các bước EDA, kiểm tra trùng lặp, chia Stratified Split 90/10, đo Mean/Std tập Train | `src.datasets.eda`<br>`src.utils.plots` | `outputs/eda/split_indices.npz`<br>`outputs/eda/preprocessing_config.json`<br>`outputs/eda/tables/*.csv` & `figures/*.png` |
| **3** | **DataLoaders & Guard Assert** | Xây dựng pipeline `ToTensor` + `Normalize`, kiểm tra 4 tiêu chí bàn giao dữ liệu | `src.datasets.base`<br>`src.datasets.eda` | Bộ nạp `train_loader`, `val_loader`, `test_loader`<br>Cổng `verify_pipeline_handoff` PASSED 100% |
| **4** | **Model & Sanity Overfit** | Khởi tạo mạng `FashionMLP` (235,146 tham số), kiểm tra ép overfit 1 mini-batch (8 mẫu) kéo loss về sát 0 | `src.models.mlp`<br>`src.training.trainer` | Cấu trúc mô hình `FashionMLP`<br>Sanity Overfit Test PASSED (Loss < 0.15) |
| **5** | **Huấn Luyện (Training)** | Huấn luyện mạng `FashionMLP` 10 epochs bằng Adam optimizer kết hợp bộ lập lịch `CosineAnnealingLR` | `src.training.trainer` | Trọng số `best.pt`<br>Nhật ký `history.json` & `metrics.json` |
| **6** | **Đánh Giá & Trực Quan Hóa** | Vẽ đường cong học tập, lập ma trận nhầm lẫn đếm số nguyên 10x10 và phân tích sai số | `src.utils.plots`<br>`src.training.trainer` | `outputs/runs/<run_id>/learning_curves.png`<br>`outputs/runs/<run_id>/confusion_matrix.png` |
| **7** | **So Sánh 3 Mô Hình** | So sánh `SmallMLP`, `MainMLP`, và `MainMLP_FE` (kết hợp 10 đặc trưng hình học thủ công) | `src.training.comparison`<br>`src.utils.comparison_plots` | `confusion_comparison.png`<br>`MainMLP_FE_ovr_curves.png` |
| **8** | **Khảo Sát ROC-AUC Đa Lớp** | Tính toán phân phối xác suất softmax, vẽ đường cong One-vs-Rest ROC và tính điểm AUC cho 10 lớp | `src.utils.comparison_plots` | `roc_ovr_curves.csv`<br>`pr_ovr_curves.csv` |
| **9** | **Tuning Siêu Tham Số (Optuna)** | Thử nghiệm 3 optimizer (SGD, RMSprop, Adam) qua dò tìm siêu tham số Optuna TPE + Pruner | `src.training.tuning` | `studies.sqlite3`<br>`tuning_progress.png`<br>`best_configs.json` |
| **10** | **Refit & Tự Động Thăng Hạng** | Huấn luyện lại cấu hình tối ưu nhất, tự động lưu vào `models/final.pt` và `best_model.pt` | `src.utils.helpers` | `labs/lab01/models/best_model.pt`<br>`labs/lab01/models/manifest.json` |
