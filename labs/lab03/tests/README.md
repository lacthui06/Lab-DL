# Bộ Kiểm Thử Tự Động (labs/lab03/tests)

## 1. Thông tin chung
- **Tác giả:** Thái Anh Lạc
- **Ngày cập nhật:** 2026-11-11
- **Vị trí:** `labs/lab03/tests/`

## 2. Mục đích & Vai trò
- Hoạt động như **"Cổng an toàn kỹ thuật" (Safety Gates)** kiểm tra toàn diện pipeline trước khi đẩy mã nguồn lên môi trường huấn luyện GPU từ xa (Kaggle).
- Đảm bảo tính toán đúng đắn của mọi thành phần: từ kiểm tra hợp đồng dữ liệu, suy luận mô hình, hàm tính loss, cơ chế dừng sớm đến logic quét ngưỡng xác suất.

## 3. Bản đồ phân loại 25 tệp kiểm thử

| Nhóm Kiểm Thử | Tệp Kiểm Thử | Mục Đích & Kiểm Tra Cốt Lõi |
|---|---|---|
| **Dữ liệu & Hợp đồng** | `test_datasets.py`<br>`test_data_preparation.py`<br>`test_augmentation.py` | Kiểm tra đúng số lượng mẫu các phân vùng (8530/1066/1066), không có chuỗi rỗng; kiểm tra thuật toán Augmentation không làm rò rỉ dữ liệu sang tập heldout. |
| **Mô hình & Suy luận** | `test_models.py`<br>`test_exercise1.py` | Kiểm tra chiều tensor đầu ra, ánh xạ nhãn đúng `{0: NEGATIVE, 1: POSITIVE}`; kiểm tra tính đồng nhất giữa HF Pipeline và Manual PyTorch Forward (< 1e-5). |
| **Huấn luyện & Dừng sớm** | `test_training.py`<br>`test_early_stopping.py`<br>`test_epoch_training.py`<br>`test_frozen.py` | Kiểm tra vòng lặp Trainer; kiểm tra cơ chế dừng sớm kích hoạt chính xác sau 2 epoch không cải thiện validation loss; kiểm tra đóng băng tham số. |
| **Ngưỡng quyết định** | `test_thresholds.py`<br>`test_decision.py` | Kiểm tra tính hợp lệ của vector `P(POSITIVE)`; kiểm tra thuật toán quét 19 ngưỡng; xác nhận ngưỡng tối ưu chọn từ validation được khóa và áp dụng đúng lên test. |
| **Hệ thống & Tái lập** | `test_runs.py`<br>`test_evidence.py`<br>`test_performance.py`<br>`test_plots.py` | Kiểm tra Context Manager `Run`, tính toàn vẹn mã băm SHA-256; đo lường tài nguyên RAM/VRAM và xuất biểu đồ không bị lỗi backend. |
| **Notebook & Kiểm toán** | `test_notebook_phases.py`<br>`test_notebook_runtime.py`<br>`test_final_review.py`<br>`test_published_artifacts.py`<br>`test_kaggle_kernel_export.py` | Kiểm tra các cell notebook chạy mượt mà ở cả hai chế độ `view` và `train`; kiểm toán tính toàn vẹn của gói artifact trước khi nộp bài. |

## 4. Cách chạy kiểm thử
Từ thư mục gốc của dự án:
```powershell
pytest labs/lab03/tests
```
