# Mô-đun Tiện Ích & Giám Sát Thực Nghiệm (src/utils)

## 1. Thông tin chung
- **Tác giả:** Thái Anh Lạc
- **Ngày cập nhật:** 2026-11-11
- **Vị trí:** `labs/lab03/src/utils/`
- **Tệp mã nguồn:** [`runs.py`](runs.py), [`plots.py`](plots.py)

## 2. Mục đích & Vai trò
- Quản lý toàn bộ vòng đời của các lần chạy thực nghiệm (Experiment Runs) thông qua cơ chế Context Manager nguyên tử.
- Đảm bảo khả năng tái lập kết quả khoa học (Strict Reproducibility): chụp ảnh môi trường thực thi, ghi nhật ký console và tự động băm mã SHA-256 cho mọi tệp dữ liệu đầu ra.
- Cung cấp bộ công cụ trực quan hóa chuyên sâu cho bài toán Xử lý Ngôn ngữ Tự nhiên (NLP): đường cong học tập, ma trận nhầm lẫn, đồ thị quét ngưỡng và đường cong cân chuẩn độ tin cậy (Calibration Curve).

## 3. Khởi tạo & Cơ chế hoạt động

### 3.1. Quản Lý Thí Nghiệm & Bằng Chứng Số Liệu (`runs.py`)
- `class Run`: Tạo thư mục thực nghiệm duy nhất với dấu thời gian và mã định danh ngẫu nhiên (`outputs/runs/<timestamp>_<name>_<hash>`).
  - Tự động chuyển hướng toàn bộ luồng xuất màn hình (`sys.stdout` và `sys.stderr`) vào file `console.log` thông qua lớp `Tee`.
  - Tự động gọi `snapshot()` để ghi nhận phiên bản hệ điều hành, GPU CUDA, Python và các thư viện cốt lõi (`torch`, `transformers`, `datasets`, `accelerate`).
  - Ghi nhận trạng thái hoàn thành (`status.json`) và tự động tính mã băm SHA-256 cho toàn bộ file artifact khi kết thúc để chống giả mạo số liệu.
- `save_json(path, data)`: Cơ chế ghi file nguyên tử (Atomic write) — ghi vào tệp tạm thời `.tmp` trước khi đổi tên đè vào tệp đích, loại bỏ triệt để nguy cơ tệp bị hỏng (corrupted) khi tiến trình bị ngắt đột ngột.
- `hash_text_file(path)`: Chuẩn hóa ký tự xuống dòng `\r\n` thành `\n` (LF normalization) trước khi băm SHA-256, đảm bảo mã băm hoàn toàn trùng khớp khi chuyển đổi qua lại giữa Windows, Linux và Kaggle.

### 3.2. Trực Quan Hóa Dữ Liệu & Chẩn Đoán Lỗi (`plots.py`)
- `plot_epoch_history(history, save_path)`: Vẽ đường cong diễn biến Loss và Accuracy trên cả hai tập Train và Validation qua các epoch để phát hiện sớm dấu hiệu học vẹt.
- `plot_threshold_scan(scan_results, save_path)`: Vẽ sự đánh đổi giữa Precision, Recall và F1-score khi di chuyển ngưỡng quyết định từ 0.05 đến 0.95.
- `plot_confusion_matrix(cm, labels, save_path)`: Vẽ ma trận nhầm lẫn 2×2 thể hiện số lượng dự đoán đúng/sai kèm thanh đo màu sắc trực quan.
- `plot_calibration_curve(y_true, y_prob, save_path)`: Vẽ biểu đồ độ tin cậy (Reliability Diagram) so sánh giữa xác suất dự đoán của mô hình và tần suất thực tế, giúp đánh giá mức độ tin cậy của Softmax.
- `plot_probabilities(y_true, y_prob, save_path)`: Vẽ biểu đồ phân bố xác suất của mô hình tách biệt theo nhãn thực tế, giúp nhìn rõ độ phân tách biên quyết định.

## 4. Thảo luận kỹ thuật & Quyết định thiết kế
- **Tầm quan trọng của Calibration Curve trong Transformer:** Các mô hình Transformer tiền huấn luyện lớn thường có xu hướng "quá tự tin" (Overconfident — dự đoán xác suất sát 0.99 ngay cả khi đoán sai). Việc vẽ đồ thị cân chuẩn là công cụ chẩn đoán bắt buộc để biết mô hình có cần hiệu chỉnh nhiệt độ (Temperature Scaling) trước khi đưa vào thực tế hay không.
- **Tính toán độc lập nền tảng (Cross-platform Reproducibility):** Việc chuẩn hóa LF trước khi băm SHA-256 giúp bảo đảm rằng bài nộp hoặc checkpoint sinh ra trên Kaggle GPU (Linux) khi tải về máy tính Windows cá nhân vẫn vượt qua bài kiểm tra toàn vẹn dữ liệu 100%.
