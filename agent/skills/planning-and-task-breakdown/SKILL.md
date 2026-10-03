---
name: planning-and-task-breakdown
description: Phân rã bài toán Data Science và AI từ bản đặc tả (Spec) thành các tác vụ nhỏ nguyên tử theo từng chặng Pipeline ML có thứ tự phụ thuộc và tiêu chí nghiệm thu rõ ràng.
---

# planning-and-task-breakdown (DS & AI Edition)

## Tổng quan
Kỹ năng này chuyển hóa bản đặc tả kỹ thuật thành một **Kế hoạch triển khai Pipeline Học máy (ML Implementation Plan)** gồm các tác vụ độc lập, có thể kiểm thử riêng biệt trước khi ráp nối thành pipeline hoàn chỉnh.

## Khi nào sử dụng
- Ngay sau khi hoàn thành `spec-driven-development`.
- Trước khi bắt đầu viết bất kỳ module code nào.

## Cấu trúc chuẩn của ML Pipeline Tasks

Kế hoạch phải chia nhỏ công việc thành 6 chặng chuẩn mực:

### Chặng 1: Nạp & Kiểm định Dữ liệu (Data Ingestion & Validation)
- Tác vụ 1.1: Tải dữ liệu, kiểm tra định dạng và kích thước ban đầu.
- Tác vụ 1.2: Viết hàm kiểm định Schema (kiểu dữ liệu, dải giá trị, kiểm tra cột mục tiêu).

### Chặng 2: Tiền xử lý & Trích xuất Đặc trưng (Preprocessing & Feature Engineering)
- Tác vụ 2.1: Xử lý giá trị khuyết thiếu (Imputation) và dữ liệu ngoại lai (Outliers).
- Tác vụ 2.2: Mã hóa đặc trưng (One-Hot, Target Encoding) hoặc Tokenization / Augmentation.
- *Quy tắc:* Mọi phép biến đổi (Fit) chỉ được thực hiện trên tập Train, sau đó Transform trên tập Validation/Test để chống Data Leakage.

### Chặng 3: Xây dựng Mô hình Cơ sở (Baseline Model)
- Tác vụ 3.1: Huấn luyện mô hình đơn giản (Baseline).
- Tác vụ 3.2: Đo đạc chỉ số cơ sở làm mốc so sánh (Benchmark).

### Chặng 4: Triển khai Mô hình Chính (Core Model / Deep Learning)
- Tác vụ 4.1: Định nghĩa kiến trúc mô hình (Model Architecture / Neural Network Layers).
- Tác vụ 4.2: Thiết lập vòng lặp huấn luyện (Training Loop: Optimizer, Scheduler, Loss).
- Tác vụ 4.3: Tích hợp Early Stopping và Model Checkpointing (lưu trọng số tốt nhất).

### Chặng 5: Đánh giá Toàn diện & Phân tích Lỗi (Evaluation & Error Analysis)
- Tác vụ 5.1: Đo đạc các chỉ số trên tập kiểm thử độc lập (Test Set).
- Tác vụ 5.2: Vẽ biểu đồ trực quan (Confusion Matrix, ROC Curve, Feature Importance).
- Tác vụ 5.3: Phân tích các mẫu dự đoán sai (Error Analysis).

### Chặng 6: Đóng gói Pipeline (Packaging & Clean Code)
- Tác vụ 6.1: Đóng gói thành hàm `predict()` hoặc script pipeline hoàn chỉnh.
- Tác vụ 6.2: Dọn dẹp mã nguồn theo chuẩn module tái sử dụng.

## Tiêu chí nghiệm thu (Acceptance Criteria)
- Mỗi tác vụ trong kế hoạch phải có:
  - Input và Output rõ ràng.
  - Tiêu chí kiểm thử để xác nhận tác vụ đó đã hoàn thành đúng trước khi chuyển sang tác vụ kế tiếp.
