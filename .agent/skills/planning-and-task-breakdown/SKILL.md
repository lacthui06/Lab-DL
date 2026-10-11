---
name: planning-and-task-breakdown
description: Phân rã đặc tả kỹ thuật (Spec) thành các giai đoạn pipeline nguyên tử, có thể kiểm thử, với thứ tự phụ thuộc, hợp đồng rõ ràng và tiêu chí nghiệm thu đã xác thực cho các dự án Data Science và AI.
---

# planning-and-task-breakdown (Bản DS & AI)

## Tổng quan

Skill này chuyển hóa một đặc tả kỹ thuật thành một **Kế hoạch triển khai Machine Learning** có thể hành động được. Nó phân rã các workflow phức tạp thành những giai đoạn module độc lập có thể kiểm thử riêng, trước khi kết nối chúng lại thành một pipeline end-to-end.

Nguyên lý: một pipeline lớn không thể kiểm thử được. Một pipeline được chia thành các module nhỏ, mỗi module có input/output rõ ràng, thì mỗi phần đều có thể xác thực độc lập, và lỗi được cô lập về đúng module gây ra nó.

## Khi nào sử dụng

- Ngay sau khi hoàn thành `spec-driven-development`.
- Trước khi tạo hoặc chỉnh sửa gói Python module trong `src/`.

## Phân rã pipeline chuẩn 6 giai đoạn

### Giai đoạn 1: Nạp dữ liệu và Xác thực (Data Ingestion and Validation)
- Task 1.1: Nạp dữ liệu thô, kiểm tra chiều kích thước ban đầu, và xác thực tính toàn vẹn.
- Task 1.2: Triển khai các assertion về schema và kiểu dữ liệu (kiểm tra khoảng giá trị nhãn, ràng buộc null).

### Giai đoạn 2: Tiền xử lý và Pipeline đặc trưng (Preprocessing and Feature Pipeline)
- Task 2.1: Xử lý giá trị thiếu (missing values) và ngoại lai (outliers).
- Task 2.2: Áp dụng mã hóa biến phân loại, chuẩn hóa đặc trưng, hoặc augmentation ảnh.
- **Quy tắc:** Mọi thao tác fit biến đổi phải được tính toán nghiêm ngặt trên tập train và chỉ transform sang tập validation/test, nhằm ngăn chặn rò rỉ dữ liệu.

### Giai đoạn 3: Xây dựng mô hình Baseline
- Task 3.1: Triển khai baseline tối thiểu theo heuristic hoặc chuẩn (ví dụ MLP 2 lớp).
- Task 3.2: Ghi nhận metric benchmark ban đầu để thiết lập baseline so sánh.

### Giai đoạn 4: Mô hình lõi và Pipeline Deep Learning
- Task 4.1: Định nghĩa kiến trúc mô hình (các layer mạng nơ-ron, normalization, activation head).
- Task 4.2: Xây dựng vòng lặp huấn luyện (forward, criterion, backward, optimizer, scheduler).
- Task 4.3: Tích hợp cơ chế checkpointing (`best_model.pt`) và early stopping.

### Giai đoạn 5: Đánh giá và Phân tích lỗi chẩn đoán
- Task 5.1: Đo metric cuối cùng trên tập test độc lập.
- Task 5.2: Kết xuất các artifact trực quan (đường cong loss/accuracy, confusion matrix).
- Task 5.3: Kiểm tra các lát cắt mẫu bị phân loại sai để xác định chế độ thất bại (failure mode).

### Giai đoạn 6: Đóng gói và Module hóa
- Task 6.1: Tổ chức các module tái sử dụng dưới `src/` (`config.py`, `dataset.py`, `model.py`, `engine.py`, `utils.py`).
- Task 6.2: Đảm bảo ranh giới giao diện sạch và script có thể tái lập.

## Tiêu chí nghiệm thu

- Mỗi task trong kế hoạch phải định nghĩa rõ:
  - Input và output tường minh.
  - Các assertion xác thực xác nhận task thành công trước khi chuyển sang giai đoạn kế tiếp.
