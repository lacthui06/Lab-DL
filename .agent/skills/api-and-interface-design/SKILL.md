---
name: api-and-interface-design
description: Thiết kế và triển khai dịch vụ inference API production cho mô hình AI và Deep Learning bằng FastAPI, áp đặt schema xác thực input/output chặt chẽ và quản lý vòng đời mô hình an toàn.
---

# api-and-interface-design (Bản DS & AI)

## Tổng quan

Skill này đóng gói các mô hình machine learning và deep learning đã huấn luyện thành dịch vụ Web API chuẩn công nghiệp (thường dùng `FastAPI`). Nó cho phép các ứng dụng bên ngoài truy vấn mô hình với độ trễ thấp, xử lý lỗi mạnh mẽ, và payload dữ liệu đã được xác thực.

Mục tiêu là biến một "artifact thí nghiệm" thành một "dịch vụ có thể tích hợp", với các hợp đồng request/response rõ ràng.

## Khi nào sử dụng

- Đóng gói một mô hình đã hoàn thành thành inference API để demo, nộp đồ án môn học, hoặc tích hợp hệ thống.
- Định nghĩa hợp đồng Request và Response chính thức cho việc phục vụ mô hình.

## Nguyên tắc cốt lõi khi phục vụ mô hình

### 1. Nạp trọng số mô hình một lần lúc khởi động (Lifespan Loading)
- **Quy tắc vàng:** Trọng số mô hình (`.pt`, `.onnx`, `.pkl`) phải được nạp vào bộ nhớ (RAM/VRAM) **đúng một lần trong lúc ứng dụng khởi động** bằng FastAPI lifespan context manager. Không bao giờ nạp lại trọng số bên trong từng request handler.
- Lý do: nạp lại trọng số mỗi request sẽ làm độ trễ tăng vọt và lãng phí tài nguyên một cách nghiêm trọng.

### 2. Xác thực schema chặt chẽ bằng Pydantic
- Mọi đặc trưng đầu vào và dự đoán đầu ra phải áp đặt an toàn kiểu dữ liệu và ràng buộc miền giá trị. Ví dụ schema cần có:
  - `PredictRequest` chứa vector đặc trưng đầu vào dạng danh sách số thực kèm mô tả.
  - `PredictResponse` chứa chỉ số lớp dự đoán, tên lớp người đọc được, độ tin cậy (ràng buộc trong khoảng 0.0 đến 1.0), và độ trễ inference tính bằng mili-giây.
- Pydantic giúp chặn dữ liệu sai định dạng ngay tại biên hệ thống, thay vì để lỗi lan sâu vào trong mô hình.

### 3. Kiến trúc endpoint chuẩn
Một dịch vụ mô hình production tối thiểu phải phơi bày:
1. `GET /health`: Healthcheck xác minh trạng thái server và sự sẵn có của mô hình đã nạp trong bộ nhớ.
2. `GET /metadata`: Metadata mô hình (phiên bản kiến trúc, ngày huấn luyện, metric đánh giá mục tiêu).
3. `POST /predict`: Endpoint chính nhận payload đặc trưng và trả về kết quả inference có cấu trúc.

### 4. Xử lý ngoại lệ kiên cường (Resilient Exception Handling)
- Bắt các dữ liệu đầu vào hỏng (ví dụ buffer ảnh không hợp lệ, tensor dị dạng) và trả về phản hồi HTTP 400/422 có thông tin, thay vì để xảy ra crash HTTP 500 không được xử lý.

## Tiêu chí xác thực

- Dịch vụ khởi tạo sạch và `GET /health` trả về `{"status": "healthy", "model_loaded": true}`.
- Payload mẫu gửi tới `/predict` trả về thành công phản hồi có kiểu dữ liệu đúng trong ngân sách độ trễ chấp nhận được (< 200 ms).
