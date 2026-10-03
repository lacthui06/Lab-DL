---
name: api-and-interface-design
description: Thiết kế giao diện và dịch vụ API phục vụ suy luận mô hình AI (Model Serving) bằng FastAPI, chuẩn hóa lược đồ đầu vào/đầu ra và xử lý tải trọng mô hình an toàn.
---

# api-and-interface-design (DS & AI Edition)

## Tổng quan
Kỹ năng này chịu trách nhiệm đóng gói mô hình đã được huấn luyện thành một dịch vụ Web API chuẩn công nghiệp (thường dùng `FastAPI`), cho phép các ứng dụng bên ngoài truyền dữ liệu vào và nhận kết quả dự đoán với độ trễ thấp và độ tin cậy cao.

## Khi nào sử dụng
- Khi huấn luyện xong mô hình và cần tạo giao diện API để làm demo, nộp đồ án hoặc tích hợp vào hệ thống phần mềm.
- Cần định nghĩa chuẩn định dạng Request/Response dữ liệu cho model.

## Nguyên tắc thiết kế API phục vụ mô hình AI (Model Serving Principles)

### 1. Nạp mô hình một lần duy nhất (Lifespan / Startup Loading)
- **Quy tắc vàng:** Trọng số mô hình (`.pt`, `.pkl`, `.onnx`) phải được nạp vào bộ nhớ (RAM/VRAM) **duy nhất một lần khi khởi động server** (dùng `lifespan` của FastAPI), tuyệt đối KHÔNG nạp lại file model trong từng request dự đoán!

### 2. Chuẩn hóa Schema bằng Pydantic (Input/Output Validation)
- Mọi dữ liệu đầu vào phải được kiểm tra chặt chẽ kiểu dữ liệu và giới hạn biên:
  ```python
  from pydantic import BaseModel, Field
  from typing import List

  class PredictRequest(BaseModel):
      features: List[float] = Field(..., example=[5.1, 3.5, 1.4, 0.2], description="Danh sách các thuộc tính đầu vào")

  class PredictResponse(BaseModel):
      prediction: int = Field(..., description="Nhãn lớp dự đoán")
      class_name: str = Field(..., description="Tên lớp dự đoán")
      confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy của dự đoán")
      latency_ms: float = Field(..., description="Thời gian suy luận (ms)")
  ```

### 3. Cung cấp bộ Endpoint tiêu chuẩn
Một dịch vụ API mô hình chuẩn phải có tối thiểu 3 endpoint:
1. `GET /health`: Kiểm tra trạng thái hoạt động của server và xác nhận mô hình đã nạp thành công vào bộ nhớ.
2. `GET /metadata`: Trả về thông tin mô hình (phiên bản, tên mô hình, ngày huấn luyện, chỉ số F1 đạt được).
3. `POST /predict`: Endpoint chính nhận dữ liệu và trả kết quả suy luận.

### 4. Xử lý ngoại lệ an toàn (Safe Exception Handling)
- Bắt các lỗi dữ liệu đầu vào không hợp lệ (ví dụ ảnh bị hỏng, chuỗi rỗng) và trả về mã lỗi HTTP 422/400 rõ ràng thay vì để server sập (HTTP 500).

## Tiêu chí nghiệm thu (Verification)
- Server khởi động và endpoint `/health` trả về `{"status": "healthy", "model_loaded": true}`.
- Gửi một request mẫu tới `/predict` và nhận kết quả phản hồi có độ trễ $< 200\text{ms}$.
