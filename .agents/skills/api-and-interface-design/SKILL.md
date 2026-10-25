---
name: api-and-interface-design
description: Thiết kế và triển khai dịch vụ inference API production cho mô hình AI và Deep Learning bằng FastAPI, áp đặt schema xác thực input/output chặt chẽ và quản lý vòng đời mô hình an toàn.
---

# api-and-interface-design (Bản DS & AI)

## 1. Tổng quan

Skill này đóng gói các mô hình machine learning và deep learning đã huấn luyện thành dịch vụ Web API chuẩn công nghiệp (thường dùng FastAPI). Nó cho phép các ứng dụng bên ngoài truy vấn mô hình với độ trễ thấp, xử lý lỗi mạnh mẽ, và payload dữ liệu đã được xác thực.

Mục tiêu là biến một "artifact thí nghiệm" thành một "dịch vụ có thể tích hợp", với các hợp đồng request/response rõ ràng.

Điểm khác biệt cốt lõi giữa notebook thí nghiệm và dịch vụ production là tính bền vững: dịch vụ phải chạy liên tục, xử lý dữ liệu bất thường một cách an toàn, và không sụp đổ khi nhận input không mong đợi.

Nguyên lý then chốt: nạp trọng số một lần lúc khởi động, xác thực mọi input, và trả về lỗi có thông tin thay vì crash.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Đóng gói mô hình đã huấn luyện thành inference API để demo hoặc nộp đồ án.
- Cần tích hợp mô hình vào một hệ thống hoặc ứng dụng khác.
- Cần định nghĩa hợp đồng Request và Response chính thức cho việc phục vụ mô hình.
- Cần xử lý input từ người dùng cuối một cách an toàn.

**Điều kiện tiên quyết:**
- Đã có checkpoint mô hình tốt nhất đã xác thực.
- Đã biết schema của input và output mô hình.
- Đã có môi trường có thể chạy web service.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Checkpoint mô hình đã huấn luyện | Dịch vụ inference API chạy được |
| Schema input/output mô hình | Hợp đồng Request và Response có xác thực |
| Yêu cầu về độ trễ | Các endpoint health, metadata, predict |
| | Xử lý ngoại lệ an toàn |

## 4. Quy trình chi tiết

### Bước 1: Nạp trọng số một lần lúc khởi động
Nạp trọng số mô hình (`.pt`, `.onnx`, `.pkl`) vào bộ nhớ đúng một lần trong lúc ứng dụng khởi động bằng lifespan context manager. Không bao giờ nạp lại trọng số bên trong từng request handler, vì việc này làm độ trễ tăng vọt và lãng phí tài nguyên nghiêm trọng.

### Bước 2: Định nghĩa schema xác thực chặt chẽ
Định nghĩa schema Pydantic cho request và response. Request chứa vector đặc trưng đầu vào với mô tả rõ ràng. Response chứa chỉ số lớp dự đoán, tên lớp người đọc được, độ tin cậy (ràng buộc trong khoảng 0.0 đến 1.0), và độ trễ inference. Pydantic chặn dữ liệu sai định dạng ngay tại biên hệ thống.

### Bước 3: Xây dựng các endpoint chuẩn
Phơi bày tối thiểu ba endpoint: `GET /health` xác minh trạng thái server và sự sẵn có của mô hình; `GET /metadata` trả về phiên bản kiến trúc, ngày huấn luyện, và metric đánh giá; `POST /predict` nhận payload đặc trưng và trả về kết quả inference có cấu trúc.

### Bước 4: Xử lý ngoại lệ kiên cường
Bắt các dữ liệu đầu vào hỏng (buffer ảnh không hợp lệ, tensor dị dạng) và trả về phản hồi HTTP 400 hoặc 422 có thông tin, thay vì để xảy ra crash HTTP 500 không được xử lý. Dịch vụ production phải chịu được input xấu mà không sụp đổ.

### Bước 5: Đảm bảo tiền xử lý nhất quán với huấn luyện
Đảm bảo pipeline tiền xử lý trong dịch vụ giống hệt pipeline dùng khi huấn luyện. Sai lệch giữa hai pipeline (train/serve skew) là nguyên nhân phổ biến khiến mô hình cho kết quả sai trong production dù đã hoạt động tốt khi test.

### Bước 6: Xác thực dịch vụ
Khởi động dịch vụ và xác nhận `GET /health` trả về trạng thái khỏe mạnh với mô hình đã nạp. Gửi payload mẫu tới `/predict` và xác nhận phản hồi có kiểu dữ liệu đúng trong ngân sách độ trễ chấp nhận được.

## 5. Deliverables

- Dịch vụ inference API chạy được với các endpoint health, metadata, predict.
- Schema Pydantic cho request và response.
- Xử lý ngoại lệ trả về lỗi có thông tin.
- Bằng chứng xác thực dịch vụ với payload mẫu.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `code-simplification` (cung cấp module mô hình sạch) | `documentation-and-adrs` (nhận thông tin dịch vụ để viết tài liệu) |
| `performance-optimization` (cung cấp mô hình tối ưu cho inference) | `version-control-and-artifacts` (commit code dịch vụ) |
| `test-driven-development` (cung cấp xác thực mô hình) | |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Nạp trọng số mỗi request:** Độ trễ tăng vọt và lãng phí tài nguyên. Dấu hiệu: mỗi request chậm vài giây. Phòng tránh: nạp một lần lúc khởi động ở Bước 1.
- **Không xác thực input:** Input sai định dạng gây crash HTTP 500. Phòng tránh: dùng schema Pydantic ở Bước 2.
- **Train/serve skew:** Tiền xử lý trong dịch vụ khác với huấn luyện. Dấu hiệu: kết quả sai dù test tốt. Phòng tránh: đảm bảo nhất quán ở Bước 5.
- **Thiếu endpoint health:** Không biết dịch vụ có đang hoạt động không. Phòng tránh: thêm endpoint ở Bước 3.
- **Bắt lỗi quá rộng:** Nuốt mọi lỗi và trả về kết quả mặc định. Dấu hiệu: lỗi thật bị che giấu. Phòng tránh: trả lỗi có thông tin ở Bước 4.

## 8. Definition of Done

- [ ] Trọng số được nạp một lần lúc khởi động.
- [ ] Schema Pydantic cho request và response đã được định nghĩa.
- [ ] Ba endpoint health, metadata, predict đã được xây dựng.
- [ ] Xử lý ngoại lệ trả về lỗi có thông tin.
- [ ] Tiền xử lý trong dịch vụ nhất quán với huấn luyện.
- [ ] `GET /health` trả về trạng thái khỏe mạnh với mô hình đã nạp.
- [ ] Payload mẫu tới `/predict` trả về phản hồi đúng trong ngân sách độ trễ.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Nạp trọng số trong mỗi request cho đơn giản"* -> **Bác bỏ:** Nạp lại trọng số mỗi request làm độ trễ tăng vọt và lãng phí tài nguyên. Nạp một lần lúc khởi động là bắt buộc.
- [X] *"Không cần xác thực input, người dùng sẽ gửi đúng"* -> **Bác bỏ:** Dịch vụ production luôn nhận input bất thường. Không xác thực dẫn đến crash và lỗi khó truy nguyên.
- [X] *"Tiền xử lý trong dịch vụ viết lại cho gọn"* -> **Bác bỏ:** Viết lại dễ gây sai lệch so với lúc huấn luyện, dẫn đến train/serve skew và kết quả sai. Dùng chung pipeline.
- [X] *"Bắt mọi lỗi và trả kết quả mặc định cho an toàn"* -> **Bác bỏ:** Che giấu lỗi khiến vấn đề thật không được phát hiện và người dùng nhận kết quả sai. Trả lỗi có thông tin.
