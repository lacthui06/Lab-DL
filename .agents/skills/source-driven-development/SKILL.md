---
name: source-driven-development
description: Neo mọi quyết định triển khai AI và Data Science vào tài liệu chính thức của thư viện (PyTorch, Hugging Face, Scikit-learn, Torchvision), loại bỏ pattern deprecated và keyword argument bị ảo tưởng.
---

# source-driven-development (Bản DS & AI)

## 1. Tổng quan

Các framework deep learning và AI tiến hóa rất nhanh với những thay đổi API phá vỡ tương thích thường xuyên. Skill này bắt buộc AI Agent phải xác thực việc triển khai dựa trên tài liệu chính thức cập nhật, loại bỏ các hàm đã deprecated và các argument bị ảo tưởng.

Vấn đề gốc rễ: các mô hình ngôn ngữ lớn có xu hướng "ảo tưởng" ra những keyword argument nghe có vẻ hợp lý nhưng không tồn tại trong phiên bản thư viện hiện tại. Việc kiểm tra chữ ký hàm trước khi dùng là cách duy nhất để chống lại điều này.

Skill cũng chống lại thói quen dùng cú pháp cũ đã bị deprecated, vốn có thể chạy được nhưng sẽ bị gỡ bỏ trong bản phát hành tương lai, gây lỗi khó lường.

Nguyên lý then chốt: nếu không chắc một hàm nhận argument nào, phải kiểm tra chữ ký của nó thay vì đoán.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Triển khai pipeline với các framework hiện đại (PyTorch 2.x, Torchvision, Hugging Face Transformers, Scikit-learn, Polars).
- Cấu hình các tính năng huấn luyện nâng cao (AMP autocast, DistributedDataParallel, quantization, scheduler).
- Gặp lỗi TypeError về argument không mong đợi.
- Cần xác nhận cú pháp API trước khi dùng.

**Điều kiện tiên quyết:**
- Đã biết framework và phiên bản mục tiêu.
- Có khả năng truy cập tài liệu chính thức hoặc kiểm tra chữ ký hàm.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Framework và phiên bản mục tiêu | Cú pháp API đã xác thực |
| Đoạn code cần xác thực | Loại bỏ pattern deprecated |
| Câu hỏi về argument của hàm | Xác nhận chữ ký hàm chính xác |

## 4. Quy trình chi tiết

### Bước 1: Xác định framework và phiên bản
Xác định chính xác framework và phiên bản đang dùng trong môi trường (local và Kaggle). Hành vi API khác nhau giữa các phiên bản, nên phải biết phiên bản trước khi tra cứu cú pháp đúng.

### Bước 2: Tra cứu tài liệu chính thức
Với mỗi API không tầm thường, tra cứu tài liệu chính thức của phiên bản tương ứng. Ưu tiên tài liệu chính thức hơn là ví dụ trên blog hoặc câu trả lời cũ, vì tài liệu chính thức phản ánh API hiện hành.

### Bước 3: Thay thế cú pháp deprecated
Áp dụng các chuẩn framework hiện đại: dùng `torch.amp.autocast('cuda')` thay cho `torch.cuda.amp.autocast()` cũ; dùng transforms v2 thay cho transforms v1 deprecated; dùng `optimizer.zero_grad(set_to_none=True)` để tối ưu bộ nhớ GPU. Ghi nhận lý do đằng sau mỗi thay thế.

### Bước 4: Xác thực chữ ký hàm khi không chắc chắn
Nếu không chắc một hàm nhận keyword argument cụ thể nào, kiểm tra chữ ký của nó bằng `inspect.signature` hoặc xác minh bằng một script ngắn trước khi nhúng vào module production. Bước này chống lại argument ảo tưởng.

### Bước 5: Ghi nguồn tài liệu trong comment triển khai
Khi áp dụng hyperparameter không tầm thường hoặc logic scheduling phức tạp, tham chiếu tài liệu chính thức ngay trong docstring. Việc ghi nguồn giúp người đọc sau này và chính agent trong phiên sau có thể kiểm chứng lại lý do đằng sau mỗi lựa chọn API.

### Bước 6: Kiểm chứng bằng chạy thử
Chạy thử đoạn code đã xác thực trên một input nhỏ để đảm bảo cú pháp đúng và hành vi như kỳ vọng trước khi tích hợp vào pipeline chính.

## 5. Deliverables

- Danh sách API đã xác thực với phiên bản tương ứng.
- Code loại bỏ hoàn toàn pattern deprecated.
- Comment ghi nguồn tài liệu cho các API không tầm thường.
- Bằng chứng chạy thử xác nhận cú pháp đúng.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `planning-and-task-breakdown` (xác định module cần triển khai) | `test-driven-development` (nhận code đã xác thực để viết test) |
| `spec-driven-development` (cung cấp ràng buộc phiên bản) | `environment-and-reproducibility` (nhận thông tin phiên bản API) |
| | `debugging-and-error-recovery` (chuyển lỗi API phức tạp) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Argument ảo tưởng:** Dùng keyword argument không tồn tại gây lỗi TypeError. Dấu hiệu: lỗi "unexpected keyword argument". Phòng tránh: kiểm tra chữ ký hàm ở Bước 4.
- **Cú pháp deprecated:** Dùng API cũ vẫn chạy nhưng sẽ bị gỡ bỏ. Dấu hiệu: cảnh báo deprecation trong output. Phòng tránh: thay thế ở Bước 3.
- **Tra cứu sai phiên bản:** Áp dụng cú pháp của phiên bản khác. Dấu hiệu: hành vi không như tài liệu mô tả. Phòng tránh: xác định phiên bản chính xác ở Bước 1.
- **Dựa vào blog cũ:** Dùng ví dụ lỗi thời từ nguồn không chính thức. Phòng tránh: ưu tiên tài liệu chính thức ở Bước 2.

## 8. Definition of Done

- [ ] Đã xác định chính xác framework và phiên bản.
- [ ] Mọi API không tầm thường đã được tra cứu từ tài liệu chính thức.
- [ ] Đã thay thế toàn bộ cú pháp deprecated.
- [ ] Đã xác thực chữ ký hàm cho các API không chắc chắn.
- [ ] Đã ghi nguồn tài liệu cho các API không tầm thường.
- [ ] Đã chạy thử xác nhận cú pháp và hành vi đúng.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Tôi nhớ parameter này từng hoạt động trong PyTorch hồi 2021"* -> **Bác bỏ:** Các bản phát hành hiện đại (PyTorch 2.x, Transformers 4.x) đã tái cấu trúc chữ ký hàm. Luôn dùng cú pháp chính thức hiện hành.
- [X] *"Code chạy được rồi, khỏi tra tài liệu"* -> **Bác bỏ:** Code chạy được không có nghĩa là dùng API đúng. Cú pháp deprecated vẫn chạy nhưng sẽ gây lỗi trong tương lai và có thể cho hành vi sai.
- [X] *"Đoán argument cũng được, sai thì sửa"* -> **Bác bỏ:** Đoán argument dẫn đến lỗi runtime tốn thời gian gỡ. Kiểm tra chữ ký hàm mất vài giây và loại bỏ hoàn toàn rủi ro này.
- [X] *"Ví dụ trên blog cũng giống tài liệu chính thức"* -> **Bác bỏ:** Blog thường dựa trên phiên bản cũ và không được cập nhật. Tài liệu chính thức là nguồn duy nhất phản ánh API hiện hành.
