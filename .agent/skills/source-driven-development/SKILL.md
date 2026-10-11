---
name: source-driven-development
description: Neo mọi quyết định triển khai AI và Data Science vào tài liệu chính thức của thư viện (PyTorch, Hugging Face, Scikit-learn, Torchvision). Loại bỏ nghiêm ngặt các pattern đã bị deprecated và các keyword argument bị ảo tưởng.
---

# source-driven-development (Bản DS & AI)

## Tổng quan

Các framework deep learning và AI tiến hóa rất nhanh với những thay đổi API phá vỡ tương thích (breaking changes) thường xuyên. Skill này bắt buộc AI Agent phải xác thực việc triển khai dựa trên tài liệu chính thức cập nhật, loại bỏ các hàm đã deprecated và các argument bị ảo tưởng.

Vấn đề gốc rễ: các mô hình ngôn ngữ lớn có xu hướng "ảo tưởng" (hallucinate) ra những keyword argument nghe có vẻ hợp lý nhưng không tồn tại trong phiên bản thư viện hiện tại. Việc kiểm tra chữ ký hàm (function signature) trước khi dùng là cách duy nhất để chống lại điều này.

## Khi nào sử dụng

- Triển khai pipeline với các framework hiện đại: `PyTorch 2.x`, `Torchvision`, `Hugging Face Transformers`, `Scikit-learn`, `Polars`.
- Cấu hình các tính năng huấn luyện nâng cao: `torch.amp.autocast`, `DistributedDataParallel`, lượng tử hóa `BitsAndBytes`, Learning Rate Scheduler.

## Các quy tắc thực thi

### 1. Xác thực phiên bản API và cú pháp
- Luôn áp dụng chuẩn framework hiện đại:
  - Dùng `torch.amp.autocast('cuda')` thay cho `torch.cuda.amp.autocast()` đã cũ.
  - Dùng `from torchvision.transforms.v2 import ...` thay cho transforms v1 đã deprecated.
  - Dùng `optimizer.zero_grad(set_to_none=True)` để tối ưu bộ nhớ GPU thay cho `optimizer.zero_grad()`.

### 2. Ghi nguồn tài liệu trong comment triển khai
- Khi áp dụng các hyperparameter không tầm thường hoặc logic scheduling phức tạp, tham chiếu tài liệu chính thức ngay trong docstring.
- Việc ghi nguồn giúp người đọc sau này (và chính agent trong phiên sau) có thể kiểm chứng lại lý do đằng sau mỗi lựa chọn API.

### 3. Nghiêm cấm tuyệt đối argument ảo tưởng (No Hallucinated Kwargs)
- Nếu không chắc chắn một hàm có nhận một keyword argument cụ thể hay không, hãy kiểm tra chữ ký của nó thông qua `inspect.signature` hoặc xác minh bằng một script ngắn một dòng trước khi nhúng vào module production.

## Chống biện minh (Anti-Rationalization)

- [X] *"Tôi nhớ parameter này từng hoạt động trong PyTorch hồi 2021"* -> **Bác bỏ:** Các bản phát hành hiện đại (PyTorch 2.x, Transformers 4.x) đã tái cấu trúc chữ ký hàm. Luôn dùng cú pháp chính thức hiện hành.
