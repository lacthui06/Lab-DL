---
name: source-driven-development
description: Căn cứ mọi quyết định mã nguồn AI và Data Science vào tài liệu chính thức của thư viện (PyTorch, Hugging Face, Scikit-learn, Pandas). Ngăn chặn tuyệt đối việc sử dụng API lỗi thời hoặc tham số bịa đặt.
---

# source-driven-development (DS & AI Edition)

## Tổng quan
Các thư viện AI và Deep Learning phát triển với tốc độ chóng mặt và liên tục thay đổi API. Kỹ năng này bắt buộc AI Agent phải đối chiếu trực tiếp với tài liệu chuẩn (Official Documentation) mới nhất, loại bỏ hoàn toàn các đoạn code lỗi thời (deprecated) hoặc tham số bị ảo giác.

## Khi nào sử dụng
- Khi sử dụng các framework AI phổ biến: `PyTorch`, `Hugging Face Transformers`, `Scikit-learn`, `PyTorch Lightning`, `LangChain`, `Polars`.
- Khi cấu hình các kỹ thuật huấn luyện nâng cao: `torch.amp.autocast`, `DistributedDataParallel`, `BitsAndBytes` Quantization, Learning Rate Schedulers.

## Quy tắc thực thi (Enforcement Rules)

### 1. Đối chiếu phiên bản và API chính thức
- Luôn sử dụng cú pháp của phiên bản hiện đại:
  - Dùng `torch.amp.autocast(device_type='cuda')` thay cho cú pháp cũ `torch.cuda.amp.autocast()`.
  - Sử dụng `from torchvision.transforms.v2` (Torchvision v2) thay cho phiên bản v1 cũ kỹ.
  - Sử dụng cú pháp `optimizer.zero_grad(set_to_none=True)` để tối ưu bộ nhớ GPU thay vì `optimizer.zero_grad()`.

### 2. Trích dẫn nguồn tài liệu trong mã nguồn
- Khi áp dụng một cấu hình tham số quan trọng hoặc kiến trúc phức tạp, phải ghi chú URL hoặc tên API chuẩn trong docstring:
  ```python
  # Theo tài liệu chính thức của PyTorch: torch.optim.lr_scheduler.CosineAnnealingLR
  # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  ```

### 3. Nghiêm cấm bịa đặt tham số (No Hallucinated Kwargs)
- Nếu không chắc chắn một hàm trong Hugging Face hoặc PyTorch có hỗ trợ tham số đó hay không, AI phải tra cứu hoặc kiểm tra qua `inspect.signature` hoặc script kiểm chứng ngắn trước khi đưa vào pipeline chính.

## Anti-Rationalization
- ❌ *"Tôi nhớ tham số này trong PyTorch năm 2021 là như vậy"* ➔ **Bác bỏ:** PyTorch 2.x và Transformers 4.x đã thay đổi nhiều chữ ký hàm. Phải dùng cú pháp chuẩn hiện hành.
