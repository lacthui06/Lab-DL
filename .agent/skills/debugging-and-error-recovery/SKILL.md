---
name: debugging-and-error-recovery
description: Chẩn đoán và khắc phục các chế độ thất bại chuyên biệt trong vòng lặp huấn luyện Machine Learning và Deep Learning. Liên kết trực tiếp với dl_workflows/dl_training_and_optimization_guide.md để xử lý CUDA OOM, loss NaN/Inf, quyền hệ thống file trên Kaggle, và gradient tiêu biến.
---

# debugging-and-error-recovery (Bản DS & AI)

## Tổng quan

Skill này cung cấp một giao thức phân loại 5 bước (5-step triage protocol) và một sổ tay chẩn đoán để xử lý các chế độ thất bại kinh điển gặp phải trong quá trình huấn luyện deep learning trên máy trạm cục bộ và các nền tảng đám mây như Kaggle.

Mục tiêu là biến việc gỡ lỗi từ "thử mò" thành một quy trình có hệ thống: nhận diện triệu chứng, khoanh vùng nguyên nhân, áp dụng giải pháp kỹ thuật, và phòng ngừa tái diễn.

## Tham chiếu workflow

Khi gặp bất thường kỹ thuật, agent **BẮT BUỘC** phải tham khảo và áp dụng các giải pháp chuẩn từ:
- [`dl_workflows/dl_training_and_optimization_guide.md`](../../dl_workflows/dl_training_and_optimization_guide.md): Quản lý VRAM GPU, Automatic Mixed Precision (AMP FP16), gradient clipping, và giao thức early stopping.

---

## Sổ tay chẩn đoán kỹ thuật cho Deep Learning & Kaggle

### 1. Cạn kiệt bộ nhớ GPU (`RuntimeError: CUDA out of memory`)
- **Giải pháp kỹ thuật:**
  1. Giảm `batch_size` (ví dụ từ 64 xuống 32 hoặc 16) và bù lại bằng **Gradient Accumulation** qua $N$ bước để bảo toàn động lực batch hiệu dụng.
  2. Bật Automatic Mixed Precision: `torch.amp.autocast('cuda', dtype=torch.float16)` -> Giảm VRAM khoảng 50%.
  3. Ngăn rò rỉ đồ thị tính toán (computational graph leak): Tích lũy giá trị vô hướng bằng `total_loss += loss.item()` thay vì `total_loss += loss`.
  4. Xóa bộ nhớ cache tường minh: `torch.cuda.empty_cache()`.

### 2. Bất ổn định loss (Loss = NaN hoặc Inf)
- **Giải pháp kỹ thuật:**
  1. Thêm gradient clipping trước khi optimizer cập nhật: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
  2. Xác thực tensor đầu vào: Thực thi `assert not torch.isnan(inputs).any()`.
  3. Ngăn chia cho 0 hoặc bất ổn định $\log(0)$ bằng cách kẹp giá trị số (numeric clamping): `torch.clamp(x, min=1e-7)`.
  4. Giảm learning rate đi một bậc độ lớn (ví dụ từ $10^{-2}$ xuống $10^{-3}$ hoặc $10^{-4}$).

### 3. Lỗi runtime đặc thù trên Kaggle
- **Lỗi 1 (Bus error - DataLoader bị kill đột ngột):**
  - *Nguyên nhân gốc:* `num_workers` quá cao làm cạn kiệt phân vùng bộ nhớ chia sẻ `/dev/shm` của máy ảo Kaggle.
  - *Giải pháp:* Giới hạn `num_workers = 2` hoặc `num_workers = 0`.
- **Lỗi 2 (Permission denied - Hệ thống file chỉ đọc):**
  - *Nguyên nhân gốc:* Cố gắng tạo hoặc ghi thư mục dưới `/kaggle/input/`.
  - *Giải pháp:* Hướng mọi mục tiêu ghi, checkpoint mô hình, và log về `/kaggle/working/`.
- **Lỗi 3 (Connection refused khi tải trọng số pretrained):**
  - *Nguyên nhân gốc:* Quyền truy cập internet bị tắt trong cài đặt notebook Kaggle.
  - *Giải pháp:* Bật `Settings -> Internet -> Internet on` trong giao diện notebook Kaggle.

---

## Giao thức phân loại 5 bước

1. **Reproduce (Tái hiện):** Chạy lại trên một mini-batch duy nhất để cô lập lỗi trong trạng thái tối giản.
2. **Localize (Khoanh vùng):** Xác định nguyên nhân gốc xuất phát từ DataLoader, Model Forward, Loss, hay bước Backward.
3. **Isolate (Cô lập):** In shape tensor và khẳng định `torch.isnan().any()` tại các ranh giới module bị nghi ngờ.
4. **Fix (Sửa):** Triển khai các chỉnh sửa mang tính cấu trúc, bền vững, thay vì các giải pháp tình thế tạm thời.
5. **Guard (Phòng ngừa):** Thêm các assertion test tự động hoặc case pytest để ngăn chặn tái phát (regression).
