---
name: debugging-and-error-recovery
description: Chẩn đoán và xử lý các lỗi đặc thù trong huấn luyện mô hình Machine Learning và Deep Learning. Liên kết trực tiếp với dl_workflows/dl_training_and_optimization_guide.md để xử lý CUDA OOM, Loss NaN/Inf, lỗi môi trường Kaggle và đạo hàm triệt tiêu.
---

# debugging-and-error-recovery (DS & AI Edition)

## Tổng quan
Kỹ năng này cung cấp quy trình 5 bước và sổ tay chẩn đoán nhanh các lỗi kinh điển khi huấn luyện Deep Learning trên máy cá nhân và trên môi trường đám mây Kaggle.

## Liên kết Quy trình Nghiệp vụ (Flow References)
Khi gặp lỗi kỹ thuật, AI **BẮT BUỘC** phải tra cứu và đối chiếu giải pháp từ:
- 🛠️ [**`dl_workflows/dl_training_and_optimization_guide.md`**](../../dl_workflows/dl_training_and_optimization_guide.md): Cẩm nang xử lý tràn bộ nhớ GPU, kỹ thuật AMP FP16, cắt tỉa gradient và cơ chế Early Stopping.

---

## Sổ Tay Xử Lý Lỗi Kinh Điển Trong Deep Learning & Kaggle

### 1. Tràn bộ nhớ GPU (`RuntimeError: CUDA out of memory`)
- **Giải pháp kỹ thuật:**
  1. Giảm `batch_size` (từ 64 xuống 32 hoặc 16) và bù đắp bằng **Gradient Accumulation** (tích lũy gradient qua $N$ bước) để giữ nguyên hiệu năng học.
  2. Bật Automatic Mixed Precision: `torch.amp.autocast('cuda', dtype=torch.float16)` ➔ Giảm $50\%$ VRAM ngay lập tức!
  3. Kiểm tra rò rỉ bộ nhớ lịch sử: Dùng `total_loss += loss.item()` thay vì `total_loss += loss` (tuyệt đối không giữ nguyên đồ thị tính toán).
  4. Giải phóng cache thủ công: `torch.cuda.empty_cache()`.

### 2. Hàm mất mát ra NaN hoặc Inf (Loss = NaN / Inf)
- **Giải pháp kỹ thuật:**
  1. Thêm cắt tỉa đạo hàm (Gradient Clipping) trước bước optimizer:
     ```python
     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
     ```
  2. Kiểm tra dữ liệu đầu vào: Chạy `assert not torch.isnan(inputs).any()`.
  3. Tránh phép toán chia cho 0 hoặc $\log(0)$ bằng cách kẹp giá trị: `torch.clamp(x, min=1e-7)`.
  4. Giảm `learning_rate` xuống 10 lần (từ $10^{-2}$ xuống $10^{-3}$ hoặc $10^{-4}$).

### 3. Các Lỗi Đặc Thù Khi Chạy Trên Kaggle
- **Lỗi 1 (Bus error - DataLoader bị kill đột ngột):**
  - *Nguyên nhân:* Đặt `num_workers` quá lớn làm tràn bộ nhớ chia sẻ `/dev/shm` của máy ảo Kaggle.
  - *Cách sửa:* Đặt cố định `num_workers = 2` hoặc `num_workers = 0`.
- **Lỗi 2 (Permission denied - Read-only file system):**
  - *Nguyên nhân:* Cố tình ghi file hoặc tạo folder trong `/kaggle/input/`.
  - *Cách sửa:* Chuyển toàn bộ đường dẫn lưu checkpoint sang `/kaggle/working/`.
- **Lỗi 3 (Connection refused khi tải mô hình Pretrained):**
  - *Nguyên nhân:* Tắt kết nối Internet trong cài đặt Notebook.
  - *Cách sửa:* Bật `Settings ➔ Internet ➔ Internet on` trên giao diện Kaggle.

---

## Quy trình 5 bước gỡ lỗi (5-Step Triage)
1. **Tái hiện (Reproduce):** Chạy lại trên đúng 1 batch nhỏ để cô lập lỗi.
2. **Khoanh vùng (Localize):** Xác định lỗi thuộc về DataLoader, Model Architecture, Loss hay Optimizer.
3. **Cô lập (Isolate):** In kích thước Tensor và kiểm tra giá trị `NaN` tại tầng bị nghi vấn.
4. **Sửa tận gốc (Fix):** Áp dụng giải pháp chuẩn mực (không sửa tạm bợ).
5. **Thiết lập rào chắn (Guard):** Bổ sung test case hoặc assertion để lỗi không tái diễn.
