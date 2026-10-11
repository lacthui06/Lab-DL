---
name: test-driven-development
description: Áp dụng giao thức Test-Driven Development (TDD) chuyên biệt cho Data Science và Deep Learning. Liên kết trực tiếp với dl_workflows/dl_data_and_augmentation_guide.md để xác thực tính toàn vẹn dữ liệu, chống rò rỉ dữ liệu, kiểm tra chiều tensor, và bài test 1-batch sanity overfit.
---

# test-driven-development (Bản DS & AI)

## Tổng quan

Skill này chuyển thể nguyên lý TDD sang bản chất ngẫu nhiên (stochastic) của machine learning và deep learning. Trong kỹ thuật deep learning, kiểm thử tự động là **lá chắn chính yếu chống lại các lỗi câm (silent bugs)** - những lỗi như nhiễm bẩn tập train-val, lỗi broadcasting tensor âm thầm, rò rỉ bộ nhớ GPU, hoặc kiến trúc không hội tụ.

Điểm khác biệt cốt lõi so với TDD phần mềm truyền thống: trong deep learning, một pipeline có thể chạy trơn tru không văng lỗi nhưng vẫn cho ra kết quả sai hoàn toàn. Test không chỉ kiểm tra "code có chạy không" mà phải kiểm tra "code có chạy đúng về mặt toán học và dữ liệu không".

## Tham chiếu workflow

Khi triển khai test, agent **BẮT BUỘC** phải áp dụng chuẩn kỹ thuật từ:
- [`dl_workflows/dl_data_and_augmentation_guide.md`](../../dl_workflows/dl_data_and_augmentation_guide.md): Quét tính toàn vẹn file, xác thực lazy-loading, và khẳng định chiều batch của DataLoader.
- [`dl_workflows/dl_training_and_optimization_guide.md`](../../dl_workflows/dl_training_and_optimization_guide.md): Khẳng định luồng gradient (gradient flow), kiểm tra clipping, và bài test 1-batch sanity overfit.

---

## 5 bài test bắt buộc cho Deep Learning

### 1. Test toàn vẹn dữ liệu và chống rò rỉ (Data Integrity and Leakage Prevention)
- Quét và khẳng định không có file ảnh hoặc file dạng bảng bị hỏng trước khi nạp dữ liệu vào loader.
- Xác thực tính rời rạc của các phân vùng: $\text{Train} \cap \text{Val} = \emptyset$ (không có mẫu hoặc định danh nào chồng lấn).
- Khẳng định ranh giới fit của tiền xử lý: Các scaler hoặc encoder tiền xử lý chỉ được gọi `fit()` trên tập train mà thôi.

### 2. Test lazy-loading của Dataset tùy chỉnh (Custom Dataset Lazy-Loading)
- Khẳng định rằng `Dataset.__init__` chỉ lưu mảng index hoặc đường dẫn file, **không bao giờ đệm trực tiếp ma trận ảnh thô vào RAM của host**.
- Truy cập `dataset[0]` để xác thực kiểu dữ liệu: ảnh phải là `torch.Tensor` với dtype `torch.float32`, nhãn phải là `torch.Tensor` với dtype `torch.int64`.

### 3. Test chiều tensor và shape (Tensor Dimension and Shape)
- Truyền một batch giả tổng hợp (synthetic dummy batch) qua đường forward để bắt lỗi sai chiều trước khi lặp toàn bộ dữ liệu.
- Khẳng định output logits có shape đúng bằng `(batch_size, num_classes)`, và toàn bộ giá trị là hữu hạn (không chứa NaN hoặc Inf).

### 4. Năng lực học - Test 1-Batch Sanity Overfit (QUAN TRỌNG NHẤT)
- Cô lập một mini-batch duy nhất (4 đến 8 mẫu) và huấn luyện mô hình trong 30 đến 50 vòng lặp chỉ trên đúng batch đó.
- **Tiêu chí bắt buộc:** Training loss **PHẢI GIẢM VỀ SÁT 0 ($\approx 0.00$)** và độ chính xác trên batch phải đạt **100%**.
- **Quy tắc:** Nếu một mô hình không thể overfit 8 mẫu, thì kiến trúc, công thức loss, hoặc optimizer đang chứa lỗi nghiêm trọng. **NGHIÊM CẤM HUẤN LUYỆN TRÊN CLOUD/KAGGLE** cho tới khi test này pass.

### 5. Test tương thích đường dẫn môi trường (Environment Path Compatibility)
- Khẳng định rằng các đường dẫn động (dynamic path) thích ứng trơn tru khi chạy ở môi trường cục bộ so với Kaggle (`/kaggle/input/` và `/kaggle/working/`).

---

## Chống biện minh (Anti-Rationalization)

- [X] *"Mô hình Deep Learning chạy lâu lắm, bỏ qua test overfit rồi train thẳng trên Kaggle"* -> **Bác bỏ:** Huấn luyện hàng giờ trên Kaggle chỉ để phát hiện mô hình không bao giờ hội tụ sẽ bào mòn hạn mức GPU hàng tuần. Một bước xác thực overfit cục bộ 5 giây là bắt buộc.
