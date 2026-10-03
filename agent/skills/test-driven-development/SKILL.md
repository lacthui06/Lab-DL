---
name: test-driven-development
description: Thực hiện kiểm thử theo hướng phát triển (TDD) chuyên sâu cho Data Science, Machine Learning và Deep Learning. Liên kết trực tiếp với dl_workflows/dl_data_and_augmentation_guide.md để kiểm thử dữ liệu, rò rỉ (leakage), kích thước Tensor và overfit 1 batch.
---

# test-driven-development (DS & AI Edition)

## Tổng quan
Kỹ năng này áp dụng TDD vào đặc thù của Học máy và Học sâu. Trong Deep Learning, kiểm thử là **lá chắn bảo vệ bạn khỏi các lỗi ngầm** (như rò rỉ dữ liệu, sai lệch kích thước Tensor, tràn bộ nhớ GPU hoặc mô hình không có khả năng học).

## Liên kết Quy trình Nghiệp vụ (Flow References)
Khi thực hiện kiểm thử, AI **BẮT BUỘC** phải áp dụng các chuẩn mực kỹ thuật từ:
- 🖼️ [**`dl_workflows/dl_data_and_augmentation_guide.md`**](../../dl_workflows/dl_data_and_augmentation_guide.md): Cài đặt kiểm thử tính toàn vẹn file ảnh, kiểm tra Lazy Loading của Dataset và kiểm thử kích thước batch của DataLoader.
- ⚙️ [**`dl_workflows/dl_training_and_optimization_guide.md`**](../../dl_workflows/dl_training_and_optimization_guide.md): Cài đặt kiểm thử gradient, cắt tỉa đạo hàm và kiểm thử Sanity Overfit 1 batch.

---

## Bộ 5 Bài Kiểm Thử Bắt Buộc (The 5 Mandatory DL Tests)

### 1. Kiểm thử Tính toàn vẹn & Chống rò rỉ Dữ liệu (Integrity & Leakage Test)
- Quét và đảm bảo không có file dữ liệu nào bị lỗi/corrupt trước khi nạp vào DataLoader.
- Khẳng định tính độc lập giữa các tập: $Train \cap Val = \emptyset$ (không trùng lặp ID hoặc mẫu).
- Kiểm tra tiền xử lý: Hàm `fit()` của Scaler/Encoding chỉ được gọi trên tập Train.

### 2. Kiểm thử Cơ chế Lazy Loading của Custom Dataset
- Viết test khẳng định: Việc khởi tạo `Dataset` trong hàm `__init__` chỉ lưu danh sách đường dẫn, **không được nạp mảng ảnh thô vào RAM**.
- Gọi thử `dataset[0]` và kiểm tra:
  ```python
  image, label = dataset[0]
  assert isinstance(image, torch.Tensor), "Đầu ra phải là Tensor!"
  assert image.dtype == torch.float32, "Kiểu dữ liệu phải là float32!"
  ```

### 3. Kiểm thử Kích thước Tensor (Tensor Shape Test)
- Trước khi chạy toàn bộ tập dữ liệu, truyền một Dummy Tensor qua mô hình để xác nhận không bị lỗi lệch chiều:
  ```python
  def test_forward_shape():
      model = build_model(num_classes=10)
      dummy = torch.randn(2, 3, 224, 224) # 2 ảnh mẫu
      out = model(dummy)
      assert out.shape == (2, 10), f"Sai kích thước đầu ra: {out.shape}"
  ```

### 4. Kiểm thử Khả năng Học — Sanity Overfit Test (TỐI QUAN TRỌNG)
- Lấy đúng 1 batch nhỏ (4 đến 8 mẫu dữ liệu) và cho mô hình huấn luyện trong 30 - 50 epoch trên đúng batch đó.
- **Tiêu chí bắt buộc:** Hàm mất mát (Loss) **PHẢI GIẢM DẦN VỀ GẦN 0 ($\approx 0.00$)** và Accuracy trên 8 mẫu đó phải đạt **100%**.
- *Quy tắc:* Nếu mô hình không thể overfit được 8 mẫu, kiến trúc mô hình hoặc hàm loss/optimizer đang có bug nghiêm trọng; **TUYỆT ĐỐI KHÔNG ĐƯỢC MANG LÊN KAGGLE TRAIN**.

### 5. Kiểm thử Tương thích Đường dẫn Kaggle
- Viết test kiểm tra xem đường dẫn `DATA_DIR` và `OUTPUT_DIR` có tự động thích ứng khi chạy trên Kaggle (`/kaggle/input/` và `/kaggle/working/`) hay không.

---

## Anti-Rationalization
- ❌ *"Mô hình Deep Learning chạy lâu lắm, bỏ qua test overfit để lên Kaggle train luôn"* ➔ **Bác bỏ:** Lên Kaggle train 2 tiếng rồi mới phát hiện model không học (Loss đứng im) sẽ làm lãng phí toàn bộ quota 30 tiếng GPU miễn phí trong tuần! Test trước 30 giây trên local là bắt buộc.
