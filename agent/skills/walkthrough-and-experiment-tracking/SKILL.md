---
name: walkthrough-and-experiment-tracking
description: Tự động ghi chép, theo dõi và cập nhật toàn bộ quy trình thử nghiệm (Full-Flow Experiment Lineage) vào file walkthrough_lab_0X.md sau mỗi lần huấn luyện. Ghi nhận các thay đổi về Dữ liệu, Tiền xử lý, Mô hình, Huấn luyện và Đánh giá để sẵn sàng xuất báo cáo nộp bài.
---

# walkthrough-and-experiment-tracking

## Tổng quan
Kỹ năng này đóng vai trò là "Nhật ký thí nghiệm tự động" (Experiment Tracker & Lineage Logger) cho toàn bộ các bài Lab Deep Learning. Sau mỗi lần chạy thử nghiệm (Run #1, Run #2,... dù chạy trên máy Local hay tải kết quả từ Kaggle về), kỹ năng này bắt buộc AI phải cập nhật chi tiết **sự thay đổi của toàn bộ 5 khâu trong Pipeline** vào file `walkthrough_lab_0X.md`.

## Khi nào kích hoạt
- Khi hoàn thành một lần chạy huấn luyện và có kết quả đo lường (Metrics/Loss).
- Khi người dùng điều chỉnh bất kỳ khâu nào trong Flow (thêm Data Augmentation, đổi mạng CNN sang ResNet, đổi Optimizer).
- Khi chuẩn bị xuất số liệu và biểu đồ để làm báo cáo nộp bài.

---

## Quy trình 4 bước cập nhật nhật ký Flow (The Tracking Protocol)

### Bước 1: Ghi nhận trạng thái cấu hình 5 khâu (Capture Full-Flow State)
Thu thập và ghi lại trạng thái của lần chạy hiện tại (`Run #X`):
1. **Dữ liệu:** Tên dataset, số lượng mẫu, tỷ lệ chia Train/Val/Test.
2. **Tiền xử lý & Augmentation:** Phép biến đổi đã dùng (`RandomResizedCrop`, `ColorJitter`, `Normalize`).
3. **Mô hình:** Tên kiến trúc mạng, số lượng tham số (`Total Params`), số tầng.
4. **Chiến lược Huấn luyện:** Hàm mất mát, Optimizer, Learning Rate, Scheduler, Batch Size, Epochs.
5. **Kết quả Đo lường:** Train Loss/Acc, Val Loss/Acc, F1-Score, thời gian chạy.

### Bước 2: Xác định độ chênh lệch kỹ thuật (Log Technical Delta)
Trả lời 3 câu hỏi bắt buộc vào mục **Changelog**:
- *Đã thay đổi ở khâu nào so với lần chạy trước?* (Dữ liệu, Tiền xử lý, Mô hình, hay Huấn luyện?)
- *Tại sao lại thay đổi (Why)?* (Nhằm giải quyết lỗi Overfit, Loss ra NaN hay tăng tốc độ?)
- *Tác động định lượng là gì?* (Metric tăng/giảm bao nhiêu %, độ chênh lệch Train-Val cải thiện ra sao?)

### Bước 3: Cập nhật Bảng Ma trận So sánh (Update Comparison Matrix)
- Thêm một cột mới cho `Run #X` vào bảng **Flow Comparison Matrix** trong file `walkthrough_lab_0X.md`.
- Đánh dấu phiên bản nào đang nắm giữ kỷ lục điểm số cao nhất (**Best Checkpoint**).

### Bước 4: Đúc kết dữ liệu & Thu hoạch Artifact từ Kaggle (Kaggle Harvest & Report-Ready Findings)
- **Cơ chế thu hoạch từ Kaggle:** Khi người dùng chạy trên Kaggle GPU:
  - Code Notebook lưu toàn bộ kết quả vào `/kaggle/working/`:
    - File trọng số mô hình: `outputs/checkpoints/best_model.pt`
    - Lưới ảnh dự đoán Input vs Output (Predicted vs Actual): `outputs/predictions_best.png`
    - Ma trận nhầm lẫn: `outputs/confusion_matrix.png`
    - Đồ thị huấn luyện: `outputs/loss_curves.png`
  - Người dùng tải trực tiếp (hoặc nén `!zip -r outputs.zip /kaggle/working/outputs`) về thư mục `labs/lab0X/outputs/` ở máy Local.
- **Trích xuất báo cáo:**
  - AI đọc các artifact vừa tải về để lấy số liệu thực tế cập nhật vào file `walkthrough_lab_0X.md`.
  - Trích xuất 3 kết luận then chốt:
    - Yếu tố kỹ thuật nào tạo ra bước ngoặt cải thiện điểm số lớn nhất?
    - Mô hình tốt nhất vẫn còn dự đoán sai ở những trường hợp nào?
    - Đường dẫn tới các file biểu đồ và checkpoint tốt nhất (`outputs/best_model.pt`).

---

## Tiêu chí nghiệm thu (Verification)
- File `walkthrough_lab_0X.md` được cập nhật đồng bộ, đầy đủ số liệu thực tế, không có các ô trống hoặc thông tin bịa đặt.
- Người dùng chỉ cần mở file là có đầy đủ toàn bộ bằng chứng thí nghiệm để nộp báo cáo môn học.
