---
name: walkthrough-and-experiment-tracking
description: Tự động theo dõi, ghi nhận và cập nhật dòng chảy thí nghiệm đầy đủ trong Section 2 của walkthrough_lab_0X.md sau mỗi lần chạy huấn luyện. Nắm bắt chênh lệch về kiến trúc, hyperparameter và metric qua cả 5 giai đoạn pipeline.
---

# walkthrough-and-experiment-tracking (Bản DS & AI)

## Tổng quan

Skill này hoạt động như bộ theo dõi dòng chảy thí nghiệm tự động cho mọi bài lab deep learning. Sau mỗi lần thực thi huấn luyện (Run #1 Baseline, Run #2 Tuned MLP, Run #3 FashionCNN), skill bắt buộc agent phải ghi lại **các thay đổi qua cả 5 giai đoạn pipeline** trực tiếp vào **Section 2 của tài liệu sống duy nhất `walkthrough_lab_0X.md`** (chịu sự điều phối của Rule 2 trong Kiến trúc 3 Rule).

Chuẩn áp dụng: Mọi log thí nghiệm và lý do điều chỉnh hyperparameter đều được hợp nhất trực tiếp vào Section 2 của `walkthrough_lab_0X.md`. Không tạo file changelog riêng.

Giá trị cốt lõi: nếu không ghi lại lý do đằng sau mỗi thay đổi, thì các lần chạy sau sẽ lặp lại sai lầm cũ hoặc không thể giải thích được vì sao một cấu hình lại tốt hơn cấu hình khác.

## Khi nào kích hoạt

- Sau khi hoàn thành bất kỳ lần chạy huấn luyện nào khi đã có metric đánh giá và giá trị loss.
- Bất cứ khi nào chỉnh sửa một thành phần bất kỳ của pipeline (data transforms, kiến trúc backbone, optimizer schedule).
- Khi tổng hợp metric và hình ảnh cho việc nộp bài môn học.

---

## Giao thức theo dõi 4 bước

### Bước 1: Nắm bắt trạng thái toàn dòng chảy (Capture Full-Flow State)
Ghi lại ảnh chụp cấu hình (configuration snapshot) cho lần thực thi hiện tại (`Run #X`):
1. **Data:** Tên dataset, số lượng mẫu, tỷ lệ chia Train/Val/Test.
2. **Preprocessing & Augmentation:** Các biến đổi được áp dụng (`RandomHorizontalFlip`, `Normalize`, v.v.).
3. **Model:** Tên kiến trúc, tổng số tham số, độ sâu/số layer.
4. **Training Strategy:** Hàm loss, optimizer, learning rate, scheduler, batch size, số epoch.
5. **Evaluation Metrics:** Train Loss/Acc, Val Loss/Acc, Test Loss/Acc, Macro F1-Score, thời gian thực thi.

### Bước 2: Ghi log chênh lệch kỹ thuật (Log Technical Delta)
Trả lời 3 câu hỏi bắt buộc trong **Changelog**:
- *Những giai đoạn pipeline nào đã thay đổi so với lần chạy trước?* (Data, Preprocessing, Architecture, hay Training Strategy?)
- *Lý do kỹ thuật (Why) là gì?* (Giảm overfitting, tăng tốc hội tụ, hay giải quyết gradient tiêu biến?)
- *Tác động định lượng là gì?* (Mức tăng accuracy tuyệt đối, mức giảm khoảng cách tổng quát hóa, chênh lệch loss?)

### Bước 3: Cập nhật ma trận so sánh (Update Comparison Matrix)
- Thêm một cột mới cập nhật cho `Run #X` trong **Flow Comparison Matrix** của `walkthrough_lab_0X.md`.
- Đánh dấu tường minh ứng viên nào đang giữ kỷ lục **Best Checkpoint**.

### Bước 4: Thu hoạch artifact từ Kaggle và Tổng hợp phát hiện (Harvesting and Findings Synthesis)
- **Thu hoạch từ Kaggle:** Khi chạy trên Kaggle GPU:
  - Notebook ghi các artifact đầu ra vào `/kaggle/working/outputs/`:
    - Model checkpoint: `outputs/checkpoints/best_model.pt`
    - Lưới dự đoán (Input so với Predicted): `outputs/predictions_best.png`
    - Confusion matrix: `outputs/confusion_matrix.png`
    - Đường cong huấn luyện: `outputs/loss_curves.png`
  - Đồng bộ thông qua GitHub hoặc tải `outputs.zip` trở về `labs/lab0X/outputs/` cục bộ.
- **Tổng hợp phát hiện (Findings Synthesis):**
  - Đọc các artifact đầu ra và ghi lại các con số thực nghiệm đã xác thực vào `walkthrough_lab_0X.md`.
  - Trích xuất nhân tố đột phá chính, các pattern của lát cắt lỗi, và đường dẫn tới trọng số mô hình đã lưu.

## Tiêu chí xác thực

- `walkthrough_lab_0X.md` được đồng bộ với các con số thực nghiệm chính xác, không còn chỗ trống placeholder hay tuyên bố chưa được xác thực.
- Tài liệu sống cung cấp một dấu vết kiểm toán (audit trail) end-to-end sẵn sàng cho việc nộp bài.
