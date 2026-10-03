---
name: documentation-and-adrs
description: Soạn thảo Model Card, báo cáo thí nghiệm kết quả, hồ sơ quyết định kiến trúc (ADR) và tài liệu bàn giao hoàn chỉnh cho dự án Data Science và AI.
---

# documentation-and-adrs (DS & AI Edition)

## Tổng quan
Kỹ năng này chịu trách nhiệm hoàn thiện khâu tài liệu chuyên môn cao nhất cho dự án Data Science và AI. Nó tạo ra các tài liệu chuẩn công nghiệp gồm: **Model Card (Thẻ thông tin mô hình)**, **Báo cáo thí nghiệm (Experiment Report)** và **Quyết định kiến trúc (ADRs)** để bất kỳ ai cũng có thể đọc hiểu, tái lập và nộp bài với điểm số tối đa.

## Khi nào sử dụng
- Khi kết thúc một dự án hoặc hoàn thành huấn luyện mô hình.
- Khi cần chuẩn bị tài liệu nộp bài tập lớn, khóa luận tốt nghiệp hoặc bàn giao cho nhóm sản phẩm.
- Khi cần giải thích lý do tại sao lại chọn thuật toán/mô hình này thay vì thuật toán khác.

## 3 Đầu ra tài liệu bắt buộc

### 1. Thẻ thông tin mô hình chuẩn mực (MODEL_CARD.md)
Áp dụng khung tiêu chuẩn *Model Card* của Google/Hugging Face:
- **Thông tin cơ bản:** Tên mô hình, kiến trúc cốt lõi, ngày hoàn thành, tác giả.
- **Mục đích sử dụng (Intended Use):** Mô hình được thiết kế để giải quyết bài toán gì? Không được dùng cho những trường hợp nào?
- **Dữ liệu huấn luyện & Đánh giá:** Nguồn dữ liệu, kích thước tập dữ liệu, phân bố nhãn.
- **Kết quả định lượng (Quantitative Results):**
  - Bảng so sánh giữa Baseline Model và Mô hình chính (Accuracy, Precision, Recall, F1-Macro, ROC-AUC).
  - Ma trận nhầm lẫn (Confusion Matrix).
- **Hạn chế & Lưu ý đạo đức (Limitations & Biases):** Những trường hợp mô hình có thể dự đoán sai hoặc dữ liệu bị thiếu hụt.

### 2. Hồ sơ Quyết định Kiến trúc (ADR - Architecture Decision Records)
Ghi lại lý do chọn các giải pháp kỹ thuật vào thư mục `docs/adr/`:
- *Ví dụ ADR 001:* Tại sao chọn `Focal Loss` thay cho `Binary Cross-Entropy`? (Giải thích: Vì dữ liệu bị mất cân bằng tỷ lệ 95:5).
- *Ví dụ ADR 002:* Tại sao chọn `ResNet-18` thay vì `ViT (Vision Transformer)`? (Giải thích: Vì tập dữ liệu chỉ có 2000 ảnh, dùng ViT sẽ bị Overfitting nặng nề và tài nguyên GPU có hạn).

### 3. Báo cáo bàn giao phiên làm việc (HANDOFF.md)
Tóm tắt ngắn gọn dành cho người dùng và các phiên làm việc tiếp theo:
- Danh sách các tệp mã nguồn và thư mục đã tạo.
- Vị trí lưu trữ file trọng số mô hình tốt nhất (`checkpoints/best_model.pt`).
- Lệnh một dòng để tái lập kết quả:
  ```bash
  python train.py --config config.yaml
  python evaluate.py --checkpoint checkpoints/best_model.pt
  ```

## Tiêu chí nghiệm thu (Verification)
- Mọi con số metric ghi trong tài liệu phải phản ánh đúng 100% kết quả từ log thực tế khi chạy mã nguồn, không bịa đặt hoặc làm tròn sai lệch.
