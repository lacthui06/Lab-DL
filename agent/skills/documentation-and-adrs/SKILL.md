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

## Đầu ra tài liệu cho các bài Lab (Tích hợp vào Mục 3 của walkthrough_lab_0X.md)

Đối với các bài Lab, toàn bộ Báo cáo định lượng, Model Card và Quyết định kỹ thuật đều được tổng hợp trực tiếp vào **Mục 3 của file duy nhất `walkthrough_lab_0X.md`** (ứng với Rule 3 trong kiến trúc 3 Rule):
- **Kết quả định lượng & Ma trận so sánh:** Bảng đối đầu giữa Baseline Model và các Iteration nâng cao (Test Loss, Test Accuracy, F1-Macro).
- **Chẩn đoán Ma trận Nhầm lẫn (Confusion Matrix):** Phân tích chi tiết các cặp lớp dễ bị nhầm lẫn nhất (ví dụ Shirt vs T-shirt/Pullover).
- **Phân tích lỗi sai thực tế (Error Analysis):** Trích xuất các lát cắt mẫu dữ liệu bị dự đoán sai, nguyên nhân và bài học rút ra.
- **Quyết định Kỹ thuật & Kết luận (Key Findings):** Lý do chuyển đổi kiến trúc thành công, đánh giá hiệu quả của các kỹ thuật điều hòa (Regularization), LR Scheduler, Data Augmentation và hướng phát triển tiếp theo.
- **Kiểm chứng nạp Checkpoint (Save/Load Verification):** Xác nhận trọng số `best_model.pt` nạp lại suy luận cho kết quả đồng nhất 100%.

> Tuyệt đối KHÔNG sinh thêm file `REPORT.md` riêng lẻ, toàn bộ báo cáo hoàn chỉnh được tích hợp trực tiếp tại Mục 3 của `walkthrough_lab_0X.md`.

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
