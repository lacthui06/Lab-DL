---
name: documentation-and-adrs
description: Soạn Model Card, báo cáo đánh giá kỹ thuật, Architecture Decision Record (ADR) và tài liệu nộp bài cho các dự án Data Science và AI. Ánh xạ trực tiếp vào Section 3 của walkthrough_lab_0X.md.
---

# documentation-and-adrs (Bản DS & AI)

## Tổng quan

Skill này xử lý tài liệu kỹ thuật chuyên nghiệp cho các dự án DS và AI, cung cấp các artifact chuẩn công nghiệp: **Model Card**, **Báo cáo đánh giá so sánh**, và **Architecture Decision Record (ADR)** nhằm đảm bảo khả năng tái lập đầy đủ, khả năng diễn giải, và độ chặt chẽ học thuật.

Điểm cốt lõi: tài liệu không phải là thủ tục hành chính mà là bằng chứng kỹ thuật. Mọi con số trong tài liệu phải khớp 100% với log thực nghiệm.

## Khi nào sử dụng

- Kết thúc các lần chạy thí nghiệm và tạo báo cáo môn học hoặc dự án cuối cùng.
- Ghi lại lý do vì sao chọn một kiến trúc, hàm loss, hoặc lịch hyperparameter cụ thể thay vì các lựa chọn thay thế khác.
- Tổng hợp confusion matrix và các chế độ thất bại thành phân tích lỗi có thể hành động.

## Các tài liệu bàn giao cho bài lab môn học (tích hợp vào Section 3 của `walkthrough_lab_0X.md`)

Với các bài lab thực hành, toàn bộ đánh giá định lượng, tóm tắt chẩn đoán, và quyết định kỹ thuật được hợp nhất trực tiếp vào **Section 3 của tài liệu sống duy nhất `walkthrough_lab_0X.md`** (chịu sự điều phối của Rule 3 trong Kiến trúc 3 Rule), bao gồm:
- **Ma trận đánh giá định lượng và so sánh:** Bảng so sánh trực tiếp giữa tất cả các lần chạy (Baseline so với các vòng lặp nâng cao), bao gồm Test Loss, Test Accuracy, và Macro F1-Score.
- **Chẩn đoán Confusion Matrix:** Phân tích sâu các cặp lớp nhầm lẫn cao (ví dụ Shirt so với T-shirt/Pullover).
- **Lát cắt phân tích lỗi (Error Analysis Slices):** Kiểm tra trực quan và thống kê các mẫu bị phân loại sai, xác định nguyên nhân gốc của thất bại.
- **Phát hiện kỹ thuật chính:** Giải thích các đột phá về hiệu năng, đánh giá tác động của regularization, learning rate scheduler, và data augmentation.
- **Xác thực Checkpoint (Save/Load):** Xác nhận rằng việc nạp lại `best_model.pt` tái lập 100% metric inference giống hệt.

Nghiêm cấm tạo file `REPORT.md` riêng biệt. Mọi phát hiện cuối cùng được tích hợp trực tiếp vào Section 3 của `walkthrough_lab_0X.md`.

### 2. Architecture Decision Record (ADR)
Khi duy trì các dự án kỹ thuật lâu dài, ghi lại các lựa chọn kiến trúc quan trọng dưới `docs/adr/`:
- *ADR 001:* Vì sao chọn Focal Loss thay vì Binary Cross-Entropy? (Lý do: Giảm mất cân bằng lớp nghiêm trọng 95:5).
- *ADR 002:* Vì sao chọn ResNet thay vì Vision Transformer (ViT)? (Lý do: Kích thước mẫu nhỏ 2,000 ảnh gây ViT overfitting, cộng với bộ nhớ GPU hạn chế).

### 3. Báo cáo bàn giao phiên làm việc (`HANDOFF.md`)
Bản tóm tắt bàn giao ngắn gọn cho các phiên phát triển tiếp theo:
- Danh sách các module source và artifact đầu ra đã tạo.
- Đường dẫn file tới checkpoint mô hình tốt nhất đã xác thực (`outputs/checkpoints/best_model.pt`).
- Hướng dẫn tái lập kết quả bằng một lệnh duy nhất:
  ```bash
  python train.py --config config.yaml
  python evaluate.py --checkpoint checkpoints/best_model.pt
  ```

## Tiêu chí xác thực

- Mọi metric được ghi tài liệu khớp 100% với log thực nghiệm và confusion matrix đầu ra; không có con số ảo tưởng hay làm tròn thiếu chính xác.
