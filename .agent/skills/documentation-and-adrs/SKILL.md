---
name: documentation-and-adrs
description: Soạn Model Card, báo cáo đánh giá kỹ thuật, Architecture Decision Record và tài liệu nộp bài cho các dự án Data Science và AI, ánh xạ trực tiếp vào Section 3 của walkthrough_lab_0X.md.
---

# documentation-and-adrs (Bản DS & AI)

## 1. Tổng quan

Skill này xử lý tài liệu kỹ thuật chuyên nghiệp cho các dự án DS và AI, cung cấp các artifact chuẩn công nghiệp: **Model Card**, **Báo cáo đánh giá so sánh**, và **Architecture Decision Record (ADR)** nhằm đảm bảo khả năng tái lập đầy đủ, khả năng diễn giải, và độ chặt chẽ học thuật.

Điểm cốt lõi: tài liệu không phải là thủ tục hành chính mà là bằng chứng kỹ thuật. Mọi con số trong tài liệu phải khớp 100% với log thực nghiệm.

Skill này chịu sự điều phối của Rule 3 trong Kiến trúc 3 Rule và ghi trực tiếp vào Section 3 của tài liệu sống duy nhất.

Nguyên lý then chốt: tài liệu phải trả lời được ba câu hỏi - kết quả là gì, vì sao đạt được, và còn hạn chế nào.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Kết thúc các lần chạy thí nghiệm và tạo báo cáo môn học hoặc dự án cuối cùng.
- Cần ghi lại lý do chọn một kiến trúc, hàm loss, hoặc lịch hyperparameter cụ thể.
- Cần tổng hợp confusion matrix và các chế độ thất bại thành phân tích lỗi.
- Chuẩn bị nộp bài hoặc bàn giao dự án.

**Điều kiện tiên quyết:**
- Đã có Section 1 (đặc tả) và Section 2 (changelog) hoàn tất.
- Đã có đầy đủ artifact và số liệu đã xác thực.
- Đã chạy kiểm toán `doubt-driven-development` không phát hiện rò rỉ.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Section 1 và Section 2 đã hoàn tất | Section 3 đầy đủ trong `walkthrough_lab_0X.md` |
| Artifact và số liệu đã xác thực | Ma trận so sánh và chẩn đoán confusion matrix |
| Phán quyết kiểm toán từ doubt-driven | Phân tích lỗi và kết luận cuối |
| | ADR cho các quyết định kiến trúc quan trọng |

## 4. Quy trình chi tiết

### Bước 1: Xây dựng ma trận đánh giá định lượng
Tạo bảng so sánh trực tiếp giữa baseline và các vòng lặp nâng cao, bao gồm Test Loss, Test Accuracy, và Macro F1-Score. Bảng này là bằng chứng định lượng cho toàn bộ tiến bộ của dự án.

### Bước 2: Chẩn đoán confusion matrix
Phân tích sâu các cặp lớp nhầm lẫn cao (ví dụ Shirt so với T-shirt/Pullover) và giải thích nguyên nhân trực quan. Chẩn đoán này giải thích mô hình yếu ở đâu và vì sao.

### Bước 3: Phân tích lát cắt lỗi
Kiểm tra trực quan và thống kê các mẫu bị phân loại sai, xác định nguyên nhân gốc của thất bại. Phân tích lỗi biến kết quả thô thành hiểu biết có thể hành động.

### Bước 4: Tổng hợp phát hiện kỹ thuật chính
Giải thích các đột phá về hiệu năng, đánh giá tác động của regularization, learning rate scheduler, và data augmentation. Phần này trả lời câu hỏi "vì sao đạt được kết quả".

### Bước 5: Truy vết quyết định về điểm thảo luận và chuẩn hóa ngôn ngữ
Với mỗi quyết định kỹ thuật quan trọng (kiến trúc, hàm loss, metric, chiến lược chia tập), ghi rõ điểm thảo luận tương ứng ở Mục 1.0 của `walkthrough_lab_0X.md` và hướng giải quyết đã chốt, thường dưới dạng bảng Decision Traceability. Dùng văn phong kỹ thuật trung tính, ngôi thứ ba; không dùng lối kể chuyện hay từ chỉ người như "bạn", "tôi", "chúng ta". Ví dụ đúng: "Chiến lược chia tập đã được thảo luận và chốt dùng StratifiedKFold do phân bố lớp cân bằng."

### Bước 6: Xác thực checkpoint (Save/Load)
Xác nhận rằng việc nạp lại checkpoint tốt nhất tái lập 100% metric inference giống hệt. Đây là bằng chứng cuối cùng về tính tái lập của kết quả.

### Bước 7: Ghi Architecture Decision Record
Với các dự án kỹ thuật lâu dài, ghi lại các lựa chọn kiến trúc quan trọng dưới `docs/adr/`. Ví dụ: vì sao chọn Focal Loss thay vì Binary Cross-Entropy; vì sao chọn ResNet thay vì Vision Transformer cho dataset nhỏ. ADR lưu lại lý do đằng sau các quyết định để không phải suy đoán lại về sau.

### Bước 8: Tạo báo cáo bàn giao phiên
Tạo bản tóm tắt bàn giao ngắn gọn cho phiên phát triển tiếp theo: danh sách module source và artifact đã tạo, đường dẫn checkpoint tốt nhất, và hướng dẫn tái lập kết quả bằng một lệnh duy nhất.

### Bước 9: Kiểm chứng tính chính xác của tài liệu
Xác nhận mọi metric được ghi tài liệu khớp 100% với log thực nghiệm và confusion matrix đầu ra. Không có con số ảo tưởng hay làm tròn thiếu chính xác.

## 5. Deliverables

- Section 3 của `walkthrough_lab_0X.md` đầy đủ: ma trận đánh giá, chẩn đoán, phân tích lỗi, truy vết quyết định về Mục 1.0, kết luận.
- Architecture Decision Record cho các quyết định kiến trúc quan trọng.
- Báo cáo bàn giao phiên với hướng dẫn tái lập.
- Xác nhận checkpoint tái lập 100% metric.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `walkthrough-and-experiment-tracking` (cung cấp số liệu và changelog) | `version-control-and-artifacts` (commit tài liệu cuối) |
| `doubt-driven-development` (cung cấp phán quyết kiểm toán) | Checklist `pre_submission.md` (cổng nộp bài) |
| `api-and-interface-design` (cung cấp thông tin dịch vụ nếu có) | |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Tạo REPORT.md riêng:** Vi phạm kiến trúc tài liệu sống duy nhất. Dấu hiệu: phân mảnh tài liệu. Phòng tránh: ghi vào Section 3.
- **Số liệu không khớp log:** Chép tay sai hoặc làm tròn khác. Dấu hiệu: số trong tài liệu khác số trong log. Phòng tránh: kiểm chứng ở Bước 8.
- **Bỏ qua phân tích lỗi:** Chỉ trình bày kết quả tốt mà không phân tích thất bại. Dấu hiệu: báo cáo thiếu chiều sâu. Phòng tránh: thực hiện Bước 2 và 3.
- **Không ghi ADR:** Quên lý do đằng sau các quyết định kiến trúc. Phòng tránh: ghi ADR ở Bước 6.
- **Bỏ qua xác thực checkpoint:** Không chứng minh được tái lập. Phòng tránh: thực hiện Bước 5.

## 8. Definition of Done

- [ ] Mỗi quyết định kỹ thuật quan trọng đã được truy vết về điểm thảo luận ở Mục 1.0.
- [ ] Ma trận đánh giá định lượng đã hoàn tất với đầy đủ các run.
- [ ] Chẩn đoán confusion matrix đã phân tích các cặp lớp nhầm lẫn cao.
- [ ] Phân tích lát cắt lỗi đã xác định nguyên nhân gốc.
- [ ] Phát hiện kỹ thuật chính đã được tổng hợp.
- [ ] Checkpoint đã được xác thực tái lập 100% metric.
- [ ] ADR đã được ghi cho các quyết định kiến trúc quan trọng.
- [ ] Mọi metric khớp 100% với log thực nghiệm.
- [ ] Section 3 hoàn tất, không có file REPORT.md riêng.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Tách REPORT.md riêng cho dễ đọc"* -> **Bác bỏ:** Gây phân mảnh tài liệu và trùng lặp nội dung. Kiến trúc tài liệu sống duy nhất yêu cầu hợp nhất vào Section 3.
- [X] *"Chỉ cần trình bày kết quả tốt, khỏi phân tích lỗi"* -> **Bác bỏ:** Phân tích lỗi là phần thể hiện hiểu biết sâu về mô hình. Bỏ qua nó làm báo cáo thiếu chiều sâu và kém thuyết phục.
- [X] *"Số liệu gần đúng cũng được, không cần chính xác tuyệt đối"* -> **Bác bỏ:** Số liệu không khớp log là dấu hiệu của ghi chép cẩu thả và làm mất độ tin cậy của toàn bộ báo cáo.
- [X] *"Không cần xác thực checkpoint, chắc là giống"* -> **Bác bỏ:** Xác thực tái lập là bằng chứng cuối cùng cho tính khoa học của kết quả. Không xác thực thì không chứng minh được gì.
- [X] *"Khỏi trích dẫn thảo luận, viết kết luận là đủ"* -> **Bác bỏ:** Báo cáo không truy vết được về thảo luận ban đầu thì người chấm không biết quyết định nào đã được cân nhắc và chốt. Thiếu vết truy vết làm báo cáo mất tính kiểm chứng.
- [X] *"Viết theo lối kể chuyện cho gần gũi"* -> **Bác bỏ:** Tài liệu kỹ thuật phải trung tính và ngôi thứ ba. Lối kể chuyện và từ chỉ người làm giảm độ chuyên nghiệp và khó kiểm chứng.
