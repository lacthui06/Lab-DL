---
name: interview-me
description: Thực hiện phỏng vấn có cấu trúc, mỗi lần một câu hỏi, nhằm làm rõ yêu cầu bài toán, đặc điểm dataset, tỷ lệ cân bằng lớp, metric đánh giá và ràng buộc tính toán trước khi viết code.
---

# interview-me (Bản DS & AI)

## 1. Tổng quan

Skill này hướng dẫn AI Agent đóng vai một **Lead Data Scientist**, tiến hành phỏng vấn tập trung theo kiểu mỗi lần một câu hỏi. Mục đích là làm rõ các sắc thái quan trọng của dữ liệu và yêu cầu đánh giá trước khi viết bất kỳ dòng code pipeline nào.

Hiểu sai yêu cầu ngay từ đầu là nguyên nhân số một dẫn đến việc phải làm lại toàn bộ pipeline. Skill này đóng vai trò "khóa chặt" các giả định trước khi chúng trở thành lỗi tốn kém.

Skill cũng ngăn chặn một lỗi phổ biến khác: giả định sai về metric. Ví dụ, tối ưu Accuracy trên một dataset mất cân bằng nghiêm trọng có thể cho ra mô hình vô dụng trong thực tế dù chỉ số trông đẹp.

Nguyên lý then chốt: mỗi câu hỏi phải nhắm vào một quyết định thiết kế cụ thể, và chỉ dừng khi đủ 95% thông tin cần thiết để đặc tả.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Người dùng đưa ra prompt mơ hồ (ví dụ: "Xây dựng bộ dự đoán khách hàng rời bỏ", "Phân loại ảnh X-quang y tế").
- Schema dữ liệu, tỷ lệ cân bằng lớp, hoặc môi trường compute chưa được xác định.
- Người dùng yêu cầu rõ ràng: "Hãy phỏng vấn tôi" hoặc "Đặt câu hỏi làm rõ cho tôi".
- Sau khi `idea-refine` đã chọn hướng nhưng chi tiết còn thiếu.

**Điều kiện tiên quyết:**
- Đã xác định được lĩnh vực và hướng bài toán sơ bộ.
- Người dùng sẵn sàng trả lời các câu hỏi làm rõ.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Prompt mơ hồ của người dùng | Bản tóm tắt 4 điểm đã được làm rõ |
| Thông tin dữ liệu hiện có | Xác nhận về đặc điểm dữ liệu và phân bố lớp |
| Bối cảnh tính toán | Xác nhận về metric mục tiêu và ràng buộc compute |

## 4. Quy trình chi tiết

### Bước 1: Xác định các chiều thông tin còn thiếu
Đọc prompt và xác định chiều nào trong bốn chiều bắt buộc còn thiếu: đặc điểm dataset, phân bố lớp, metric mục tiêu, và ràng buộc compute. Chỉ hỏi về những chiều thực sự chưa rõ, tránh hỏi lại thông tin đã có.

### Bước 2: Hỏi từng câu một theo thứ tự ưu tiên
Hỏi từng câu, bắt đầu từ chiều quan trọng nhất (thường là đặc điểm dataset và metric). Mỗi câu hỏi đi kèm 2 đến 3 lựa chọn cụ thể để người dùng quyết định nhanh. Không dồn dập nhiều câu hỏi cùng lúc vì gây quá tải và giảm chất lượng câu trả lời.

### Bước 3: Làm rõ đặc điểm dataset
Hỏi về khối lượng mẫu, mức độ làm sạch, và tình trạng nhãn. Xác định dataset đã sẵn sàng dùng hay còn cần tiền xử lý lớn. Thông tin này quyết định khối lượng công việc ở giai đoạn ingestion.

### Bước 4: Làm rõ phân bố lớp
Hỏi tỷ lệ giữa các lớp là cân bằng hay mất cân bằng, và mức độ cụ thể. Điều này quyết định việc có cần Focal Loss, Class Weights, hay chia tập Stratified hay không. Bỏ qua bước này thường dẫn đến chọn sai hàm loss.

### Bước 5: Làm rõ metric mục tiêu
Hỏi Accuracy tổng thể có đủ không, hay Recall là tối quan trọng (ví dụ chẩn đoán y tế, phát hiện gian lận). Làm rõ hình phạt của False Positive so với False Negative. Metric sai khiến mô hình tối ưu nhầm mục tiêu.

### Bước 6: Làm rõ ràng buộc compute
Hỏi môi trường chạy là local (CPU/GPU) hay đám mây (Kaggle, Colab, AWS). Điều này quyết định kích thước batch, kiến trúc khả thi, và ngân sách VRAM.

### Bước 7: Tổng hợp và chuyển tiếp
Sau khi đã đủ 95% thông tin, tổng hợp bản tóm tắt 4 điểm và chuyển tiếp sang `spec-driven-development`. Không tiếp tục hỏi những tiểu tiết triển khai không ảnh hưởng đến quyết định thiết kế.

## 5. Deliverables

- Bản tóm tắt 4 điểm: đặc điểm dataset, phân bố lớp, metric mục tiêu, ràng buộc compute.
- Danh sách các giả định đã được xác nhận với người dùng.
- Điểm chuyển tiếp rõ ràng sang `spec-driven-development`.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `idea-refine` (cung cấp hướng bài toán sơ bộ) | `spec-driven-development` (nhận bản tóm tắt 4 điểm để viết spec) |
| `using-ds-ai-skills` (phát hiện yêu cầu mơ hồ) | `planning-and-task-breakdown` (nhận ràng buộc để lập kế hoạch) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Hỏi dồn dập nhiều câu:** Gây quá tải, người dùng trả lời sơ sài. Dấu hiệu: câu trả lời chung chung. Phòng tránh: hỏi từng câu một ở Bước 2.
- **Bỏ qua phân bố lớp:** Không hỏi tỷ lệ lớp dẫn đến chọn sai loss. Dấu hiệu: mô hình đạt Accuracy cao nhưng Recall lớp thiểu số thấp. Phòng tránh: luôn hỏi ở Bước 4.
- **Giả định metric mặc định:** Mặc định dùng Accuracy cho mọi bài toán. Phòng tránh: làm rõ metric ở Bước 5.
- **Hỏi quá nhiều tiểu tiết:** Kéo dài phỏng vấn vô ích. Phòng tránh: dừng ở 95% rõ ràng ở Bước 7.

## 8. Definition of Done

- [ ] Đã xác định được các chiều thông tin còn thiếu.
- [ ] Đã hỏi từng câu một, mỗi câu kèm lựa chọn cụ thể.
- [ ] Đã làm rõ đặc điểm dataset và tình trạng nhãn.
- [ ] Đã làm rõ phân bố lớp và mức độ mất cân bằng.
- [ ] Đã làm rõ metric mục tiêu và hình phạt FP/FN.
- [ ] Đã làm rõ ràng buộc compute.
- [ ] Đã tổng hợp bản tóm tắt 4 điểm.
- [ ] Đã chuyển tiếp sang `spec-driven-development`.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Hỏi nhiều câu cùng lúc cho nhanh"* -> **Bác bỏ:** Dồn dập gây quá tải nhận thức, người dùng trả lời hời hợt và bỏ sót chiều quan trọng. Hỏi từng câu cho câu trả lời chất lượng cao hơn.
- [X] *"Cứ mặc định dùng Accuracy cho mọi bài toán"* -> **Bác bỏ:** Trên dataset mất cân bằng, Accuracy cao có thể che giấu việc mô hình bỏ qua hoàn toàn lớp thiểu số. Phải làm rõ metric.
- [X] *"Phân bố lớp chắc cân bằng, khỏi hỏi"* -> **Bác bỏ:** Giả định sai về phân bố lớp dẫn đến chọn sai hàm loss và metric. Hỏi một câu rẻ hơn nhiều so với làm lại pipeline.
- [X] *"Hỏi cho hết mọi thứ rồi mới làm"* -> **Bác bỏ:** Hỏi quá nhiều tiểu tiết triển khai kéo dài vô ích. Dừng ở 95% thông tin cần thiết cho thiết kế.
