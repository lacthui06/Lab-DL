---
name: spec-driven-development
description: Soạn đặc tả kỹ thuật ML/DL trước khi triển khai code, liên kết và trích xuất chuẩn từ docs/dl_flow.md, docs/eda.md và docs/feature.md, ánh xạ trực tiếp vào Section 1 của walkthrough_lab_0X.md.
---

# spec-driven-development (Bản DS & AI)

## 1. Tổng quan

Skill này bắt buộc AI Agent phải soạn một **Đặc tả Machine Learning / Deep Learning (ML/DL Spec)** toàn diện trước khi viết code. Đặc tả này neo dự án vào các hợp đồng dữ liệu chuẩn hóa, công thức loss khách quan, kiến trúc baseline và tiêu chí đánh giá định lượng.

Nguyên lý đằng sau là: code là hệ quả của đặc tả, không phải ngược lại. Khi đặc tả rõ ràng, việc triển khai chỉ còn là công việc cơ học; khi đặc tả mơ hồ, mọi quyết định code đều trở thành phỏng đoán.

Đặc tả cũng là cơ sở cho mọi so sánh về sau: không có baseline và ngưỡng chấp nhận được định trước, không thể chứng minh mô hình nâng cao thực sự tốt hơn.

Nguyên lý then chốt: mọi quyết định thiết kế phải được chốt và ghi lại trước khi dòng code đầu tiên được viết.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Bắt đầu một lab mới hoặc dự án mới.
- Sau khi `interview-me` đã làm rõ yêu cầu.
- Cần chốt baseline, metric, và ngưỡng chấp nhận trước khi code.
- Cần trích xuất chuẩn kỹ thuật từ playbook workflow tương ứng.

**Điều kiện tiên quyết:**
- Đã biết modality và loại bài toán.
- Đã xác định dataset và môi trường compute.
- Đã có bản tóm tắt yêu cầu từ `interview-me` (nếu có).

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Yêu cầu đã làm rõ từ `interview-me` | Section 1 đầy đủ trong `walkthrough_lab_0X.md` |
| Đề bài gốc trong `description_lab_0X.md` | Bản tóm tắt cách hiểu đề đã xác nhận (Mục 1.0) |
| Playbook workflow tương ứng | Data contract và chiến lược chia tập |
| Ràng buộc compute và bộ nhớ | Baseline model, metric, ngưỡng chấp nhận |
| Đặc điểm dataset | Thiết kế loss, optimizer, scheduler |

## 4. Quy trình chi tiết

### Bước 0: Làm rõ đề bài gốc trước khi đặc tả
Đọc `description_lab_0X.md` và tóm tắt lại cách hiểu đề thành các gạch đầu dòng: mục tiêu, phạm vi, ràng buộc kỹ thuật, và sản phẩm cần nộp. Liệt kê mọi điểm mơ hồ, thiếu thông tin, hoặc giả định chưa được xác nhận, rồi trình bày cho người dùng và chờ xác nhận trước khi viết Section 1. Nếu còn điểm mơ hồ ảnh hưởng đến thiết kế, kích hoạt `interview-me` để làm rõ. Bản tóm tắt đã xác nhận được ghi vào Mục 1.0 của `walkthrough_lab_0X.md`. Bước này bắt buộc kể cả khi đề bài trông đã rõ ràng.

### Bước 1: Chọn playbook workflow tham chiếu
Xác định modality của bài toán. Với bài toán Deep Learning, tham chiếu `docs/dl_flow.md` cho modality, chiều tensor, backbone, và hàm loss chuyên biệt. Tham chiếu `docs/eda.md` và `docs/feature.md` cho hướng dẫn phân tích dữ liệu và kỹ thuật đặc trưng. Trích xuất chuẩn kỹ thuật từ playbook thay vì tự phát minh.

### Bước 2: Định hình bài toán và modality
Ghi rõ loại tác vụ (Classification, Regression, Segmentation, Object Detection, Sequence Modeling) và modality đầu vào (Tabular DataFrame, tensor ảnh 2D `[B, C, H, W]`, chuỗi token `[B, Seq_Len]`). Xác định hợp đồng đầu vào và đầu ra về shape và dtype.

### Bước 3: Thiết kế hợp đồng dữ liệu và chiến lược chia tập
Ghi rõ schema dữ liệu, kiểu dữ liệu, tỷ lệ cân bằng lớp. Chọn chiến lược chia tập không rò rỉ: mẫu độc lập dùng `StratifiedKFold`; chuỗi thời gian dùng `TimeSeriesSplit`; thực thể có nhóm dùng `GroupKFold`. Lựa chọn sai chiến lược chia tập là nguyên nhân rò rỉ dữ liệu phổ biến nhất.

### Bước 4: Chốt baseline model và metric mục tiêu
Chọn mô hình baseline tối thiểu khả thi để neo tiến bộ so sánh. Chọn metric chính (Macro F1-Score, ROC-AUC, mAP, IoU, RMSE). Định nghĩa ngưỡng chấp nhận: biên định lượng tối thiểu mà mô hình nâng cao phải đạt so với baseline.

### Bước 5: Thiết kế hàm loss và tối ưu
Chọn hàm loss phù hợp: phân loại đa lớp chuẩn dùng `CrossEntropyLoss` có label smoothing; mất cân bằng nghiêm trọng dùng Focal Loss hoặc class weights. Chọn optimizer (AdamW) và learning rate scheduler (CosineAnnealingLR hoặc OneCycleLR).

### Bước 6: Ghi ràng buộc môi trường thực thi
Ghi rõ mục tiêu compute (local hay Kaggle GPU Tesla T4), tương thích hệ thống file (`/kaggle/input/` chỉ đọc và `/kaggle/working/` ghi được), và ngân sách bộ nhớ (VRAM dưới 15GB, bật AMP FP16).

### Bước 7: Ghi đặc tả vào Section 1 của tài liệu sống
Tất cả nội dung trên được soạn trực tiếp trong Section 1 của `walkthrough_lab_0X.md`. Nghiêm cấm tạo file `SPEC.md` riêng. Đặc tả này là điểm tham chiếu cố định cho mọi quyết định code về sau.

## 5. Deliverables

- Section 1 đầy đủ trong `walkthrough_lab_0X.md` gồm: master flow, problem formulation, data contract, baseline, metric, loss, và ràng buộc môi trường.
- Danh sách chuẩn kỹ thuật trích xuất từ playbook workflow.
- Ngưỡng chấp nhận định lượng rõ ràng.
- Bản tóm tắt cách hiểu đề bài đã xác nhận với người dùng, ghi ở Mục 1.0 của `walkthrough_lab_0X.md`.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `interview-me` (cung cấp yêu cầu đã làm rõ) | `planning-and-task-breakdown` (nhận spec để phân rã công việc) |
| `idea-refine` (cung cấp hướng bài toán) | `test-driven-development` (nhận data contract để viết test) |
| `using-ds-ai-skills` (định tuyến) | `environment-and-reproducibility` (nhận ràng buộc môi trường) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Không có baseline:** Bỏ qua baseline khiến không có mốc so sánh. Dấu hiệu: không biết mô hình nâng cao có thực sự tốt hơn không. Phòng tránh: luôn định nghĩa baseline ở Bước 4.
- **Chọn sai chiến lược chia tập:** Dùng chia ngẫu nhiên cho dữ liệu chuỗi thời gian hoặc có nhóm. Dấu hiệu: metric validation cao bất thường. Phòng tránh: chọn đúng chiến lược ở Bước 3.
- **Metric không phù hợp:** Dùng Accuracy cho dataset mất cân bằng. Phòng tránh: chốt metric dựa trên phân bố lớp ở Bước 4.
- **Tạo file SPEC.md riêng:** Vi phạm kiến trúc tài liệu sống duy nhất. Phòng tránh: ghi vào Section 1 ở Bước 7.
- **Bỏ qua ràng buộc môi trường:** Code chạy local nhưng lỗi trên Kaggle. Phòng tránh: ghi ràng buộc ở Bước 6.

## 8. Definition of Done

- [ ] Đã đọc `description_lab_0X.md` và xác nhận làm rõ đề bài với người dùng.
- [ ] Đã chọn và trích xuất chuẩn từ playbook workflow tương ứng.
- [ ] Đã ghi rõ loại tác vụ và modality với hợp đồng tensor.
- [ ] Đã thiết kế data contract và chiến lược chia tập không rò rỉ.
- [ ] Đã chốt baseline model và metric chính.
- [ ] Đã định nghĩa ngưỡng chấp nhận định lượng.
- [ ] Đã chọn hàm loss, optimizer, và scheduler.
- [ ] Đã ghi ràng buộc môi trường và ngân sách bộ nhớ.
- [ ] Section 1 của `walkthrough_lab_0X.md` đã hoàn tất, không có file SPEC.md riêng.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Bài lab này ngắn, code luôn cho nhanh"* -> **Bác bỏ:** Triển khai không có spec sẽ đánh mất benchmark baseline, có nguy cơ chọn sai metric, và làm giảm độ chặt chẽ của báo cáo cuối.
- [X] *"Tách SPEC.md riêng cho gọn"* -> **Bác bỏ:** Gây phân mảnh tài liệu và trùng lặp nội dung. Kiến trúc tài liệu sống duy nhất yêu cầu hợp nhất vào Section 1.
- [X] *"Không cần baseline, mô hình chính là đủ"* -> **Bác bỏ:** Không có baseline thì không có cơ sở để chứng minh mô hình chính thực sự cải thiện. Mọi tuyên bố về hiệu năng trở nên vô căn cứ.
- [X] *"Chia tập ngẫu nhiên là đủ cho mọi loại dữ liệu"* -> **Bác bỏ:** Dữ liệu chuỗi thời gian hoặc có nhóm cần chiến lược chia chuyên biệt, nếu không sẽ rò rỉ dữ liệu và cho metric ảo.
- [X] *"Đề bài đã rõ ràng, khỏi cần tóm tắt lại"* -> **Bác bỏ:** Đề bài rõ với người giao không có nghĩa là rõ với người làm. Hiểu sai đề là nguyên nhân số một khiến phải làm lại; tóm tắt và xác nhận tốn vài phút nhưng chặn được hàng giờ làm sai.
