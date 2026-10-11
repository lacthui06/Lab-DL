---
name: idea-refine
description: Tinh chỉnh và xác định phạm vi cho đề xuất nghiên cứu, đồ án tốt nghiệp và bài toán hackathon trong Data Science và AI, hình thành định nghĩa bài toán khả thi và đề xuất kiến trúc phù hợp năng lực tính toán.
---

# idea-refine (Bản DS & AI)

## 1. Tổng quan

Skill này chuyển hóa các ý tưởng rộng hoặc đề bài mở thành những đề xuất dự án Machine Learning / Deep Learning cụ thể, khả thi về mặt kỹ thuật, đồng thời xác định được dataset mở truy cập được và các backbone mô hình thực tế.

Skill giải quyết vấn đề phổ biến nhất của người mới: ý tưởng quá tham vọng nhưng không có dữ liệu, hoặc dữ liệu không đủ để huấn luyện kiến trúc đã chọn. Nhiệm vụ của nó là "hạ cánh" ý tưởng về mặt thực tiễn trước khi tiêu tốn thời gian.

Skill này cũng đóng vai trò lá chắn chống lại cám dỗ chọn mô hình SOTA phức tạp cho mọi bài toán, vốn là nguyên nhân hàng đầu gây overfitting trên dataset nhỏ.

Nguyên lý then chốt: một ý tưởng chỉ khả thi khi nó có dữ liệu truy cập được và kiến trúc vừa với năng lực tính toán hiện có.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Người dùng trình bày một chủ đề dự án còn chưa định hình (ví dụ: "Tôi muốn làm dự án AI cho y tế hoặc nông nghiệp").
- Đang xác định phạm vi cho bài dự thi hackathon hoặc đề xuất nghiên cứu.
- Cần cân nhắc đánh đổi giữa baseline Machine Learning cổ điển và Deep Learning.
- Người dùng hỏi "nên làm đề tài gì".

**Điều kiện tiên quyết:**
- Đã biết sơ bộ lĩnh vực quan tâm của người dùng.
- Chưa có ràng buộc cứng về dataset hay kiến trúc.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Lĩnh vực hoặc chủ đề quan tâm | 2-3 hướng định hình bài toán khả thi |
| Ràng buộc về thời gian và compute | Dataset công khai được khuyến nghị |
| Trình độ kỹ thuật của người dùng | Đề xuất cuối gồm tên, dataset, baseline, ứng viên nâng cao |

## 4. Quy trình chi tiết

### Bước 1: Brainstorm phân kỳ
Đề xuất 2 đến 3 hướng định hình bài toán khác nhau, bám sát lĩnh vực người dùng quan tâm. Với mỗi hướng, làm rõ modality đầu vào (Image, Text, Tabular, Audio) và đầu ra mục tiêu (nhãn lớp rời rạc, giá trị vô hướng liên tục, bounding box, văn bản sinh tạo). Mục đích là mở rộng không gian lựa chọn, tránh khóa cứng vào một hướng ngay từ đầu.

### Bước 2: Tìm nguồn dataset
Xác định các kho dữ liệu công khai đã được kiểm chứng (Kaggle, Hugging Face Datasets, PapersWithCode, UCI Machine Learning Repository). Cảnh báo người dùng nếu bài toán đòi hỏi dữ liệu độc quyền hoặc chưa được làm sạch mà không thể thu thập thuận lợi. Một ý tưởng hay nhưng không có dữ liệu khả dụng chỉ là ý tưởng chết.

### Bước 3: Đánh giá khả thi phần cứng và tính toán
Đánh giá yêu cầu compute: kiến trúc này có thể huấn luyện trong hạn mức GPU miễn phí (Kaggle/Colab Tesla T4 15GB VRAM) không. Đề xuất kiến trúc "vừa vặn": tránh foundation model khổng lồ cho bài toán chỉ cần ResNet hoặc gradient-boosted trees.

### Bước 4: Đánh giá chất lượng và quy mô dữ liệu
Kiểm tra sơ bộ: dataset có đủ mẫu để huấn luyện mô hình dự kiến không, có nhãn sẵn hay cần gán nhãn, và có vấn đề bản quyền hay không. Bước này ngăn chặn việc chọn một bài toán mà dữ liệu không đủ để mô hình học được gì.

### Bước 5: Hội tụ và chọn lựa
Trình bày đề xuất đã tinh chỉnh cho người dùng, gồm tiêu đề dự án, dataset công khai được khuyến nghị, mô hình baseline ban đầu, và ứng viên vòng lặp nâng cao. Xác nhận người dùng đồng ý trước khi chuyển sang `interview-me` hoặc `spec-driven-development`.

## 5. Deliverables

- Danh sách 2-3 hướng định hình bài toán kèm modality và đầu ra.
- Danh sách dataset công khai khả dụng kèm nguồn.
- Đánh giá khả thi compute và kích thước dữ liệu.
- Đề xuất cuối cùng được người dùng xác nhận.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| Không có (điểm vào sau định tuyến) | `interview-me` (làm rõ chi tiết yêu cầu) |
| `using-ds-ai-skills` (cung cấp ngữ cảnh ban đầu) | `spec-driven-development` (chốt đặc tả kỹ thuật) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Chọn bài toán không có dữ liệu:** Ý tưởng hay nhưng không có dataset truy cập được. Dấu hiệu: phải tự thu thập và gán nhãn quy mô lớn. Phòng tránh: kiểm tra nguồn dữ liệu ở Bước 2 trước khi cam kết.
- **Chọn kiến trúc quá lớn:** Đề xuất foundation model cho dataset nhỏ. Dấu hiệu: overfitting ngay từ epoch đầu. Phòng tránh: đánh giá khả thi ở Bước 3.
- **Bỏ qua bản quyền dữ liệu:** Dùng dataset không được phép. Phòng tránh: kiểm tra giấy phép ở Bước 4.
- **Khóa cứng quá sớm:** Chỉ đề xuất một hướng duy nhất. Phòng tránh: luôn brainstorm 2-3 hướng ở Bước 1.

## 8. Definition of Done

- [ ] Đã đề xuất 2-3 hướng định hình bài toán khác nhau.
- [ ] Mỗi hướng đã xác định rõ modality và đầu ra mục tiêu.
- [ ] Đã xác định dataset công khai khả dụng cho hướng được chọn.
- [ ] Đã đánh giá khả thi compute và kích thước dữ liệu.
- [ ] Đã trình bày đề xuất cuối gồm tên, dataset, baseline, ứng viên nâng cao.
- [ ] Người dùng đã xác nhận hướng đi.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Luôn chọn mô hình SOTA phức tạp nhất để tối đa hóa điểm số"* -> **Bác bỏ:** Kiến trúc quá phức tạp trên dataset nhỏ gây overfitting nghiêm trọng và kéo dài chu kỳ huấn luyện. Ưu tiên baseline khả thi, có thể kiểm chứng.
- [X] *"Không cần kiểm tra dữ liệu trước, cứ chọn đề tài rồi tính"* -> **Bác bỏ:** Chọn đề tài không có dữ liệu là nguyên nhân số một khiến dự án đổ vỡ giữa chừng. Kiểm tra nguồn dữ liệu là bắt buộc trước khi cam kết.
- [X] *"Chỉ cần một hướng là đủ, khỏi brainstorm"* -> **Bác bỏ:** Không so sánh nhiều hướng thì không biết hướng đã chọn có phải tốt nhất. Brainstorm giúp phát hiện hướng tốt hơn với ít công sức.
