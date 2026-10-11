---
name: idea-refine
description: Tinh chỉnh và xác định phạm vi cho đề xuất nghiên cứu, đồ án tốt nghiệp và bài toán hackathon trong Data Science và AI. Hình thành định nghĩa bài toán khả thi, xác định dataset truy cập được, và đề xuất kiến trúc mô hình phù hợp.
---

# idea-refine (Bản DS & AI)

## Tổng quan

Chuyển hóa các ý tưởng rộng hoặc đề bài mở thành những đề xuất dự án Machine Learning / Deep Learning cụ thể, khả thi về mặt kỹ thuật, đồng thời xác định được dataset mở truy cập được và các backbone mô hình thực tế.

Skill này giải quyết vấn đề phổ biến nhất của người mới: ý tưởng quá tham vọng nhưng không có dữ liệu, hoặc dữ liệu không đủ để huấn luyện kiến trúc đã chọn. Nhiệm vụ của nó là "hạ cánh" ý tưởng về mặt thực tiễn trước khi tiêu tốn thời gian.

## Khi nào sử dụng

- Người dùng trình bày một chủ đề dự án còn chưa định hình (ví dụ: *"Tôi muốn làm một dự án AI cho y tế hoặc nông nghiệp"*).
- Đang xác định phạm vi cho bài dự thi hackathon hoặc đề xuất nghiên cứu học thuật.
- Cần cân nhắc đánh đổi giữa baseline Machine Learning cổ điển và hướng tiếp cận Deep Learning.

## Quy trình 4 bước

### Bước 1: Brainstorm phân kỳ
- Đề xuất 2 đến 3 hướng định hình bài toán khác nhau, bám sát lĩnh vực mà người dùng quan tâm.
- Với mỗi hướng, làm rõ:
  - Modality của đầu vào (Image, Text, Tabular, Audio).
  - Đầu ra mục tiêu (nhãn lớp rời rạc, giá trị vô hướng liên tục, bounding box, văn bản sinh tạo).
- Mục đích của bước này là mở rộng không gian lựa chọn, tránh khóa cứng vào một hướng duy nhất ngay từ đầu.

### Bước 2: Tìm nguồn dataset
- Xác định các kho dữ liệu công khai đã được kiểm chứng (Kaggle, Hugging Face Datasets, PapersWithCode, UCI Machine Learning Repository).
- Cảnh báo người dùng nếu bài toán đòi hỏi dữ liệu độc quyền (proprietary) hoặc chưa được làm sạch mà không thể thu thập được một cách thuận lợi.
- Một ý tưởng hay nhưng không có dữ liệu khả dụng chỉ là ý tưởng chết; bước này quyết định tính khả thi thực sự.

### Bước 3: Đánh giá khả thi về phần cứng và tính toán
- Đánh giá yêu cầu compute: Kiến trúc này có thể huấn luyện trong hạn mức GPU miễn phí (Kaggle/Colab Tesla T4 15GB VRAM) không?
- Đề xuất kiến trúc "vừa vặn" (right-sized): Tránh đề xuất các foundation model khổng lồ cho những bài toán chỉ cần ResNet hoặc gradient-boosted trees là đủ.

### Bước 4: Hội tụ và chọn lựa
- Trình bày đề xuất đã tinh chỉnh cho người dùng, bao gồm:
  - Tiêu đề dự án đề xuất.
  - Dataset công khai được khuyến nghị.
  - Mô hình baseline ban đầu và ứng viên vòng lặp nâng cao.

## Chống biện minh (Anti-Rationalization)

- [X] *"Luôn chọn mô hình SOTA phức tạp nhất để tối đa hóa điểm số"* -> **Bác bỏ:** Kiến trúc quá phức tạp trên dataset nhỏ gây overfitting nghiêm trọng và kéo dài chu kỳ huấn luyện. Ưu tiên các baseline khả thi, có thể kiểm chứng trước.
