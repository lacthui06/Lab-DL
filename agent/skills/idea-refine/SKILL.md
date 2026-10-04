---
name: idea-refine
description: Mài giũa và định hình các ý tưởng đề tài nghiên cứu, đồ án tốt nghiệp, bài toán Hackathon trong lĩnh vực Data Science và AI. Sử dụng khi người dùng có ý tưởng sơ khai hoặc cần tìm hướng tiếp cận giải pháp AI.
---

# idea-refine (DS & AI Edition)

## Tổng quan
Chuyển hóa một ý tưởng sơ bộ hoặc một đề tài mở thành một đề xuất giải pháp kỹ thuật Machine Learning / Deep Learning có tính khả thi cao, xác định nguồn dữ liệu sẵn có và kiến trúc mô hình phù hợp.

## Khi nào sử dụng
- Người dùng chưa có đề tài cụ thể (ví dụ: *"Tôi muốn làm đồ án AI về nông nghiệp / y tế"*).
- Tham gia cuộc thi Hackathon AI hoặc bắt đầu dự án nghiên cứu mới.
- Cần so sánh tính khả thi giữa các hướng tiếp cận (Deep Learning vs Mô hình truyền thống).

## Quy trình 4 bước (Process)

### Bước 1: Tư duy phân kỳ (Divergent Brainstorming)
- Liệt kê 2 - 3 bài toán cụ thể có thể giải quyết được bằng AI trong lĩnh vực người dùng quan tâm.
- Với mỗi bài toán, chỉ rõ:
  - Input dữ liệu là gì (Ảnh, Văn bản, Bảng tabular, Audio)?
  - Output mong đợi là gì (Nhãn phân loại, Giá trị liên tục, Bounding box, Text sinh ra)?

### Bước 2: Khảo sát nguồn dữ liệu mở (Dataset Sourcing)
- Kiểm tra tính sẵn có của dữ liệu công khai (Kaggle, Hugging Face Datasets, PapersWithCode, UCI Machine Learning Repository).
- Cảnh báo ngay nếu bài toán đòi hỏi dữ liệu riêng tư hoặc quá hiếm mà người dùng không thể tự thu thập.

### Bước 3: Đánh giá tính khả thi phần cứng & Thời gian (Resource Check)
- Đánh giá yêu cầu tài nguyên: Bài toán này cần GPU gì? Chạy trên Google Colab miễn phí (T4 15GB VRAM) có train nổi không, hay cần máy chủ chuyên dụng?
- Đề xuất kiến trúc phù hợp: Tránh đề xuất các mô hình khổng lồ (như LLM 70B hoặc mô hình quá nặng) cho các bài toán chỉ cần ResNet hoặc XGBoost.

### Bước 4: Tư duy hội tụ (Convergent Selection)
- Trình bày 1 đề xuất giải pháp tối ưu nhất cho người dùng kèm:
  - Tên đề tài gợi ý.
  - Bộ dữ liệu khuyến nghị sử dụng.
  - Kiến trúc mô hình thử nghiệm ban đầu (Baseline) và mô hình nâng cao.

## Anti-Rationalization
- [X] *"Cứ chọn mô hình mới nhất (SOTA) phức tạp nhất là điểm sẽ cao"* -> **Bác bỏ:** Mô hình phức tạp trên tập dữ liệu nhỏ sẽ bị Overfitting nặng nề và không kịp train. Ưu tiên giải pháp khả thi với tài nguyên hiện có.
