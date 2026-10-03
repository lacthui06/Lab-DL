---
name: interview-me
description: Phỏng vấn từng câu một để làm rõ yêu cầu, đặc tính dữ liệu, độ lệch nhãn và chỉ số mục tiêu của bài toán Data Science & AI. Sử dụng khi đề bài còn mơ hồ hoặc thiếu thông tin kỹ thuật quan trọng.
---

# interview-me (DS & AI Edition)

## Tổng quan
Kỹ năng này biến AI thành một Lead Data Scientist tiến hành vấn đáp người dùng **từng câu một**. Mục đích là làm rõ 95% các chi tiết ngầm của dữ liệu và bài toán trước khi bắt tay vào xây dựng pipeline.

## Khi nào sử dụng
- Người dùng giao một đề bài ngắn ngủn (ví dụ: *"Làm mô hình dự đoán churn"*, *"Phân loại ảnh X-quang"*).
- Chưa rõ cấu trúc dữ liệu, tỷ lệ mất cân bằng nhãn hoặc tài nguyên tính toán.
- Người dùng nói "hãy phỏng vấn tôi" hoặc "hỏi tôi các câu hỏi cần thiết".

## Quy tắc cốt lõi (Core Rules)
1. **Hỏi từng câu một (One Question at a Time):** Tuyệt đối không bắn ra một danh sách 5-10 câu hỏi cùng lúc làm người dùng choáng ngợp.
2. **Cung cấp gợi ý kèm câu hỏi:** Mỗi câu hỏi phải có sẵn 2 - 3 lựa chọn hoặc ví dụ minh họa để người dùng dễ trả lời.
3. **Dừng lại khi đạt 95% độ rõ ràng:** Không hỏi lan man các chi tiết phụ về code; chỉ tập trung vào nghiệp vụ dữ liệu, metric và tài nguyên.

## Các khía cạnh bắt buộc phải làm rõ qua phỏng vấn:
1. **Đặc tính Dữ liệu:** Kích thước bao nhiêu dòng/cột/ảnh? Dữ liệu đã được gán nhãn sạch sẽ chưa hay còn nhiều missing value/nhiễu?
2. **Phân phối Nhãn (Class Balance):** Các lớp có cân bằng không? (Ví dụ: 50-50 hay 99-1? Điều này quyết định việc có cần dùng SMOTE, Focal Loss, Class Weights hay không).
3. **Chỉ số đo lường (Metrics):** Ưu tiên độ chính xác tổng thể (Accuracy) hay ưu tiên bắt trúng ca bệnh/gian lận (Recall)? Cái giá của việc đoán nhầm (False Positive vs False Negative) là gì?
4. **Môi trường tính toán:** Chạy trên máy cá nhân (CPU/GPU) hay cloud (Google Colab, Kaggle, AWS)?

## Tiêu chí kết thúc
- Sau khi có đủ câu trả lời, tóm tắt lại bài toán thành bản tóm tắt 4 ý chính và tự động chuyển giao sang `spec-driven-development`.
