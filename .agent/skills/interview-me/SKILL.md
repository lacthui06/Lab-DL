---
name: interview-me
description: Thực hiện phỏng vấn có cấu trúc, mỗi lần một câu hỏi, nhằm làm rõ yêu cầu bài toán, đặc điểm dataset, tỷ lệ cân bằng lớp và metric đánh giá cho các dự án Data Science và AI.
---

# interview-me (Bản DS & AI)

## Tổng quan

Skill này hướng dẫn AI Agent đóng vai một **Lead Data Scientist**, tiến hành phỏng vấn tập trung theo kiểu **mỗi lần một câu hỏi**. Mục đích là làm rõ các sắc thái quan trọng của dữ liệu và yêu cầu đánh giá trước khi viết bất kỳ dòng code pipeline nào.

Việc hiểu sai yêu cầu ngay từ đầu là nguyên nhân số một dẫn đến việc phải làm lại toàn bộ pipeline. Skill này đóng vai trò "khóa chặt" các giả định trước khi chúng trở thành lỗi tốn kém.

## Khi nào sử dụng

- Người dùng đưa ra một prompt mơ hồ (ví dụ: *"Xây dựng bộ dự đoán khách hàng rời bỏ"*, *"Phân loại ảnh X-quang y tế"*).
- Schema dữ liệu, tỷ lệ cân bằng lớp, hoặc môi trường compute chưa được xác định.
- Người dùng yêu cầu rõ ràng: *"Hãy phỏng vấn tôi"* hoặc *"Đặt câu hỏi làm rõ cho tôi"*.

## Nguyên tắc cốt lõi

1. **Mỗi lần một câu hỏi:** Không bao giờ dồn dập người dùng bằng một loạt 5-10 câu hỏi cùng lúc. Điều này gây quá tải nhận thức và làm giảm chất lượng câu trả lời.
2. **Đưa ra lựa chọn cụ thể:** Mỗi câu hỏi phải đi kèm 2 đến 3 giá trị mặc định hoặc ví dụ khả thi để hỗ trợ ra quyết định nhanh.
3. **Dừng ở mức 95% rõ ràng:** Tập trung nghiêm ngặt vào đặc điểm dữ liệu, metric chính và ràng buộc compute; không thẩm vấn những tiểu tiết triển khai không cần thiết.

## Các chiều thông tin bắt buộc phải làm rõ

1. **Đặc điểm Dataset:** Khối lượng mẫu/hàng/ảnh là bao nhiêu? Dataset đã được làm sạch và gán nhãn sẵn, hay còn nhiễu và thiếu đặc trưng?
2. **Phân bố lớp:** Cân bằng hay mất cân bằng? (ví dụ 50:50 so với 99:1, quyết định việc có cần Focal Loss, Class Weights, hay chia Stratified hay không).
3. **Metric mục tiêu:** Accuracy tổng thể có đủ không, hay Recall là tối quan trọng (ví dụ chẩn đoán y tế, phát hiện gian lận)? Hình phạt của False Positive so với False Negative là gì?
4. **Ràng buộc compute:** Chạy trên phần cứng cục bộ (CPU/GPU) hay môi trường đám mây (Google Colab, Kaggle GPU, AWS)?

## Tiêu chí hoàn thành

- Sau khi các chiều thông tin then chốt đã được giải quyết, tổng hợp một bản tóm tắt ngắn gọn 4 điểm và chuyển tiếp mượt mà việc thực thi sang `spec-driven-development`.
- Không được bắt đầu viết code khi vẫn còn chiều thông tin then chốt chưa được làm rõ.
