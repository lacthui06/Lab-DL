---
name: doubt-driven-development
description: Cơ chế kiểm toán đối kháng được kích hoạt khi mô hình Machine Learning hoặc Deep Learning đạt hiệu năng cao bất thường (Accuracy > 98%), nhằm phát hiện rò rỉ nhãn (target leakage), nhiễm bẩn tập train-test, và overfitting tiềm ẩn.
---

# doubt-driven-development (Bản DS & AI)

## Tổng quan

Trong AI và Machine Learning, chế độ thất bại nguy hiểm nhất là **Tự tin sai lệch (False Confidence)**. Khi một mô hình đạt độ chính xác 99.9% ngay lần thử đầu tiên, điều đó gần như chắc chắn chỉ ra một khiếm khuyết âm thầm trong pipeline, chứ không phải sự vượt trội của thuật toán. Skill này buộc AI Agent đóng vai một kiểm toán viên hoài nghi, tìm cách phủ định thiết lập thí nghiệm trước khi chấp nhận hiệu năng được báo cáo.

Triết lý đằng sau: một kết quả tốt bất thường cần bằng chứng mạnh mẽ hơn mức bình thường. Trách nhiệm của kiểm toán viên là cố gắng "bác bỏ" kết quả, chứ không phải xác nhận nó.

## Khi nào kích hoạt

- Mô hình đạt Accuracy, F1-Score, hoặc ROC-AUC cao bất thường (> 95%) trong các vòng lặp huấn luyện đầu tiên.
- Trước khi hoàn thiện bài nộp môn học hoặc triển khai production.
- Khi metric validation có vẻ tối ưu một cách phi thực tế bất chấp miền dữ liệu đầu vào còn nhiễu.

## Giao thức kiểm toán hoài nghi 4 bước

### Bước 1: Kiểm toán rò rỉ dữ liệu (Data Leakage Audit)
- **Rò rỉ nhãn (Target Leakage):**
  - Có đặc trưng nào trong $X$ được sinh ra *sau* khi biến cố mục tiêu $y$ xảy ra trong dòng thời gian thực tế hay không?
  - Có cột định danh nào (ID, row hash) tương quan 1-1 với nhãn mục tiêu hay không?
- **Nhiễm bẩn tập chia (Split Contamination):**
  - Có phải `fit_transform()` đã được thực thi trên toàn bộ dataset *trước* `train_test_split()`? (Nếu có -> RÒ RỈ NGHIÊM TRỌNG: thống kê của tập test đã nhiễm vào tập train).
- **Rò rỉ mẫu trùng lặp (Duplicate Sample Leakage):**
  - Có hàng hoặc ảnh trùng lặp giống hệt nhau xuất hiện đồng thời ở cả tập train lẫn tập test hay không?

### Bước 2: Kiểm toán bẫy mất cân bằng lớp (Class Imbalance Trap Audit)
- Kiểm tra **Confusion Matrix**:
  - Trong một dataset có 99 mẫu âm và 1 mẫu dương, một bộ phân loại đa số tầm thường đạt tới 99% accuracy!
  - Các kiểm tra bắt buộc: Macro F1-Score, Precision và Recall của lớp thiểu số, và Precision-Recall AUC (PR-AUC).

### Bước 3: Kiểm toán overfitting và khoảng cách tổng quát hóa (Generalization Gap Audit)
- So sánh các đường cong học tập (learning curves):
  - Train Loss = 0.02 trong khi Val Loss = 1.20 -> Mạng đang ghi nhớ (memorize) mẫu train và thất bại trong việc tổng quát hóa.

### Bước 4: Kiểm tra hoán vị ngẫu nhiên (Random Permutation Sanity Check)
- **Test hoán vị nhãn (Label Permutation Test):**
  - Xáo trộn ngẫu nhiên cột mục tiêu $y$ rồi huấn luyện lại mô hình.
  - **Kết quả kỳ vọng:** Hiệu năng giảm xuống mức ngẫu nhiên (ví dụ ~50% cho phân loại nhị phân, ~10% cho FashionMNIST 10 lớp).
  - **Cảnh báo đỏ:** Nếu mô hình vẫn đạt độ chính xác cao trên nhãn bị xáo trộn ngẫu nhiên -> Pipeline chứa lỗi rò rỉ dữ liệu đã được xác nhận 100%.

## Tiêu chí hoàn thành

- Kết quả mô hình chỉ được chấp nhận là hợp lệ sau khi vượt qua cả 4 kiểm toán mà không phát hiện rò rỉ dữ liệu hay artifact đánh giá nào.
