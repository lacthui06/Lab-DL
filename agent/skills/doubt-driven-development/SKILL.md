---
name: doubt-driven-development
description: Cơ chế phản biện đối kháng độc lập khi mô hình Machine Learning hoặc Deep Learning đạt điểm số cao bất thường (Accuracy > 98%), nhằm phát hiện rò rỉ dữ liệu (Data Leakage), ngộ nhận phân phối hoặc Overfitting ngầm.
---

# doubt-driven-development (DS & AI Edition)

## Tổng quan
Trong Trí tuệ Nhân tạo, hiện tượng nguy hiểm nhất là **"Ảo tưởng thành công" (False Confidence)**. Mô hình đạt độ chính xác 99.9% thường không phải vì mô hình siêu việt, mà là vì **mã nguồn đang có lỗi ngầm**. Kỹ năng này bắt buộc AI phải đóng vai "kiểm toán viên hoài nghi", tìm mọi cách chứng minh mô hình đang bị gian lận trước khi công nhận kết quả.

## Khi nào kích hoạt
- Mô hình đạt Accuracy, F1-score hoặc ROC-AUC cao bất thường ($> 95\%$) ngay từ những vòng lặp đầu tiên.
- Khi người dùng muốn thẩm định kết quả trước khi báo cáo hoặc nộp bài.
- Khi kết quả trên tập Validation quá đẹp nhưng có nguy cơ rò rỉ dữ liệu.

## Quy trình 4 bước kiểm toán hoài nghi (The Doubt Protocol)

### Bước 1: Kiểm toán Rò rỉ Dữ liệu (Data Leakage Audit)
- **Kiểm tra rò rỉ mục tiêu (Target Leakage):**
  - Có cột nào trong ma trận đặc trưng $X$ thực chất được sinh ra *sau* khi sự kiện mục tiêu $y$ diễn ra không?
  - Có cột ID hoặc mã định danh nào vô tình tương quan 1-1 với nhãn không?
- **Kiểm tra rò rỉ phân tách (Split Leakage):**
  - Kiểm tra xem `fit_transform()` có bị gọi trên toàn bộ tập dữ liệu *trước* khi chia `train_test_split()` không? (Nếu có ➔ RÒ RỈ NGHIÊM TRỌNG: giá trị trung bình/phương sai của tập test đã lọt vào tập train).
- **Kiểm tra rò rỉ mẫu trùng (Duplicate Leakage):**
  - Tập train và test có các dòng dữ liệu giống hệt nhau không?

### Bước 2: Kiểm toán Bẫy Mất cân bằng Nhãn (Class Imbalance Trap)
- Xem xét kỹ **Ma trận nhầm lẫn (Confusion Matrix)**:
  - Nếu bài toán có 99 mẫu nhãn 0 và 1 mẫu nhãn 1, mô hình chỉ cần dự đoán tất cả là 0 thì Accuracy đã là 99%!
  - Bắt buộc kiểm tra: F1-Macro, Precision, Recall của lớp thiểu số, và diện tích dưới đường cong PR (PR-AUC).

### Bước 3: Kiểm toán Độ lệch Train vs Validation (Overfitting Audit)
- Đối chiếu đường cong học (Learning Curve):
  - Train Loss = 0.02 nhưng Val Loss = 1.20 ➔ Mô hình đang học vẹt thuộc lòng dữ liệu train, hoàn toàn mất khả năng tổng quát hóa.

### Bước 4: Kiểm toán Tính ngẫu nhiên (Shuffle & Baseline Sanity Check)
- **Bài kiểm tra xáo trộn nhãn (Label Permutation Test):**
  - Thử xáo trộn ngẫu nhiên cột nhãn $y$ rồi train lại mô hình.
  - **Kết quả đúng:** Điểm số phải tụt về mức đoán ngẫu nhiên (ví dụ 50% cho bài toán 2 lớp).
  - **Cảnh báo đỏ:** Nếu nhãn đã xáo trộn ngẫu nhiên mà mô hình vẫn đạt Accuracy cao ➔ Mã nguồn chắc chắn 100% có bug rò rỉ dữ liệu!

## Tiêu chí kết thúc
- Chỉ khi vượt qua toàn bộ 4 bước kiểm toán trên mà không phát hiện dấu vết rò rỉ dữ liệu thì kết quả mô hình mới được công nhận là hợp lệ.
