---
name: doubt-driven-development
description: Cơ chế kiểm toán đối kháng được kích hoạt khi mô hình đạt hiệu năng cao bất thường (metric vượt kỳ vọng của dataset/baseline), nhằm phát hiện rò rỉ nhãn, nhiễm bẩn tập train-test, và overfitting tiềm ẩn.
---

# doubt-driven-development (Bản DS & AI)

## 1. Tổng quan

Trong AI và Machine Learning, chế độ thất bại nguy hiểm nhất là **Tự tin sai lệch (False Confidence)**. Khi một mô hình đạt độ chính xác cao ngay lần thử đầu tiên, điều đó gần như chắc chắn chỉ ra một khiếm khuyết âm thầm trong pipeline, chứ không phải sự vượt trội của thuật toán. Skill này buộc AI Agent đóng vai một kiểm toán viên hoài nghi, tìm cách phủ định thiết lập thí nghiệm trước khi chấp nhận hiệu năng được báo cáo.

Triết lý đằng sau: một kết quả tốt bất thường cần bằng chứng mạnh mẽ hơn mức bình thường. Trách nhiệm của kiểm toán viên là cố gắng "bác bỏ" kết quả, chứ không phải xác nhận nó.

Skill cũng bảo vệ trước bẫy mất cân bằng lớp, nơi một bộ phân loại đa số tầm thường có thể đạt Accuracy rất cao mà không học được gì hữu ích.

Nguyên lý then chốt: kết quả chỉ được chấp nhận sau khi vượt qua mọi kiểm toán mà không phát hiện dấu hiệu rò rỉ hay artifact đánh giá.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Metric tăng bất thường so với baseline, độ khó dataset, phân bố lớp hoặc giai đoạn huấn luyện. Không dùng ngưỡng 95%/98% chung cho Accuracy, F1 và ROC-AUC; ROC-AUC cao không đồng nghĩa Accuracy cao.
- Trước khi hoàn thiện bài nộp môn học hoặc triển khai production.
- Metric validation có vẻ tối ưu phi thực tế bất chấp miền dữ liệu nhiễu.
- Kết quả tốt hơn kỳ vọng đáng kể so với baseline lý thuyết.

**Điều kiện tiên quyết:**
- Đã có kết quả thực nghiệm với metric cụ thể.
- Có quyền truy cập vào dữ liệu, cách chia tập, và code pipeline.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Metric thực nghiệm cao bất thường | Báo cáo kiểm toán rò rỉ dữ liệu |
| Cách chia tập và pipeline | Xác nhận hoặc bác bỏ tính hợp lệ của kết quả |
| Confusion matrix và learning curves | Phân tích khoảng cách tổng quát hóa |
| | Kết quả test hoán vị nhãn |

## 4. Quy trình chi tiết

### Bước 1: Kiểm toán rò rỉ nhãn (Target Leakage)
Kiểm tra xem có đặc trưng nào trong $X$ được sinh ra sau khi biến cố mục tiêu $y$ xảy ra trong dòng thời gian thực tế không. Kiểm tra xem có cột định danh nào (ID, row hash) tương quan 1-1 với nhãn mục tiêu không. Đây là dạng rò rỉ phổ biến nhất và cũng khó phát hiện nhất.

### Bước 2: Kiểm toán nhiễm bẩn tập chia (Split Contamination)
Kiểm tra xem `fit_transform()` có được thực thi trên toàn bộ dataset trước `train_test_split()` không. Nếu có, thống kê của tập test đã nhiễm vào tập train, gây rò rỉ nghiêm trọng. Cũng kiểm tra xem có hàng hoặc ảnh trùng lặp giống hệt nhau xuất hiện đồng thời ở tập train và test không.

### Bước 3: Kiểm toán bẫy mất cân bằng lớp
Kiểm tra confusion matrix. Trong dataset có 99 mẫu âm và 1 mẫu dương, một bộ phân loại đa số tầm thường đạt tới 99% accuracy mà không học gì. Các kiểm tra bắt buộc: Macro F1-Score, Precision và Recall của lớp thiểu số, và Precision-Recall AUC.

### Bước 4: Kiểm toán overfitting và khoảng cách tổng quát hóa
So sánh learning curves giữa train và validation. Nếu train loss rất thấp trong khi validation loss cao (ví dụ train 0.02 so với val 1.20), mạng đang ghi nhớ mẫu train và thất bại trong việc tổng quát hóa. Khoảng cách càng lớn thì overfitting càng nghiêm trọng.

### Bước 5: Test hoán vị nhãn (Label Permutation Test)
Khi kiểm tra chia tập/fit-on-train và metric vẫn để lại nghi ngờ, chạy permutation test trong ngân sách đã chốt: hoán vị nhãn train, giữ tập đánh giá độc lập, huấn luyện lại và so với baseline ngẫu nhiên phù hợp phân bố lớp. Không dùng accuracy train để kết luận vì mạng có thể ghi nhớ nhãn ngẫu nhiên. Kết quả cao là dấu hiệu cần điều tra, không tự động chứng minh nguyên nhân rò rỉ.

### Bước 6: Tổng hợp phán quyết
Tổng hợp kiểm tra đã thực hiện, bằng chứng và giới hạn; ghi rõ test chưa chạy và lý do. Không tuyên bố đã hoàn thành permutation test nếu chưa có artifact. Nếu phát hiện vấn đề, ghi rõ nguyên nhân và chuyển sang `debugging-and-error-recovery` để sửa.

## 5. Deliverables

- Báo cáo kiểm toán rò rỉ nhãn và nhiễm bẩn tập chia.
- Phân tích confusion matrix và tỷ lệ lớp thiểu số.
- So sánh learning curves và khoảng cách tổng quát hóa.
- Kết quả test hoán vị nhãn.
- Phán quyết cuối về tính hợp lệ của kết quả.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `walkthrough-and-experiment-tracking` (cung cấp metric bất thường) | `documentation-and-adrs` (nhận phán quyết để viết báo cáo) |
| `test-driven-development` (cung cấp kết quả test) | `debugging-and-error-recovery` (chuyển vấn đề phát hiện để sửa) |
| `environment-and-reproducibility` (cung cấp bằng chứng tái lập) | |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Bỏ qua kiểm toán vì kết quả đẹp:** Chấp nhận metric cao mà không kiểm tra rò rỉ. Dấu hiệu: mô hình thất bại trong thực tế. Phòng tránh: kiểm tra dữ liệu và split trước, bổ sung permutation test khi cần.
- **Không kiểm tra tập chia:** Bỏ qua kiểm tra trùng lặp mẫu giữa train và test. Phòng tránh: kiểm tra ở Bước 2.
- **Bỏ qua lớp thiểu số:** Chỉ nhìn Accuracy tổng thể. Phòng tránh: kiểm tra Macro F1 và PR-AUC ở Bước 3.
- **Bỏ qua test hoán vị:** Không chạy test hoán vị nhãn vì tốn thời gian. Đây là kiểm tra quyết định nhất để xác nhận rò rỉ. Phòng tránh: ghi rõ căn cứ cần chạy hoặc chưa chạy Bước 5.

## 8. Definition of Done

- [ ] Đã kiểm toán rò rỉ nhãn và cột định danh.
- [ ] Đã kiểm toán nhiễm bẩn tập chia và mẫu trùng lặp.
- [ ] Đã kiểm tra confusion matrix và metric lớp thiểu số.
- [ ] Đã so sánh learning curves và khoảng cách tổng quát hóa.
- [ ] Đã ghi kết quả permutation test nếu thực hiện, hoặc lý do không thực hiện và giới hạn kết luận.
- [ ] Đã tổng hợp phán quyết cuối về tính hợp lệ của kết quả.
- [ ] Phán quyết đã được ghi vào `walkthrough_lab_0X.md`.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Accuracy 99% là tốt rồi, không cần kiểm tra gì thêm"* -> **Bác bỏ:** Kết quả cao cần được đối chiếu độ khó dataset/baseline và kiểm tra pipeline, không mặc định là lỗi. Kiểm toán là bắt buộc trước khi chấp nhận.
- [X] *"Test hoán vị nhãn tốn thời gian, bỏ qua"* -> **Bác bỏ:** Cân nhắc theo mức nghi ngờ và ngân sách; ghi rõ phạm vi kiểm tra, không biến việc chưa chạy thành một kết quả đã xác thực.
- [X] *"Accuracy cao là đủ, không cần xem lớp thiểu số"* -> **Bác bỏ:** Trên dataset mất cân bằng, Accuracy cao có thể che giấu việc mô hình bỏ qua hoàn toàn lớp thiểu số. Phải xem Macro F1 và PR-AUC.
- [X] *"Chắc là mô hình mình giỏi thật"* -> **Bác bỏ:** Trách nhiệm kiểm toán là tìm cách bác bỏ giả thuyết này. Chỉ khi mọi kiểm toán đều sạch mới tin vào kết quả.
