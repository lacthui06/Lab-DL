# Báo cáo Lab 0X — <tên bài lab>

Nguồn báo cáo duy nhất của lab. Viết bằng văn phong học thuật, dùng số liệu và ảnh thực từ artifacts đã chọn. Không thêm dòng phiên bản/ngày biên tập nếu không có yêu cầu. Nhật ký sửa code, lỗi môi trường và console log giữ local, ngoài báo cáo.

## 1. Bài toán và thiết kế tổng thể

### 1.0. Đề bài và phạm vi đã thống nhất
Tóm tắt đề gốc và yêu cầu mở rộng từ description, mục tiêu, ràng buộc, giả định chưa kiểm chứng. Chuyển thảo luận thành lý do phương pháp, không chép hội thoại.

### 1.1. Dữ liệu và hợp đồng tensor
Nguồn dataset, nhãn, kích thước, dtype, đầu vào/đầu ra; train/validation/test, seed, cách chống leakage.

### 1.2. Pipeline Execution Roadmap
Ánh xạ từng giai đoạn tới module/hàm có thật và artifact thực. Phân biệt notebook gọi backend gộp với cell tự thực hiện từng giai đoạn.

### 1.3. Thiết kế so sánh
Baseline và các cấu hình, câu hỏi thực nghiệm, biến được kiểm soát, metric chính và tiêu chí lựa chọn bằng validation.

## 2. Phương pháp và thiết kế thí nghiệm

### 2.1. EDA
Tính toàn vẹn, phân bố nhãn, mẫu đại diện, đặc điểm dữ liệu theo modality. Mỗi hình có diễn giải thực tế và ý nghĩa cho xử lý dữ liệu.

### 2.2. Preprocessing
Các phép biến đổi theo đúng thứ tự, scaler/statistics chỉ fit trên train; cách áp dụng cho validation/test; augmentation nếu thực sự dùng.

### 2.3. Feature engineering
Định nghĩa/công thức, lý do chọn, ví dụ tính và kiểm chứng toàn bộ dữ liệu; phân bố, tương quan và giới hạn. Ghi rõ đặc trưng nào thực sự đi vào model.

### 2.4. Mô hình
Kiến trúc, số tầng ẩn, kích thước tensor và cách tính số tham số cho từng cấu hình.

### 2.5. Huấn luyện và tìm siêu tham số
Loss, optimizer, momentum, scheduler và thời điểm step; epochs, batch, sanity test, seed, search space, sampler/pruner, ngân sách và checkpoint selection. Không chọn cấu hình bằng test.

### 2.6. Môi trường và tái lập
Local CPU/GPU hoặc Kaggle theo cấu hình; phiên bản thực từ snapshot của run, khác biệt và giới hạn kiểm chứng. Không suy diễn đã chạy Kaggle nếu chỉ chạy local.

## 3. Kết quả, thảo luận và kết luận

### 3.1. So sánh mô hình
Một bảng kết quả chính: validation selection, test loss/accuracy, macro F1/precision/recall, tham số và thời gian khi đo được. Phân biệt điểm phần trăm với phần trăm tương đối.

### 3.2. Hội tụ
Train/validation loss và accuracy; giải thích khoảng cách, epoch tốt nhất và dấu hiệu overfitting từ hình và history thực.

### 3.3. Đánh giá theo lớp
Precision/recall/F1 và support, confusion matrix có số đếm và màu chuẩn hóa. TPR là recall; FPR có thể ở bảng, không bắt buộc biểu đồ riêng. Phân tích cặp lớp nhầm lẫn và mẫu đúng/sai có confidence.

### 3.4. ROC và Precision–Recall
Dùng khi có xác suất và phù hợp bài toán; với Lab01 là một hình hai panel OvR mỗi model. Ghi rõ aggregation, ROC-AUC và AP, không nhầm AP với trapezoidal PR-AUC.

### 3.5. So sánh optimizer và tuning
Fixed LR so với search; best parameters, trạng thái trial, tiến trình và đường cong các optimizer. Giải thích tuned có thể kém baseline, mức thực sự dùng TPE và các giới hạn ngân sách.

### 3.6. Save/load và kiểm chứng test
Kiến trúc, định dạng checkpoint, full test sau load, sai số so sánh và lưới dự đoán. Nêu có/không lưu optimizer/scheduler state nếu bàn về resume.

### 3.7. Kết luận và giới hạn
Kết quả trả lời từng câu hỏi, một seed/ngân sách search nếu có, giới hạn khẳng định nhân quả và hướng thí nghiệm tiếp. Không đưa log gỡ lỗi phần mềm.

### 3.8. Nguồn và provenance
Tài liệu chính thức và nguồn dataset; dẫn ảnh/bảng trong outputs/published/, truy run gốc qua provenance. Nội dung chưa có bằng chứng phải ghi rõ, không điền số giả.

## Quy tắc biên tập PDF

- Biên tập đầy đủ phương pháp và phân tích kết quả; không biến PDF thành danh sách file, log hay code dump.
- Giữ bảng kết quả đầy đủ ở Section 3; Section 2 giải thích và dẫn chiếu để tránh lặp.
- Với Lab01, mục tiêu tối thiểu 25 trang có nội dung; render để kiểm tra số trang, chữ, ảnh, caption và bảng, không khẳng định từ số dòng Markdown.
- Không tạo REPORT.md hoặc bản sao walkthrough trong outputs/. README chỉ hướng dẫn sử dụng.
