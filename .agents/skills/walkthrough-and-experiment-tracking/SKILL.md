---
name: walkthrough-and-experiment-tracking
description: Tự động theo dõi, ghi nhận và cập nhật dòng chảy thí nghiệm đầy đủ trong Section 2 của walkthrough_lab_0X.md sau mỗi lần chạy huấn luyện, nắm bắt chênh lệch về kiến trúc, hyperparameter và metric qua cả 5 giai đoạn pipeline; với bài toán phân loại phải ghi đủ bộ metric (Accuracy, Macro F1, Precision, Recall, TPR, FPR theo lớp) và đọc, diễn giải ảnh output (đường cong, confusion matrix, lưới dự đoán, ROC/PR OvR).
---

# walkthrough-and-experiment-tracking (Bản DS & AI)

## 1. Tổng quan

Skill này hoạt động như bộ theo dõi dòng chảy thí nghiệm tự động cho mọi bài lab deep learning. Sau mỗi lần thực thi huấn luyện, skill bắt buộc agent phải ghi lại **các thay đổi qua cả 5 giai đoạn pipeline** trực tiếp vào **Section 2 của tài liệu sống duy nhất `walkthrough_lab_0X.md`** (chịu sự điều phối của Rule 2 trong Kiến trúc 3 Rule).

Giá trị cốt lõi: nếu không ghi lại lý do đằng sau mỗi thay đổi, thì các lần chạy sau sẽ lặp lại sai lầm cũ hoặc không thể giải thích được vì sao một cấu hình lại tốt hơn cấu hình khác.

Skill cũng đảm bảo mọi con số trong tài liệu đều là số liệu thực nghiệm đã xác thực, không phải placeholder hay phỏng đoán.

Nguyên lý then chốt: mỗi run phải để lại một dấu vết kiểm toán đầy đủ về cấu hình, lý do thay đổi, và tác động định lượng.

## Phạm vi báo cáo và lựa chọn biểu đồ

- Walkthrough là nguồn báo cáo duy nhất; Section 2 trình bày phương pháp/thiết kế thí nghiệm, Section 3 trình bày kết quả và so sánh. Không lặp toàn bộ bảng kết quả ở cả hai phần.
- Giữ Accuracy, Macro F1, Precision/Recall/F1 theo lớp và confusion matrix có số đếm. Recall là TPR; FPR có thể giữ trong bảng số liệu, không bắt buộc vẽ biểu đồ TPR/FPR.
- Chọn biểu đồ theo bài toán và lựa chọn đã chốt. Lab01 dùng một hình hai panel ROC và Precision–Recall One-vs-Rest cho mỗi model; phân biệt Average Precision với diện tích PR tính bằng hình thang. Không tự thêm lại histogram TPR/FPR đã bị loại.
- Đọc hình thực tế và đối chiếu bảng/predictions; nêu bằng chứng, không suy luận nhân quả chỉ từ tương quan hoặc một seed.
- Trong search, chọn hyperparameter bằng validation; không bắt buộc đánh giá test cho mọi trial. Trial chỉ có validation thì ghi đúng split, không bịa test metrics.
- Sửa code, lỗi môi trường và console log giữ local, ngoài báo cáo. Phân tích mô hình nhầm lớp vẫn thuộc kết quả thực nghiệm.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Sau khi hoàn thành bất kỳ lần chạy huấn luyện nào khi đã có metric đánh giá và giá trị loss.
- Bất cứ khi nào chỉnh sửa một thành phần của pipeline (data transforms, kiến trúc backbone, optimizer schedule).
- Khi tổng hợp metric và hình ảnh cho việc nộp bài môn học.
- Sau khi đồng bộ artifact từ Kaggle về cục bộ.
- Khi cần phân tích ảnh output (confusion matrix, lưới dự đoán, đường cong, ROC/PR OvR) để rút ra nhận xét định tính.

**Điều kiện tiên quyết:**
- Đã có Section 1 (đặc tả) trong `walkthrough_lab_0X.md`.
- Đã có artifact của run cần ghi nhận (checkpoint, hình, bảng kết quả).
- Đã có số liệu metric của run, bao gồm metric theo từng lớp cho bài toán phân loại.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Cấu hình của run hiện tại | Section 2 đã cập nhật với run mới |
| Metric đánh giá (loss, accuracy, F1, Precision, Recall, TPR, FPR) | Bảng so sánh đã thêm cột cho run mới |
| Ảnh output (đường cong, confusion matrix, lưới dự đoán, ROC/PR OvR) | Nhận xét định tính về nội dung từng ảnh |
| Artifact của run | Diễn tiến thí nghiệm với 3 câu hỏi bắt buộc |
| Cấu hình run trước đó để so sánh | Xác định run giữ kỷ lục best checkpoint |

## 4. Quy trình chi tiết

### Bước 1: Nắm bắt trạng thái toàn dòng chảy
Ghi lại ảnh chụp cấu hình cho lần thực thi hiện tại, bao gồm 5 thành phần: Data (tên dataset, số mẫu, tỷ lệ chia), Preprocessing và Augmentation (các biến đổi áp dụng), Model (tên kiến trúc, tổng số tham số, độ sâu), Training Strategy (loss, optimizer, learning rate, scheduler, batch size, epoch), và Evaluation Metrics (train và val loss/acc, test loss/acc, Macro F1, thời gian chạy).

**Bộ metric chuẩn cho bài toán phân loại (bắt buộc ghi đủ, không chỉ Accuracy):**

| Nhóm | Metric | Ghi chú |
|---|---|---|
| Tổng thể | Test Loss, Test Accuracy | Loss không có đơn vị %; Accuracy ghi rõ fraction hoặc %, làm tròn nhất quán |
| Tổng hợp lớp | Macro F1-Score | Chỉ số chính khi các lớp cần được đối xử công bằng |
| Theo từng lớp | Precision, Recall (TPR), F1-Score | Xuất từ classification report, ghi cho đủ mọi lớp |
| Theo từng lớp | False Positive Rate (FPR) | Đi kèm TPR để thấy lớp nào bị dự đoán nhầm nhiều |
| Ma trận | Confusion Matrix (đã chuẩn hóa theo hàng) | Giữ cả số đếm để tính FPR; ma trận chuẩn hóa theo hàng phục vụ đọc recall và cặp nhầm lẫn |
| Đường cong | Train/Val Loss và Accuracy theo epoch | Bằng chứng hội tụ và overfitting |

- Với bài toán phân loại nhiều lớp, **chỉ ghi Accuracy là không đủ**: Accuracy cao vẫn có thể che giấu một lớp bị mô hình bỏ quên hoàn toàn. Phải ghi Macro F1 và metric theo từng lớp để phát hiện hiện tượng đó.
- Mọi metric theo lớp phải ghi kèm tên lớp (ví dụ `Shirt`, `Sneaker`), không ghi chỉ số lớp trần.

### Bước 2: Ghi nhận thay đổi thiết kế thí nghiệm
Trả lời 3 câu hỏi bắt buộc trong diễn tiến thí nghiệm: Những giai đoạn pipeline nào đã thay đổi so với run trước? Lý do kỹ thuật của thay đổi là gì (giảm overfitting, tăng tốc hội tụ, hay giải quyết gradient tiêu biến)? Tác động định lượng là gì (mức tăng accuracy tuyệt đối, mức giảm khoảng cách tổng quát hóa, chênh lệch loss)? Ba câu hỏi này biến log thành tường thuật kỹ thuật có giá trị. Lý do thay đổi phải nhất quán với các quyết định đã được thảo luận và ghi ở Mục 1.0 của `walkthrough_lab_0X.md`.

### Bước 3: Cập nhật ma trận so sánh
Cập nhật một hàng cho mỗi run trong bảng kết quả Section 3; Section 2 dẫn chiếu và giải thích thiết kế. Đánh dấu tường minh ứng viên nào đang giữ kỷ lục best checkpoint. Ma trận này cho phép so sánh toàn cảnh mọi run chỉ trong một bảng.

### Bước 4: Thu hoạch artifact từ Kaggle
Khi chạy trên Kaggle, ghi artifact dưới `/kaggle/working/outputs/runs/<run_id>/` và thu hồi gói đầy đủ về local. Chỉ bộ evidence đã rà riêng tư trong `outputs/published/` được đưa lên Git; log thô/checkpoint/prepared data không được push cùng.

### Bước 5: Tổng hợp phát hiện
Đọc các artifact đầu ra và ghi lại các con số thực nghiệm đã xác thực vào `walkthrough_lab_0X.md`. Trích xuất nhân tố đột phá chính, các pattern của lát cắt lỗi, và đường dẫn tới trọng số mô hình đã lưu.

**Bắt buộc đọc và phân tích ảnh output, không chỉ liệt kê tên file.** Mỗi hình phải được mở và diễn giải nội dung thực tế; liệt kê đường dẫn mà không đọc hình là chưa hoàn thành bước này. Cụ thể:

- **Đường cong huấn luyện:** mô tả hình dạng hội tụ, epoch mà val loss bắt đầu tăng (dấu hiệu overfitting), và khoảng cách train/val.
- **Confusion matrix:** chỉ ra các ô sáng nhất ngoài đường chéo (cặp lớp nhầm lẫn nhiều nhất), đọc theo hàng để biết lớp thật bị dự đoán sai thành lớp nào.
- **Lưới dự đoán:** chỉ ra các mẫu bị dự đoán sai, mức tự tin của mô hình ở các mẫu đó, và pattern hình ảnh của nhóm bị lỗi.
- **ROC/PR OvR và bảng theo lớp:** giải thích lớp khó, recall/precision và AUC/AP từ artifact; không bắt buộc tạo biểu đồ TPR/FPR riêng.
- **Bảng so sánh giữa các run:** nêu rõ run nào thắng ở metric nào và mức chênh lệch tuyệt đối.

### Bước 6: Xác thực tính nhất quán số liệu
Kiểm tra rằng mọi con số trong tài liệu khớp với log thực nghiệm và bảng kết quả. Loại bỏ mọi placeholder hoặc giá trị chưa được xác thực. Số liệu không khớp là dấu hiệu của lỗi ghi chép.

## 5. Deliverables

- Section 2 của `walkthrough_lab_0X.md` đã cập nhật với run mới.
- Bảng Flow Comparison Matrix đầy đủ mọi run.
- Bảng metric theo từng lớp (Precision, Recall, F1, TPR, FPR) cho bài toán phân loại.
- Nhận xét định tính cho từng ảnh output (đường cong, confusion matrix, lưới dự đoán, ROC/PR OvR).
- Diễn tiến thí nghiệm với 3 câu hỏi bắt buộc cho mỗi run.
- Xác định rõ run giữ kỷ lục best checkpoint.
- Số liệu đã xác thực, không có placeholder.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `kaggle-remote-execution` (cung cấp artifact và số liệu) | `doubt-driven-development` (nhận metric để kiểm toán) |
| `environment-and-reproducibility` (cung cấp snapshot cấu hình) | `documentation-and-adrs` (nhận số liệu để viết báo cáo) |
| `test-driven-development` (cung cấp kết quả test) | |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Ghi vết sau khi đã quên:** Không ghi ngay sau khi chạy khiến lý do thay đổi bị mất. Dấu hiệu: diễn tiến thí nghiệm chung chung, không nhớ vì sao thay đổi. Phòng tránh: ghi ngay sau mỗi run.
- **Bỏ qua câu hỏi "vì sao":** Chỉ ghi số liệu mà không ghi lý do. Dấu hiệu: không thể giải thích cải thiện. Phòng tránh: trả lời đủ 3 câu hỏi ở Bước 2.
- **Để lại placeholder:** Quên điền số liệu thật. Dấu hiệu: tài liệu còn ô trống. Phòng tránh: xác thực ở Bước 6.
- **Không cập nhật ma trận:** Thêm run mới nhưng không thêm cột vào bảng so sánh. Phòng tránh: cập nhật ở Bước 3.
- **Số liệu không khớp log:** Ghi sai số do chép tay. Phòng tránh: đối chiếu với artifact ở Bước 6.
- **Liệt kê ảnh mà không đọc:** Chỉ chép đường dẫn hình vào tài liệu mà không mở hình để diễn giải. Dấu hiệu: tài liệu có tên file nhưng không có câu nào mô tả hình chứa gì. Phòng tránh: thực hiện đầy đủ checklist đọc ảnh ở Bước 5.
- **Chỉ ghi Accuracy:** Bỏ qua Macro F1 và metric theo từng lớp khiến lớp yếu bị che giấu. Dấu hiệu: bảng kết quả chỉ có một cột Accuracy. Phòng tránh: ghi đủ bộ metric chuẩn ở Bước 1.

## 8. Definition of Done

- [ ] Đã ghi ảnh chụp cấu hình đầy đủ 5 thành phần cho run mới.
- [ ] Run đánh giá cuối có Test Loss, Accuracy, Macro F1 và Precision/Recall/F1 theo lớp; FPR lưu bảng khi có. Trial tuning chỉ yêu cầu validation metrics đã thực sự đo.
- [ ] Đã mở và diễn giải từng ảnh output (đường cong, confusion matrix, lưới dự đoán, ROC/PR OvR), không chỉ liệt kê đường dẫn.
- [ ] Đã trả lời đủ 3 câu hỏi bắt buộc trong diễn tiến thí nghiệm.
- [ ] Đã cập nhật hàng cho run mới trong bảng kết quả Section 3; Section 2 dẫn chiếu thiết kế và không lặp toàn bộ bảng.
- [ ] Đã xác định rõ run giữ kỷ lục best checkpoint.
- [ ] Artifact của run đã được đồng bộ về `outputs/` cục bộ.
- [ ] Mọi con số trong tài liệu khớp với log thực nghiệm.
- [ ] Không còn placeholder hoặc giá trị chưa xác thực.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Ghi vết để sau, giờ tập trung chạy thêm thí nghiệm"* -> **Bác bỏ:** Ký ức về lý do thay đổi phai nhạt rất nhanh. Ghi ngay sau mỗi run rẻ hơn nhiều so với truy ngược về sau.
- [X] *"Chỉ cần ghi số liệu, không cần ghi lý do"* -> **Bác bỏ:** Số liệu không có lý do thì không thể giải thích cải thiện và không thể học được bài học cho run sau. Lý do là phần giá trị nhất của diễn tiến thí nghiệm.
- [X] *"Để placeholder rồi điền sau"* -> **Bác bỏ:** Placeholder dễ bị bỏ quên và dẫn đến tài liệu nộp bài còn ô trống. Điền số liệu thật ngay khi có.
- [X] *"Ma trận so sánh dài dòng, khỏi cập nhật"* -> **Bác bỏ:** Ma trận là công cụ so sánh toàn cảnh duy nhất. Không cập nhật thì mất khả năng đánh giá tiến bộ giữa các run.
- [X] *"Đã ghi đường dẫn hình vào tài liệu là coi như phân tích xong"* -> **Bác bỏ:** Đường dẫn không phải bằng chứng. Chỉ khi mở hình và mô tả nội dung (cặp lớp nhầm lẫn, mẫu dự đoán sai, lớp có TPR thấp) thì người chấm mới thấy hình đã được đọc và hiểu.
- [X] *"Accuracy cao là đủ, khỏi ghi metric theo lớp"* -> **Bác bỏ:** Accuracy trung bình có thể cao trong khi một lớp bị dự đoán sai gần hết. Macro F1 và metric theo lớp là cách duy nhất phát hiện lớp bị bỏ quên.
