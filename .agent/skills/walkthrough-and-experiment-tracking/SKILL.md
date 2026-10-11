---
name: walkthrough-and-experiment-tracking
description: Tự động theo dõi, ghi nhận và cập nhật dòng chảy thí nghiệm đầy đủ trong Section 2 của walkthrough_lab_0X.md sau mỗi lần chạy huấn luyện, nắm bắt chênh lệch về kiến trúc, hyperparameter và metric qua cả 5 giai đoạn pipeline.
---

# walkthrough-and-experiment-tracking (Bản DS & AI)

## 1. Tổng quan

Skill này hoạt động như bộ theo dõi dòng chảy thí nghiệm tự động cho mọi bài lab deep learning. Sau mỗi lần thực thi huấn luyện, skill bắt buộc agent phải ghi lại **các thay đổi qua cả 5 giai đoạn pipeline** trực tiếp vào **Section 2 của tài liệu sống duy nhất `walkthrough_lab_0X.md`** (chịu sự điều phối của Rule 2 trong Kiến trúc 3 Rule).

Giá trị cốt lõi: nếu không ghi lại lý do đằng sau mỗi thay đổi, thì các lần chạy sau sẽ lặp lại sai lầm cũ hoặc không thể giải thích được vì sao một cấu hình lại tốt hơn cấu hình khác.

Skill cũng đảm bảo mọi con số trong tài liệu đều là số liệu thực nghiệm đã xác thực, không phải placeholder hay phỏng đoán.

Nguyên lý then chốt: mỗi run phải để lại một dấu vết kiểm toán đầy đủ về cấu hình, lý do thay đổi, và tác động định lượng.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Sau khi hoàn thành bất kỳ lần chạy huấn luyện nào khi đã có metric đánh giá và giá trị loss.
- Bất cứ khi nào chỉnh sửa một thành phần của pipeline (data transforms, kiến trúc backbone, optimizer schedule).
- Khi tổng hợp metric và hình ảnh cho việc nộp bài môn học.
- Sau khi đồng bộ artifact từ Kaggle về cục bộ.

**Điều kiện tiên quyết:**
- Đã có Section 1 (đặc tả) trong `walkthrough_lab_0X.md`.
- Đã có artifact của run cần ghi nhận (checkpoint, hình, bảng kết quả).
- Đã có số liệu metric của run.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Cấu hình của run hiện tại | Section 2 đã cập nhật với run mới |
| Metric đánh giá (loss, accuracy, F1) | Bảng so sánh đã thêm cột cho run mới |
| Artifact của run | Changelog với 3 câu hỏi bắt buộc |
| Cấu hình run trước đó để so sánh | Xác định run giữ kỷ lục best checkpoint |

## 4. Quy trình chi tiết

### Bước 1: Nắm bắt trạng thái toàn dòng chảy
Ghi lại ảnh chụp cấu hình cho lần thực thi hiện tại, bao gồm 5 thành phần: Data (tên dataset, số mẫu, tỷ lệ chia), Preprocessing và Augmentation (các biến đổi áp dụng), Model (tên kiến trúc, tổng số tham số, độ sâu), Training Strategy (loss, optimizer, learning rate, scheduler, batch size, epoch), và Evaluation Metrics (train và val loss/acc, test loss/acc, Macro F1, thời gian chạy).

### Bước 2: Ghi log chênh lệch kỹ thuật
Trả lời 3 câu hỏi bắt buộc trong changelog: Những giai đoạn pipeline nào đã thay đổi so với run trước? Lý do kỹ thuật của thay đổi là gì (giảm overfitting, tăng tốc hội tụ, hay giải quyết gradient tiêu biến)? Tác động định lượng là gì (mức tăng accuracy tuyệt đối, mức giảm khoảng cách tổng quát hóa, chênh lệch loss)? Ba câu hỏi này biến log thành tường thuật kỹ thuật có giá trị.

### Bước 3: Cập nhật ma trận so sánh
Thêm một cột mới cho run hiện tại trong Flow Comparison Matrix của `walkthrough_lab_0X.md`. Đánh dấu tường minh ứng viên nào đang giữ kỷ lục best checkpoint. Ma trận này cho phép so sánh toàn cảnh mọi run chỉ trong một bảng.

### Bước 4: Thu hoạch artifact từ Kaggle
Khi chạy trên Kaggle GPU, notebook ghi artifact vào `/kaggle/working/outputs/`: checkpoint tốt nhất, lưới dự đoán, confusion matrix, và đường cong huấn luyện. Đồng bộ thông qua Git hoặc tải file nén về `labs/lab0X/outputs/` cục bộ.

### Bước 5: Tổng hợp phát hiện
Đọc các artifact đầu ra và ghi lại các con số thực nghiệm đã xác thực vào `walkthrough_lab_0X.md`. Trích xuất nhân tố đột phá chính, các pattern của lát cắt lỗi, và đường dẫn tới trọng số mô hình đã lưu.

### Bước 6: Xác thực tính nhất quán số liệu
Kiểm tra rằng mọi con số trong tài liệu khớp với log thực nghiệm và bảng kết quả. Loại bỏ mọi placeholder hoặc giá trị chưa được xác thực. Số liệu không khớp là dấu hiệu của lỗi ghi chép.

## 5. Deliverables

- Section 2 của `walkthrough_lab_0X.md` đã cập nhật với run mới.
- Bảng Flow Comparison Matrix đầy đủ mọi run.
- Changelog với 3 câu hỏi bắt buộc cho mỗi run.
- Xác định rõ run giữ kỷ lục best checkpoint.
- Số liệu đã xác thực, không có placeholder.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `kaggle-remote-execution` (cung cấp artifact và số liệu) | `doubt-driven-development` (nhận metric để kiểm toán) |
| `environment-and-reproducibility` (cung cấp snapshot cấu hình) | `documentation-and-adrs` (nhận số liệu để viết báo cáo) |
| `test-driven-development` (cung cấp kết quả test) | |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Ghi vết sau khi đã quên:** Không ghi ngay sau khi chạy khiến lý do thay đổi bị mất. Dấu hiệu: changelog chung chung, không nhớ vì sao thay đổi. Phòng tránh: ghi ngay sau mỗi run.
- **Bỏ qua câu hỏi "vì sao":** Chỉ ghi số liệu mà không ghi lý do. Dấu hiệu: không thể giải thích cải thiện. Phòng tránh: trả lời đủ 3 câu hỏi ở Bước 2.
- **Để lại placeholder:** Quên điền số liệu thật. Dấu hiệu: tài liệu còn ô trống. Phòng tránh: xác thực ở Bước 6.
- **Không cập nhật ma trận:** Thêm run mới nhưng không thêm cột vào bảng so sánh. Phòng tránh: cập nhật ở Bước 3.
- **Số liệu không khớp log:** Ghi sai số do chép tay. Phòng tránh: đối chiếu với artifact ở Bước 6.

## 8. Definition of Done

- [ ] Đã ghi ảnh chụp cấu hình đầy đủ 5 thành phần cho run mới.
- [ ] Đã trả lời đủ 3 câu hỏi bắt buộc trong changelog.
- [ ] Đã thêm cột cho run mới trong Flow Comparison Matrix.
- [ ] Đã xác định rõ run giữ kỷ lục best checkpoint.
- [ ] Artifact của run đã được đồng bộ về `outputs/` cục bộ.
- [ ] Mọi con số trong tài liệu khớp với log thực nghiệm.
- [ ] Không còn placeholder hoặc giá trị chưa xác thực.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Ghi vết để sau, giờ tập trung chạy thêm thí nghiệm"* -> **Bác bỏ:** Ký ức về lý do thay đổi phai nhạt rất nhanh. Ghi ngay sau mỗi run rẻ hơn nhiều so với truy ngược về sau.
- [X] *"Chỉ cần ghi số liệu, không cần ghi lý do"* -> **Bác bỏ:** Số liệu không có lý do thì không thể giải thích cải thiện và không thể học được bài học cho run sau. Lý do là phần giá trị nhất của changelog.
- [X] *"Để placeholder rồi điền sau"* -> **Bác bỏ:** Placeholder dễ bị bỏ quên và dẫn đến tài liệu nộp bài còn ô trống. Điền số liệu thật ngay khi có.
- [X] *"Ma trận so sánh dài dòng, khỏi cập nhật"* -> **Bác bỏ:** Ma trận là công cụ so sánh toàn cảnh duy nhất. Không cập nhật thì mất khả năng đánh giá tiến bộ giữa các run.
