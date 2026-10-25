---
name: test-driven-development
description: Áp dụng giao thức Test-Driven Development chuyên biệt cho Data Science và Deep Learning, xác thực tính toàn vẹn dữ liệu, chống rò rỉ dữ liệu, kiểm tra chiều tensor, và bài test 1-batch sanity overfit trước khi huấn luyện từ xa.
---

# test-driven-development (Bản DS & AI)

## 1. Tổng quan

Skill này chuyển thể nguyên lý TDD sang bản chất ngẫu nhiên của machine learning và deep learning. Trong kỹ thuật deep learning, kiểm thử tự động là **lá chắn chính yếu chống lại các lỗi câm (silent bugs)** - những lỗi như nhiễm bẩn tập train-val, lỗi broadcasting tensor âm thầm, rò rỉ bộ nhớ GPU, hoặc kiến trúc không hội tụ.

Điểm khác biệt cốt lõi so với TDD phần mềm truyền thống: trong deep learning, một pipeline có thể chạy trơn tru không văng lỗi nhưng vẫn cho ra kết quả sai hoàn toàn. Test không chỉ kiểm tra "code có chạy không" mà phải kiểm tra "code có chạy đúng về mặt toán học và dữ liệu không".

Bài test quan trọng nhất là 1-batch sanity overfit: nếu mô hình không thể học vẹt được 8 mẫu, thì không có lý do gì tin nó sẽ học được toàn bộ dataset.

Nguyên lý then chốt: không được đốt GPU quota cho một pipeline chưa vượt qua xác thực cục bộ.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Triển khai Dataset, DataLoader, hoặc Model mới.
- Cần xác thực chiều tensor và kiểu dữ liệu.
- Cần kiểm tra chống rò rỉ dữ liệu.
- Trước khi đẩy job huấn luyện lên Kaggle GPU.
- Sau khi thay đổi cấu trúc dữ liệu hoặc kiến trúc mô hình.

**Điều kiện tiên quyết:**
- Đã có data contract từ `spec-driven-development`.
- Đã có cấu trúc module trong `src/`.
- Đã có seed cố định từ `environment-and-reproducibility`.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Data contract và chiều tensor kỳ vọng | Bộ test trong `tests/test_pipeline.py` |
| Module Dataset, Model, Engine trong `src/` | Kết quả test pass 100% cục bộ |
| Batch nhỏ để kiểm tra overfit | Bằng chứng 1-batch overfit đạt loss gần 0 |

## 4. Quy trình chi tiết

### Bước 1: Viết test toàn vẹn dữ liệu và chống rò rỉ
Quét và khẳng định không có file hỏng trước khi nạp vào loader. Xác thực tính rời rạc giữa tập train và tập val (không chồng lấn mẫu hoặc định danh). Khẳng định các scaler và encoder chỉ được fit trên tập train. Đây là tuyến phòng thủ đầu tiên chống lại lỗi rò rỉ dữ liệu.

### Bước 2: Viết test lazy-loading của Dataset
Khẳng định `Dataset.__init__` chỉ lưu index hoặc đường dẫn file, không đệm ma trận ảnh thô vào RAM. Truy cập một mẫu để xác thực kiểu dữ liệu: ảnh phải là tensor float32, nhãn phải là tensor int64. Test này ngăn chặn việc tràn RAM khi dataset lớn.

### Bước 3: Viết test chiều tensor và shape
Truyền một batch giả tổng hợp qua đường forward để bắt lỗi sai chiều trước khi lặp toàn bộ dữ liệu. Khẳng định output logits có shape đúng bằng số lớp và toàn bộ giá trị hữu hạn (không NaN, không Inf). Test này chạy trong tích tắc nhưng bắt được lỗi dimension tốn kém nhất.

### Bước 4: Viết test 1-batch sanity overfit (quan trọng nhất)
Cô lập một mini-batch nhỏ (4 đến 8 mẫu) và huấn luyện mô hình trong 30 đến 50 vòng lặp chỉ trên batch đó. Tiêu chí bắt buộc: loss giảm về sát 0 và accuracy trên batch đạt 100%. Nếu mô hình không thể overfit 8 mẫu, kiến trúc, công thức loss, hoặc optimizer đang chứa lỗi nghiêm trọng. Nghiêm cấm huấn luyện trên cloud cho tới khi test này pass.

### Bước 5: Viết test tương thích đường dẫn môi trường
Khẳng định các đường dẫn động thích ứng trơn tru giữa môi trường cục bộ và Kaggle (`/kaggle/input/` và `/kaggle/working/`). Test này ngăn chặn lỗi chỉ xuất hiện khi chuyển lên cloud.

### Bước 6: Chạy toàn bộ test và xác nhận pass
Chạy bộ test bằng pytest và xác nhận tất cả pass. Nếu có test thất bại, sửa nguyên nhân gốc thay vì bỏ qua test. Chỉ khi tất cả pass mới chuyển sang thực thi từ xa.

## 5. Deliverables

- File `tests/test_pipeline.py` với 5 bài test bắt buộc.
- Bằng chứng tất cả test pass cục bộ.
- Bằng chứng 1-batch sanity overfit đạt loss gần 0 và accuracy 100%.
- Kết quả test được ghi nhận trong `walkthrough_lab_0X.md`.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `spec-driven-development` (cung cấp data contract) | `kaggle-remote-execution` (nhận xác nhận pipeline đủ điều kiện chạy) |
| `planning-and-task-breakdown` (cung cấp danh sách module) | `debugging-and-error-recovery` (chuyển lỗi phát hiện được để gỡ) |
| `environment-and-reproducibility` (cung cấp seed) | `walkthrough-and-experiment-tracking` (ghi nhận kết quả test) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Bỏ qua test 1-batch:** Chạy thẳng lên Kaggle và phát hiện mô hình không hội tụ sau hàng giờ. Phòng tránh: luôn chạy test 1-batch ở Bước 4 trước.
- **Test không khẳng định gì:** Test chỉ chạy code mà không có assertion. Dấu hiệu: test luôn pass dù code sai. Phòng tránh: mỗi test phải có khẳng định cụ thể.
- **Bỏ qua kiểm tra rò rỉ:** Không test tính rời rạc của tập chia. Dấu hiệu: metric validation cao bất thường. Phòng tránh: viết test ở Bước 1.
- **Bỏ qua kiểu dữ liệu nhãn:** Nhãn sai dtype gây lỗi CrossEntropyLoss. Phòng tránh: kiểm tra dtype ở Bước 2.
- **Test phụ thuộc dữ liệu lớn:** Test chậm vì dùng toàn bộ dataset. Phòng tránh: dùng batch nhỏ hoặc dữ liệu tổng hợp.

## 8. Definition of Done

- [ ] Đã viết test toàn vẹn dữ liệu và chống rò rỉ.
- [ ] Đã viết test lazy-loading và kiểm tra dtype.
- [ ] Đã viết test chiều tensor với batch giả tổng hợp.
- [ ] Đã viết và chạy test 1-batch sanity overfit đạt loss gần 0 và accuracy 100%.
- [ ] Đã viết test tương thích đường dẫn môi trường.
- [ ] Toàn bộ test pass cục bộ.
- [ ] Kết quả test đã được ghi vào `walkthrough_lab_0X.md`.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Mô hình Deep Learning chạy lâu lắm, bỏ qua test overfit rồi train thẳng trên Kaggle"* -> **Bác bỏ:** Huấn luyện hàng giờ trên Kaggle chỉ để phát hiện mô hình không bao giờ hội tụ sẽ bào mòn hạn mức GPU hàng tuần. Một bước xác thực overfit cục bộ 5 giây là bắt buộc.
- [X] *"Test chỉ cần chạy không lỗi là được"* -> **Bác bỏ:** Test không có assertion không bắt được lỗi logic. Trong deep learning, code có thể chạy không lỗi nhưng cho kết quả sai hoàn toàn.
- [X] *"Kiểm tra rò rỉ dữ liệu mất công, chắc không sao đâu"* -> **Bác bỏ:** Rò rỉ dữ liệu cho ra metric ảo và mô hình vô dụng trong thực tế. Đây là lỗi tốn kém nhất và cũng dễ kiểm tra nhất.
- [X] *"Dữ liệu nhỏ nên không cần lazy-loading"* -> **Bác bỏ:** Thói quen đệm toàn bộ dữ liệu vào RAM sẽ sụp đổ khi chuyển sang dataset lớn. Viết đúng ngay từ đầu.
