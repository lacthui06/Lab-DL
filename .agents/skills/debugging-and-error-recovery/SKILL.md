---
name: debugging-and-error-recovery
description: Chẩn đoán và khắc phục các chế độ thất bại chuyên biệt trong vòng lặp huấn luyện Machine Learning và Deep Learning, bao gồm CUDA OOM, loss NaN/Inf, quyền hệ thống file trên Kaggle, và gradient tiêu biến.
---

# debugging-and-error-recovery (Bản DS & AI)

## 1. Tổng quan

Skill này cung cấp một giao thức phân loại 5 bước và một sổ tay chẩn đoán để xử lý các chế độ thất bại kinh điển gặp phải trong quá trình huấn luyện deep learning trên máy trạm cục bộ và các nền tảng đám mây như Kaggle.

Mục tiêu là biến việc gỡ lỗi từ "thử mò" thành một quy trình có hệ thống: nhận diện triệu chứng, khoanh vùng nguyên nhân, áp dụng giải pháp kỹ thuật, và phòng ngừa tái diễn.

Điểm khó của gỡ lỗi deep learning là lỗi thường không rõ ràng: một loss bất thường có thể đến từ dữ liệu, kiến trúc, optimizer, hoặc learning rate. Quy trình có hệ thống giúp thu hẹp không gian tìm kiếm nhanh chóng.

Nguyên lý then chốt: sửa nguyên nhân gốc bằng thay đổi cấu trúc, không dùng giải pháp tình thế che giấu triệu chứng.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Lỗi `RuntimeError: CUDA out of memory`.
- Loss trở thành NaN hoặc Inf.
- DataLoader bị kill đột ngột (bus error).
- Lỗi permission denied khi ghi file trên Kaggle.
- Gradient bùng nổ hoặc tiêu biến khiến mô hình không hội tụ.
- Mô hình không học được (loss không giảm).

**Điều kiện tiên quyết:**
- Đã có pipeline chạy được tới điểm phát sinh lỗi.
- Có thể tái hiện lỗi trên một batch nhỏ.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Thông báo lỗi và stack trace | Chẩn đoán nguyên nhân gốc |
| Trạng thái pipeline khi lỗi | Giải pháp cấu trúc, không tình thế |
| Log huấn luyện | Assertion hoặc test phòng ngừa tái diễn |

## 4. Quy trình chi tiết

### Bước 1: Tái hiện lỗi trong trạng thái tối giản
Chạy lại trên một mini-batch duy nhất để cô lập lỗi. Việc giảm trạng thái giúp loại bỏ nhiễu và làm lỗi xuất hiện nhanh hơn, dễ quan sát hơn.

### Bước 2: Khoanh vùng nguyên nhân
Xác định nguyên nhân gốc xuất phát từ DataLoader, Model Forward, Loss, hay bước Backward. Chia pipeline thành các ranh giới rõ ràng và kiểm tra từng ranh giới để thu hẹp phạm vi.

### Bước 3: Cô lập bằng assertion
In shape tensor và khẳng định `torch.isnan().any()` tại các ranh giới module bị nghi ngờ. Việc đặt assertion tại đúng điểm giúp xác định chính xác nơi giá trị bất thường xuất hiện lần đầu.

### Bước 4: Xử lý cạn kiệt bộ nhớ GPU
Nếu gặp CUDA out of memory: giảm batch size và bù bằng gradient accumulation để bảo toàn động lực batch hiệu dụng; bật Automatic Mixed Precision để giảm VRAM khoảng 50%; tích lũy loss bằng giá trị vô hướng (không giữ đồ thị); và xóa cache bộ nhớ tường minh.

### Bước 5: Xử lý bất ổn định loss
Nếu loss là NaN hoặc Inf: thêm gradient clipping trước khi optimizer cập nhật; xác thực tensor đầu vào không chứa NaN; kẹp giá trị số để tránh chia cho 0 hoặc log(0); và giảm learning rate đi một bậc độ lớn.

### Bước 6: Xử lý lỗi đặc thù Kaggle
Với bus error do DataLoader: giảm `num_workers` xuống 2 hoặc 0 để tránh cạn `/dev/shm`. Với permission denied: chuyển mọi mục tiêu ghi về `/kaggle/working/`. Với connection refused khi tải pretrained: bật Internet trong cài đặt notebook.

### Bước 7: Sửa bằng thay đổi cấu trúc
Triển khai các chỉnh sửa mang tính cấu trúc, bền vững, thay vì giải pháp tình thế. Ví dụ, thay vì chỉ giảm learning rate tạm thời, xem xét lại thiết kế scheduler.

### Bước 8: Phòng ngừa tái diễn
Thêm assertion test tự động hoặc case pytest để ngăn chặn lỗi tái phát. Mỗi lỗi đã gỡ nên để lại một "hàng rào" ngăn nó quay lại.

## 5. Deliverables

- Chẩn đoán nguyên nhân gốc cho mỗi lỗi.
- Giải pháp cấu trúc đã áp dụng.
- Assertion hoặc test phòng ngừa tái diễn.
- Ghi chú lỗi và cách xử lý vào tài liệu.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `test-driven-development` (phát hiện lỗi qua test) | `test-driven-development` (nhận test phòng ngừa mới) |
| `kaggle-remote-execution` (phát hiện lỗi môi trường) | `performance-optimization` (nhận thông tin điểm nghẽn) |
| | `walkthrough-and-experiment-tracking` (ghi nhận sự cố và cách xử lý) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Che giấu triệu chứng:** Chỉ giảm learning rate mà không tìm nguyên nhân NaN. Dấu hiệu: lỗi tái xuất hiện ở run sau. Phòng tránh: tìm nguyên nhân gốc ở Bước 2 và 3.
- **Giữ đồ thị tính toán:** Tích lũy loss bằng tensor thay vì giá trị vô hướng gây rò rỉ bộ nhớ. Dấu hiệu: VRAM tăng dần qua các batch. Phòng tránh: dùng giá trị vô hướng ở Bước 4.
- **`num_workers` quá cao trên Kaggle:** Gây bus error. Phòng tránh: đặt thấp ở Bước 6.
- **Không thêm test phòng ngừa:** Lỗi cũ quay lại ở phiên sau. Phòng tránh: thêm assertion ở Bước 8.
- **Sửa tình thế không bền vững:** Giải pháp tạm thời che lấp vấn đề thiết kế. Phòng tránh: ưu tiên thay đổi cấu trúc ở Bước 7.

## 8. Definition of Done

- [ ] Đã tái hiện lỗi trong trạng thái tối giản.
- [ ] Đã khoanh vùng được nguyên nhân gốc.
- [ ] Đã cô lập bằng assertion tại ranh giới nghi ngờ.
- [ ] Đã áp dụng giải pháp cấu trúc, không tình thế.
- [ ] Đã thêm assertion hoặc test phòng ngừa tái diễn.
- [ ] Đã xác nhận lỗi không còn xuất hiện.
- [ ] Đã ghi chú sự cố và cách xử lý vào tài liệu.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Cứ giảm learning rate là hết NaN"* -> **Bác bỏ:** Giảm learning rate có thể che triệu chứng mà không sửa nguyên nhân (ví dụ log(0) hoặc dữ liệu hỏng). Lỗi sẽ tái xuất hiện.
- [X] *"Tăng batch size cho nhanh dù VRAM căng"* -> **Bác bỏ:** Gây CUDA OOM giữa chừng và mất toàn bộ tiến trình. Cân nhắc gradient accumulation thay vì tăng batch.
- [X] *"Lỗi đã hết rồi, khỏi thêm test"* -> **Bác bỏ:** Không có test phòng ngừa thì lỗi sẽ quay lại trong phiên sau. Mỗi lỗi đã gỡ nên để lại một hàng rào.
- [X] *"Sửa tạm cho chạy được đã, tính sau"* -> **Bác bỏ:** Giải pháp tình thế tích lũy nợ kỹ thuật và che giấu vấn đề thiết kế thật. Sửa cấu trúc ngay khi có thể.
