---
name: performance-optimization
description: Tối ưu hóa throughput tính toán, pipeline nạp dữ liệu, mức sử dụng GPU và mức tiêu thụ bộ nhớ trên các workload Data Science, Machine Learning và Deep Learning.
---

# performance-optimization (Bản DS & AI)

## 1. Tổng quan

Skill này tập trung vào việc chẩn đoán và loại bỏ các điểm nghẽn về tính toán, bộ nhớ và I/O. Trong kỹ thuật deep learning, tối ưu hóa có hệ thống giúp giảm thời gian huấn luyện từ hàng ngày xuống hàng giờ, và cho phép huấn luyện những kiến trúc lớn hơn trong ngân sách GPU hạn chế.

Điểm mấu chốt: tối ưu hóa phải dựa trên đo lường, không dựa trên phỏng đoán. Phải xác định đúng điểm nghẽn trước khi tối ưu, nếu không công sức sẽ đổ vào nhầm chỗ.

Skill cũng nhấn mạnh đánh đổi: mỗi tối ưu có thể ảnh hưởng đến độ chính xác hoặc tính tái lập. Người tối ưu phải cân nhắc và ghi nhận đánh đổi.

Nguyên lý then chốt: đo lường trước, tối ưu sau, và xác nhận cải thiện bằng số liệu.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Điểm nghẽn nạp dữ liệu khiến GPU bị bỏ không (GPU Utilization < 50%).
- Ràng buộc VRAM GPU ngăn cản việc đặt batch size hiệu quả.
- Độ trễ inference cao vượt quá mục tiêu cam kết dịch vụ.
- Thời gian huấn luyện quá lâu so với kỳ vọng.
- Cần chạy kiến trúc lớn hơn trong ngân sách VRAM hiện có.

**Điều kiện tiên quyết:**
- Đã có pipeline huấn luyện chạy đúng và cho kết quả hợp lệ.
- Đã có cách đo lường throughput và mức sử dụng tài nguyên.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Pipeline huấn luyện đang chạy đúng | Điểm nghẽn đã được xác định |
| Số liệu throughput và mức sử dụng tài nguyên | Tối ưu đã áp dụng kèm đánh đổi |
| Ngân sách VRAM | Bằng chứng cải thiện throughput |

## 4. Quy trình chi tiết

### Bước 1: Đo lường trước khi tối ưu
Xác định mức sử dụng GPU, tốc độ nạp dữ liệu, và thời gian mỗi epoch. Việc đo lường trước giúp tránh tối ưu nhầm chỗ. Nếu GPU bị bỏ không dưới 50%, điểm nghẽn nằm ở pipeline dữ liệu chứ không phải ở tính toán.

### Bước 2: Tối ưu pipeline dữ liệu và CPU host
Thay thế vòng lặp Python và Pandas `.apply()` bằng tính toán vector hóa NumPy hoặc engine hiệu năng cao như Polars. Tinh chỉnh DataLoader: đặt `num_workers` phù hợp để tiền xử lý song song, bật `pin_memory` để tăng tốc truyền dữ liệu, và dùng định dạng lưu trữ dạng cột (Parquet, Feather) thay cho CSV.

### Bước 3: Áp dụng Automatic Mixed Precision
Kết hợp số học dấu phẩy động 16-bit và 32-bit để giảm VRAM khoảng 50% và tăng gấp đôi throughput trên Tensor Cores. Quy trình gồm: tạo GradScaler, bọc forward và loss trong autocast, sau đó scale loss, step optimizer, và update scaler. Đây là tối ưu có tác động lớn nhất trên GPU hiện đại.

### Bước 4: Hợp nhất kernel với torch.compile
Trên PyTorch 2.0+, dùng `torch.compile` để tối ưu đồ thị thực thi. Bước này có thể tăng tốc đáng kể nhưng cần thời gian biên dịch lần đầu, nên cân nhắc đánh đổi giữa thời gian biên dịch và lợi ích tốc độ.

### Bước 5: Giảm dấu chân bộ nhớ
Áp dụng lượng tử hóa sau huấn luyện xuống 8-bit hoặc 4-bit nếu cần giảm kích thước mô hình cho inference. Cân nhắc đánh đổi về độ chính xác.

### Bước 6: Tối ưu runtime inference
Tắt autograd engine bằng `torch.inference_mode()` khi chạy inference. Xuất mô hình sang ONNX hoặc TensorRT nếu cần độ trễ thấp hơn nữa.

### Bước 7: Xác nhận cải thiện bằng số liệu
Đo lại throughput và mức sử dụng tài nguyên sau mỗi tối ưu. Xác nhận cải thiện thực sự bằng số liệu, không dựa trên cảm giác. Nếu một tối ưu không cải thiện hoặc làm giảm độ chính xác, cân nhắc loại bỏ.

## 5. Deliverables

- Báo cáo điểm nghẽn đã xác định.
- Danh sách tối ưu đã áp dụng kèm đánh đổi.
- Số liệu throughput trước và sau tối ưu.
- Ghi chú về ảnh hưởng đến độ chính xác và tính tái lập.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `debugging-and-error-recovery` (phát hiện điểm nghẽn) | `kaggle-remote-execution` (áp dụng tối ưu khi chạy từ xa) |
| `environment-and-reproducibility` (cung cấp ràng buộc) | `walkthrough-and-experiment-tracking` (ghi nhận tác động lên metric) |
| | `api-and-interface-design` (nhận mô hình tối ưu cho inference) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Tối ưu không đo lường:** Áp dụng tối ưu mà không biết điểm nghẽn thật. Dấu hiệu: không cải thiện throughput. Phòng tránh: đo ở Bước 1.
- **`num_workers` quá cao:** Gây bus error trên Kaggle do cạn `/dev/shm`. Phòng tránh: đặt giá trị an toàn.
- **AMP gây NaN:** Mixed precision có thể gây bất ổn định số học. Dấu hiệu: loss NaN sau khi bật AMP. Phòng tránh: dùng GradScaler đúng cách và theo dõi loss.
- **torch.compile chậm lần đầu:** Thời gian biên dịch lâu khiến tưởng không cải thiện. Phòng tránh: đo qua nhiều epoch sau khi biên dịch xong.
- **Quên `inference_mode`:** Chạy inference trong chế độ có autograd gây tốn bộ nhớ và chậm. Phòng tránh: dùng `torch.inference_mode()`.

## 8. Definition of Done

- [ ] Đã đo lường throughput và mức sử dụng tài nguyên ban đầu.
- [ ] Đã xác định chính xác điểm nghẽn.
- [ ] Đã tối ưu pipeline dữ liệu nếu đó là điểm nghẽn.
- [ ] Đã cân nhắc và áp dụng AMP nếu phù hợp.
- [ ] Đã đo lại và xác nhận cải thiện bằng số liệu.
- [ ] Đã ghi nhận đánh đổi về độ chính xác và tính tái lập.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Cứ tối ưu hết mọi thứ cho chắc"* -> **Bác bỏ:** Tối ưu không đo lường có thể đổ công sức vào chỗ không phải điểm nghẽn và làm phức tạp code vô ích. Đo trước, tối ưu sau.
- [X] *"Đặt `num_workers` cao cho nhanh"* -> **Bác bỏ:** Trên Kaggle, `num_workers` cao gây bus error do cạn `/dev/shm`. Đặt thấp là an toàn và đủ nhanh.
- [X] *"Bật AMP là chắc chắn nhanh hơn"* -> **Bác bỏ:** AMP có thể gây bất ổn định số học và NaN nếu không dùng GradScaler đúng. Phải theo dõi loss sau khi bật.
- [X] *"Tối ưu xong rồi, khỏi đo lại"* -> **Bác bỏ:** Không đo lại thì không biết tối ưu có thực sự cải thiện hay vô tình làm chậm. Xác nhận bằng số liệu là bắt buộc.
