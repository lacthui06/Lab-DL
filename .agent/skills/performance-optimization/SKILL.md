---
name: performance-optimization
description: Tối ưu hóa throughput tính toán, pipeline nạp dữ liệu, mức sử dụng GPU và mức tiêu thụ bộ nhớ trên các workload Data Science, Machine Learning và Deep Learning.
---

# performance-optimization (Bản DS & AI)

## Tổng quan

Skill này tập trung vào việc chẩn đoán và loại bỏ các điểm nghẽn (bottleneck) về tính toán, bộ nhớ và I/O. Trong kỹ thuật deep learning, tối ưu hóa có hệ thống giúp giảm thời gian huấn luyện từ hàng ngày xuống hàng giờ, và cho phép huấn luyện những kiến trúc lớn hơn trong ngân sách GPU hạn chế.

Điểm mấu chốt: tối ưu hóa phải dựa trên đo lường, không dựa trên phỏng đoán. Phải xác định đúng điểm nghẽn trước khi tối ưu, nếu không công sức sẽ đổ vào nhầm chỗ.

## Khi nào sử dụng

- Điểm nghẽn nạp dữ liệu khiến GPU bị bỏ không (GPU Utilization < 50%).
- Ràng buộc VRAM GPU ngăn cản việc đặt batch size hiệu quả.
- Độ trễ inference cao vượt quá mục tiêu cam kết dịch vụ (SLO) production.

## 4 trụ cột tối ưu trong DS & AI

### 1. Tối ưu pipeline dữ liệu và CPU host
- **Tính toán vector hóa (Vectorized Computation):** Thay thế vòng lặp `for` Python và lệnh gọi Pandas `.apply()` bằng các routine **NumPy** đã vector hóa hoặc các engine hiệu năng cao như **Polars**.
- **Tinh chỉnh PyTorch `DataLoader`:**
  - Cấu hình `num_workers = 2` hoặc `4` để bật tiền xử lý song song chạy nền.
  - Đặt `pin_memory = True` để tăng tốc truyền bộ nhớ page-locked từ host sang device.
  - Áp dụng các định dạng lưu trữ dạng cột throughput cao (`Parquet` hoặc `Feather`) thay cho `.csv` thuần.

### 2. Tối ưu huấn luyện tăng tốc bằng GPU
- **Automatic Mixed Precision (AMP FP16/BF16):**
  - Kết hợp số học dấu phẩy động 16-bit và 32-bit để giảm yêu cầu VRAM khoảng 50% và tăng gấp đôi throughput trên Tensor Cores.
  - Quy trình gồm: tạo `torch.amp.GradScaler('cuda')`, bọc forward và loss trong `torch.amp.autocast('cuda')`, sau đó gọi `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, và `scaler.update()`.
- **Hợp nhất kernel với `torch.compile()`:** Trên PyTorch 2.0+, tối ưu đồ thị thực thi với `model = torch.compile(model)`.

### 3. Giảm dấu chân bộ nhớ (Quantization & Pruning)
- Áp dụng lượng tử hóa sau huấn luyện (post-training quantization) xuống độ chính xác 8-bit hoặc 4-bit (thông qua `bitsandbytes` hoặc PyTorch Quantization).

### 4. Tối ưu runtime inference
- **Tắt Autograd Engine:** Bao toàn bộ lượt inference trong `with torch.inference_mode():` (hiệu quả hơn `torch.no_grad()`).
- **Xuất sang runtime tối ưu:** Xuất trọng số PyTorch đã huấn luyện sang **ONNX** hoặc **TensorRT** để phục vụ qua `onnxruntime`.

## Tiêu chí xác thực

- Throughput đo được (samples/giây) cho thấy cải thiện rõ rệt, và mức sử dụng GPU compute duy trì ổn định ở mức cao (> 85%).
