---
name: performance-optimization
description: Tối ưu hóa hiệu năng tính toán, tốc độ nạp dữ liệu và dung lượng bộ nhớ GPU trong các tác vụ Data Science, Machine Learning và Deep Learning.
---

# performance-optimization (DS & AI Edition)

## Tổng quan
Kỹ năng này tập trung vào việc loại bỏ "nút thắt cổ chai" (bottleneck) về tốc độ và bộ nhớ. Trong AI, tối ưu hóa có thể rút ngắn thời gian huấn luyện từ vài ngày xuống vài giờ, đồng thời cho phép huấn luyện các mô hình lớn hơn trên phần cứng giới hạn.

## Khi nào sử dụng
- Quá trình nạp dữ liệu (Data Loading) chậm khiến GPU phải chờ CPU (GPU Utilization < 50%).
- Bộ nhớ GPU bị giới hạn, không tăng được batch size.
- Thời gian suy luận (Inference Latency) quá lâu, không đạt yêu cầu thực tế.

## 4 Trục tối ưu hóa hiệu năng trong DS & AI

### 1. Tối ưu hóa Dữ liệu (CPU & Data Pipeline)
- **Triệt tiêu vòng lặp Python thuần:** Thay thế toàn bộ vòng lặp `for` và hàm `.apply()` trong Pandas bằng tính toán vector hóa của **NumPy** hoặc chuyển sang dùng **Polars**.
- **Tối ưu hóa `DataLoader` PyTorch:**
  - Thiết lập `num_workers = 2` hoặc `4` (tùy số nhân CPU) để nạp dữ liệu đa tiến trình song song.
  - Bật `pin_memory = True` để tăng tốc độ chuyển tensor từ RAM CPU sang VRAM GPU.
  - Sử dụng định dạng lưu trữ nhanh: Dùng `Parquet` hoặc `Feather` thay cho file `.csv` cồng kềnh.

### 2. Tối ưu hóa Huấn luyện GPU (Deep Learning Training)
- **Huấn luyện độ chính xác hỗn hợp (Automatic Mixed Precision - AMP):**
  - Sử dụng FP16/BF16 kết hợp FP32 để giảm 50% dung lượng VRAM và tăng tốc độ tính toán gấp 2 - 3 lần trên GPU Tensor Cores:
  ```python
  scaler = torch.amp.GradScaler('cuda')
  for inputs, labels in dataloader:
      optimizer.zero_grad(set_to_none=True)
      with torch.amp.autocast('cuda'):
          outputs = model(inputs)
          loss = criterion(outputs, labels)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
  ```
- **Tận dụng `torch.compile()`:** Trên PyTorch 2.0+, biên dịch mô hình với `model = torch.compile(model)` để dung hợp các kernel tính toán.

### 3. Tối ưu hóa Kích thước Mô hình (Quantization & Pruning)
- Lượng tử hóa mô hình sang 8-bit hoặc 4-bit (sử dụng thư viện `bitsandbytes` cho LLM hoặc PyTorch Dynamic Quantization).

### 4. Tối ưu hóa Suy luận (Inference Optimization)
- **Tắt theo dõi đạo hàm:** Luôn bọc suy luận bằng `with torch.inference_mode():` (nhanh hơn `torch.no_grad()`).
- **Xuất sang định dạng tối ưu:** Đổi mô hình PyTorch sang **ONNX** hoặc **TensorRT** để chạy suy luận bằng `onnxruntime` trên production.

## Tiêu chí nghiệm thu (Verification)
- Đo đạc trước và sau tối ưu: Tốc độ xử lý (Samples/giây) tăng lên rõ rệt và GPU Utilization duy trì ở mức cao ($> 85\%$).
