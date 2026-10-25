# EDA cho dữ liệu ảnh

Flow chung cho bước EDA của bài toán phân loại ảnh trong repo này. Tài liệu này là
**quy trình**, không chứa kết quả đo. Mọi con số đo được ghi trong
`labs/labNN/walkthrough_lab_NN.md` kèm `split_version` và `preproc_version`.

Bước tiếp theo: [feature-image.md](feature-image.md).
Kỹ thuật chung: [eda.md](eda.md). Luồng Deep Learning: [dl_flow.md](dl_flow.md).
Quy tắc project: [AGENTS.md](../.agents/AGENTS.md).

## 1. Nguyên tắc

1. Chia tập **một lần**, dùng lại cho EDA, Feature Engineering và mọi run training.
2. Fit mọi tham số học từ dữ liệu (mean/std, PCA, scaler) trên **train only**.
3. Validation dùng để chọn cấu hình. Test chỉ mở sau khi khóa cấu hình.
4. Không hard-code thống kê. Đo, ghi artifact, đọc lại từ artifact.
5. Không tự xóa mẫu. Báo cáo trước, quyết định sau.
6. Mỗi kết quả phải tái lập được và ghi rõ version của split và preprocessing.
7. Nếu một phép kiểm tra không chạy, ghi `not computed`. Đừng ghi `0` hoặc bỏ
   trống, vì người đọc sẽ hiểu là "đã kiểm tra và sạch".

## 2. Chia tập

Dùng stratified split để giữ tỷ lệ lớp giữa train và validation:

```python
train_test_split(test_size=0.1, random_state=42, shuffle=True, stratify=y)
```

- Với dữ liệu ảnh, phân tầng là bắt buộc. Split ngẫu nhiên không phân tầng cho
  validation lệch lớp, làm per-class metric nhiễu theo cách không liên quan đến
  chất lượng mô hình.
- Tạo split **một lần**, lưu indices kèm `dataset identity`, `seed`, `split_version`.
- Module đọc split phải **raise lỗi** khi artifact vắng mặt, không được chia lại.
  Một split khác làm mọi thống kê đã đo trở nên vô nghĩa.

## 3. Ba view của dữ liệu

Mỗi phép đo phải ghi rõ chạy trên view nào. Trộn view là nguồn sai số phổ biến nhất
khi báo cáo thống kê ảnh.

| View | Kiểu | Shape | Khoảng | Dùng cho |
|---|---|---|---|---|
| Raw | `uint8` | `[H, W]` | `[0, 255]` | integrity, hash, duplicate |
| Scaled | `float32` | `[H, W]` | `[0, 1]` | tính mean/std của train |
| Model | `float32` | `[B, C, H, W]` | ngoài `[0, 1]` | forward/backward |

Loader dùng cho EDA phải `shuffle=False`, `drop_last=False`, không augmentation.
Loader training thường `drop_last=True`, nên một vòng qua nó không đại diện đủ cho
toàn bộ train. Với mẫu nghi vấn, lưu `source_dataset`, `original_index`, `split`,
`label`; index cục bộ của subset không đủ để truy ngược ảnh.

## 4. Các bước thực hiện EDA cho ảnh

Mọi bước dưới đây đều là quy chuẩn bắt buộc thực hiện đồng nhất trong pipeline EDA ảnh.

### Step 1. Structural Overview
Kiểm tra số lượng ảnh và nhãn, array shape, image size, số kênh, dtype, min/max pixel,
memory usage, và mapping label sang class name.

Đầu ra: `structural_overview.csv`.

### Step 2. Data Integrity Check
Kiểm tra label ngoài miền, pixel ngoài miền, NaN/Inf, ảnh toàn một giá trị, và ảnh
trùng lặp (trùng nội bộ Train, trùng nội bộ Test và rò rỉ chéo Train-Test).

Đầu ra: `integrity_report.csv` kèm danh sách index nghi vấn.

### Step 3. Class Distribution
Đếm số ảnh và phần trăm mỗi class; so sánh train với validation. Đảm bảo split phân tầng
đều đặn giữa các lớp.

Đầu ra: `class_distribution.csv`, `class_distribution.png`.

### Step 4. Visual Inspection
Hiển thị ảnh mẫu ngẫu nhiên đại diện cho từng class kèm nhãn; tính mean image theo class.
Chọn mẫu bằng random seed cố định.

Đầu ra: `sample_images.png`, `class_mean_images.png`.

### Step 5. Pixel Intensity Statistics
Vẽ histogram pixel; tính global mean/std, tỷ lệ pixel bằng 0, mean/std từng ảnh;
so sánh brightness giữa các class.

Đầu ra: `pixel_statistics.csv`, `pixel_histogram.png`.

### Step 6. Spatial Analysis
Tính mean-pixel heatmap, pixel-wise variance heatmap, vùng pixel thường xuyên bằng 0,
foreground ratio, và bounding box (width, height, aspect ratio) theo class.

Đầu ra: `mean_pixel_heatmap.png`, `pixel_variance_heatmap.png`, `spatial_features.csv`.

### Step 7. Handcrafted Feature Analysis
Đặc trưng mỗi ảnh: mean, std, foreground ratio, edge density, bounding box. Phân tích
phương sai và phân phối đặc trưng theo từng lớp.

Đầu ra: `handcrafted_features.csv`.

### Step 8. Inter-Class Similarity
Tính ma trận tương đồng Cosine hoặc correlation giữa mean image của các class; vẽ heatmap
để định vị các cặp lớp dễ nhầm lẫn nhất.

Đầu ra: `class_similarity_matrix.csv`, `class_similarity_heatmap.png`.

### Step 9. Findings and Logging
Tổng hợp kết quả và chuyển thành quyết định tiền xử lý, kiến trúc mô hình và ngưỡng phân loại.

Đầu ra: nhận xét trong `walkthrough_lab_NN.md`, bảng CSV, biểu đồ PNG và file cấu hình tiền xử lý `preprocessing_config.json`.

## 5. Quyết định sau EDA

- **Preprocessing:** chuyển Tensor, scale `[0, 1]`, có dùng Normalize hay không và dựa trên thống kê đo đạc thực tế.
- **Augmentation:** phép nào phù hợp ngữ nghĩa của loại ảnh; không mặc định bật bừa bãi.
- **Kiến trúc:** ghi rõ model nhận dạng input nào; `Flatten` nằm trong model hay ngoài.
- **Độ khó của class:** similarity chỉ là giả thuyết, xác nhận bằng confusion matrix và per-class metric sau khi huấn luyện.

## 6. Đầu ra

Bảng CSV và biểu đồ PNG vào `outputs/eda/`, cộng một file JSON tổng hợp để notebook
training đọc lại. Nhận xét và quyết định ghi vào `walkthrough_lab_NN.md`.
