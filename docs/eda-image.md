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

## 4. Các bước EDA

Phân hai nhóm: **core** là bắt buộc cho mọi bài ảnh; **mở rộng** chỉ làm khi có câu
hỏi cụ thể và ngân sách phù hợp. Bỏ một bước mở rộng không làm EDA thất bại, nhưng
phải ghi lý do.

### Step 1. Structural Overview — core

Kiểm tra số lượng ảnh và nhãn, array shape, image size, số kênh, dtype, min/max pixel,
memory usage, và mapping label sang class name.

Đầu ra: `structural_overview.csv`.

### Step 2. Data Integrity Check — core

Kiểm tra label ngoài miền, pixel ngoài miền, NaN/Inf, ảnh toàn một giá trị, và ảnh
trùng lặp.

Ba điểm dễ làm sai:

- **Exact duplicate và near-duplicate là hai phép đo khác nhau.** Báo riêng.
- **Duplicate trong một split khác với duplicate giữa các split.** Báo riêng.
- **Kiểm tra indices rời nhau khác với kiểm tra trùng nội dung.** Cả hai đều cần;
  indices rời nhau không bảo đảm nội dung không trùng.

Đầu ra: `integrity_report.csv` kèm danh sách index nghi vấn. Không xóa ảnh tự động;
ảnh nghi vấn phải được kiểm tra trước khi quyết định.

### Step 3. Class Distribution — core

Đếm số ảnh và phần trăm mỗi class; so sánh train với validation. Bảng của test chỉ
lập sau khi khóa cấu hình. Không mặc định phân phối cân bằng; tính từ dữ liệu.

Đầu ra: `class_distribution.csv`, `class_distribution.png`.

### Step 4. Visual Inspection — core

Hiển thị 10–20 ảnh ngẫu nhiên mỗi class kèm label; tính mean image (và median nếu
phù hợp) theo class. Chọn mẫu bằng random seed, không lấy những ảnh đầu tiên.

**Mean image làm mất phương sai trong lớp.** Hai lớp có mean gần giống nhau vẫn có
thể phân biệt hoàn hảo, và ngược lại. Mean image chỉ tạo giả thuyết.

Đầu ra: `sample_images.png`, `class_mean_images.png`.

### Step 5. Pixel Intensity Statistics — core

Vẽ histogram pixel; tính global mean/std, tỷ lệ pixel bằng 0, mean/std từng ảnh;
so sánh brightness giữa các class. Mọi thống kê phải ghi rõ tính trên tập nào và
theo thang đo nào.

Đầu ra: `pixel_statistics.csv`, `pixel_histogram.png`,
`image_brightness_distribution.png`.

### Step 6. Spatial Analysis — core

Tính mean-pixel heatmap, pixel-wise variance heatmap, vùng pixel thường xuyên bằng 0,
foreground ratio, và bounding box (width, height, aspect ratio) theo class.

**Foreground threshold phải là hằng số có tên.** Với ảnh có nền lớn, ngưỡng 0 sẽ gộp
nhiễu nền vào bounding box và làm mọi aspect ratio vô nghĩa. Không mặc định mọi pixel
khác 0 là foreground khi chưa đánh giá.

Đầu ra: `mean_pixel_heatmap.png`, `pixel_variance_heatmap.png`, `spatial_features.csv`.

### Step 7. Handcrafted Feature Analysis — mở rộng

Đặc trưng mỗi ảnh: mean, std, foreground ratio, edge density (Sobel), bounding box
width/height/aspect ratio, symmetry. Vẽ box/violin theo class; phân tích skewness,
kurtosis, IQR.

Các đặc trưng này phục vụ EDA, không mặc định là đầu vào của MLP.

Đầu ra: `handcrafted_features.csv`, `feature_boxplots.png`.

### Step 8. Inter-Class Similarity — mở rộng

Cosine hoặc correlation giữa mean image của các class; vẽ heatmap. Đây là máy tạo
giả thuyết, không phải kết luận: xác nhận phải bằng per-class metric trên validation.

Đầu ra: `class_similarity_matrix.csv`, `class_similarity_heatmap.png`.

### Step 9. Dimensionality Reduction — mở rộng

PCA: flatten mỗi ảnh, **fit trên train**, vẽ cumulative explained variance, xác định
số component cho 90% và 95%, scatter PC1–PC2 theo class.

t-SNE/UMAP: subsample, giảm chiều, ghi rõ seed và Hyperparameters. **Khoảng cách trên
t-SNE không có ý nghĩa định lượng**, nên không dùng để khẳng định class nào dễ phân
biệt. Nếu thư viện chưa có, ghi rõ và bỏ qua, không cài ngầm.

Đầu ra: `pca_explained_variance.png`, `pca_components_summary.csv`, `pca_scatter.png`.

### Step 10. Outlier and Potential Label Issues — mở rộng

Xếp hạng ảnh nghi vấn bằng PCA reconstruction error, Mahalanobis distance, hoặc kNN
(hàng xóm thuộc class khác). Hiển thị ảnh kèm label và điểm bất thường.

Không kết luận nhãn sai hoặc loại ảnh chỉ từ điểm outlier. Phân biệt "hợp lệ nhưng
hiếm" với "có khả năng lỗi dữ liệu".

Đầu ra: `potential_outliers.csv`, `potential_outliers.png`.

### Step 11. Distribution Shift Check — mở rộng

So sánh class distribution, global pixel mean/std, mean image, brightness distribution
giữa các tập. Có thể dùng KS test cho brightness.

Không chỉ dựa vào p-value; không xem mọi khác biệt là data drift. Với split phân tầng
lấy từ cùng một tập gốc, train và validation được kỳ vọng gần như đồng nhất — khác
biệt lớn ở đây là dấu hiệu lỗi split, không phải drift thật.

Đầu ra: `distribution_comparison.csv`, `distribution_comparison.png`.

### Step 12. Optional Baseline Separability — mở rộng

Fit Logistic Regression hoặc kNN trên ảnh flatten hoặc đặc trưng PCA; đánh giá trên
validation; báo accuracy, macro F1, confusion matrix. Không dùng test để chọn mô hình.

Đầu ra: `baseline_separability.csv`, `baseline_confusion_matrix.png`.

### Step 13. Findings and Logging — core

Tổng hợp kết quả và chuyển thành quyết định. Ghi mỗi phát hiện theo bảng:
`phát hiện | split/view | bằng chứng/artifact | đề xuất | cách kiểm chứng | trạng thái`.

Tách rõ ba loại: kết quả đã chạy, giả thuyết, và phần chưa triển khai. Không viết sẵn
kết luận định lượng.

Đầu ra: nhận xét trong `walkthrough_lab_NN.md`, bảng CSV, biểu đồ PNG. Không tạo
report Markdown riêng.

## 5. Quyết định sau EDA

- **Preprocessing:** chuyển Tensor, scale `[0, 1]`, có dùng Normalize hay không và
  dựa trên thống kê nào.
- **Augmentation:** phép nào phù hợp ngữ nghĩa của loại ảnh; không mặc định bật.
- **Kiến trúc:** ghi rõ model nhận dạng input nào; `Flatten` nằm trong model hay ngoài.
- **Độ khó của class:** similarity chỉ là giả thuyết, xác nhận bằng confusion matrix
  và per-class metric sau khi huấn luyện.
- **Giảm chiều:** nếu PCA giữ phần lớn variance với ít chiều, có thể cân nhắc cho
  visualization hoặc model truyền thống; không mặc định dùng làm đầu vào.

## 6. Đầu ra

Bảng CSV và biểu đồ PNG vào `outputs/eda/`, cộng một file JSON tổng hợp để notebook
training đọc lại. Nhận xét và quyết định ghi vào `walkthrough_lab_NN.md`.
