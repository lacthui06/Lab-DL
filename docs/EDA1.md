# Fashion-MNIST EDA Pipeline

## 1. Project Context

Tôi đang thực hiện một dự án Deep Learning sử dụng bộ dữ liệu Fashion-MNIST để so sánh hai thí nghiệm:

1. Baseline MLP
2. Tuned MLP

> **Ghi chú phạm vi (cập nhật):** Bản trước của tài liệu này liệt kê ba thí nghiệm, bao gồm
> `FashionCNN`. Kiến trúc CNN đã bị loại khỏi phạm vi Lab 01 (xem commit `498b017`,
> "remove FashionCNN and retarget optimizer sweep to TunedMLP"). `src/model.py` chỉ cung cấp
> `BaselineMLP` và `TunedMLP`, và `build_model()` từ chối mọi `model_type` khác. Mọi phân tích
> trong tài liệu này chỉ phục vụ hai kiến trúc đó.

Mục tiêu của tài liệu này là xây dựng một pipeline EDA có hệ thống, giúp hiểu dữ liệu ảnh, phát hiện vấn đề về chất lượng dữ liệu và đưa ra quyết định có cơ sở cho các bước Preprocessing, Feature Engineering và Model Training.

Phạm vi chỉ là `labs/lab01/`, không áp dụng thiết kế so sánh optimizer của mini-lab Adam. Đây là hướng dẫn triển khai; danh sách đầu ra bên dưới không có nghĩa các phân tích đã được chạy.

Tham chiếu: [Feature Engineering](FEATURE-ENGINEERING.md), [quy tắc project](../.agents/AGENTS.md), [EDA playbook](eda.md), [FE playbook](feature.md), [DL flow](dl_flow.md), [DL data](dl_data.md) và [DL training](dl_training.md). Các playbook chung cần được áp dụng theo dữ liệu ảnh và phạm vi Lab 01, không biến mọi kỹ thuật thành yêu cầu bắt buộc.

### Dataset Information

Fashion-MNIST có các đặc điểm sau:

- Total images: 70,000.
- Original training set: 60,000 images.
- Original test set: 10,000 images.
- Image size: 28 × 28 pixels.
- Image type: grayscale.
- Number of classes: 10.
- Original pixel data type: `uint8`.
- Original pixel value range: `[0, 255]`.
- Image data is non-tabular data, so there are no traditional feature columns.

### Class Mapping

| Label | Class Name |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 2. General Requirements

Hãy triển khai pipeline EDA theo các nguyên tắc sau:

1. Sử dụng Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn và PyTorch/Torchvision khi phù hợp.
2. Viết code có cấu trúc rõ ràng, chia thành các hàm có nhiệm vụ cụ thể.
3. Giải thích mục tiêu, đầu vào, đầu ra và ý nghĩa của từng bước.
4. Không hard-code các kết quả thống kê chưa được tính từ dữ liệu thực tế.
5. Không tự giả định kết quả của biểu đồ, kiểm định thống kê hoặc thí nghiệm.
6. Mọi kết luận phải được hỗ trợ bởi kết quả thực thi hoặc dữ liệu đã kiểm tra.
7. Phân biệt rõ EDA bắt buộc, EDA mở rộng và các phân tích tùy chọn.
8. Lưu các bảng thống kê dưới dạng CSV và biểu đồ dưới dạng PNG.
9. Đảm bảo có thể chạy lại pipeline và tái tạo kết quả.
10. Không thay đổi dữ liệu gốc hoặc loại bỏ ảnh chỉ vì chúng bị đánh dấu là outlier.
11. Không sử dụng test set để lựa chọn mô hình, điều chỉnh Hyperparameters hoặc quyết định Preprocessing.
12. Nếu môi trường thiếu thư viện hoặc dữ liệu không khả dụng, hãy báo lỗi rõ ràng thay vì tạo ra kết quả giả.

## 3. Data Splitting and Leakage Prevention

Sử dụng stratified split để chia original training set theo nhãn:

- Training set: 90% của 60,000 ảnh ban đầu, tương đương 54,000 ảnh.
- Validation set: 10% của 60,000 ảnh ban đầu, tương đương 6,000 ảnh.
- Test set: giữ nguyên 10,000 ảnh gốc.

Chia indices của 60,000 ảnh original train bằng `sklearn.model_selection.train_test_split`, với `test_size=0.1`, `random_state=42`, `shuffle=True` và `stratify=original_train_targets`. Tham số `test_size` ở đây tạo validation, không thay đổi official test set. Không dùng random split không phân tầng. Kiểm tra số lượng và tỷ lệ từng lớp sau khi chia.

EDA và cả hai run training phải dùng cùng train/validation indices; không chia lại trong Feature Engineering. Khi triển khai, lưu indices kèm dataset identity, seed, tỷ lệ split và phiên bản thư viện để truy vết. Không khẳng định artifact indices đã tồn tại nếu chưa kiểm tra.

**Trạng thái triển khai (đã đồng bộ):** `src/split.py` tạo split bằng `train_test_split(stratify=...)` và ghi `outputs/eda/split_indices.npz` cùng `split_metadata.json`; `src/dataset.py` chỉ đọc lại artifact đó qua `load_split()` và **không** chia lại. Lý do thay `torch.utils.data.random_split`: split ngẫu nhiên có seed vẫn cho validation lệch lớp (537–658 ảnh/lớp thay vì 600, đo trên dữ liệu thật). Split được đặt version `split_v2_stratified`; kết quả Phase 1 cũ mang metadata `split_v1` và được giữ riêng, không so sánh trực tiếp với kết quả chạy trên split mới.

Yêu cầu:

- Sử dụng random seed cố định.
- Giữ nguyên test set cho đánh giá cuối cùng.
- Tách validation từ original training set.
- Thực hiện các phép tính phục vụ việc lựa chọn mô hình dựa trên training set.
- Không sử dụng validation hoặc test để tính các tham số chuẩn hóa dùng trong training.
- Trong phát triển, phân tích chi tiết trên train; dùng validation để đánh giá giả thuyết. Chỉ mở phân tích chi tiết test sau khi khóa cấu hình. Kiểm tra kỹ thuật về số lượng, shape, label và exact duplicates không được dùng để tuning trên test.
- Ghi lại số lượng mẫu và tỷ lệ từng lớp sau khi split.

### Data Views and Coverage

Ba view dưới đây là hợp đồng dữ liệu giữa các tầng trong pipeline. Mỗi phép đo phải ghi rõ nó chạy trên view nào, vì trộn lẫn view là nguồn sai số phổ biến khi báo cáo thống kê ảnh.

| View | Kiểu | Shape | Khoảng giá trị | Dùng cho |
|---|---|---|---|---|
| Raw | `uint8` | `[28, 28]` | `[0, 255]` | integrity, hash, duplicate |
| Scaled | `float32` | `[28, 28]` | `[0, 1]` | tính mean/std của train |
| Model | `float32` | `[B, 1, 28, 28]` | ngoài `[0, 1]` sau Normalize | forward/backward |

- Raw view: ảnh `uint8` `[28, 28]`, pixel `[0, 255]`, dùng kiểm tra integrity và hash; pixel 0 là nền, không mặc định là missing value.
- Scaled view: ảnh `float32` trên `[0, 1]`, chưa augmentation và chưa Normalize, dùng tính mean/std của train.
- Model view: batch `float32` `[B, 1, 28, 28]` đã Normalize; giá trị có thể ngoài `[0, 1]`. Không dùng view này để kiểm tra raw pixel range.
- Đọc đủ raw train subset hoặc dùng loader EDA `shuffle=False`, `drop_last=False`, không augmentation. Loader training hiện dùng `drop_last=True`; một vòng qua loader đó không đại diện đầy đủ cho toàn train.
- Lưu `source_dataset` (original train hoặc original test), `original_index`, `split`, `label` cho mẫu nghi vấn; index cục bộ của subset không đủ để truy ngược ảnh.

Với dữ liệu ảnh, thứ tự kênh cũng là một phần của hợp đồng. `ToTensor()` của torchvision nhận `HWC` và trả `CHW`; nếu đầu vào là `numpy` shape `(1, 28, 28)` thì nó hiểu nhầm là `HWC` và báo `Expected 1 channel(s) at dim -3, got 28`. Vì vậy mọi phép chuyển đổi phải nêu rõ layout đầu vào, và test phải khẳng định shape sau transform.

### Required and Optional Analysis

Core: Steps 1–5 và Step 13, gồm structure, integrity, exact duplicate review, class counts, ảnh mẫu và pixel statistics. Median image trong Step 4 là tùy chọn.

Mở rộng khi có câu hỏi và ngân sách phù hợp: Steps 6–12, gồm spatial features, handcrafted features, similarity, PCA, t-SNE/UMAP, outlier scoring, distribution diagnostics và baseline truyền thống. Bỏ qua một nhánh mở rộng không làm core EDA thất bại; ghi rõ lý do không thực hiện.

## 4. EDA Pipeline

### Step 1. Structural Overview

Mục tiêu: hiểu cấu trúc cơ bản của dữ liệu ảnh.

Thực hiện:

- Kiểm tra số lượng ảnh và nhãn.
- Kiểm tra tensor shape hoặc array shape.
- Kiểm tra image size và number of channels.
- Kiểm tra data type.
- Kiểm tra minimum và maximum pixel values.
- Tính memory usage.
- Kiểm tra mapping từ label sang class name.

Đầu ra:

- Bảng tổng quan dataset.
- Bảng thống kê shape, dtype và pixel range.
- File `structural_overview.csv`.

### Step 2. Data Integrity Check

Mục tiêu: phát hiện vấn đề về tính toàn vẹn dữ liệu.

Thực hiện:

- Kiểm tra số lượng nhãn hợp lệ trong khoảng 0 đến 9.
- Kiểm tra giá trị pixel ngoài khoảng `[0, 255]`.
- Kiểm tra NaN hoặc Inf nếu biểu diễn dữ liệu hỗ trợ các giá trị này.
- Phát hiện ảnh toàn đen hoặc ảnh có cường độ pixel bất thường.
- Phát hiện ảnh trùng lặp bằng hash hoặc phương pháp phù hợp.
- Phân biệt exact duplicates và near-duplicates nếu cần.
- Phân biệt duplicate trong một split và giữa các split; báo riêng ảnh giống nhau nhưng nhãn khác nhau. Kiểm tra không giao nhau về indices và kiểm tra trùng nội dung là hai kiểm tra khác nhau. Không tự xóa hoặc chuyển mẫu, đặc biệt không thay đổi official test set.

Đầu ra:

- Bảng số lượng lỗi theo từng loại.
- Danh sách index của các ảnh nghi vấn.
- File `integrity_report.csv`.

Lưu ý: không xóa ảnh tự động. Các ảnh nghi vấn cần được kiểm tra trước khi đưa ra quyết định.

### Step 3. Class Distribution

Mục tiêu: kiểm tra phân phối nhãn.

Thực hiện:

- Đếm số lượng ảnh theo từng class.
- Tính phần trăm của từng class.
- Vẽ bar chart.
- So sánh số lượng lớp giữa train và validation; bảng test phục vụ báo cáo cuối sau khi khóa cấu hình.

Đầu ra:

- `class_distribution.csv`
- `class_distribution.png`

Không mặc định rằng phân phối luôn cân bằng. Hãy tính toán từ dữ liệu thực tế.

### Step 4. Visual Inspection

Mục tiêu: hiểu đặc điểm trực quan của các class.

Thực hiện:

- Hiển thị 10 đến 20 ảnh ngẫu nhiên cho mỗi class.
- Hiển thị class name và label.
- Tính mean image theo từng class.
- Tính median image theo từng class nếu phù hợp.
- Kiểm tra ảnh mờ, ảnh bất thường và những trường hợp khó phân biệt.

Đầu ra:

- `sample_images.png`
- `class_mean_images.png`
- `class_median_images.png`, nếu được triển khai.

Không chỉ hiển thị những ảnh đầu tiên trong dataset. Cần chọn mẫu có tính đại diện và có thể tái tạo bằng random seed.

Mean image làm mất phương sai trong lớp: hai lớp có thể có mean gần giống nhau nhưng phân bố rất khác. Vì vậy mean image chỉ dùng để tạo giả thuyết, không dùng để kết luận hai lớp khó phân biệt.

### Step 5. Pixel Intensity Statistics

Mục tiêu: hiểu phân phối cường độ pixel.

Thực hiện:

- Vẽ histogram của pixel intensity.
- Tính global mean và global standard deviation.
- Tính tỷ lệ pixel bằng 0.
- Tính mean và standard deviation của từng ảnh.
- So sánh brightness distribution giữa các class.
- Kiểm tra các ảnh có tỷ lệ pixel bằng 0 hoặc độ sáng bất thường.

Chuẩn hóa pixel về `[0, 1]` khi cần thống kê theo thang đo này.

Đầu ra:

- `pixel_statistics.csv`
- `pixel_histogram.png`
- `image_brightness_distribution.png`

Lưu ý: mọi thống kê phải ghi rõ được tính trên tập dữ liệu nào và theo thang đo nào.

Mean/std dùng cho Normalize phải tính trên toàn bộ pixel của train scaled view trước augmentation, theo channel, dùng population variance (`ddof=0` hoặc `correction=0`). Không lấy trung bình std của từng ảnh để thay thế global std. Dùng tích lũy theo batch nếu cần tiết kiệm RAM; kiểm tra số pixel đã đếm, giá trị finite và std > 0.

Code hiện dùng mean `(0.2860,)`, std `(0.3205,)` trong `src/config.py`. Đây là cấu hình hiện có, chưa phải bằng chứng đã tính trên đúng split train. Ghi kết quả đo mới và chênh lệch; không tự ghi đè cấu hình hoặc diễn giải lại kết quả cũ. Nếu chấp nhận thống kê mới, cần version preprocessing và chạy lại các run liên quan.

Đã đo trên dữ liệu thật: global std của train scaled view là `0.353160`, còn `0.320661` là trung bình std của từng ảnh. Hai giá trị này khác nhau về ý nghĩa, không phải sai số làm tròn. Dùng `0.3205` làm mẫu số khiến dữ liệu sau Normalize có std `≈ 1.1019` thay vì `1.0`, tức là bước chuẩn hóa không đạt mục tiêu. Vì vậy `src/config.py` đọc mean/std từ artifact `preprocessing_config.json` do notebook Feature Engineering ghi ra, và chỉ rơi về hằng số dự phòng kèm `RuntimeWarning` khi artifact vắng mặt.

### Step 6. Spatial Analysis

Mục tiêu: hiểu cấu trúc không gian của ảnh.

Thực hiện:

- Tính mean-pixel heatmap.
- Tính pixel-wise variance heatmap.
- Xác định các vùng pixel thường xuyên bằng 0.
- Tính foreground ratio.
- Xác định bounding box của foreground.
- Phân tích width, height và aspect ratio của bounding box theo class.

Đầu ra:

- `mean_pixel_heatmap.png`
- `pixel_variance_heatmap.png`
- `spatial_features.csv`
- Các biểu đồ phân phối theo class.

Cần định nghĩa rõ foreground threshold. Không mặc định rằng mọi pixel khác 0 đều thuộc foreground nếu chưa đánh giá cách định nghĩa đó. Với Fashion-MNIST, pixel nền chiếm khoảng một nửa tổng số pixel, nên ngưỡng thấp sẽ gộp cả vùng nhiễu nền vào foreground và làm sai bounding box. Ngưỡng phải là hằng số có tên trong code, không phải số viết trực tiếp trong hàm.

### Step 7. Handcrafted Feature Analysis

Mục tiêu: khảo sát các đặc trưng thủ công để hiểu khả năng phân biệt giữa các class.

Tạo các đặc trưng:

- Image mean.
- Image standard deviation.
- Foreground ratio.
- Edge density bằng Sobel hoặc một phương pháp phát hiện cạnh phù hợp.
- Bounding box width.
- Bounding box height.
- Bounding box aspect ratio.
- Image symmetry.

Thực hiện:

- Tính các đặc trưng cho từng ảnh.
- Vẽ box plot hoặc violin plot theo class.
- Phân tích skewness, kurtosis và IQR.
- Phát hiện những giá trị đặc trưng bất thường.

Đầu ra:

- `handcrafted_features.csv`
- `feature_boxplots.png`
- `feature_distributions.png`

Lưu ý: các đặc trưng này phục vụ EDA và không mặc định là đầu vào bắt buộc của MLP hoặc CNN.

### Step 8. Inter-Class Similarity

Mục tiêu: khảo sát mức độ tương đồng giữa các class.

Thực hiện:

- Tính mean image của từng class.
- Tính cosine similarity hoặc correlation giữa các mean image.
- Vẽ similarity matrix dưới dạng heatmap.
- Phân tích những cặp class có mức độ tương đồng cao.

Đặc biệt quan sát:

- T-shirt/top và Shirt.
- Pullover và Coat.
- Sandal và Sneaker.
- Sneaker và Ankle boot.

Đầu ra:

- `class_similarity_matrix.csv`
- `class_similarity_heatmap.png`

Không kết luận rằng hai class khó phân loại chỉ dựa trên similarity của mean image. Cần kết hợp với kết quả phân loại và confusion matrix nếu có.

### Step 9. Dimensionality Reduction

Mục tiêu: khảo sát cấu trúc dữ liệu trong không gian nhiều chiều.

#### PCA

Thực hiện:

- Chuyển mỗi ảnh thành vector 784 chiều.
- Fit PCA trên training set.
- Vẽ cumulative explained variance.
- Tính số lượng Principal Components cần thiết để đạt 90% và 95% explained variance.
- Vẽ scatter plot của PC1 và PC2 theo class.

Đầu ra:

- `pca_explained_variance.png`
- `pca_components_summary.csv`
- `pca_scatter.png`

Không giả định trước số lượng Principal Components cần thiết. Hãy tính từ dữ liệu thực tế. PCA phải fit trên train và chỉ transform lên validation/test; fit trên toàn bộ dữ liệu là rò rỉ thông tin.

#### t-SNE hoặc UMAP

Tùy chọn:

- Lấy một subsample khoảng 5,000 đến 10,000 ảnh.
- Giảm chiều xuống hai chiều.
- Tô màu điểm theo class.
- Ghi rõ random seed, preprocessing và các Hyperparameters.

Không dùng biểu đồ t-SNE hoặc UMAP để khẳng định chắc chắn rằng các class có thể phân loại tốt trong không gian ban đầu. Khoảng cách giữa các cụm trên t-SNE không có ý nghĩa định lượng, và `umap` không có trong môi trường hiện tại nên nhánh này cần cài thêm hoặc bỏ qua có ghi chú.

### Step 10. Outlier and Potential Label Issues

Mục tiêu: tìm các ảnh khác thường cần được kiểm tra thủ công.

Phương pháp có thể sử dụng:

- PCA reconstruction error.
- Mahalanobis distance trên không gian đặc trưng phù hợp.
- k-Nearest Neighbors để tìm những ảnh có hàng xóm thuộc class khác.
- Khoảng cách hoặc độ bất thường của handcrafted features.

Thực hiện:

- Xếp hạng các ảnh nghi vấn.
- Hiển thị ảnh, label và chỉ số bất thường.
- Kiểm tra trực quan những ảnh có điểm bất thường cao.
- Phân biệt ảnh hợp lệ nhưng hiếm với ảnh có khả năng lỗi dữ liệu.

Đầu ra:

- `potential_outliers.csv`
- `potential_outliers.png`

Không tự động kết luận nhãn sai hoặc loại bỏ ảnh chỉ từ điểm outlier.

### Step 11. Distribution Shift Check

Đây là nhánh mở rộng: so sánh train/validation trong phát triển; bổ sung test sau khi khóa cấu hình.

Mục tiêu: mô tả sự khác biệt giữa các tập dữ liệu.

Thực hiện:

- So sánh class distribution.
- So sánh global pixel mean và standard deviation.
- So sánh mean image.
- So sánh brightness distribution.
- Có thể sử dụng Kolmogorov-Smirnov test để so sánh phân phối brightness khi phù hợp.

Đầu ra:

- `distribution_comparison.csv`
- `distribution_comparison.png`

Lưu ý:

- Báo cáo rõ ý nghĩa thống kê và mức độ khác biệt thực tế.
- Không chỉ dựa vào p-value.
- Không xem mọi khác biệt giữa các tập là data drift.
- Không sử dụng kết quả trên test set để điều chỉnh mô hình.

Với split phân tầng lấy từ cùng một tập gốc, train và validation được kỳ vọng gần như đồng nhất. Một khác biệt lớn ở đây là dấu hiệu lỗi split, không phải drift thật.

### Step 12. Optional Baseline Separability

Mục tiêu: ước lượng sơ bộ khả năng phân biệt các class bằng một mô hình đơn giản.

Có thể thử:

- Logistic Regression.
- k-Nearest Neighbors.

Dữ liệu đầu vào có thể là ảnh đã Flatten hoặc đặc trưng PCA.

Yêu cầu:

- Fit mô hình trên training set.
- Đánh giá trên validation set.
- Báo cáo accuracy, macro F1-score và confusion matrix.
- Không dùng test set để chọn mô hình.
- Ghi rõ đây là bước đánh giá bổ sung, không phải EDA bắt buộc.

Đầu ra:

- `baseline_separability.csv`
- `baseline_confusion_matrix.png`

### Step 13. Findings and Logging

Mục tiêu: tổng hợp kết quả EDA và chuyển kết quả thành các quyết định có căn cứ.

Báo cáo ghi kết quả core và trạng thái thực hiện của các mục mở rộng:

1. Dataset overview.
2. Data integrity findings.
3. Class distribution.
4. Pixel intensity findings.
5. Spatial analysis findings.
6. Handcrafted feature findings.
7. Inter-class similarity.
8. PCA findings.
9. Potential outliers.
10. Distribution comparison.
11. Limitations.
12. Recommendations for subsequent experiments.

Đầu ra:

- Phần EDA trong `labs/lab01/walkthrough_lab_01.md`; không tạo report Markdown riêng.
- Các bảng CSV.
- Các biểu đồ PNG.
- File cấu hình để tái tạo kết quả nếu cần.

Không viết sẵn các kết luận định lượng. Hãy chạy code, đọc kết quả và tạo nhận xét từ những gì thực sự quan sát được.

Mỗi quyết định bàn giao sang Feature Engineering ghi theo bảng: `phát hiện | split/view | bằng chứng/artifact | đề xuất | cách kiểm chứng trên validation | trạng thái xác nhận`. Tách rõ kết quả đã chạy, giả thuyết và phần chưa triển khai.

## 5. Decisions After EDA

Sau khi hoàn thành EDA, hãy đưa ra khuyến nghị cho hai thí nghiệm.

### Preprocessing

- Chuyển ảnh sang Tensor.
- Scale pixel về `[0, 1]`.
- Nếu sử dụng Normalize, tính mean và standard deviation trên training set sau khi scale.
- Áp dụng cùng quy tắc preprocessing phù hợp cho validation và test.
- Ghi lại preprocessing configuration.

Thứ tự cố định cho dữ liệu ảnh: `ToTensor` → `Normalize`, và với train có thêm augmentation đặt **trước** `ToTensor`. Lý do: các phép augmentation hình học của torchvision nhận `PIL` hoặc tensor, còn `Normalize` cần tensor `float`; đặt augmentation sau `Normalize` sẽ tạo ra giá trị ngoài miền hợp lệ và làm sai thống kê.

### Data Augmentation

Đánh giá các phép biến đổi có phù hợp với ngữ nghĩa của ảnh quần áo hay không.

- RandomHorizontalFlip: cân nhắc và kiểm tra tính phù hợp.
- RandomVerticalFlip: không mặc định sử dụng.
- RandomRotation: dùng góc nhỏ nếu có lý do phù hợp.
- Không áp dụng augmentation ngẫu nhiên cho validation và test trong quy trình đánh giá thông thường.

Với Fashion-MNIST, phép lật và xoay làm thay đổi hình dạng quần áo theo cách không tồn tại trong dữ liệu thật. Vì vậy augmentation chỉ nên áp dụng cho run được thiết kế để kiểm chứng tác dụng của nó, và mọi so sánh phải giữ nguyên phần còn lại của pipeline.

### Model Architecture

- Baseline MLP và Tuned MLP: nhận cùng ảnh `[B, 1, 28, 28]`; `nn.Flatten()` đã nằm trong model, không cần flatten thêm ở data layer.
- Tuned MLP: thử nghiệm các Hyperparameters và kỹ thuật Regularization.

Phạm vi Lab 01 chỉ có hai kiến trúc trên. Không có run CNN, nên không có kết luận nào về việc CNN tốt hơn hay kém hơn MLP trên dữ liệu này — đó là câu hỏi chưa được đo trong lab này.

### Class Difficulty

Nếu EDA cho thấy các class có đặc điểm tương đồng, hãy ghi nhận đây là giả thuyết. Sau khi huấn luyện, kiểm tra confusion matrix và per-class metrics để xác minh.

### Dimensionality Reduction

Nếu PCA cho thấy phần lớn variance được giữ lại với số chiều thấp hơn, có thể cân nhắc PCA cho visualization, clustering hoặc mô hình truyền thống. Không mặc định sử dụng PCA làm đầu vào cho CNN.

## 6. Code Quality Requirements

- Chia code thành các hàm nhỏ, có tên rõ ràng.
- Tách phần loading data, analysis, visualization và reporting.
- Sử dụng random seed để tái tạo kết quả.
- Kiểm tra shape và dtype trước khi tính toán.
- Tránh lặp lại các phép tính tốn kém.
- Với các phép tính trên toàn bộ ảnh, cân nhắc giới hạn bộ nhớ.
- Khi triển khai, lưu bảng/ảnh vào `labs/lab01/outputs/eda/`; trên Kaggle dùng `OUTPUTS_DIR / "eda"`.
- Tách `required EDA` khỏi `optional analysis`.
- Không cài đặt thêm thư viện nếu thư viện hiện có đáp ứng được yêu cầu.
- Nếu có lỗi môi trường hoặc thiếu dependency, giải thích nguyên nhân và đề xuất cách khắc phục trước khi thay đổi môi trường.

Lưu ý về API thư viện: các phiên bản mới thay đổi tham số theo hướng không tương thích ngược, nên code phải kiểm chứng với phiên bản đang cài thay vì theo tài liệu cũ. Ví dụ đã gặp trong lab này:

| API | Cách gọi cũ | Cách gọi đúng ở phiên bản hiện tại |
|---|---|---|
| `torch.Tensor.std` | `std(ddof=0)` | `std(correction=0)` |
| `Axes.boxplot` | `labels=` | `tick_labels=` |
| `numpy.ndarray.std` | `std(axis=0, ddof=1)` trên mảng 2 chiều trả về mảng | `.reshape(-1).std(ddof=1)` để có scalar |

## 7. Expected Folder Structure

```text
labs/lab01/
├── data/
├── notebooks/
│   ├── lab01_eda.ipynb                     # EDA, sinh split + thống kê
│   ├── lab01_feature_engineering.ipynb     # chốt normalization + augmentation
│   └── lab01_fashionmnist_walkthrough.ipynb
├── src/
│   ├── config.py
│   ├── split.py
│   ├── dataset.py
│   ├── eda.py
│   ├── model.py
│   ├── engine.py
│   └── utils.py
├── scripts/
│   ├── build_eda_notebook.py
│   ├── build_fe_notebook.py
│   └── patch_training_notebook.py
├── outputs/
│   └── eda/
│       ├── split_indices.npz
│       ├── split_metadata.json
│       ├── eda_findings.json
│       ├── preprocessing_config.json
│       ├── tables/
│       └── figures/
└── walkthrough_lab_01.md     # nhận xét, quyết định và báo cáo
```

Tái sử dụng notebook và module hiện có; không tạo thêm cây `src/eda/` song song. Chỉ đề xuất tách module nếu khối lượng thực tế cần và được xác nhận. File phụ trợ AI nếu cần giữ lại phải theo vị trí `.agents/<task>/` đã được xác nhận, không trộn vào artifact sản phẩm.

Notebook sinh ra từ script Python (không sửa tay) để cell luôn khớp với `src/`. Lý do: notebook sửa tay sẽ lệch khỏi module và không tái tạo được.

## 8. Expected Final Report

Báo cáo cuối cùng trả lời các câu hỏi core; với phân tích mở rộng chưa chạy, ghi rõ chưa thực hiện thay vì suy đoán:

- Dataset có cấu trúc như thế nào?
- Có vấn đề gì về chất lượng hoặc tính toàn vẹn dữ liệu không?
- Các class phân bố ra sao?
- Pixel intensity và spatial structure có đặc điểm gì?
- Các handcrafted features có khả năng phân biệt class ở mức nào?
- Những class nào có mean image tương đồng?
- PCA cho thấy dữ liệu có cấu trúc như thế nào?
- Có ảnh bất thường nào cần kiểm tra thủ công không?
- Có sự khác biệt đáng chú ý giữa các tập dữ liệu không?
- Những phát hiện nào ảnh hưởng đến Preprocessing, Feature Engineering và Model Training?
- Những nhận định nào mới chỉ là giả thuyết và cần xác minh bằng thí nghiệm?

## 9. Execution Workflow

Trước khi viết hoặc chỉnh sửa code:

1. Kiểm tra repository và cấu trúc dữ liệu hiện có.
2. Xác định framework và các dependency đã cài đặt.
3. Đề xuất kế hoạch triển khai, chia thành các bước nhỏ.
4. Nêu rõ đầu vào, đầu ra và mục đích của từng bước.
5. Phân biệt phần bắt buộc và phần tùy chọn.

Sau khi kế hoạch được xác nhận:

1. Triển khai từng nhóm chức năng.
2. Chạy kiểm tra để xác nhận code hoạt động.
3. Kiểm tra các bảng và biểu đồ được tạo ra.
4. Tổng hợp kết quả thực tế vào phần EDA của `labs/lab01/walkthrough_lab_01.md`.
5. Báo cáo các lỗi, giới hạn và phần chưa thực hiện.

Không tự động thay đổi kiến trúc dự án, cài đặt hàng loạt thư viện hoặc triển khai toàn bộ các phân tích mở rộng khi chưa đánh giá nhu cầu và chi phí thực hiện.

Trước khi ghi đè bất kỳ file nào bằng script, cần commit hoặc sao lưu file đó. Bài học từ lab này: hai tài liệu chưa được Git theo dõi đã bị hỏng encoding khi đi qua pipeline `Get-Content | Set-Content` của PowerShell, và không có bản sao nào để phục hồi. Với file text có ký tự ngoài ASCII, dùng công cụ so khớp literal thay vì round-trip qua shell.
