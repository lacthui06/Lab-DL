# Feature Engineering Pipeline for Fashion-MNIST

## 1. Project Context

Tôi đang thực hiện một dự án Deep Learning sử dụng Fashion-MNIST để nghiên cứu và so sánh hai thí nghiệm chính:

1. Baseline MLP
2. Tuned MLP

> **Ghi chú phạm vi (cập nhật):** Bản trước của tài liệu này liệt kê ba thí nghiệm, bao gồm
> `FashionCNN`. Kiến trúc CNN đã bị loại khỏi phạm vi Lab 01 (commit `498b017`). Mọi nhánh
> liên quan đến CNN trong tài liệu này (ví dụ Conv Autoencoder ở Step 6B) vì vậy cũng nằm
> ngoài phạm vi, không chỉ là "tùy chọn chưa xác nhận".

Mục tiêu là xây dựng một pipeline Feature Engineering và Representation Learning có hệ thống, bao gồm:

- Data Transformation.
- Data Scaling.
- Data Augmentation.
- Handcrafted Features, nếu có cơ sở.
- PCA Baseline.
- Deep Feature Extraction.
- Representation Selection.
- Regularization.
- Validation-based Model Selection.
- Final Test Evaluation.

Không coi tất cả kỹ thuật là bắt buộc. Cần phân biệt rõ phần cốt lõi của dự án và các nhánh mở rộng.

Phạm vi chỉ là `labs/lab01/` với hai model trên; không phải mini-lab so sánh optimizer SGD–Adam. Core là transform ảnh, normalization có nguồn, augmentation theo cấu hình từng run và đánh giá hai model. PCA, handcrafted features, Conv Autoencoder, pretrained backbone và input variants là mở rộng chỉ thực hiện khi được xác nhận.

Tài liệu này là kế hoạch, không chứng minh các nhánh đã được triển khai hoặc chạy. Tham chiếu [EDA](EDA1.md), [quy tắc project](../.agents/AGENTS.md), [FE playbook](feature.md), [DL flow](dl_flow.md), [DL data](dl_data.md) và [DL training](dl_training.md). Không áp dụng máy móc mọi kỹ thuật trong playbook chung.

## 2. Dataset Information

Fashion-MNIST có các đặc điểm:

- Total images: 70,000.
- Original training set: 60,000 images.
- Original test set: 10,000 images.
- Image size: 28 × 28 pixels.
- Image type: grayscale.
- Number of classes: 10.
- Original pixel data type: `uint8`.
- Original pixel value range: `[0, 255]`.

Class mapping:

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

## 3. Main Principles

### 3.1. Distinguish the Concepts

Không đánh đồng các khái niệm sau:

- Feature Engineering: thiết kế, tạo hoặc biến đổi đặc trưng.
- Data Transformation: thay đổi định dạng hoặc cách biểu diễn dữ liệu.
- Data Scaling: điều chỉnh thang đo của dữ liệu.
- Data Augmentation: tạo các biến thể dữ liệu phục vụ huấn luyện.
- Feature Extraction: trích xuất đặc trưng bằng phương pháp thủ công hoặc mô hình.
- Representation Learning: mô hình tự học biểu diễn từ dữ liệu.
- Feature Selection: chọn tập đặc trưng hoặc biến thể đầu vào phù hợp.
- Model Selection: lựa chọn kiến trúc và cấu hình mô hình.
- Regularization: hạn chế Overfitting và cải thiện khả năng tổng quát hóa.

Với Deep Learning, CNN có thể tự học đặc trưng từ ảnh. Vì vậy, không bắt buộc phải áp dụng Handcrafted Feature Engineering trước khi huấn luyện CNN. Với MLP, mô hình không có khả năng chia sẻ trọng số theo vị trí, nên mọi cấu trúc không gian đều phải học lại từ đầu — đây là lý do Baseline MLP là mốc so sánh cần thiết.

### 3.2. No Unsupported Assumptions

- Không hard-code kết quả thống kê chưa được tính.
- Không mặc định PCA cần 80 đến 90 components để giữ 90% variance.
- Không giả định augmentation luôn cải thiện kết quả.
- Không khẳng định CNN hoặc pretrained backbone luôn tốt hơn MLP.
- Không tự động loại ảnh chỉ vì được đánh dấu là outlier.
- Không tạo kết quả Accuracy, F1-score hoặc biểu đồ giả.
- Mọi kết luận phải được hỗ trợ bởi kết quả thực thi.

## 4. Data Splitting and Leakage Prevention

Tái sử dụng split đã dùng cho EDA và training của Lab 01:

- Training set: 54,000 images.
- Validation set: 6,000 images.
- Test set: giữ nguyên 10,000 images.

Pipeline bắt buộc dùng stratified split theo [EDA](EDA1.md): chia indices của original train bằng `sklearn.model_selection.train_test_split` với `test_size=0.1`, `random_state=42`, `shuffle=True`, `stratify=original_train_targets`. Phần 10% là validation; giữ nguyên official test. Không dùng random split không phân tầng.

Tạo split một lần và tái sử dụng cùng indices cho EDA và cả hai run; lưu indices, dataset identity và cấu hình tái lập. Không chia lại sau EDA.

**Trạng thái triển khai:** `src/split.py` ghi `outputs/eda/split_indices.npz` và `split_metadata.json`; `src/dataset.py` chỉ đọc lại qua `load_split()`. `load_split()` nhận `expected_pool_size` và raise `FileNotFoundError` kèm hướng dẫn chạy notebook EDA khi artifact vắng mặt — nó không bao giờ tự chia lại. Split mang version `split_v2_stratified`; kết quả Phase 1 cũ mang metadata `split_v1` và được giữ riêng, không so sánh như thể chúng dùng cùng split.

Yêu cầu:

1. Giữ nguyên test set trong quá trình phát triển mô hình.
2. Chỉ sử dụng train để fit các bước học tham số từ dữ liệu, chẳng hạn scaler, PCA và model weights.
3. Chỉ sử dụng validation để lựa chọn cấu hình, kiến trúc và Hyperparameters.
4. Không áp dụng Data Augmentation ngẫu nhiên cho validation/test trong quy trình đánh giá thông thường.
5. Không dùng test để lựa chọn mô hình hoặc điều chỉnh preprocessing.
6. Kiểm tra duplicate images giữa các tập để tránh leakage.
7. Ghi lại split configuration và random seed.

Nếu có ảnh trùng giữa các tập, hãy báo cáo và đề xuất cách xử lý. Không tự động loại bỏ dữ liệu trước khi xác định quy tắc rõ ràng.

## 5. Pipeline Overview

```text
Raw integrity check -> Split một lần -> EDA trên train -> Review quyết định
    |
    +-> Core: cùng split và normalization
    |     +-> Baseline MLP: không augmentation -> ToTensor -> Normalize -> MLP
    |     +-> Tuned MLP: augmentation train -> ToTensor -> Normalize -> MLP
    |     Validation/test: ToTensor -> Normalize, không augmentation ngẫu nhiên
    |
    +-> Optional PCA: deterministic scaling -> Flatten -> PCA fit train -> classifier
    +-> Optional AE/pretrained: pipeline riêng, chỉ khi được xác nhận

Core contract + 1-batch sanity -> Train -> Validation + best checkpoint
-> Khóa cấu hình và các run so sánh -> Final test -> Walkthrough
```

Đây là các nhánh thực nghiệm. Không cần thực hiện tuần tự mọi nhánh nếu vượt quá phạm vi dự án.

Thứ tự transform là ràng buộc kỹ thuật, không phải lựa chọn phong cách. `Normalize` yêu cầu tensor `float`, nên nó phải đứng sau `ToTensor`. Augmentation hình học của torchvision nhận `PIL` image hoặc tensor, nên đặt **trước** `ToTensor` là an toàn và giữ nguyên độ chính xác số học. Đảo thứ tự sẽ hoặc gây lỗi runtime, hoặc làm giá trị đã chuẩn hóa bị biến dạng bởi phép nội suy.

## 6. Step-by-Step Pipeline

### Step 0. Guard

Mục tiêu: xác nhận dữ liệu và EDA đã sẵn sàng.

Thực hiện:

- Kiểm tra EDA outputs.
- Kiểm tra dataset availability.
- Kiểm tra train/validation/test split.
- Kiểm tra label mapping.
- Kiểm tra random seed.
- Kiểm tra duplicate leakage.
- Kiểm tra dependencies và repository structure.
- Kiểm tra train/validation indices không giao nhau, tổng số mẫu đúng và mỗi mẫu có `source_dataset`, `original_index`, `split`, `label` khi cần truy vết. Exact duplicate content phải được báo cáo riêng; không tự đổi official test set.

Đầu ra:

- Dataset validation report.
- Split summary.
- Danh sách các điều kiện chưa đạt.

Không tiếp tục huấn luyện nếu các điều kiện quan trọng chưa được xác minh.

Guard phải kiểm tra thêm hai điều kiện về tính nhất quán giữa artifact và code, vì đây là nơi lỗi âm thầm hay xảy ra nhất:

1. `preproc_version` trong artifact Feature Engineering phải khớp `split_version` của split đang mount. Nếu artifact được đo trên một split khác, mean/std đang dùng không thuộc tập đang train.
2. `FASHION_MNIST_MEAN`/`FASHION_MNIST_STD` mà `src/config.py` nạp được phải bằng giá trị ghi trong artifact, và `NORMALIZATION_SOURCE` phải bắt đầu bằng `artifact:`. Nếu không, module đã rơi về hằng số dự phòng và thống kê chưa được kiểm chứng.

Cả hai điều kiện phải là `assert`, không phải `print`. Một guard chỉ in cảnh báo sẽ bị bỏ qua khi notebook chạy batch trên Kaggle.

### Step 1. Data Integrity Review

Mục tiêu: xử lý các vấn đề dữ liệu đã được phát hiện từ EDA.

Thực hiện:

- Review exact duplicate images.
- Review empty or abnormal images.
- Review potential label issues.
- Ghi lại các quyết định giữ, loại hoặc đánh dấu mẫu.
- Không sửa nhãn tự động nếu chưa có bằng chứng.
- Đảm bảo cùng một quy tắc xử lý được áp dụng nhất quán.

Đầu ra:

- `data_review.csv`
- `data_cleaning_log.csv`

Lưu ý: việc loại mẫu phải có lý do rõ ràng. Không xóa ảnh hợp lệ chỉ vì chúng hiếm hoặc khó phân loại.

Trên Fashion-MNIST, kết quả đo được là không có mẫu nào cần loại: 0 exact duplicate trong train, 0 trong test, 0 overlap train↔test, 0 ảnh toàn đen, 0 nhãn ngoài `[0, 9]`, 0 pixel ngoài `[0, 255]`. Vì vậy bước này kết thúc bằng quyết định "giữ nguyên toàn bộ", và quyết định đó phải được ghi lại chứ không bỏ trống.

### Step 2. Basic Transformation

Mục tiêu: chuyển dữ liệu sang định dạng phù hợp với mô hình.

Thực hiện:

- Chuyển pixel từ `uint8` sang `float32`.
- Scale pixel từ `[0, 255]` về `[0, 1]`.
- Giữ batch `[B, 1, 28, 28]` cho cả hai model hiện có. BaselineMLP và TunedMLP đã có `nn.Flatten()` bên trong; chỉ flatten ngoài model cho nhánh PCA/classifier truyền thống.
- Chỉ thay đổi kích thước ảnh khi có mục tiêu thực nghiệm cụ thể.

Các biểu diễn:

Chỉ `raw_28` là đầu vào chung của core hiện tại. Các biến thể khác là mở rộng cần kế hoạch và kiểm tra model riêng; tên `raw_28` chỉ kích thước/biểu diễn, không có nghĩa model nhận trực tiếp pixel uint8.

| Input Variant | Representation |
|---|---|
| `raw_28` | `[1, 28, 28]` |
| `flat_784` | `[784]` |
| `pad_32` | `[1, 32, 32]` |
| `resize_224x3` | `[3, 224, 224]` |
| `raw_28+edge` | `[2, 28, 28]`, nếu dùng hai kênh |

Yêu cầu:

- Đảm bảo input shape phù hợp với model architecture.
- Không Flatten ảnh trước các Convolutional Layers.
- Không resize hoặc pad nếu chưa có lý do thực nghiệm.
- Ghi lại cấu hình transformation cho mỗi experiment.

Data contract core: ảnh batch `torch.float32` `[B, 1, 28, 28]`; nhãn `torch.long` `[B]` trong `[0, 9]`; model trả logits `[B, 10]` cho CrossEntropyLoss. Kiểm tra ảnh/logits finite; không thêm Softmax trước loss. Tensor đã Normalize không còn bị giới hạn trong `[0, 1]`.

Layout là một phần của contract. `ToTensor()` của torchvision nhận `HWC` và trả `CHW`; truyền vào `numpy` shape `(1, 28, 28)` sẽ bị hiểu là `HWC` và báo `Expected 1 channel(s) at dim -3, got 28`. Với ảnh grayscale một kênh, nguồn dữ liệu phải là `(28, 28)` hoặc tensor đã đúng `CHW`. Test phải khẳng định shape sau transform thay vì giả định.

Vì phạm vi Lab 01 chỉ có MLP, không có biến thể nào yêu cầu thay đổi số kênh hay feature-map size. Các biến thể `pad_32`, `resize_224x3`, `raw_28+edge` chỉ có nghĩa nếu sau này thêm kiến trúc tích chập, và khi đó phải kiểm tra lại contract một cách tường minh.

### Step 3. Scaling

Mục tiêu: chuẩn hóa thang đo dữ liệu phù hợp với mô hình.

#### Standard Normalization

Thực hiện:

1. Scale pixel về `[0, 1]`.
2. Tính mean và standard deviation theo channel từ toàn bộ pixel training set trước augmentation, dùng population variance (`ddof=0` hoặc `correction=0`), không dùng trung bình std từng ảnh.
3. Dùng cùng các giá trị này để transform validation và test.

Công thức:

$$
x_{\text{normalized}}=\frac{x-\mu_{\text{train}}}{\sigma_{\text{train}}}
$$

Không tính lại mean/std riêng cho validation và test.

Phép đo phải thỏa các điều kiện sau, nếu không kết quả không dùng được:

- Đọc đủ 54,000 ảnh train, không `drop_last`, không augmentation.
- Dùng scaled view `[0, 1]`, không dùng raw `uint8`.
- Gộp toàn bộ pixel thành một mảng phẳng rồi mới lấy std; lấy std theo trục 0 của mảng 2 chiều sẽ trả về mảng chứ không phải scalar.
- Ghi lại số ảnh, số pixel và kiểm tra `std > 0`, finite.

Đã đo trên dữ liệu thật: `mean = 0.286209`, `std(ddof=0) = 0.353160`. Giá trị `0.320661` là trung bình std của từng ảnh — một đại lượng khác về ý nghĩa, và không được dùng làm mẫu số. `src/config.py` trước đây dùng `std = 0.3205`, gần đúng bằng giá trị sai này; hệ quả là dữ liệu sau Normalize có std `≈ 1.1019` thay vì `1.0`. Cấu hình hiện tại đọc mean/std từ `preprocessing_config.json` do notebook Feature Engineering ghi ra, với version `preproc_v2_train_global_std`.

Sự khác nhau giữa hai đại lượng không phải sai số làm tròn. Trung bình của các std từng ảnh bỏ qua phương sai **giữa** các ảnh; global std bao gồm cả phương sai trong ảnh và giữa ảnh. Với dữ liệu ảnh có nền chiếm khoảng một nửa số pixel, hai giá trị này chênh nhau đáng kể và luôn theo hướng global std lớn hơn.

#### Pretrained Normalization

Nếu dùng pretrained backbone với ImageNet normalization:

- Chuyển ảnh grayscale thành ba kênh theo cách đã xác định.
- Resize ảnh theo yêu cầu của backbone.
- Sử dụng ImageNet mean/std tương ứng với mô hình.
- Ghi rõ input transformation và normalization configuration.

Không trộn ImageNet normalization với cấu hình chuẩn hóa của Fashion-MNIST mà không có lý do rõ ràng. Nhánh này nằm ngoài phạm vi Lab 01 vì không có pretrained backbone trong hai model hiện tại.

### Step 4. Data Augmentation

Mục tiêu: tăng tính đa dạng của training data và giảm nguy cơ Overfitting.

Hiện `get_transforms(augment=True)` dùng `RandomHorizontalFlip(p=0.5)` và `RandomRotation(degrees=10)` trước `ToTensor -> Normalize`. RUN1 không augmentation; RUN2 bật augmentation. Giữ mô tả này khi tái lập run hiện có. Crop/erasing bên dưới là đề xuất mở rộng, chưa được hỗ trợ bởi cờ `augment` hiện tại.

Các biến thể có thể thử:

1. `none`
2. `flip`
3. `flip + crop`
4. `flip + crop + erasing`

Các kỹ thuật:

- `RandomHorizontalFlip`
- `RandomCrop` với padding nhỏ, ví dụ 2 đến 4 pixels
- `RandomErasing` với tỷ lệ và xác suất nhẹ

Quy tắc:

- Chỉ áp dụng augmentation cho train.
- Không áp dụng augmentation ngẫu nhiên cho validation/test.
- Không mặc định dùng `RandomVerticalFlip`.
- Không sử dụng góc xoay lớn nếu có thể làm thay đổi ngữ nghĩa ảnh.
- Kiểm tra trực quan ảnh sau augmentation.
- Chỉ chọn augmentation dựa trên validation performance.

Không bắt buộc phải thử mọi tổ hợp augmentation. Bắt đầu với baseline không augmentation rồi bổ sung từng kỹ thuật.

Một điểm dễ sai: nếu train và validation dùng chung một `Dataset` object và chỉ khác cờ transform, việc bật augmentation có thể lan sang validation. Cách an toàn là tạo hai `Dataset` riêng biệt trên cùng indices, như `TransformedSubset` đang làm, và kiểm tra rằng hai lần gọi trên cùng một ảnh validation cho kết quả giống nhau.

### Step 5. PCA Feature Extraction Baseline

Nhánh tùy chọn sau core, không phải điều kiện để hoàn thành Lab 01. Dùng ảnh với preprocessing xác định, không nhận augmentation ngẫu nhiên từ nhánh neural network. Tái sử dụng kết quả PCA EDA nếu cùng split và preprocessing; nếu khác phải fit và ghi cấu hình riêng.

Mục tiêu: tạo baseline bằng phương pháp giảm chiều truyền thống.

Thực hiện:

1. Flatten mỗi ảnh thành vector 784 chiều.
2. Fit PCA trên training set.
3. Tính cumulative explained variance.
4. Xác định số components cần thiết để đạt 90% và 95% explained variance.
5. Transform validation bằng PCA đã fit.
6. Huấn luyện Logistic Regression hoặc kNN trên PCA representation.
7. Đánh giá trên validation set.

Đầu ra:

- `pca_explained_variance.png`
- `pca_components.csv`
- `pca_baseline_results.csv`

Không fit PCA trên validation/test. Không mặc định PCA là lựa chọn tốt nhất cho CNN; đây là baseline để nghiên cứu và so sánh.

### Step 6. Deep Feature Extraction

Thực hiện theo các nhánh độc lập.

Core gồm BaselineMLP và TunedMLP qua `build_model` hiện có. Cả hai học biểu diễn trên vector sau `nn.Flatten()` nội bộ. Không cần handcrafted feature extraction trước hai model. AE và pretrained bên dưới là mở rộng ngoài core.

#### Branch A. MLP from Scratch

Mục tiêu: xây dựng mô hình tự học biểu diễn trực tiếp từ vector 784 chiều.

Thực hiện:

- Sử dụng ảnh grayscale 28 × 28 đã Normalize.
- `BaselineMLP` giữ cấu trúc tối thiểu để làm mốc so sánh: 101,770 parameters.
- `TunedMLP` thêm capacity và Regularization: 235,914 parameters.
- Huấn luyện trên training set, chọn Hyperparameters bằng validation set.
- Ghi lại model architecture, số parameters và kết quả.

Đây là hai nhánh core duy nhất của Lab 01. Việc so sánh chúng cho phép tách phần đóng góp của capacity và Regularization khỏi phần đóng góp của kiến trúc, trong giới hạn rằng hai run còn khác nhau về epoch budget, augmentation và scheduler.

#### Branch B. Conv Autoencoder

Nhánh này nằm ngoài phạm vi Lab 01 vì phụ thuộc kiến trúc tích chập đã bị loại. Nếu được xác nhận mở rộng, các yêu cầu sau vẫn áp dụng:

- Chỉ dùng training images để fit Autoencoder.
- Không dùng validation/test để học encoder.
- Ghi lại latent dimension.
- Xem đây là nhánh tùy chọn, không phải điều kiện bắt buộc.

#### Branch C. Pretrained Backbone

Nhánh này nằm ngoài phạm vi Lab 01. Nếu được xác nhận mở rộng, cần lưu ý:

- Chuyển ảnh thành định dạng đầu vào phù hợp, thường là 3 kênh và 224 × 224.
- Sử dụng normalization tương ứng với pretrained weights, không dùng mean/std của Fashion-MNIST.
- Thử frozen backbone trước, chỉ fine-tuning nếu có đủ tài nguyên và lý do rõ ràng.
- So sánh số parameters được train, thời gian và validation metrics.

Không mặc định resize 224 × 224 luôn cần thiết hoặc tốt hơn ảnh gốc. Ghi rõ chi phí tính toán và giới hạn khi so sánh pretrained backbone với MLP nhỏ.

#### Branch Selection

Không cần triển khai tất cả nhánh ngay từ đầu.

Ưu tiên:

1. Baseline MLP.
2. Tuned MLP.
3. Optional PCA baseline.
4. Optional Conv Autoencoder.
5. Optional pretrained backbone.

Các nhánh bổ sung chỉ triển khai sau khi nhánh chính hoạt động ổn định.

### Step 7. Feature and Model Selection

Mục tiêu: lựa chọn cấu hình phù hợp dựa trên validation performance.

Các trục có thể thử:

| Axis | Candidate Values |
|---|---|
| Input | `raw_28`, `raw_28+edge`, `pad_32`, `resize_224x3` |
| Scaling | `/255`, `train mean/std`, `ImageNet mean/std` cho pretrained |
| Augmentation | `none`, `flip`, `flip+crop`, `flip+crop+erasing` |
| Extraction | Core: `BaselineMLP`, `TunedMLP`; mở rộng: `PCA + head`, `Conv-AE + head`, `ResNet frozen`, `ResNet fine-tune` |
| Regularization | `none`, `dropout`, `weight decay` |

Đây là danh sách các hướng thử nghiệm, không phải yêu cầu chạy toàn bộ Cartesian product.

Quy tắc:

- Bắt đầu bằng một baseline đã xác định.
- Thay đổi từng yếu tố hoặc dùng một experiment plan có kiểm soát.
- Giữ split nhất quán.
- Ghi lại random seed và Hyperparameters.
- Dùng cùng tiêu chí đánh giá.
- Ghi lại số parameters và training time.
- Chọn cấu hình bằng validation, không bằng test.

### Step 8. Regularization and Internal Selection

Mục tiêu: kiểm soát Overfitting.

Có thể thử:

- Dropout.
- Weight Decay.
- Early Stopping.
- Điều chỉnh model depth hoặc model width.

Pruning là bước tối ưu bổ sung và không bắt buộc trong giai đoạn đầu.

Đánh giá:

- Training loss.
- Validation loss.
- Validation accuracy.
- Macro F1-score.
- Train-validation gap.

Không chọn mô hình chỉ dựa vào training accuracy.

### Step 9. Representation Comparison

Tùy chọn khi đã được xác nhận các nhánh PCA/Autoencoder; không bắt buộc cho so sánh hai model core.

Mục tiêu: so sánh ảnh gốc với các biểu diễn đặc trưng có số chiều thấp hơn.

Các candidate:

- Raw image representation.
- PCA representation, ví dụ PCA-50 nếu phù hợp với kết quả thực tế.
- Autoencoder latent representation, ví dụ latent dimension 64 nếu được xác định trong experiment plan.

Thực hiện:

- Sử dụng cùng training/validation split.
- Dùng cùng metric.
- Giữ quy trình huấn luyện phù hợp với từng loại representation.
- Ghi rõ dimensionality và model architecture.
- Không mặc định PCA-50 hoặc latent-64 là lựa chọn tốt nhất.

Đầu ra:

- `representation_comparison.csv`
- `representation_comparison.png`

### Step 10. Distribution Shift Diagnostics

Tùy chọn, tái sử dụng EDA thay vì tính lặp. Phát triển chỉ phân tích train/validation; phân tích chi tiết test sau khi khóa cấu hình.

Mục tiêu: kiểm tra sự khác biệt giữa các tập dữ liệu.

Có thể phân tích:

- Pixel mean/std.
- Mean image.
- Brightness distribution.
- Class distribution.
- KS test cho brightness nếu phù hợp.

Báo cáo kết quả như một bước chẩn đoán.

Không tự động kết luận có distribution drift chỉ từ một khác biệt thống kê. Không sử dụng kết quả trên test để thay đổi pipeline.

### Step 11. Final Selection and Test Evaluation

Mục tiêu: đánh giá khách quan cấu hình đã lựa chọn.

Thực hiện:

1. Chọn cấu hình cuối dựa trên validation.
2. Ghi lại model architecture và Hyperparameters.
3. Xác định trước cách huấn luyện mô hình cuối.
4. Với quy trình core, nạp best-validation checkpoint của từng run đã khóa để đánh giá. Huấn luyện lại trên train + validation là thay đổi quy trình cần xác nhận riêng; phải định trước số epoch và cách chọn checkpoint khi không còn validation, fit lại preprocessing trên tập được phép sử dụng và ghi là run mới.
5. Giữ test hoàn toàn ngoài quá trình lựa chọn.
6. Đánh giá test sau khi đã khóa cấu hình.

Khóa danh sách hai run và quy tắc so sánh trước khi mở test. Có thể báo test metrics của cả hai checkpoint đã khóa; lựa chọn cấu hình dựa trên validation. Không dùng test để mở vòng tuning mới hoặc chọn model rồi gọi đó là đánh giá độc lập. Nếu test đã được dùng để điều chỉnh trong lịch sử, công khai giới hạn đó.

Báo cáo:

- Test accuracy.
- Test loss.
- Test macro F1-score.
- Per-class precision, recall và F1-score.
- Confusion matrix.
- Training time.
- Number of parameters.

Test evaluation chỉ được thực hiện sau khi đã chốt cấu hình. Không điều chỉnh mô hình dựa trên test metrics rồi tiếp tục gọi đó là đánh giá test độc lập.

### Step 12. Logging and Reporting

Mục tiêu: đảm bảo mọi experiment có thể được kiểm tra và tái tạo.

Lưu:

- Experiment ID.
- Input variant.
- Scaling configuration.
- Augmentation configuration.
- Feature extraction method.
- Model architecture.
- Random seed.
- Hyperparameters.
- Number of parameters.
- Training time.
- Best epoch.
- Validation metrics.
- Final test metrics, chỉ sau bước đánh giá cuối.
- Ghi chú về các quyết định.

Đầu ra:

- `experiment_results.csv`
- Phần preprocessing, experiment changelog và kết quả trong `labs/lab01/walkthrough_lab_01.md`; không tạo report Markdown riêng.
- Các biểu đồ PNG.
- Các cấu hình thực nghiệm.

Không hard-code kết quả. Báo cáo phải được tạo từ các kết quả thực tế.

Với mỗi kết quả đã công bố, phải ghi kèm `split_version` và `preproc_version`. Hai con số accuracy đo trên hai split khác nhau không so sánh được với nhau, kể cả khi kiến trúc giống hệt.

## 7. Evaluation Metrics

Sử dụng các metrics sau:

### Primary Metrics

- Validation Accuracy.
- Validation Macro F1-score.

Core hiện chọn best checkpoint bằng validation accuracy (`val_acc > best_val_acc` trong `src/engine.py`); giữ metric này để khớp code. Macro F1 là metric bổ sung. Nếu muốn đổi tiêu chí chọn, cần xác nhận và đồng bộ engine trước khi chạy thí nghiệm mới.

### Diagnostic Metrics

- Per-class Recall.
- Per-class Precision.
- Per-class F1-score.
- Confusion Matrix.
- Training Loss.
- Validation Loss.
- Train-validation gap.

Đặc biệt theo dõi:

- T-shirt/top.
- Pullover.
- Coat.
- Shirt.

Đây là các class đáng chú ý trong phân tích nhầm lẫn, nhưng không giả định trước rằng chúng luôn là những class khó nhất. EDA đã đo cosine similarity giữa các mean image: `Coat/Shirt = 0.9873`, `Pullover/Coat = 0.9841`, `Pullover/Shirt = 0.9830`, `T-shirt/top/Shirt = 0.9497`. Bốn lớp này tạo thành một cụm tương đồng cao, khớp với cụm nhầm lẫn quan sát trong confusion matrix của Phase 1. Đây vẫn chỉ là giả thuyết cho đến khi được xác nhận bằng per-class metrics trên split mới.

## 8. Experiment Design

Không chạy toàn bộ các tổ hợp ngay từ đầu.

Giai đoạn 1: Core experiments.

- Baseline MLP.
- Tuned MLP.

Giai đoạn 2: Traditional feature baseline (tùy chọn, sau khi được xác nhận).

- PCA.
- Logistic Regression hoặc kNN.

Giai đoạn 3: Controlled improvements (thí nghiệm mới khi cần).

- Normalization.
- Data Augmentation.
- Dropout.
- Weight Decay.

Giai đoạn 4: Optional representation experiments.

- Edge channel.
- Conv Autoencoder.
- Pretrained backbone.
- Representation dimensionality comparison.

Sau mỗi giai đoạn, tổng hợp kết quả và xác định bước tiếp theo dựa trên validation performance, training cost và mục tiêu nghiên cứu.

### Current Runs and Comparison Limits

| Run | Epoch budget | Augmentation | Scheduler | Weight decay |
|---|---:|---|---|---:|
| BaselineMLP | 5 | Không | Không | 0.0 |
| TunedMLP | 10 | Flip + Rotation | Bật | 0.0001 |

Đây là giá trị trong `src/config.py`, không phải kết quả thực thi. Hai run thay đổi nhiều yếu tố nên chỉ cho phép kết luận về các cấu hình hoàn chỉnh; không quy toàn bộ chênh lệch cho riêng kiến trúc hoặc feature engineering. Nếu cần xác minh tác động một yếu tố, đề xuất ablation riêng với split, preprocessing và ngân sách so sánh phù hợp; không tự sửa hai run cũ.

Kết quả Phase 1 đã công bố (85.14% và 87.81%, Macro F1 84.93 và 87.71) được đo dưới `split_v1` và `preproc_v1`. Chúng không so sánh trực tiếp được với run trên `split_v2_stratified` + `preproc_v2_train_global_std`, vì cả thành phần train/validation và thang đo đầu vào đều đã thay đổi. Muốn so sánh, phải chạy lại toàn bộ hai run trên split mới.

### Validation Gates Before Full Training

- Kiểm tra split indices, coverage và duplicate report; dừng khi có lỗi contract chưa giải quyết.
- Kiểm tra train-only fitting cho normalization/PCA; validation/test dùng cùng tham số đã fit.
- Kiểm tra shape/dtype, label range, giá trị finite và ảnh sau transform của cả hai model.
- Với cùng ảnh validation/test, transform phải xác định; kiểm tra train augmentation không lan sang validation qua shared dataset.
- Chạy forward/loss smoke test, 1-batch sanity overfit và kiểm tra save/load checkpoint trước full training. Khi lỗi, báo expected/actual, input/config liên quan và lỗi cụ thể; không tạo kết quả thay thế.
- Kiểm tra inference dùng `model.eval()` và tắt gradient; tính metrics trên đủ validation/test.
- Seed Python, NumPy, PyTorch và worker khi dùng; ghi phiên bản môi trường và cấu hình. Local chỉ smoke/test/sanity; full training trên Kaggle GPU hoặc compute tương đương.

Đây là checklist cần thực thi khi triển khai code, không phải tuyên bố tests hiện có đã cover hoặc pass tất cả mục.

## Feature Engineering Pipeline Mapping

| Pipeline Step | Nhóm chính | Ghi chú |
|---|---|---|
| Step 0: Guard | Data Validation | Kiểm tra dữ liệu, data split và data leakage |
| Step 1: Data Integrity Review | Data Cleaning | Kiểm tra và xử lý các vấn đề dữ liệu |
| Step 2: Basic Transformation | Feature Transformation | Chuyển kiểu dữ liệu, reshape, padding và resize |
| Step 3: Scaling | Feature Scaling | `/255`, Standard Normalization và ImageNet Normalization |
| Step 4: Data Augmentation | Data Augmentation | Tạo các biến thể của training images |
| Step 5: PCA Baseline | Feature Extraction + Dimensionality Reduction | PCA tạo representation có số chiều thấp hơn |
| Step 6A: MLP from Scratch | Feature Extraction + Representation Learning | MLP tự học biểu diễn trên vector 784 chiều |
| Step 6B: Conv Autoencoder | Feature Extraction + Representation Learning | Ngoài phạm vi: encoder tạo latent features |
| Step 6C: Pretrained Backbone | Feature Extraction | Ngoài phạm vi: đặc trưng từ backbone đã pretrained |
| Step 7: Feature and Model Selection | Feature Selection + Model Selection | Chọn input, representation và cấu hình mô hình |
| Step 8: Regularization | Regularization | Dropout, Weight Decay và Early Stopping |
| Step 9: Representation Comparison | Feature/Representation Evaluation | So sánh chất lượng các representation |
| Step 10: Distribution Shift | Data Diagnostics | Kiểm tra sự khác biệt giữa các tập dữ liệu |
| Step 11: Final Evaluation | Model Evaluation | Đánh giá trên test set sau khi khóa cấu hình |
| Step 12: Logging | Experiment Tracking | Lưu cấu hình thực nghiệm và kết quả |

## 9. Code Quality Requirements

- Kiểm tra repository trước khi tạo file.
- Tái sử dụng code hiện có nếu phù hợp.
- Không tạo cấu trúc thư mục trùng lặp không cần thiết.
- Tách data transformation, feature extraction, model training và evaluation.
- Sử dụng random seed.
- Lưu configuration cho mỗi experiment.
- Không tải hoặc cài đặt thêm dependency khi chưa kiểm tra môi trường.
- Không triển khai các nhánh tùy chọn khi nhánh chính chưa ổn định.
- Kiểm tra input shape trước khi chạy model.
- Đảm bảo preprocessing nhất quán giữa train, validation và test.
- Báo lỗi rõ ràng khi thiếu dữ liệu hoặc dependency.

Không hard-code thống kê trong hàm vẽ. `plot_prediction_grid` từng hard-code `mean=0.2860, std=0.3205` để `unnormalize` ảnh trước khi hiển thị; khi thống kê thay đổi, ảnh hiển thị lệch màu nhưng không có lỗi nào báo. `unnormalize()` hiện đọc từ `src/config.py`, và đây là pattern bắt buộc cho mọi chỗ cần đảo một phép biến đổi đã áp dụng lúc training.

Với API thư viện, kiểm chứng theo phiên bản đang cài thay vì theo tài liệu cũ. Các trường hợp đã gặp trong lab này:

| API | Cách gọi cũ | Cách gọi đúng ở phiên bản hiện tại |
|---|---|---|
| `torch.Tensor.std` | `std(ddof=0)` | `std(correction=0)` |
| `Axes.boxplot` | `labels=` | `tick_labels=` |
| `numpy.ndarray.std` | `std(axis=0, ddof=1)` trên mảng 2 chiều trả về mảng | `.reshape(-1).std(ddof=1)` để có scalar |

## 10. Suggested Folder Structure

```text
labs/lab01/
|-- data/
|-- src/
|   |-- config.py             # paths, normalization, run configs
|   |-- split.py              # stratified split + artifact
|   |-- dataset.py            # transforms, DataLoaders, đọc split artifact
|   |-- eda.py                # thống kê và biểu đồ EDA
|   |-- model.py              # BaselineMLP và TunedMLP
|   |-- engine.py             # training, evaluation, best checkpoint
|   `-- utils.py              # seed, checkpoint và visualization
|-- notebooks/
|   |-- lab01_eda.ipynb
|   |-- lab01_feature_engineering.ipynb
|   `-- lab01_fashionmnist_walkthrough.ipynb
|-- tests/
|   `-- test_pipeline.py
|-- scripts/
|   |-- build_eda_notebook.py
|   |-- build_fe_notebook.py
|   `-- patch_training_notebook.py
|-- outputs/
|   |-- figures/
|   |-- tables/
|   `-- eda/                  # split artifact, thống kê, bảng và hình EDA
`-- walkthrough_lab_01.md     # spec, changelog, report
```

Đây là ánh xạ vào cấu trúc Lab 01; các artifact mới chỉ tạo khi triển khai được xác nhận. Tái sử dụng source/notebook hiện có, không tạo framework `src/data`, `src/features`, `src/models`, `src/training` song song. Bảng/ảnh dùng `OUTPUTS_DIR` (local: `labs/lab01/outputs/`, Kaggle theo config); nhận xét nằm trong walkthrough. File phụ trợ AI không trộn vào source/artifact sản phẩm; nếu cần giữ phải theo vị trí `.agents/<task>/` đã xác nhận.

## 11. Expected Deliverables

Sau khi hoàn thành, cần có:

1. Feature Engineering pipeline có thể chạy lại.
2. Cấu hình preprocessing và augmentation rõ ràng.
3. Split metadata/indices và nguồn thống kê normalization để tái lập.
4. Kết quả Baseline MLP và Tuned MLP.
5. Bảng so sánh validation metrics.
6. Confusion matrix và per-class metrics.
7. Experiment log.
8. Phần kết quả và quyết định trong `labs/lab01/walkthrough_lab_01.md`.
9. Danh sách giới hạn và các hướng cải thiện.

PCA baseline results, handcrafted features, Conv Autoencoder, pretrained backbone, input variants và pruning chỉ cần thực hiện nếu được xác nhận mở rộng phạm vi. Không coi các đầu ra tùy chọn là điều kiện nghiệm thu core.

## 12. AI Collaboration Workflow

Trước khi viết hoặc chỉnh sửa code:

1. Kiểm tra repository, dataset và môi trường hiện tại.
2. Đọc EDA outputs nếu đã tồn tại.
3. Xác định các thành phần có thể tái sử dụng.
4. Đề xuất kế hoạch triển khai theo từng giai đoạn.
5. Nêu rõ mục tiêu, input, output, chi phí và tiêu chí kiểm tra của từng bước.
6. Phân biệt core requirements và optional experiments.
7. Chờ xác nhận kế hoạch trước khi thực hiện các thay đổi lớn.

Sau khi kế hoạch được xác nhận:

1. Triển khai từng nhóm chức năng.
2. Chạy kiểm tra để xác nhận tính đúng đắn.
3. Kiểm tra data leakage.
4. Kiểm tra input shape và preprocessing.
5. Chạy baseline trước khi mở rộng.
6. Tổng hợp kết quả thực tế.
7. Báo cáo phần đã hoàn thành, phần chưa hoàn thành và các vấn đề phát sinh.

Không tự ý mở rộng phạm vi, thay đổi kiến trúc dự án hoặc chạy toàn bộ experiment grid khi chưa có kế hoạch và sự đồng ý phù hợp.

Không dùng shell round-trip để sửa file text có ký tự ngoài ASCII. `Get-Content | Set-Content` của Windows PowerShell đọc file UTF-8 không BOM bằng codepage ANSI và thay mọi byte không định nghĩa bằng `U+FFFD`; tổn thất này không phục hồi được. Dùng công cụ so khớp literal, hoặc đọc/ghi tường minh bằng `UTF8Encoding($false)`. Trước khi ghi đè file chưa được Git theo dõi, commit hoặc sao lưu trước.
