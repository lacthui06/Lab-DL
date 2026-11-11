# Lab02 · Module datasets

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-07 | 0.1.0 | Thường FEAI | Tạo hướng dẫn đọc pipeline dữ liệu |
| 2026-11-11 | 0.1.1 | Thường FEAI | Sắp xếp theo hợp đồng dữ liệu, cơ chế, output và thực hành; bổ sung lý do và giới hạn |
| 2026-11-11 | 0.1.2 | Thường FEAI | Bổ sung shape/dtype, split theo lớp, công thức normalization và đặc trưng EDA sau đối chiếu Lab01 |
| 2026-11-11 | 0.1.3 | Thường FEAI | Viết lại từ mục 2 theo cách dẫn giải Lab01; đối chiếu source, cấu hình và artifact; giữ nguyên mục 1 |
| 2026-11-11 | 0.1.4 | Thường FEAI, cùng Codex | Đánh số các thảo luận kỹ thuật để thống nhất cách điều hướng với Lab01 |

## 1. bản đồ module

Module giữ cho EDA, training và đánh giá sử dụng đúng ảnh, nhãn và split. Nó tạo bằng chứng để đọc lại cách dữ liệu được chuẩn bị, đồng thời cung cấp batch cho model.

| File | Trách nhiệm | Đoạn code và vị trí |
| --- | --- | --- |
| [eda.py](eda.py) | Đọc các batch CIFAR-10 gốc và kiểm tra checksum. Quản lý split đã lưu, tính thống kê, kiểm tra ảnh trùng và xuất bằng chứng EDA. | `read_raw(…)` · [eda.py, dòng 21–32](eda.py#L21) · `indices = locked_split(output, labels, cfg['seed'])` · [eda.py, dòng 177–196](eda.py#L177) |
| [data.py](data.py) | Chọn chỉ số cho run, tạo transform, đọc ảnh theo chỉ số gốc và gom thành batch. | `def prepare_indices(…)` · [data.py, dòng 36–48](data.py#L36) · `return DataLoader(IndexedImages(…), …)` · [data.py, dòng 73–79](data.py#L73) |
| [__init__.py](__init__.py) | Khai báo hai module con. Dùng `eda` khi cần phân tích và `data` khi cần pipeline PyTorch. | `__all__ = ["data", "eda"]` · [__init__.py, dòng 7](__init__.py#L7) |

### Bản đồ hàm EDA

| Hàm | Trách nhiệm và dữ liệu trả về |
| --- | --- |
| [`rgb_statistics()`](eda.py#L66) | Đo moment RGB, histogram 256 mức và thống kê không gian trên các ảnh train đã chọn; trả dictionary, không đổi normalization của model. |
| [`handcrafted()`](eda.py#L114) | Trả DataFrame gồm 7 đặc trưng mô tả từng ảnh train, kèm index, nhãn, tên lớp và split. |
| [`duplicate_audit()`](eda.py#L135) | Kiểm tra nhãn ngoài miền, pixel không hữu hạn, ảnh hằng; so hash byte ảnh giữa các split và trả hai bảng kiểm tra/nghi trùng. |
| [`export_summary()`](eda.py#L94) | Ghi phân bố lớp, thống kê RGB train và ảnh ví dụ cho subset của một run; không nạp model. |
| [`build()`](eda.py#L177) | Điều phối EDA đầy đủ, gọi các phép đo và ghi bảng/hình vào `outputs/eda/`. |

## 2. Mục Đích (Purpose)

Module khảo sát dữ liệu CIFAR-10 trước khi huấn luyện để xác định kích thước ảnh, phân bố lớp, đặc điểm màu sắc và các dấu hiệu bất thường. EDA là phân tích khám phá dữ liệu; kết quả của bước này giúp kiểm tra đầu vào và đọc lại cách dữ liệu được chuẩn bị.

CIFAR-10 gồm 60.000 ảnh RGB 32×32 thuộc 10 lớp. Module chia 50.000 ảnh train chính thức thành train/validation, giữ riêng 10.000 ảnh test và lưu chỉ số để sử dụng lại. Với pipeline PyTorch, module biến đổi ảnh rồi trả từng batch gồm tensor ảnh, nhãn và index gốc để truy vết dự đoán.

## 3. Khởi Tạo Dữ Liệu & Luồng Tiền Xử Lý (Init Dataset & Pipeline)

### 3.1. Cấu Trúc Luồng Tiền Xử Lý (Data Pipeline Structure)

Quy trình có hai nhánh dùng chung cơ chế chia tập: EDA đo trên ảnh gốc, còn training biến đổi ảnh khi lấy mẫu. Các bước dưới đây mô tả source hiện tại; số lượng split được đối chiếu từ [class_distribution.csv](../../outputs/eda/class_distribution.csv).

```text
1. Đọc CIFAR-10 gốc: 50.000 ảnh train và 10.000 ảnh test RGB 32×32.
   EDA kiểm tra MD5 từng batch trước khi giải mã bằng read_raw().
2. Tạo hoặc kiểm tra locked_split(), với seed 42 trong config EDA.
   Train: 45.000 ảnh │ Validation: 5.000 ảnh │ Test chính thức: 10.000 ảnh.
3. Nhánh EDA: kiểm tra nhãn/ảnh trùng, đo thống kê RGB và 7 đặc trưng.
   Ghi bảng, hình và hồ sơ tiền xử lý vào outputs/eda/.
4. Nhánh training: prepare_indices() chọn toàn bộ split hoặc subset có phân tầng.
   Lưu split_indices.npz và selected_indices.npz trong thư mục run.
5. IndexedImages đọc ảnh theo index gốc, áp dụng transform.
   image_loader() gom tensor ảnh, nhãn và index thành batch trong bộ nhớ.
```

Chia phân tầng nghĩa là giữ tỷ lệ lớp khi tách dữ liệu. Split EDA có 4.500 ảnh/lớp ở train, 500 ảnh/lớp ở validation và 1.000 ảnh/lớp ở test. Mười lớp theo thứ tự nhãn `0..9` là airplane, automobile, bird, cat, deer, dog, frog, horse, ship và truck.

[`locked_split()`](eda.py#L35) kiểm tra seed, hash nhãn và hash file chỉ số khi sử dụng lại split. Hàm cũng kiểm tra train/validation không giao nhau, không lặp index và hợp lại phủ pool train. Nếu kiểm tra thất bại, hàm báo lỗi thay vì âm thầm chia lại.

### 3.2. Đầu Vào và Đầu Ra Dữ Liệu (Input / Output Data)

NHWC biểu diễn thứ tự số ảnh, chiều cao, chiều rộng và kênh màu; CHW đặt kênh màu trước chiều cao và chiều rộng. `B` là số mẫu thực tế trong một batch; batch cuối có thể nhỏ hơn batch size cấu hình.

| Tên thành phần | Kích thước (Shape) | Kiểu dữ liệu (Dtype) | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| Ảnh gốc từ `read_raw()` | Train `(50000,32,32,3)`; test `(10000,32,32,3)` | NumPy `uint8` | Ảnh NHWC, giá trị kênh màu trong `[0,255]`. |
| Nhãn gốc | Train `(50000,)`; test `(10000,)` | NumPy `int64` | Mã lớp `0..9`, cùng thứ tự với ảnh nguồn. |
| Chỉ số split/subset | `(N_split,)` | NumPy integer; subset được chuyển sang `int64` | Vị trí ảnh trong pool train hoặc test tương ứng. |
| Một ảnh sau transform | `(3,H,W)` | Torch floating, mặc định `float32` | Ảnh CHW đã resize và normalize; giá trị có thể âm hoặc lớn hơn 1. |
| Batch ảnh | `(B,3,H,W)` | Cùng dtype ảnh sau transform | Với cấu hình scheduled hiện tại, `H=W=64`. |
| Batch nhãn / index | `(B,)` / `(B,)` | Torch `int64` | Nhãn phân loại và index gốc, đi cùng ảnh sau shuffle. |
| Bảng đặc trưng EDA | Một hàng mỗi ảnh train; 7 cột đặc trưng và 4 cột metadata | Pandas DataFrame | Mô tả ảnh gốc; không nối vào tensor đầu vào của model. |

Ảnh ở vị trí `i` trong subset được đọc bằng `original = self.indices[i]`, sau đó lấy `self.raw[original]`. Ví dụ dò tay: với `indices=[8,42]`, phần tử `i=1` lấy ảnh gốc số `42`. Index train `42` và index test `42` thuộc hai pool khác nhau, nên cần lưu cả tên tập khi truy vết. [Source](data.py#L51).

### 3.3. Biến Đổi Ảnh và Chuẩn Hóa (Transforms & Normalization)

Ở kích thước 64×64, [`image_transform()`](data.py#L15) thực hiện `Resize → ToTensor → Normalize`. `ToTensor()` đổi ảnh byte sang CHW và chia giá trị cho 255. Với từng kênh `c`, normalization tính `z_c = (pixel_c / 255 − mean_c) / std_c`.

Mean RGB là `[0.485,0.456,0.406]`, std RGB là `[0.229,0.224,0.225]`, theo [quy ước ImageNet được ghi trong config EDA](../../config/eda.json). Source lấy các giá trị từ transform của pretrained weights trong [`SPECS`](../models/models.py#L6), không lấy từ thống kê CIFAR-10 vừa đo. Khi size bằng 224, hàm trả nguyên transform của weights, gồm quy trình resize/crop của preset.

Ví dụ tính tay: giá trị kênh đỏ bằng `255` sau resize trở thành `(1−0.485)/0.229 ≈ 2.249`. Kết quả này giải thích vì sao tensor đã normalize không còn bị giới hạn trong `[0,1]`. Nhãn và index không chịu phép chuẩn hóa.

Khi bật augmentation cho train, [`IndexedImages`](data.py#L51) thêm `RandomCrop(32,padding=4,padding_mode="reflect")` và `RandomHorizontalFlip(p=0.5)` trước transform chung. Augmentation tạo biến thể ảnh lúc lấy mẫu, không tăng số index gốc. Yêu cầu augmentation cho validation/test gây `ValueError`.

[Config scheduled hiện tại](../../config/scheduled.json) đặt size 64, batch size 128, ba limit bằng `0` và `train_augmentation=true`. Limit `0` nghĩa là giữ toàn bộ split; khi giới hạn nhỏ hơn kích thước tập, [`balanced_subset()`](data.py#L27) lấy mẫu có phân tầng trong đúng split.

[Config lưu của run được chọn](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/config.json) cũng ghi các giá trị trên. Theo [nhánh training hiện tại](../training/experiment.py#L354), augmentation áp dụng cho fine-tune train; head-only dùng cache đặc trưng với transform xác định. Trường không augmentation trong [config EDA](../../config/eda.json) không mô tả toàn bộ cấu hình training của run này.

### 3.4. Thống Kê RGB và 7 Đặc Trưng Khảo Sát

Mục tiêu khảo sát là mô tả mức sáng, độ tương phản, màu sắc và biến thiên cục bộ của ảnh train trước resize/augmentation. [`rgb_statistics()`](eda.py#L66) cộng dồn giá trị và bình phương giá trị sau chia 255 để tính mean và độ lệch chuẩn tổng thể từng kênh; histogram dùng 256 mức byte gốc.

[Artifact tiền xử lý](../../outputs/eda/preprocessing_config.json) ghi phép đo trên 45.000 ảnh, tương ứng 46.080.000 giá trị mỗi kênh. Mean RGB là `[0.491137,0.482111,0.446449]`; std RGB là `[0.246869,0.243430,0.261603]`, làm tròn 6 chữ số thập phân. Đây là thống kê mô tả train, không thay thế normalization ImageNet.

Để đọc công thức của [`handcrafted()`](eda.py#L114), gọi `X = ảnh_RGB / 255` và `G = mean_RGB(X)`. Mỗi ảnh có 1.024 pixel; `G` lấy trung bình đều ba kênh, không dùng trọng số cảm nhận màu.

| Đặc trưng | Công thức hoặc phép tính | Ý nghĩa |
| :--- | :--- | :--- |
| `brightness` | `mean(G)` | Độ sáng trung bình toàn ảnh. |
| `contrast` | `sqrt(mean((G−mean(G))²))` | Mức phân tán độ sáng quanh giá trị trung bình. |
| `zero_fraction` | `count(X==0)/(32×32×3)` | Tỷ lệ giá trị kênh bằng 0, không phải diện tích nền. |
| `edge_density` | `mean(sqrt(dx²+dy²)>0.1)` | Tỷ lệ vị trí có biến thiên cục bộ vượt ngưỡng. |
| `red_mean` | `mean(X[:,:,0])` | Trung bình kênh đỏ. |
| `green_mean` | `mean(X[:,:,1])` | Trung bình kênh xanh lá. |
| `blue_mean` | `mean(X[:,:,2])` | Trung bình kênh xanh dương. |

Ở công thức edge density, `dx[r,c]=G[r,c+1]−G[r,c]` và `dy[r,c]=G[r+1,c]−G[r,c]`, với `r,c=0..30`. Phép đo dùng 961 vị trí trên lưới 31×31; đây không phải mặt nạ đường biên vật thể đã gán nhãn. Các cột `original_index`, `label`, `class`, `split` là metadata để đối chiếu ảnh nguồn.

### 3.5. Tệp Mã Nguồn & Bản Đồ Điều Hướng Code (Source Files)

Giữ **Ctrl + Click** để mở ký hiệu hoặc đoạn code cần đọc.

| Tệp mã nguồn | Ký hiệu / Hàm chính (Ctrl+Click) | Dòng code | Việc đảm nhiệm |
| :--- | :--- | :--- | :--- |
| [eda.py](eda.py) | [`read_raw()`](eda.py#L21), [`locked_split()`](eda.py#L35) | [21–63](eda.py#L21-L63) | Kiểm tra checksum, đọc ảnh/nhãn và tạo hoặc xác thực split đã lưu. |
| [eda.py](eda.py) | [`rgb_statistics()`](eda.py#L66), [`class_counts()`](eda.py#L88), [`export_summary()`](eda.py#L94) | [66–111](eda.py#L66-L111) | Đo thống kê train và xuất bằng chứng dữ liệu riêng cho từng run. |
| [eda.py](eda.py) | [`handcrafted()`](eda.py#L114), [`duplicate_audit()`](eda.py#L135) | [114–156](eda.py#L114-L156) | Tính 7 đặc trưng; kiểm tra nhãn, pixel, ảnh hằng và ảnh trùng byte. |
| [eda.py](eda.py) | [`build()`](eda.py#L177), [`preprocessing_preview()`](eda.py#L268) | [177–279](eda.py#L177-L279) | Điều phối EDA, xuất bảng/hình và hồ sơ tiền xử lý. |
| [data.py](data.py) | [`image_transform()`](data.py#L15), [`balanced_subset()`](data.py#L27), [`prepare_indices()`](data.py#L36) | [15–48](data.py#L15-L48) | Tạo transform, chọn subset và lưu chỉ số thực sự sử dụng. |
| [data.py](data.py) | [`IndexedImages`](data.py#L51), [`image_loader()`](data.py#L73) | [51–79](data.py#L51-L79) | Đọc ảnh theo index gốc, biến đổi ảnh và gom batch. |

Các artifact chính gồm [split và metadata](../../outputs/eda/split_metadata.json), [phân bố lớp](../../outputs/eda/class_distribution.csv), [thống kê pixel](../../outputs/eda/pixel_statistics.csv), [7 đặc trưng](../../outputs/eda/handcrafted_features.csv), [kiểm tra tính hợp lệ](../../outputs/eda/integrity_report.csv) và [ảnh trùng](../../outputs/eda/duplicate_images.csv). Hình tương ứng được ghi trong `outputs/eda/`; tensor và DataLoader được trả trong bộ nhớ.

Mỗi run có bằng chứng riêng trong `run/eda/`, gồm `class_counts.csv`, `summary.json` và `training_examples.png`. Khi đọc kết quả subset, cần dùng bằng chứng của chính run đó thay vì áp số lượng của EDA đầy đủ. [Source xuất bằng chứng](eda.py#L94).

## 4. Thảo Luận Kỹ Thuật & Quyết Định Thiết Kế

### 4.1. Vì sao khóa split và giữ index gốc?

Dùng lại cùng chỉ số giúp so sánh các lần chạy trên cùng tập ảnh; index gốc giúp truy ngược dự đoán sau shuffle. Tuy nhiên, [metadata EDA hiện có](../../outputs/eda/split_metadata.json) ghi split được dựng lại theo thuật toán đã mô tả vì thiếu index run gốc. Cùng seed và số lượng chưa đủ để khẳng định split EDA trùng từng ảnh với run lịch sử.

### 4.2. Vì sao lấy subset có phân tầng?

Cách lấy này giữ gần đúng tỷ lệ lớp khi giảm số mẫu, thay vì lấy các ảnh đầu tiên. Nó không bảo đảm đại diện mọi kiểu ảnh khó/dễ; limit quá nhỏ có thể khiến phép chia báo lỗi. Kết quả subset cần ghi rõ số mẫu thực dùng. [Source](data.py#L27).

### 4.3. Vì sao giữ RGB và normalization của weights?

Pipeline duy trì ba kênh màu và quy ước đầu vào của backbone pretrained. Size 64 làm giảm số vị trí không gian so với 224, nhưng không bổ sung chi tiết mới cho ảnh gốc 32×32. Source không chứng minh size hoặc normalization này tối ưu; muốn kết luận cần thí nghiệm so sánh riêng.

### 4.4. Vì sao augmentation chỉ áp dụng cho train?

Crop/flip tạo biến thể khi học; validation/test giữ transform xác định để đánh giá nhất quán. Cần đọc cả cấu hình run và đường thực thi cache trước khi mô tả augmentation, vì head-only và fine-tune không nhận dữ liệu qua cùng đường xử lý. [Source training](../training/experiment.py#L354).

### 4.5. Vì sao 7 đặc trưng chỉ dùng khảo sát?

Module lưu chúng để đọc phân phối và tương quan, còn model nhận tensor ảnh RGB. Pixel bằng 0 không xác định nền của ảnh tự nhiên, nên module không suy ra bounding box như ảnh có mặt nạ vật thể. Tương quan đặc trưng cũng không chứng minh khả năng cải thiện phân loại. [Hồ sơ tiền xử lý](../../outputs/eda/preprocessing_config.json).

### 4.6. Kiểm tra dữ liệu bảo đảm đến đâu?

MD5 kiểm tra file batch gốc; hash ảnh phát hiện trùng byte chính xác. Audit chưa phát hiện mọi ảnh gần trùng hoặc mọi dạng rò rỉ dữ liệu, và không tự xóa ảnh nghi trùng. Khi kiểm tra pipeline, cần đối chiếu metadata, phân bố lớp và shape/nhãn/index của batch; các phép kiểm tra này không thay thế đánh giá mô hình.
