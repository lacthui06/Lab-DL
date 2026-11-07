# Lab02 · Module datasets

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-07 | 0.1.0 | Thường FEAI, cùng Codex | Tạo hướng dẫn đọc pipeline dữ liệu |

## 1. Pipeline dữ liệu

```text
CIFAR-10 gốc → locked_split() → train / validation / test
  ├─ EDA → kiểm tra cấu trúc, nhãn và ảnh → xuất bảng và hình
  └─ Training → chọn subset nếu có giới hạn → đọc ảnh theo index gốc
              → augmentation nếu bật cho train → tiền xử lý → DataLoader
              → batch ảnh đã biến đổi, nhãn và index gốc
```

Cả hai luồng dùng chung cách tạo và kiểm tra split. EDA chạy qua `scripts/analyze_data.py`. Training gọi `prepare_indices()` để lấy các chỉ số dùng cho run. Train và validation lấy từ 50.000 ảnh của tập train chính thức. Tập test dùng 10.000 ảnh chính thức riêng. [`print(build(ROOT, args.data_dir))` · scripts/analyze_data.py, dòng 14](../../scripts/analyze_data.py#L14) · [`selected = prepare_indices(…)` · experiment.py, dòng 546](../training/experiment.py#L546)

## 2. Mục đích và các file

Module giữ cho EDA, training và đánh giá sử dụng đúng ảnh, nhãn và split. Nó tạo bằng chứng để đọc lại cách dữ liệu được chuẩn bị, đồng thời cung cấp batch cho model.

| File | Trách nhiệm | Đoạn code và vị trí |
| --- | --- | --- |
| [eda.py](eda.py) | Đọc các batch CIFAR-10 gốc và kiểm tra checksum. Quản lý split đã lưu, tính thống kê, kiểm tra ảnh trùng và xuất bằng chứng EDA. | `read_raw(…)` · [eda.py, dòng 21–32](eda.py#L21) · `indices = locked_split(output, labels, cfg['seed'])` · [eda.py, dòng 177–196](eda.py#L177) |
| [data.py](data.py) | Chọn chỉ số cho run, tạo transform, đọc ảnh theo chỉ số gốc và gom thành batch. | `def prepare_indices(…)` · [data.py, dòng 36–48](data.py#L36) · `return DataLoader(IndexedImages(…), …)` · [data.py, dòng 73–79](data.py#L73) |
| [__init__.py](__init__.py) | Khai báo hai module con. Dùng `eda` khi cần phân tích và `data` khi cần pipeline PyTorch. | `__all__ = ["data", "eda"]` · [__init__.py, dòng 7](__init__.py#L7) |

## 3. Input, output và nơi lưu

Các link có tên run dẫn tới ví dụ từ registry. Output dạng tensor hoặc loader được trả cho caller trong bộ nhớ. Các file EDA là bằng chứng riêng với dữ liệu mà model nhận. [Run đang được chọn](../../outputs/selected_run.json)

| Giai đoạn | Input | Output và thông tin cần có | Nơi lưu | Tiêu chí kiểm tra | Đoạn code và vị trí |
| --- | --- | --- | --- | --- | --- |
| Đọc dữ liệu gốc | Năm file train và một file test của CIFAR-10 | Mảng ảnh RGB dạng `uint8` theo thứ tự `[N, 32, 32, 3]`, nhãn số nguyên và train/test riêng | Mảng trả về trong bộ nhớ | Checksum từng file khớp. Số ảnh và số nhãn tương ứng. | `hashlib.md5(content).hexdigest()` và `reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)` · [eda.py, dòng 25–32](eda.py#L25) |
| Tạo hoặc kiểm tra split | Nhãn train gốc, seed và thư mục lưu split | Chỉ số 45.000 train, 5.000 validation, 10.000 test. Metadata ghi seed, hash nhãn, hash chỉ số, phiên bản và nguồn gốc split. | [split_indices.npz](../../outputs/eda/split_indices.npz) · [split_metadata.json](../../outputs/eda/split_metadata.json) | Train và validation không giao nhau, không lặp index và hợp lại phủ tập train gốc. Khi dùng lại, hash và seed phải khớp. Chỉ số test thuộc tập test riêng. | `train_test_split(…, test_size=5000, …, stratify=labels)` · [eda.py, dòng 49–56](eda.py#L49) · `np.intersect1d(indices['train'], indices['val'])` · [eda.py, dòng 57–62](eda.py#L57) · `meta['label_sha256'] != identity` · [eda.py, dòng 41–47](eda.py#L41) |
| Chọn dữ liệu cho run | Split đầy đủ, nhãn đúng tập, các giới hạn mẫu và seed | Chỉ số đầy đủ và chỉ số thực sự dùng. Khi cần lấy ít hơn, chia có phân tầng để giữ gần đúng tỷ lệ lớp của tập tương ứng. | [split_indices.npz của run](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/split_indices.npz) · [selected_indices.npz](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/selected_indices.npz) | Mỗi subset thuộc đúng split và khớp số lượng yêu cầu khi có giới hạn. Khi không giới hạn hoặc giới hạn đủ lớn, dùng toàn bộ split. | `stratify=np.asarray(labels)[indices]` · [data.py, dòng 27–33](data.py#L27) · `np.savez(Path(run_dir) / "selected_indices.npz", **selected)` · [data.py, dòng 41–48](data.py#L41) |
| Xuất EDA | Ảnh gốc, nhãn và split | Bảng cấu trúc có số mẫu, shape, dtype, khoảng giá trị pixel và bộ nhớ. Báo cáo kiểm tra dữ liệu, bảng phân bố lớp và biểu đồ đếm lớp train/validation. | [structural_overview.csv](../../outputs/eda/structural_overview.csv) · [integrity_report.csv](../../outputs/eda/integrity_report.csv) · [class_distribution.csv](../../outputs/eda/class_distribution.csv) · [class_distribution.png](../../outputs/eda/class_distribution.png) | Tổng số mẫu trong bảng phân bố khớp split. Trục dọc biểu đồ là số ảnh, nhóm theo lớp và tập train/validation. | `structural.to_csv(…)`, `duplicate_audit(…)`, `counts[['train', 'val']].plot.bar(…)` · [eda.py, dòng 187–200](eda.py#L187) |
| Ghi bằng chứng tiền xử lý | Ảnh mẫu train, thống kê pixel và cấu hình EDA | Ảnh preview, phiên bản split và tiền xử lý, mean/std mô tả của train và số ảnh dùng để thống kê | [preprocessing_preview.png](../../outputs/eda/preprocessing_preview.png) · [preprocessing_config.json](../../outputs/eda/preprocessing_config.json) · [summary.json của run](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/eda/summary.json) | Phân biệt mean/std mô tả dữ liệu train với mean/std ImageNet dùng để chuẩn hóa input. | `preprocessing_preview(pool[indices['train'][0]], cfg, output)` · [eda.py, dòng 257–264](eda.py#L257) · `normalization_policy='Descriptive train statistics only; ImageNet normalization defines input'` · [eda.py, dòng 99–103](eda.py#L99) |
| Tạo ảnh đầu vào | Ảnh RGB, tên model, kích thước ảnh và cờ augmentation | Tensor `[3, H, W]` đã chuẩn hóa, giữ nhãn và index gốc. Size 224 dùng transform của weights. Ví dụ size 64 resize, đổi sang tensor rồi chuẩn hóa theo ImageNet. | Trả từng mẫu cho loader trong bộ nhớ | Shape khớp kích thước cấu hình. Random crop/flip chỉ chạy khi bật cho train. | `return weights_transform` và `transforms.Normalize(weights_transform.mean, weights_transform.std)` · [data.py, dòng 15–24](data.py#L15) · `if augmentation and split != 'train':` · [data.py, dòng 52–62](data.py#L52) |
| Gom batch | Dataset đã chọn chỉ số, batch size, seed và cờ shuffle | Batch ảnh `[B, 3, H, W]`, nhãn `[B]` và chỉ số gốc `[B]`. Batch cuối có thể nhỏ hơn batch size. | Iterator của `DataLoader` trong bộ nhớ | Sau khi gom hoặc shuffle, ảnh, nhãn và index vẫn đi cùng nhau. | `return self.transform(image), label, original` · [data.py, dòng 67–70](data.py#L67) · `torch.Generator().manual_seed(seed)` · [data.py, dòng 73–79](data.py#L73) |

## 4. Dò một ảnh và tiêu chí hoàn thành

Ví dụ dò tay với `indices = [8, 42]`, lấy phần tử `i = 1` ở tập train với size 64 và tắt augmentation.

| Bước | Trạng thái | Đoạn code và vị trí |
| --- | --- | --- |
| Đổi vị trí thành index gốc | Vị trí `1` trong subset trỏ tới ảnh `42` trong tập train gốc. | `original = int(self.indices[i])` · [data.py, dòng 68](data.py#L68) |
| Đọc và biến đổi | Lấy ảnh RGB và nhãn tại index `42`. Ảnh sau transform có shape `[3, 64, 64]`. | `image, label = self.raw[original]` · [data.py, dòng 69](data.py#L69) · `transforms.Resize((image_size, image_size), antialias=True)` · [data.py, dòng 21–23](data.py#L21) |
| Trả và gom | Mẫu trả tensor, nhãn thật và `42`. Nếu gom cả hai ảnh với shuffle tắt, batch có shape `[2, 3, 64, 64]` và index `[8, 42]`. | `return self.transform(image), label, original` · [data.py, dòng 70](data.py#L70) · `batch_size=batch` · [data.py, dòng 77–79](data.py#L77) |

Kiểm tra hoàn thành bằng cách đối chiếu split và metadata, kiểm tra phân bố của subset, mở các bảng/hình EDA và xác minh shape cùng nhãn/index của một batch. Chỉ số trùng giữa train và validation là lỗi vì cả hai dùng cùng tập gốc. Chỉ số có cùng giá trị giữa train và test có thể trỏ tới hai ảnh khác nhau vì thuộc hai tập gốc riêng. [Chọn tập nguồn](data.py#L55) · [Kiểm tra trùng train và validation](eda.py#L57)

## 5. Cách đọc và thảo luận với AI

1. Đi theo hai nhánh trong sơ đồ. Với mỗi bước, chỉ ra input, output trong bộ nhớ và file được ghi.
2. Dò một ảnh bằng tay qua `IndexedImages.__getitem__()`. Hỏi vì sao dùng `self.indices[i]`. Nếu đọc trực tiếp `self.raw[i]`, loader có thể lấy ảnh khác với ảnh đã chọn trong subset.
3. Đối chiếu `split_metadata.json`, phân bố lớp và một batch. Ghi rõ bằng chứng nào đọc từ artifact, dò tay hoặc thực thi. Khi cập nhật nội dung đã duyệt, ghi ngày, người cập nhật và phiên bản ở đầu file.
