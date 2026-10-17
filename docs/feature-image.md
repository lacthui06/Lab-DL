# Feature Engineering cho dữ liệu ảnh

Flow chung cho bước Feature Engineering của bài toán phân loại ảnh trong repo này.
Tài liệu này là **quy trình**, không chứa kết quả đo. Mọi con số đo được ghi trong
`labs/labNN/walkthrough_lab_NN.md` kèm `split_version` và `preproc_version`.

Bước trước: [eda-image.md](eda-image.md).
Kỹ thuật chung: [feature.md](feature.md). Luồng Deep Learning: [dl_flow.md](dl_flow.md).
Quy tắc project: [AGENTS.md](../.agents/AGENTS.md).

## 1. Thứ tự transform

```text
train:      augmentation -> ToTensor -> Normalize
validation:                ToTensor -> Normalize
```

Thứ tự này là **ràng buộc kỹ thuật**, không phải lựa chọn phong cách:

- `Normalize` cần tensor `float`, nên phải đứng sau `ToTensor`.
- Augmentation hình học của torchvision nhận `PIL` hoặc tensor. Đặt trước `ToTensor`
  là an toàn; đặt sau `Normalize` sẽ làm phép nội suy chạy trên giá trị đã chuẩn hóa
  và tạo ra dữ liệu không còn nghĩa.

`ToTensor()` nhận `HWC` và trả `CHW`. Với ảnh một kênh, nguồn phải là `[H, W]` hoặc
tensor đã đúng `CHW`; truyền `numpy` shape `[1, H, W]` sẽ bị hiểu là `HWC` và báo lỗi
số kênh. Test phải khẳng định shape sau transform, không giả định.

## 2. Normalization

```text
x_norm = (x - mean_train) / std_train
```

Điều kiện để phép đo dùng được:

- Đọc đủ toàn bộ ảnh train, không `drop_last`, không augmentation.
- Dùng scaled view `[0, 1]`, không dùng raw `uint8`.
- Gộp toàn bộ pixel thành mảng phẳng rồi mới lấy std. Lấy std theo trục 0 của mảng
  2 chiều sẽ trả về mảng, không phải scalar.
- Dùng **population variance** (`ddof=0` / `correction=0`), khớp với giả định của
  `torchvision.transforms.Normalize`.
- Kiểm tra `std > 0`, giá trị finite, và ghi lại số ảnh, số pixel đã đếm.

**Không dùng trung bình std của từng ảnh.** Đó là đại lượng khác: nó bỏ qua phương
sai **giữa** các ảnh, nên luôn nhỏ hơn global std và làm dữ liệu sau Normalize không
đạt std bằng 1. Cách kiểm chứng: sau khi Normalize, std của train phải xấp xỉ 1.0.
Nếu ra khác, mẫu số đang sai.

Sau khi đo, ghi mean/std vào một artifact JSON kèm `preproc_version`, và để
`src/config.py` đọc từ đó. Nếu artifact vắng mặt, fallback phải phát `RuntimeWarning`
để không ai vô tình train bằng hằng số chưa kiểm chứng.

## 3. Augmentation

- Chỉ áp dụng cho train.
- Không mặc định dùng lật dọc; cân nhắc ngữ nghĩa của loại ảnh.
- Góc xoay nhỏ; góc lớn làm thay đổi ngữ nghĩa.
- Không bật augmentation rồi so sánh với run không augmentation mà vẫn kết luận về
  kiến trúc. Một run chỉ được phép kết luận về cấu hình hoàn chỉnh của nó.
- Train và validation phải là hai `Dataset` object riêng trên cùng indices. Nếu dùng
  chung một object và chỉ đổi cờ transform, augmentation có thể lan sang validation.
  Kiểm chứng: gọi hai lần trên cùng một ảnh validation phải cho kết quả giống nhau.

## 4. Nhánh mở rộng

PCA baseline, handcrafted features, autoencoder, pretrained backbone và các biến thể
input chỉ thực hiện khi được xác nhận. Chúng không phải điều kiện nghiệm thu của
pipeline core, và không được triển khai trước khi nhánh chính ổn định.

## 5. Bàn giao cho training

Guard ở đầu notebook training phải **assert**, không `print`. Một guard chỉ in cảnh
báo sẽ bị bỏ qua khi chạy batch. Cần kiểm tra:

1. Artifact EDA và Feature Engineering tồn tại.
2. `preproc_version` khớp `split_version` của split đang dùng. Lệch nghĩa là mean/std
   được đo trên tập khác với tập đang train.
3. Mean/std mà `src/config.py` nạp được bằng giá trị trong artifact, và nguồn phải
   là `artifact:`. Nếu không, module đã rơi về fallback.
4. Kích thước train/validation mà loader trả về khớp metadata của split.

Mọi kết quả phải ghi kèm `split_version` và `preproc_version`. Hai con số accuracy
đo trên hai split khác nhau không so sánh được, kể cả khi kiến trúc giống hệt.

## 6. Chất lượng code

- Không hard-code thống kê trong hàm vẽ. Hàm đảo một phép biến đổi phải đọc tham số
  từ cùng nguồn mà phép biến đổi đã dùng, nếu không ảnh hiển thị sẽ sai mà không báo lỗi.
- Một phép biến đổi chỉ được định nghĩa ở **một chỗ**. Chia `255` ở nhiều nơi là lỗi
  tiềm ẩn: đổi một chỗ thì số liệu lệch mà không có ngoại lệ nào.
- Kiểm chứng API theo phiên bản đang cài, không theo tài liệu cũ:

| API | Cách gọi cũ | Cách gọi đúng |
|---|---|---|
| `torch.Tensor.std` | `std(ddof=0)` | `std(correction=0)` |
| `Axes.boxplot` | `labels=` | `tick_labels=` |
| `numpy.ndarray.std` | `std(axis=0, ddof=1)` trả về mảng | `.reshape(-1).std(ddof=1)` để có scalar |

## 7. Thứ tự thực thi

```text
EDA notebook      -> outputs/eda/split_indices.npz, split_metadata.json, thống kê
Feature Eng.      -> outputs/eda/preprocessing_config.json
pytest            -> xác nhận contract trước khi tốn GPU
Training (Kaggle) -> guard assert -> train -> test sau khi khóa cấu hình
walkthrough       -> ghi kết quả kèm version
```

Local chỉ chạy EDA, Feature Engineering, test và 1-batch sanity. Full training trên
Kaggle GPU.
