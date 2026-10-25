# Feature Engineering cho dữ liệu ảnh

Flow chung cho bước Feature Engineering của bài toán phân loại ảnh trong repo này.
Tài liệu này là **quy trình**, không chứa kết quả đo. Mọi con số đo được ghi trong
`labs/labNN/walkthrough_lab_NN.md` kèm `split_version` và `preproc_version`.

Bước trước: [eda-image.md](eda-image.md).
Kỹ thuật chung: [feature.md](feature.md). Luồng Deep Learning: [dl_flow.md](dl_flow.md).
Quy tắc project: [AGENTS.md](../.agents/AGENTS.md).

## 1. Phân biệt các khái niệm

Không đánh đồng các khái niệm sau. Chúng khác nhau về mục đích và về nơi fit tham số:

| Khái niệm | Là gì | Fit trên |
|---|---|---|
| Feature Engineering | thiết kế, tạo hoặc biến đổi đặc trưng | train |
| Data Transformation | thay đổi định dạng hoặc cách biểu diễn | không fit |
| Data Scaling | điều chỉnh thang đo | train |
| Data Augmentation | tạo biến thể dữ liệu cho huấn luyện | không fit |
| Feature Extraction | trích đặc trưng bằng thủ công hoặc mô hình | train |
| Representation Learning | mô hình tự học biểu diễn | train |
| Feature Selection | chọn tập đặc trưng đầu vào | train |
| Regularization | hạn chế Overfitting | không fit |
| Model Selection | chọn kiến trúc và cấu hình | validation |

Với Deep Learning, mô hình tự học đặc trưng từ ảnh nên không bắt buộc áp dụng
handcrafted feature engineering trước khi huấn luyện. Nhưng MLP không chia sẻ trọng số
theo vị trí, nên mọi cấu trúc không gian đều phải học lại từ đầu — đây là lý do một
baseline đơn giản vẫn cần thiết để làm mốc so sánh.

## 2. Thứ tự transform

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

## 3. Biến thể đầu vào

Chỉ biến thể đầu tiên là mặc định. Các biến thể còn lại cần kế hoạch riêng và kiểm tra
lại contract của model.

| Input Variant | Representation | Ghi chú |
|---|---|---|
| `raw_28` | `[1, H, W]` | mặc định cho mọi model |
| `flat_784` | `[H*W]` | cho PCA hoặc classifier truyền thống |
| `pad_32` | `[1, 32, 32]` | đổi feature-map size, phải kiểm tra model |
| `resize_224x3` | `[3, 224, 224]` | cho pretrained backbone |
| `raw_28+edge` | `[2, H, W]` | đổi số kênh, phải kiểm tra model |

Không resize hoặc pad nếu chưa có lý do thực nghiệm. Không Flatten ảnh trước các tầng
tích chập.

## 4. Normalization

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

Với pretrained backbone dùng ImageNet normalization: chuyển ảnh sang 3 kênh theo cách
đã xác định, resize theo yêu cầu backbone, và ghi rõ cấu hình. Không trộn ImageNet
mean/std với thống kê của dataset mà không có lý do.

## 5. Augmentation

- Chỉ áp dụng cho train.
- Không mặc định dùng lật dọc; cân nhắc ngữ nghĩa của loại ảnh.
- Góc xoay nhỏ; góc lớn làm thay đổi ngữ nghĩa.
- Không bật augmentation rồi so sánh với run không augmentation mà vẫn kết luận về
  kiến trúc. Một run chỉ được phép kết luận về cấu hình hoàn chỉnh của nó.
- Train và validation phải là hai `Dataset` object riêng trên cùng indices. Nếu dùng
  chung một object và chỉ đổi cờ transform, augmentation có thể lan sang validation.
  Kiểm chứng: gọi hai lần trên cùng một ảnh validation phải cho kết quả giống nhau.
- Kiểm tra trực quan ảnh sau augmentation trước khi tin vào nó.

## 6. Nhánh mở rộng

**PCA baseline:** flatten ảnh, fit PCA trên train, transform validation, xác định số
component cho 90%/95% variance, huấn luyện Logistic Regression hoặc kNN trên
representation đó. Không fit PCA trên validation/test. Không mặc định PCA-50 là tốt
nhất; số component phải tính từ dữ liệu.

**Deep feature extraction:** autoencoder (fit encoder chỉ trên train, ghi latent
dimension) hoặc pretrained backbone (frozen trước, fine-tune sau nếu đủ tài nguyên).
So sánh số parameter được train, thời gian và validation metric.

**Representation comparison:** so sánh ảnh gốc với PCA và latent representation trên
cùng split và cùng metric. Ghi rõ dimensionality của từng loại.

Các nhánh này chỉ thực hiện khi được xác nhận, không phải điều kiện nghiệm thu của
pipeline core, và không triển khai trước khi nhánh chính ổn định.

## 7. Metrics

**Primary:** Validation Accuracy và Validation Macro F1.

Accuracy không đủ cho bài nhiều lớp vì nó có thể che một lớp mà mô hình không bao giờ
dự đoán đúng. Macro F1 tính trung bình không trọng số theo lớp nên phản ánh trường hợp
đó. Nếu tiêu chí chọn checkpoint trong code là accuracy, phải giữ nguyên cho khớp;
đổi tiêu chí cần đồng bộ engine trước khi chạy.

**Diagnostic:** per-class Precision, Recall, F1, confusion matrix, train/validation
loss, và khoảng cách train-validation. Đây là bằng chứng cho phân tích lỗi, không phải
cổng pass/fail bổ sung.

## 8. Thiết kế thí nghiệm

Không chạy toàn bộ tổ hợp ngay từ đầu. Bốn giai đoạn:

1. **Core:** baseline tối thiểu và một biến thể có cải tiến có kiểm soát.
2. **Truyền thống (tùy chọn):** PCA + Logistic Regression hoặc kNN.
3. **Cải tiến có kiểm soát:** normalization, augmentation, dropout, weight decay.
4. **Representation (tùy chọn):** edge channel, autoencoder, pretrained backbone.

Sau mỗi giai đoạn, tổng hợp kết quả và chọn bước tiếp theo dựa trên validation
performance và chi phí. Một run thay đổi nhiều yếu tố cùng lúc chỉ cho phép kết luận
về cấu hình hoàn chỉnh, không được quy toàn bộ chênh lệch cho một yếu tố.

## 9. Bàn giao cho training

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

## 10. Chất lượng code

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

## 11. Thứ tự thực thi

```text
EDA notebook      -> outputs/eda/split_indices.npz, split_metadata.json, thống kê
Feature Eng.      -> outputs/eda/preprocessing_config.json
pytest            -> xác nhận contract trước khi tốn GPU
Training (Kaggle) -> guard assert -> train -> test sau khi khóa cấu hình
walkthrough       -> ghi kết quả kèm version
```

Local chỉ chạy EDA, Feature Engineering, test và 1-batch sanity. Full training trên
Kaggle GPU.
