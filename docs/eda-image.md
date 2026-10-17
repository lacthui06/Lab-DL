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
toàn bộ train.

## 4. Kiểm tra tối thiểu

| Nhóm | Nội dung |
|---|---|
| Structure | shape, dtype, số kênh, pixel range, memory, label mapping |
| Integrity | label ngoài miền, pixel ngoài miền, NaN/Inf, ảnh toàn một giá trị |
| Duplicate | exact duplicate trong từng split và giữa các split, kèm nhãn |
| Class balance | số mẫu mỗi lớp ở train và validation |
| Pixel | global mean/std, tỷ lệ pixel nền, brightness theo lớp |
| Spatial | mean/variance map, foreground ratio, bounding box theo lớp |
| Similarity | cosine/correlation giữa mean image của các lớp |

Ba lưu ý bắt buộc:

- Pixel 0 là **nền**, không phải missing value. Không điền, không loại.
- Duplicate kiểm tra bằng nội dung là phép kiểm tra khác với kiểm tra indices rời nhau.
  Cả hai đều cần, và kết quả báo riêng.
- Nếu một phép kiểm tra không chạy, ghi `not computed`. Đừng ghi `0` hoặc bỏ trống,
  vì người đọc sẽ hiểu là "đã kiểm tra và sạch".

## 5. Điều cần cảnh giác

- **Mean image làm mất phương sai trong lớp.** Hai lớp có mean gần giống nhau vẫn có
  thể phân biệt hoàn hảo, và ngược lại. Similarity của mean image chỉ tạo giả thuyết;
  xác nhận phải bằng per-class metric trên validation.
- **Foreground threshold phải là hằng số có tên.** Với ảnh có nền lớn, ngưỡng 0 sẽ
  gộp nhiễu nền vào bounding box và làm mọi aspect ratio vô nghĩa.
- **PCA phải fit trên train**, chỉ transform lên validation/test.
- **t-SNE/UMAP không có ý nghĩa khoảng cách định lượng.** Không dùng để khẳng định
  lớp nào dễ phân biệt. Nếu thư viện chưa có, ghi rõ và bỏ qua, không cài ngầm.

## 6. Đầu ra

Bảng CSV và biểu đồ PNG vào `outputs/eda/`, cộng một file JSON tổng hợp để
notebook training đọc lại. Nhận xét và quyết định ghi vào `walkthrough_lab_NN.md`,
không tạo report Markdown riêng.
