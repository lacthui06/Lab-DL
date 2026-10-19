# Log kiểm tra EDA và Feature Engineering

Ngày thực hiện: 2026-10-19. Phạm vi đã duyệt: sửa provenance của bảng spatial,
tính đúng bounding box, bổ sung test và tái tạo EDA. Không huấn luyện đầy đủ.

## 1. Vấn đề và thay đổi

| Phần | Trước | Sau |
|---|---|---|
| `original_index` | Đánh số lại 0..53999 trong tập con | Lưu đúng chỉ số trong pool train gốc 60000 ảnh qua `train_idx` |
| `split` | Ghi cố định `unassigned` | Pipeline truyền rõ `train` cho 54000 dòng |
| Bounding box | Đếm hàng/cột có foreground | Khoảng từ tọa độ nhỏ nhất đến lớn nhất, tính cả hai đầu |

Foreground vẫn là pixel > 0.1 sau khi đưa ảnh về [0, 1]. Công thức mới:
`height = ymax - ymin + 1`, `width = xmax - xmin + 1`.
Ví dụ foreground ở hàng 2 và 5: chiều cao mới là 4, trước là 2.
Ảnh rỗng: chiều cao, rộng và diện tích bằng 0; aspect ratio là NaN;
`empty_foreground=True`.

Hàm `_foreground_bbox()` dùng chung cho `spatial_features()` và
`compute_handcrafted_features()` để hai bảng dùng cùng định nghĩa.
`spatial_features()` bắt buộc truyền keyword `original_indices` và `split`:

```python
spatial_features(
    X_train[train_idx], y_train[train_idx], class_names,
    original_indices=train_idx, split="train",
)
```

Thiếu keyword sẽ báo TypeError. Chỉ số sai chiều/số lượng, không nguyên,
âm hoặc trùng nhau sẽ báo ValueError; tên split chỉ nhận `train` hoặc `val`
vì `source_dataset` của API này là `original_train`. Số lượng nhãn phải khớp ảnh.
Caller ngoài repository phải cập nhật cách gọi theo hợp đồng mới này.

## 2. Kiểm chứng đã thực thi

- Test trước sửa: 7 ca mới thất bại, gồm trường hợp khung có khoảng trống.
- Test sau sửa: **29 passed, 1 deselected in 9.90s**; gồm sanity overfit 1 batch.
- `git diff --check` hoàn tất không báo lỗi whitespace.
- Tái tạo thành công 7 bảng CSV và 7 hình PNG bằng pipeline hiện có.
- Bảng spatial có 54000 dòng; toàn bộ `original_index` khớp `train_idx`,
  toàn bộ `split` là `train`.
- Hai mảng chỉ số train/validation giống hệt trước sửa.
- Toàn bộ nội dung JSON preprocessing giống hệt trước sửa:
  mean=0.286209, std=0.353160; split vẫn là stratified 54000/6000, seed=42.
- Dữ liệu train thực tế không có ảnh foreground rỗng; hành vi ảnh rỗng được
  kiểm tra bằng dữ liệu tổng hợp.

## 3. Chênh lệch đo được so với bảng ngay trước sửa

| Cột spatial | Số dòng thay đổi |
|---|---:|
| `original_index` | 53999 |
| `split` | 54000 |
| `bbox_height` | 527 |
| `bbox_width` | 116 |
| `bbox_aspect_ratio` | 612 |
| `bbox_area_ratio` | 612 |

Số dòng bbox được so sánh bằng `np.isclose`, không tính nhiễu biểu diễn CSV.
Một chỉ số gốc tình cờ bằng vị trí trong tập con nên không phải 54000 chỉ số đổi.

Ví dụ bảng handcrafted theo lớp:

| Lớp / chỉ số | Trước | Sau |
|---|---:|---:|
| T-shirt/top: bbox_height_mean | 27.75 | 27.79 |
| T-shirt/top: bbox_area_ratio_mean | 0.7682 | 0.7696 |
| Sneaker: bbox_height_mean | 13.41 | 13.45 |
| Sneaker: bbox_aspect_ratio_mean | 2.1499 | 2.1436 |

## 4. Phạm vi ảnh hưởng và giới hạn

- `spatial_features.csv` bị Git ignore theo chính sách có sẵn, nhưng đã tái tạo
  trên máy. `handcrafted_features.csv` được Git theo dõi và có diff.
- `class_similarity_matrix.csv` và file NPZ đã có diff từ lần tái tạo trước:
  similarity chỉ lệch tối đa khoảng 5.55e-16 so với HEAD; mảng chỉ số và payload
  NPY trong NPZ không đổi. Không quy các diff có sẵn này cho sửa bounding box.
- Không sửa requirements.txt (đã có thay đổi của người dùng), model, training,
  normalization hay thuật toán chia tập.
- Mô hình nhận tensor ảnh, không dùng các cột spatial/handcrafted nên không cần
  train lại do sửa các bảng này.
- Chỉ số gốc là vị trí trong pool FashionMNIST train gốc, không phải ID toàn cục.
- Không chạy test `test_29_experiment_runner_signatures`: test này huấn luyện
  nhiều lượt trên toàn bộ dữ liệu CPU, ngoài phạm vi kiểm tra nhỏ đã duyệt.
- Notebook giữ output lịch sử, chưa chạy lại end-to-end; kết quả lần này chạy
  trực tiếp module EDA và pytest.

## 5. Lệnh kiểm tra

Chạy từ root repository:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/lab-dl-mpl OMP_NUM_THREADS=1 .venv/bin/python -m pytest labs/lab01/tests -q -k 'not test_29_experiment_runner_signatures' -p no:cacheprovider
```

Các ca bổ sung kiểm tra foreground rời nhau, ảnh rỗng, một pixel,
khung bất đối xứng, provenance sai và sự nhất quán của bảng handcrafted.
