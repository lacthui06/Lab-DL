# Module datasets — Chuẩn Bị Dữ Liệu & 10 Đặc Trưng Hình Học

## 1. Thông Tin Mô-đun (Name & Metadata)
- **Tên mô-đun:** `src/datasets` (FashionMNIST Split, Normalization & Feature Engineering)
- **Ngày tạo (Created):** 2026-10-22 · **Tác giả (Author):** Thái Anh Lạc
- **Ngày cập nhật (Updated):** 2026-11-10 · **Người cập nhật (Author):** Thái Anh Lạc
- **Tệp tài liệu chính thức:** `labs/lab01/src/datasets/README.md`

---

## 2. Mục Đích (Purpose)
Module này chịu trách nhiệm chuẩn bị dữ liệu sạch sẽ từ đầu đến cuối cho toàn bộ bài Lab 01:
- Tải bộ dữ liệu gốc FashionMNIST (70.000 ảnh xám 28×28 thuộc 10 danh mục trang phục).
- Thực hiện phân chia phân tầng (Stratified Split) theo tỷ lệ 9:1 trên 60.000 ảnh train gốc (54.000 train / 6.000 validation) và cô lập 10.000 ảnh test chuẩn để kiểm tra công tâm.
- Đo đạc thống kê cường độ điểm ảnh, thực hiện chuẩn hóa Z-score và tự tay trích xuất thêm 10 đặc trưng hình học/độ sáng để cung cấp thêm biểu diễn cho mô hình `MainMLP_FE`.
- Đóng gói toàn bộ các mảng dữ liệu đã chuẩn hóa vào một file nén duy nhất `labs/lab01/data/prepared.npz`, giúp các bước huấn luyện sau chỉ mất 1 giây để nạp dữ liệu vào bộ nhớ.

---

## 3. Khởi Tạo Dữ Liệu & Luồng Tiền Xử Lý (Init Dataset & Pipeline)

### 3.1. Cấu Trúc Luồng Tiền Xử Lý 5 Bước (Data Pipeline Structure)

```text
1. Ảnh thô từ torchvision.datasets.FashionMNIST (70.000 ảnh xám uint8 [0, 255])
   │
   ▼ [make_stratified_split() · base.py#L9](base.py#L9)
2. Phân chia phân tầng Stratified Split (seed 42, tỷ lệ 9:1)
   Train: 54.000 ảnh (5.400 ảnh/lớp) │ Val: 6.000 ảnh (600 ảnh/lớp) │ Test: 10.000 ảnh (1.000 ảnh/lớp)
   │
   ▼ [run_eda_and_fe_pipeline() · eda.py#L18](eda.py)
3. Tính toán thống kê chuẩn hóa Z-score (chỉ đo trên 54.000 ảnh train)
   Điểm ảnh được chia 255 về [0, 1] ──> Đo mean=0.2862, std=0.3532
   │
   ▼ [extract_features() · features.py#L25](features.py#L25)
4. Trích xuất 10 đặc trưng hình học từ ảnh [0, 1]
   Tính độ sáng, độ tương phản, diện tích, bounding box, tỷ lệ dáng, trọng tâm, đối xứng, viền
   │
   ▼ [prepare_data() · preparation.py#L13](preparation.py#L13)
5. Fit StandardScaler (chỉ trên train) & Đóng gói toàn bộ ra file data/prepared.npz
```

### 3.2. Đầu Vào và Đầu Ra Dữ Liệu (Input / Output Data)

#### Bảng hợp đồng dữ liệu tiền xử lý:

| Tên thành phần | Kích thước (Shape) | Kiểu dữ liệu (Dtype) | Ý nghĩa nghiệp vụ |
| :--- | :---: | :---: | :--- |
| **Ảnh thô (Raw Images)** | `(N, 28, 28)` | `uint8` | 70.000 ảnh xám gốc trong `data/FashionMNIST/raw/`. |
| **Nhãn thô (Raw Labels)** | `(N,)` | `uint8` / `int64` | 10 lớp danh mục: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot. |
| **`train_pixels`** | `(54000, 784)` | `float32` | 784 điểm ảnh đã trừ mean=0.2862 chia std=0.3532 của tập train (Z-score). |
| **`train_features`** | `(54000, 10)` | `float32` | 10 đặc trưng hình học đã được chuẩn hóa qua `StandardScaler`. |
| **`train_labels` / `train_indices`** | `(54000,)` | `uint8` / `int64` | Nhãn phân loại và chỉ số vị trí gốc trong tập train-pool 60.000. |
| **`val_pixels` / `val_features`** | `(6000, 784)` / `(6000, 10)` | `float32` | Tập kiểm định validation độc lập (600 mẫu mỗi lớp). |
| **`test_pixels` / `test_features`** | `(10000, 784)` / `(10000, 10)` | `float32` | Tập kiểm thử test độc lập (1.000 mẫu mỗi lớp). |
| **`prepared.npz`** | File nén NumPy | — | Chứa trọn vẹn cả 12 mảng trên, sẵn sàng nạp thẳng vào DataLoader. |

#### Danh sách công thức toán học của 10 đặc trưng hình học:
Gọi $I$ là ma trận ảnh kích thước $28 \times 28$ có giá trị trong đoạn $[0, 1]$, và $M = (I > 0.1)$ là mặt nạ nhị phân đánh dấu vùng vật thể (foreground):
1. **`mean` (Mức sáng toàn ảnh):** $\frac{1}{784} \sum_{x, y} I_{x, y}$
2. **`std` (Độ tương phản ảnh):** $\sqrt{\frac{1}{784} \sum_{x, y} (I_{x, y} - \text{mean})^2}$
3. **`foreground_ratio` (Tỷ lệ diện tích chiếm chỗ):** $\frac{1}{784} \sum_{x, y} \mathbb{I}(I_{x, y} > 0.1)$
4. **`bbox_width` (Độ rộng hộp bao quanh vật thể):** $\frac{\text{width}(M)}{28}$
5. **`bbox_height` (Độ cao hộp bao quanh vật thể):** $\frac{\text{height}(M)}{28}$
6. **`aspect_ratio` (Tỷ lệ dáng ngang/dọc):** $\frac{\text{width}(M)}{\text{height}(M)}$ (nếu height = 0 thì quy ước bằng 0)
7. **`centroid_x` (Tọa độ trọng tâm ngang):** $\frac{\sum x \cdot I_{x, y}}{27 \times \sum I_{x, y}}$ (nếu tổng cường độ bằng 0 thì quy ước bằng 0)
8. **`centroid_y` (Tọa độ trọng tâm dọc):** $\frac{\sum y \cdot I_{x, y}}{27 \times \sum I_{x, y}}$ (nếu tổng cường độ bằng 0 thì quy ước bằng 0)
9. **`symmetry` (Mức độ đối xứng trục dọc):** $1 - \frac{1}{784} \sum |I - \text{flip\_horizontal}(I)|$
10. **`edge_density` (Mật độ đường viền lân cận):** Tỷ lệ các cặp điểm ảnh liền kề có chênh lệch độ sáng $|I_1 - I_2| > 0.1$ trên tổng số 1.512 cặp (756 cặp ngang + 756 cặp dọc).

### 3.3. Tệp Mã Nguồn & Bản Đồ Điều Hướng Code (Source Files)

Giữ phím **`Ctrl + Click`** để mở trực tiếp trong trình soạn thảo:

| Tệp mã nguồn | Ký hiệu / Hàm chính (Ctrl+Click) | Dòng code | Việc đảm nhiệm |
| :--- | :--- | :--- | :--- |
| [`base.py`](base.py) | [`make_stratified_split()`](base.py#L9)<br>[`save_split()`](base.py#L23)<br>[`load_split()`](base.py#L52) | [`base.py#L9-L67`](base.py#L9-L67) | Chia phân tầng giữ nguyên tỷ lệ 10 lớp, lưu và nạp lại chỉ số index để tái lập bit-exact giữa các lần chạy. |
| [`features.py`](features.py) | [`extract_features()`](features.py#L25)<br>[`foreground_bbox()`](features.py#L8)<br>[`measure_intensity()`](features.py#L17) | [`features.py#L8-L45`](features.py#L8-L45) | Tính toán 10 đặc trưng hình học từ mảng ảnh float32 `[0, 1]`, xử lý an toàn cho ảnh rỗng (tránh lỗi chia cho 0). |
| [`preparation.py`](preparation.py) | [`prepare_data()`](preparation.py#L13) | [`preparation.py#L13-L71`](preparation.py#L13-L71) | Điều phối toàn bộ quy trình tiền xử lý, fit scaler chỉ trên train, vẽ biểu đồ tương quan và ghi ra `data/prepared.npz`. |
| [`eda.py`](eda.py) | `run_eda_and_fe_pipeline()` | [`eda.py#L18`](eda.py) | Kiểm tra tính toàn vẹn (NaN, trùng lặp, dải giá trị), tính trung bình ảnh theo lớp và vẽ heatmap pixel. |

- **Artifact đầu ra phục vụ các bước tiếp theo:**
  - File dữ liệu trung tâm: [`labs/lab01/data/prepared.npz`](../../data/prepared.npz).
  - Hồ sơ tiền xử lý: `preprocessing_config.json` và `feature_scaler.json`.
