# Lab01 · Mã Nguồn Mô-đun Hóa (`labs/lab01/src/`)

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-10 | 0.1.0 | Thái Anh Lạc | Chuẩn hóa tài liệu điều phối src/ |

Thư mục `src/` chứa toàn bộ mã nguồn triển khai thuần Python cho Lab 01, tuân thủ kiến trúc Clean Architecture: không sử dụng biến toàn cục, có type annotations, docstrings đầy đủ và tách biệt hoàn toàn giữa dữ liệu, mô hình, huấn luyện, suy luận và tiện ích.

---

## 1. Cấu trúc Mô-đun & Tài liệu Chi tiết

Mỗi thư mục con trong `src/` đều có một file đặc tả lộ trình kỹ thuật riêng biệt (`README.md`) tuân thủ cấu trúc chuẩn 5 phần:

```text
labs/lab01/src/
├── datasets/          # Xử lý dữ liệu, stratified split, trích xuất 10 feature
│   └── README.md      # Chi tiết: src/datasets/README.md
├── models/            # Kiến trúc FashionMLP, khởi tạo trọng số Kaiming, 3 biến thể
│   └── README.md      # Chi tiết: src/models/README.md
├── training/          # Vòng lặp huấn luyện, 1-batch sanity check, so sánh, tuning Optuna
│   └── README.md      # Chi tiết: src/training/README.md
├── utils/             # Quản lý run, cố định seed, snapshot môi trường, vẽ đồ thị
│   └── README.md      # Chi tiết: src/utils/README.md
├── inference.py       # Module suy luận, nạp checkpoint qua alias, kiểm chứng roundtrip
└── workflow.py        # Kịch bản thực thi pipeline tự động từ đầu đến cuối
```

### Liên kết nhanh đến tài liệu của từng module:
- [**Datasets Module Documentation (`datasets/README.md`)**](datasets/README.md)
- [**Models Module Documentation (`models/README.md`)**](models/README.md)
- [**Training Module Documentation (`training/README.md`)**](training/README.md)
- [**Utils Module Documentation (`utils/README.md`)**](utils/README.md)

---

## 2. Pipeline Tổng Thể Lab 01

```text
FashionMNIST gốc
  ↓ (datasets/base.py, datasets/eda.py)
Phân tầng Stratified Split (9:1) & Khóa thông số chuẩn hóa Z-score
  ↓ (datasets/features.py, datasets/preparation.py)
Trích xuất 10 đặc trưng thủ công & Đóng gói prepared.npz
  ↓ (training/comparison.py, training/trainer.py)
Chốt chặn 1-Batch Sanity Probe → Huấn luyện 3 kiến trúc (SmallMLP, MainMLP, MainMLP_FE)
  ↓ (training/tuning.py)
Khảo sát 3 Optimizer Fixed-LR → Tuning siêu tham số bằng Optuna → Refit mô hình chiến thắng
  ↓ (inference.py)
Nạp checkpoint qua alias 'final', suy luận mẫu đơn lẻ, kiểm chứng roundtrip save/load
```

---

## 3. Lộ Trình Module Suy Luận & Phục Vụ (`inference.py`)

Quy trình nạp mô hình, kiểm chứng độ trôi và phục vụ suy luận được chuẩn hóa theo bảng lộ trình:

| Bước | Tác vụ (Phase) | Dữ liệu đầu vào (Input & Input Path) | Xử lý gì (Nội dung chi tiết & Thuật toán) | Module & Hàm thực thi | Sản phẩm đầu ra (Output & Artifact Path) | Tại sao làm vậy (Why / Lý do kỹ thuật) |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Phân giải đường dẫn Run** | - Tên alias (`"final"`, `"MainMLP"`, `"SmallMLP"`, `"MainMLP_FE"`) hoặc đường dẫn `Path`. | Tìm kiếm trong `outputs/runs/` thư mục run phù hợp nhất hoặc theo liên kết symlink/alias. Ném lỗi nếu không tồn tại. | [`inference.py`](inference.py): `resolve_run_path` (L14) | - Đối tượng `Path` trỏ chính xác đến thư mục run chứa checkpoint. | Giúp người dùng không phải gõ đường dẫn timestamp dài dòng; hỗ trợ chuyển đổi linh hoạt giữa các checkpoint. |
| **2** | **Nạp mô hình qua Alias (`load_model`)** | - Tên alias hoặc đường dẫn run.<br>- Thiết bị tính toán `device`. | Đọc `config.json` để tự động dựng đúng kiến trúc; nạp weights từ `best.pt` bằng `torch.load(weights_only=False)`; chuyển mô hình sang `model.eval()`. | [`inference.py`](inference.py): `load_model` (L64) | - `model: FashionMLP` ở `eval()` với weights tối ưu. | Cho phép nạp mô hình tức thì chỉ với một dòng lệnh; tự động khôi phục cấu hình mà không cần truyền lại tham số thủ công. |
| **3** | **Suy luận mẫu đơn lẻ (`infer_sample`)** | - Tên alias hoặc đường dẫn run.<br>- Chỉ số mẫu kiểm thử `index: int`. | Nạp mô hình, lấy ảnh tại `index` từ `data/prepared.npz`, chạy feedforward với `torch.inference_mode()`, áp dụng Softmax và lấy Argmax. | [`inference.py`](inference.py): `infer_sample` (L132) | - Dictionary kết quả:<br>  `index`, `true_label`, `predicted_label`, `probabilities`. | Phục vụ kiểm tra nhanh tính đúng đắn trên từng mẫu cụ thể; tạo thư mục run `_inference` để lưu vết lịch sử. |
| **4** | **Kiểm chứng tính toàn vẹn Nạp/Lưu (`save_load_model`)** | - Đường dẫn run hoặc alias.<br>- Lô 16 mẫu test đầu tiên. | Dựng một instance mô hình mới hoàn toàn độc lập, nạp state_dict từ `best.pt`, so sánh Softmax probability trước và sau khi lưu bằng `torch.allclose(atol=1e-6)`. | [`inference.py`](inference.py): `save_load_model` (L142) | - Trạng thái: `roundtrip_passed: True`.<br>- Ảnh lưới: `loaded_model_test_samples.png` (8 đúng xanh, 8 sai đỏ). | Chứng minh bằng toán học số học rằng quá trình serialize và deserialize không làm sai lệch dự đoán của mô hình. |
| **5** | **Đánh giá độc lập trên Test Set (`evaluate_run`)** | - Đường dẫn run hoặc alias.<br>- Tập Test 10.000 mẫu `(Batch=128)` từ `data/prepared.npz`. | Duyệt qua toàn bộ 10.000 mẫu test, tính ma trận nhầm lẫn, báo cáo phân loại, và xuất các file đánh giá chuẩn. | [`inference.py`](inference.py): `evaluate_run` (L118) | - Thư mục run `_evaluation`:<br>  `metrics.json`, `test_predictions.npz`, `classification_report.csv`. | Đánh giá độc lập, khách quan trên tập test chưa từng thấy trong quá trình huấn luyện và tuning. |

---

## 4. Hướng Dẫn Sử Dụng Nhanh (Quick Start Examples)

```python
from labs.lab01.src import load_model, infer_sample, save_load_model

# 1. Nạp mô hình tối ưu nhất
model = load_model("final")

# 2. Suy luận mẫu test thứ 0
result_path = infer_sample("final", index=0)

# 3. Kiểm chứng roundtrip save/load
save_load_model("final")
```
