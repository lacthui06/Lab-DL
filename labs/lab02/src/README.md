# Lab02 · Bản Đồ Mã Nguồn

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-11 | 0.1.0 | Thường FEAI, cùng Codex | Tạo bản đồ bốn package, luồng xử lý và điểm vào inference |

## 1. Mục đích và cách đọc

Thư mục `src/` triển khai transfer learning trên CIFAR-10 với ResNet18 và DenseNet121. Bốn package chia trách nhiệm từ chuẩn bị dữ liệu đến huấn luyện, đánh giá và trình bày bằng chứng.

Bắt đầu từ bảng dưới để tìm đúng module. README của từng package giải thích hợp đồng dữ liệu, cơ chế và quyết định thiết kế; [walkthrough](../reports/walkthrough_lab_02.md) trình bày thực nghiệm và kết quả.

## 2. Bản đồ bốn package

| Package | Việc đảm nhiệm | Đầu vào → đầu ra chính | Tài liệu |
| --- | --- | --- | --- |
| `datasets` | Đọc và kiểm tra CIFAR-10, khóa split, khảo sát dữ liệu, tạo transform và DataLoader. | Ảnh gốc + config → chỉ số split, artifact EDA và batch ảnh/nhãn/index. | [README datasets](datasets/README.md) |
| `models` | Dựng backbone, thay classifier 10 lớp và xác định phần tham số được học. | Tên model + chính sách stage → model và số tham số; ảnh → logits. | [README models](models/README.md) |
| `training` | Cache đặc trưng, huấn luyện hai stage, chọn theo validation và lưu trạng thái tiếp tục. | Model + dữ liệu + config → checkpoint, history, selection và kết quả đánh giá. | [README training](training/README.md) |
| `utils` | Xuất metric/hình, tổng hợp chi phí, kiểm chứng artifact và đo benchmark. | Dự đoán, history, run hoặc model đã nạp → báo cáo, bằng chứng và số đo. | [README utils](utils/README.md) |

## 3. Luồng xử lý tổng thể

```text
CIFAR-10 + config
  → datasets: khóa split, chọn mẫu và chuẩn bị transform
  → models: nạp backbone ImageNet, thay classifier 10 lớp
  → training: cache feature → head-only → fine-tune
  → chọn checkpoint bằng validation → đánh giá test
  → utils: metric, biểu đồ, chi phí và kiểm chứng artifact

Checkpoint đã chọn + ảnh
  → inference.load_model(): nạp model và transform một lần
  → inference.predict_image(): dự đoán từng ảnh

Artifact đã lưu
  → datasets.eda / utils.notebook_views
  → notebook trình bày bảng và hình
```

Head-only dùng backbone cố định và feature đã cache; fine-tune scheduled xử lý ảnh với augmentation chỉ trên train. Notebook chính mặc định `MODE="view"` để đọc bằng chứng; `prepare`/`train` gọi pipeline tương ứng khi được chọn rõ ràng. Benchmark vẫn là CLI riêng.

## 4. Điểm vào và ranh giới trách nhiệm

| Nhu cầu | Điểm vào | Source chính |
| --- | --- | --- |
| Tạo artifact EDA | [scripts/analyze_data.py](../scripts/analyze_data.py) | [`build()`](datasets/eda.py#L177) |
| Chạy thực nghiệm | [scripts/run.py](../scripts/run.py) | [`run_suite()`](training/experiment.py#L474) |
| Kiểm chứng và bàn giao run | [scripts/finalize.py](../scripts/finalize.py) | [`finalize_run()`](utils/delivery.py#L71), [`complete_delivery()`](utils/delivery.py#L308) |
| Nạp checkpoint | API Python trong `inference.py` | [`load_model()`](inference.py#L18) |
| Dự đoán một ảnh | API Python trong `inference.py` | [`predict_image()`](inference.py#L42) |
| Xem kết quả trong notebook | [Hai notebook chính](../notebooks/) | [`context()`](utils/notebook_views.py#L16) và các hàm trình bày tương ứng |

`inference.py` là module riêng bên cạnh bốn package. Nó dựng lại model từ metadata checkpoint và áp dụng transform phù hợp trước dự đoán; manifest bàn giao chỉ mô tả checkpoint, không thay đổi API load.

Checkpoint trong run và các tệp resume có mục đích khác nhau. Xem [chỉ mục mô hình bàn giao](../models/README.md) khi cần inference, và [README training](training/README.md) khi cần tiếp tục huấn luyện.

## 5. Kiểm chứng và giới hạn

[Bản đồ test gates](../tests/README.md) chỉ ra các bài kiểm tra dữ liệu, model, cache, lựa chọn và benchmark. Một unit test đạt không thay thế việc kiểm chứng toàn bộ run thực tế.

Khi đối chiếu kết quả, đọc config lưu trong chính run đó. Khả năng mà source hiện tại hỗ trợ không đồng nghĩa mọi cấu hình đã được chạy hoặc có kết quả để báo cáo.
