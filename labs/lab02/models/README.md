# Lab02 · Chỉ Mục Mô Hình Bàn Giao

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-11 | 0.1.0 | Thường FEAI, cùng Codex | Tạo chỉ mục checkpoint fine-tune, nguồn metric và hướng dẫn load |
| 2026-11-11 | 0.1.1 | Thường FEAI, cùng Codex | Bổ sung validation accuracy, best epoch và nguồn bằng chứng chọn checkpoint |

## 1. Phạm vi bàn giao

Thư mục này chứa tài liệu và [manifest.json](manifest.json), trỏ tới checkpoint trong run `20261030T052121Z_cifar10_scheduled64_b895bfc1`. Run được đối chiếu với [registry](../outputs/selected_run.json) khi lập manifest; đổi registry sau này không tự cập nhật bản kê này.

Mỗi model dùng `finetune/best.pt`: checkpoint được chọn theo validation trong trial fine-tune đã chọn. “Best” có phạm vi trong thực nghiệm này, chưa có nghĩa tốt nhất qua mọi run. Test metrics chỉ mô tả kết quả đánh giá sau lựa chọn.

| Model | Checkpoint trong run | Validation accuracy | Best epoch | Test accuracy | Test Macro F1 | Kích thước trọng số |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | `resnet18/finetune/best.pt` | 84,70% | 16 | 84,14% | 0,840593 | 44.795.332 byte ≈ 44,80 MB / 42,72 MiB |
| DenseNet121 | `densenet121/finetune/best.pt` | 91,86% | 18 | 91,80% | 0,917978 | 28.384.691 byte ≈ 28,38 MB / 27,07 MiB |

Nguồn metric: [ResNet18](../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/metrics.json) và [DenseNet121](../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/finetune/metrics.json). Manifest giữ giá trị gốc, số mẫu test, kích thước byte và SHA256; bảng trên làm tròn để đọc.

Validation accuracy và best epoch được đối chiếu từ training summary của [ResNet18](../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/training_summary.json) và [DenseNet121](../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/finetune/training_summary.json). Best epoch là epoch của checkpoint được chọn trong stage fine-tune, không phải số epoch thực chạy; manifest lưu hai giá trị cùng đường dẫn summary trong `selection_evidence`.

## 2. Vì sao không chép checkpoint vào thư mục này?

Hai checkpoint có tổng dung lượng khoảng 73,18 MB. Theo phạm vi bàn giao này, Git chỉ nhận manifest và README; trọng số tiếp tục nằm trong thư mục run hiện có.

Đưa các phiên bản checkpoint vào Git làm tăng dung lượng đối tượng và lịch sử repository. Clone có thể chậm hơn và cần thêm dung lượng đĩa; mỗi lần cập nhật trọng số có thể tiếp tục làm lịch sử lớn lên. Xóa tệp ở một commit mới không tự xóa phiên bản đã lưu trong lịch sử.

Đổi lại, clone repository chưa đủ để load model nếu thiếu checkpoint. Người nhận cần lấy hai tệp trọng số qua kênh bàn giao riêng, giữ đúng đường dẫn và đối chiếu SHA256 trong manifest. Hiện chưa có URL tải công khai được xác minh; manifest không cung cấp hoặc tự tải trọng số.

## 3. Chuẩn bị checkpoint và load bằng hai dòng

Đặt trọng số dưới `labs/lab02/outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/<model>/finetune/best.pt`. Chạy ví dụ từ thư mục gốc repository, trong môi trường đã cài [dependencies của Lab02](../README.md#31-chuẩn-bị-môi-trường-local).

```python
from labs.lab02.src.inference import load_model
session = load_model("labs/lab02/outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/finetune/best.pt", device="cpu")
```

Để load ResNet18, thay `densenet121` bằng `resnet18` trong đường dẫn. Hàm trả `LoadedModel`, gồm model ở chế độ đánh giá, thiết bị và transform đọc từ metadata checkpoint; không cần tự dựng lại classifier hoặc dropout.

API hiện tại nhận đường dẫn checkpoint, không đọc manifest và không tự kiểm tra SHA256 theo manifest. Thiếu tệp sẽ gây lỗi đọc checkpoint; CUDA không khả dụng sẽ báo lỗi nếu yêu cầu `device="cuda"`.

Xem [load_model()](../src/inference.py#L18) và [predict_image()](../src/inference.py#L42) để phân biệt nạp model với dự đoán một ảnh.
