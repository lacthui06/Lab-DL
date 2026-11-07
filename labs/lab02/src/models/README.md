# Lab02 · Module models

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-07 | 0.1.0 | Thường FEAI, cùng Codex | Tạo tài liệu hướng dẫn đọc module |
| 2026-11-07 | 0.1.1 | Thường FEAI, cùng Codex | Ghi rõ đầu vào, đầu ra và bằng chứng của từng giai đoạn |

## 1. Kiến trúc mô hình

```text
ResNet18
  Ảnh RGB → stem → layer1 → layer2 → layer3 → layer4
          → average pooling → flatten (512) → fc → 10 logits

DenseNet121
  Ảnh RGB → stem → denseblock1 → transition1 → denseblock2
          → transition2 → denseblock3 → transition3 → denseblock4
          → norm5 → ReLU → average pooling → flatten (1024)
          → classifier → 10 logits
```

- **Stem** là nhóm lớp xử lý ảnh đầu tiên. **Backbone** biến ảnh thành đặc trưng. Pooling và flatten đưa đặc trưng về một vector cho mỗi ảnh, sau đó classifier tính điểm cho từng lớp.
- **ResNet18** có bốn nhóm residual block, mỗi nhóm có hai block. Trong mỗi block, nhánh tắt mang đầu vào đi vòng qua một số lớp rồi cộng với đầu ra của nhánh biến đổi. Nếu kích thước khác nhau, nhánh tắt được chiếu sang kích thước phù hợp trước khi cộng. [Mã ResNet của Torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- **DenseNet121** có bốn dense block với lần lượt 6, 12, 24 và 16 lớp. Trong một block, mỗi lớp nhận các đặc trưng trước đó ghép theo chiều kênh. Các transition giữa block giảm số kênh và kích thước không gian. [Mã DenseNet của Torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)

## 2. Mục đích và các file

Module tạo mô hình phân loại CIFAR-10, trích đặc trưng và cấu hình tham số được học. Phần training gọi các hàm của module để điều phối huấn luyện và ghi checkpoint. [Điểm gọi trong training](../training/experiment.py#L337)

| File | Trách nhiệm |
| --- | --- |
| [`models.py`](models.py) | `SPECS` ánh xạ tên model sang kiến trúc và weights. `build_model()` tạo model và thay classifier. `classifier()` lấy lớp phân loại, còn `features()` lấy vector ngay trước lớp đó. `trainable_blocks()`, `configure_trainable()`, `training_mode()` và `parameter_counts()` quản lý freeze, BatchNorm và kiểm tra số tham số. `last_block()` và `CachedTail` phục vụ đường chạy sử dụng đặc trưng trung gian đã cache. |
| [`__init__.py`](__init__.py) | Công khai các tên của module để phần khác có thể import trực tiếp, ví dụ `from src.models import SPECS`. |

### Mô hình ban đầu và dữ liệu

Với `pretrained=True`, Lab02 khởi tạo cả hai kiến trúc bằng weights `IMAGENET1K_V1` từ Torchvision. ImageNet-1K là bài toán phân loại ảnh 1.000 lớp. CIFAR-10 có 60.000 ảnh màu 32×32 thuộc 10 lớp, gồm 50.000 ảnh train gốc và 10.000 ảnh test. Hai bộ dữ liệu đều dùng ảnh màu để phân loại đối tượng, nên Lab02 có thể bắt đầu từ các đặc trưng ảnh đã học như cạnh và hình dạng. Đầu phân loại được thay để dự đoán theo tập nhãn CIFAR-10. [ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) · [DenseNet121 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html) · [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

### Input và output qua một ảnh

Dò một ảnh có nhãn `cat` theo cấu hình 64×64 của run đã chọn.

| Bước | Dữ liệu |
| --- | --- |
| Đọc ảnh | Ảnh RGB 32×32 và nhãn thật đi riêng để tính loss hoặc đánh giá. |
| Tiền xử lý | Resize 64×64, đổi sang tensor và chuẩn hóa bằng mean/std của weights. Thêm chiều batch cho input `[1, 3, 64, 64]` theo thứ tự ảnh, kênh, cao, rộng. |
| Backbone, pooling và flatten | Vector `[1, 512]` với ResNet18 hoặc `[1, 1024]` với DenseNet121. |
| Classifier | Logits `[1, 10]`. Mỗi vị trí là điểm thô của một lớp. |
| Dự đoán | Lấy chỉ số của logit lớn nhất rồi ánh xạ sang tên lớp. Nếu vị trí đó là `cat`, dự đoán khớp nhãn thật. |

Ví dụ mô tả shape bằng cách đọc source và kiến trúc. Giá trị logits cụ thể cần chạy forward trên ảnh thật. [Tiền xử lý L15 data.py](../datasets/data.py#L15) · [Dự đoán inference L42](../inference.py#L42)

## 3. Mô hình sau huấn luyện và nơi lưu

### Input, output và cách kiểm tra

`models.py` cung cấp đối tượng model và tensor cho caller. Training dùng các đối tượng đó để ghi thông tin kiến trúc, graph và checkpoint. Link artifact dưới đây dẫn tới ví dụ ResNet18 trong run đã chọn.

| Giai đoạn | Input | Output và thông tin cần có | Nơi lưu | Tiêu chí kiểm tra | Đoạn code và vị trí |
| --- | --- | --- | --- | --- | --- |
| Khởi tạo | Tên model, cờ pretrained, dropout | Model PyTorch với kiến trúc tương ứng và classifier 10 đầu ra | Trong bộ nhớ, trả về cho caller | Đúng kiến trúc và weights khởi tạo theo cấu hình. Classifier nhận vector 512 hoặc 1024 phần tử. | `model = builder(weights=weights if pretrained else None)` · [models.py, dòng 15–27](models.py#L15) |
| Trích đặc trưng và phân loại | Batch ảnh RGB đã chuẩn hóa | Vector `[N, 512]` hoặc `[N, 1024]` trước classifier, logits `[N, 10]` sau classifier | Tensor trong bộ nhớ. Training có thể ghi vector thành [train_features.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/train_features.pt). | Giữ nguyên số ảnh `N`. Chiều vector khớp classifier và mỗi ảnh có 10 logits. | `torch.flatten(…)` · [models.py, dòng 34–40](models.py#L34) · `torch.save(…, dest)` · [feature_cache.py, dòng 51–52](../training/feature_cache.py#L51) |
| Ghi thông tin model | Model sau khi cấu hình head-only và cấu hình tiền xử lý | Cây lớp, tên/shape/số phần tử từng tensor tham số và cờ `head_stage_trainable`. Metadata gồm weights, dataset, RGB, kích thước ảnh, transform, augmentation và dropout. | [architecture.txt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/architecture.txt) · [parameters.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/parameters.csv) · [preprocessing.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/preprocessing.json) | Tên lớp và shape khớp model. Cờ trainable mô tả lúc xuất bảng ở head-only. | `write_text(str(model), …)`, `pd.DataFrame(params).to_csv(…)`, `write_json(…)` · [experiment.py, dòng 345–356](../training/experiment.py#L345) |
| Ghi graph và ảnh mẫu | Model và tối đa 16 ảnh train, dùng một ảnh để tạo graph | Graph phép tính, văn bản kiến trúc và lưới ảnh input trong TensorBoard | [input_preview/](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/input_preview/) | Graph thuộc đúng model. Ảnh preview được đưa về miền hiển thị bằng mean/std trước khi vẽ. | `writer.add_image("input_images", …)`, `writer.add_graph(model.eval(), batch[0][:1].to(device))` · [experiment.py, dòng 357–363](../training/experiment.py#L357) |
| Đóng gói sau train | Model và checkpoint được chọn theo validation | `state_dict`, metadata và `best_epoch` cho model hoàn chỉnh | [head_selected/best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/head_selected/best.pt) · [finetune/best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/best.pt) | Nạp lại bằng đúng kiến trúc và dropout. Hai checkpoint khớp giai đoạn trong kết quả đánh giá. | `torch.save(…)` · [experiment.py, dòng 396–399](../training/experiment.py#L396) · `restored.load_state_dict(saved["state_dict"])` · [experiment.py, dòng 419–423](../training/experiment.py#L419) |

Registry đang chọn run `20261030T052121Z_cifar10_scheduled64_b895bfc1`, dùng CPU, ảnh 64×64 và seed 42. Cả hai kiến trúc có checkpoint cho head-only và fine-tune. Khi cập nhật tài liệu sau một lần train, ghi run ID, config, giai đoạn và đường dẫn checkpoint theo bằng chứng của run được chọn. Bảng metric đầy đủ nằm trong walkthrough. [Registry selected_run](../../outputs/selected_run.json) · [Walkthrough lab 02](../../walkthrough_lab_02.md)

| Artifact của run đã chọn | ResNet18 | DenseNet121 |
| --- | --- | --- |
| Checkpoint mô hình đầy đủ sau head-only | [head_selected/best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/head_selected/best.pt) | [head_selected/best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/head_selected/best.pt) |
| Checkpoint fine-tune được chọn | [finetune/best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/best.pt) | [finetune/best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/finetune/best.pt) |
| Ví dụ checkpoint cuối epoch để resume head trial | [head_trial_00/last.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/head_trial_00/last.pt) | [head_trial_00/last.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/head_trial_00/last.pt) |
| Căn cứ chọn trial | [selection.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/selection.json) | [selection.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/selection.json) |

Các link `.pt` trỏ đến checkpoint cục bộ. Khi chỉ clone repo, cần đồng bộ checkpoint của run trước khi mở các link này.

`fit()` ghi checkpoint trong quá trình train. `run_model()` đóng gói kết quả head-only đã chọn. `search_finetune()` đưa checkpoint tốt nhất của trial được chọn vào `finetune/`. [Hướng dẫn artifact](../../README.md#24-artifact-và-tái-lập) · [`fit()`](../training/experiment.py#L90) · [`run_model()`](../training/experiment.py#L337) · [`search_finetune()`](../training/experiment.py#L259)

## 4. Tiêu chí hoàn thành và lưu ý của Lab02

- **Đầu ra.** Cả hai mô hình dùng đúng weights đã cấu hình và trả 10 logits trên mỗi ảnh. Chiều đầu vào classifier phải khớp vector đặc trưng.
- **Head-only.** Giữ nguyên tham số backbone và chỉ huấn luyện classifier mới.
- **Fine-tune.** Khởi đầu từ kết quả head-only, mở classifier và các lớp backbone theo cấu hình. Với `scheduled_tail`, ResNet18 mở `layer4.1`, còn DenseNet121 mở một tập lớp định danh trong source.
- **BatchNorm và cache.** `training_mode()` cố định running statistics của BatchNorm. Các tham số affine trong phần được mở vẫn có thể học. `CachedTail` dựa vào phần đầu đã đóng băng, transform xác định và BatchNorm cố định. Cache có thể lỗi thời nếu phần tạo đặc trưng thay đổi trong khi train.
- **Tương thích checkpoint.** Dùng cùng kiến trúc và thiết lập dropout khi khôi phục model. Bật dropout bọc classifier trong `nn.Sequential` và thay đổi tên khóa của tham số classifier trong `state_dict`. [Logic freeze và BatchNorm](models.py#L43) · [Cache đặc trưng](../training/feature_cache.py) · [Nạp checkpoint](../inference.py#L18)

## 5. Cách đọc và thảo luận với AI

1. Đọc sơ đồ kiến trúc rồi mở `SPECS` và `build_model()`. Chỉ ra dòng nạp weights và dòng thay classifier.
2. Dò một ảnh từ input đến logits. Với mỗi ký hiệu chưa rõ, hỏi AI biến đó chứa gì, lấy từ đâu, thay đổi ở dòng nào, vì sao cần và nếu sửa hoặc bỏ thì kết quả thay đổi thế nào.
3. Đọc `configure_trainable()` và `training_mode()` theo thứ tự caller. Dò cả head-only lẫn fine-tune, sau đó lần theo training tới nơi ghi checkpoint.
4. Chốt cách hiểu trước khi sửa. Ghi rõ nhận định nào đến từ đọc source, dò tay hoặc chạy thực tế. Sau khi duyệt nội dung bổ sung, ghi ngày, người cập nhật và phiên bản vào lịch sử ở đầu file.

### Đoạn code trọng tâm

```python
model.fc = nn.Linear(model.fc.in_features, 10)
```

| Câu hỏi | Dò nhánh ResNet18 trong `build_model()` |
| --- | --- |
| Ban đầu có gì? | Builder đã tạo ResNet18 với `fc` gốc. Với `pretrained=True`, weights đã nạp trước khi thay `fc`. |
| Hàm nhận gì? | Tên model, cờ pretrained và dropout. Ví dụ `build_model("resnet18", pretrained=True, dropout=0.0)`. |
| Hàm trả gì? | Đối tượng mô hình đã gắn classifier 10 đầu ra. |
| Trạng thái đổi ở đâu? | Vế phải đọc số đầu vào từ lớp cũ. Phép gán sau đó thay `fc` trên model vừa tạo. |
| Vì sao dùng `in_features`? | Để lớp mới nhận đúng 512 giá trị từ backbone. Sai chiều sẽ gây lỗi kích thước khi nhân ma trận trong forward. |
| Thứ tự chạy? | Lấy builder và weights → tạo model → đọc chiều đầu vào của head cũ → tạo và gắn head mới → bọc dropout nếu có → trả model. |
| Bỏ dòng thay lớp thì sao? | Model vẫn trả 1.000 logits. Pipeline mong đợi 10 lớp và có thể dự đoán ra chỉ số nằm ngoài danh sách CIFAR-10. |

Hình dung backbone ghi 512 dấu hiệu về một bức ảnh. Classifier đọc đủ 512 dấu hiệu rồi chấm điểm 10 ô. Đổi từ phiếu 1.000 ô sang phiếu 10 ô vẫn cần đọc đủ 512 dấu hiệu. [Đoạn code thay classifier](models.py#L12)
