# Lab02 · Module models
## 1. Thông tin module

- **Tên:** `src/models` — tạo backbone, thay classifier và cấu hình phần được học.
- **Source:** [models.py](models.py), public API ở [__init__.py](__init__.py).

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-07 | 0.1.0 | Thường FEAI, cùng Codex | Tạo tài liệu hướng dẫn đọc module |
| 2026-11-07 | 0.1.1 | Thường FEAI, cùng Codex | Ghi rõ đầu vào, đầu ra và bằng chứng của từng giai đoạn |
| 2026-11-11 | 0.1.2 | Thường FEAI, cùng Codex | Sắp xếp 8 mục; bổ sung hợp đồng model, cơ chế, tham số theo stage và giới hạn so sánh |
| 2026-11-11 | 0.1.3 | Thường FEAI, cùng Codex | Rút gọn; dùng bảng so sánh, giải thích thay classifier, freeze và tensor bằng số liệu Lab02 |
| 2026-11-11 | 0.1.4 | Thường FEAI, cùng Codex | Viết lại theo cách dẫn giải Lab01; tách kiến trúc, khởi tạo, stage, I/O; phục hồi bảng source 4 cột |
| 2026-11-11 | 0.1.5 | Thường FEAI, cùng Codex | Bổ sung khoảng trống trước danh sách để Markdown hiển thị ổn định |

## 2. Mục đích

Module models xây dựng **ResNet18 và DenseNet121** cho bài toán phân loại **10 lớp CIFAR-10**, phục vụ hai nội dung khảo sát:

1. **So sánh kiến trúc:** sử dụng ResNet18 với **11.181.642 tham số** và DenseNet121 với **6.964.106 tham số** để đối chiếu chất lượng phân loại và chi phí huấn luyện.
2. **Khảo sát khả năng thích nghi:** học classifier mới bằng head-only, sau đó fine-tune một phần backbone để đánh giá việc điều chỉnh đặc trưng pretrained ảnh hưởng thế nào đến kết quả.

Module cung cấp:

- [`build_model()`](models.py#L12): nạp backbone và thay classifier 1.000 lớp ImageNet bằng classifier 10 lớp CIFAR-10.
- [`features()`](models.py#L34): trích vector **512 chiều** với ResNet18 hoặc **1.024 chiều** với DenseNet121.
- [`configure_trainable()`](models.py#L63): xác định tham số được học ở head-only và fine-tune.
- [`parameter_counts()`](models.py#L90): đếm tổng tham số và số tham số được phép học.

## 3. Khởi tạo mô hình

### 3.1. Kiến trúc mô hình

Cả hai model xử lý ảnh qua **backbone → pooling → classifier**. ResNet18 cộng đặc trưng qua nhánh tắt, còn DenseNet121 ghép đặc trưng theo chiều kênh. Sơ đồ dưới dùng dropout **0,2** của run scheduled:

```text
ResNet18
RGB → stem → layer1 → layer2 → layer3 → layer4
    → pooling → vector 512 → Dropout(0.2) → Linear(512,10) → 10 logits

DenseNet121
RGB → stem → denseblock1 → transition1 → denseblock2 → transition2
    → denseblock3 → transition3 → denseblock4 → norm5 → ReLU
    → pooling → vector 1024 → Dropout(0.2) → Linear(1024,10) → 10 logits
```

| Thành phần | ResNet18 | DenseNet121 |
| --- | --- | --- |
| Cấu trúc chính | 4 nhóm, mỗi nhóm 2 residual block | 4 dense block gồm 6, 12, 24, 16 lớp |
| Kết hợp đặc trưng | Cộng kết quả biến đổi với đầu vào qua nhánh tắt; chiếu kích thước nếu cần | Ghép đặc trưng các lớp trước theo chiều kênh |
| Giảm kích thước | Các lớp giảm mẫu trong backbone | 3 transition giữa các dense block |
| Vector trước classifier | 512 giá trị/ảnh | 1.024 giá trị/ảnh |
| Classifier mới | `Linear(512,10)` | `Linear(1024,10)` |
| Tổng tham số sau thay classifier | 11.181.642 | 6.964.106 |

DenseNet121 có vector cuối lớn hơn nhưng tổng tham số nhỏ hơn ResNet18. Hai backbone tổ chức phép tính khác nhau, nên cần đo thời gian thực tế để so sánh chi phí, không suy ra tốc độ chỉ từ số tham số.

### 3.2. Weights và điều chỉnh classifier

Hai model dùng weights **`IMAGENET1K_V1`** tương ứng từ Torchvision: `ResNet18_Weights` và `DenseNet121_Weights`. Backbone giữ đặc trưng đã học trên ImageNet; classifier được khởi tạo mới để học **10 nhãn CIFAR-10**.

Code trọng tâm với ResNet18:
```python
model.fc = nn.Linear(model.fc.in_features, 10)
```

Dòng này thay classifier cũ bằng lớp nhận **512 giá trị** từ backbone và trả **10 điểm dự đoán**. DenseNet121 thay `model.classifier`, nhận **1.024 giá trị**. Nếu không thay classifier, model vẫn trả **1.000 đầu ra** của ImageNet; đặt sai chiều đầu vào Linear gây lỗi nhân ma trận.

Run scheduled thêm **dropout 0,2** trước Linear: khi train, ngẫu nhiên đặt khoảng **20%** phần tử vector về 0 để giảm phụ thuộc vào một số đặc trưng; khi đánh giá, dropout tắt. Hàm `build_model()` mặc định dropout bằng **0**.

### 3.3. Head-only và fine-tune

Head-only giữ nguyên backbone và chỉ học classifier mới. Fine-tune bắt đầu từ model head-only đã chọn, sau đó cho phép một phần backbone học tiếp cùng classifier.

| Trạng thái | ResNet18 | DenseNet121 |
| --- | ---: | ---: |
| Head-only: số tham số được học | 5.130 | 10.250 |
| Fine-tune scheduled: số tham số được học | 4.725.770 | 2.921.034 |
| Fine-tune scheduled: số tham số đóng băng | 6.455.872 | 4.043.072 |

Classifier ResNet18 có **512 × 10 + 10 = 5.130 tham số**; DenseNet121 có **1.024 × 10 + 10 = 10.250 tham số**. Dropout không thêm tham số; đóng băng không làm thay đổi tổng số tham số model.

| Model | Các lớp backbone được mở trong `scheduled_tail`, ngoài classifier |
| --- | --- |
| ResNet18 | `layer4.1` |
| DenseNet121 | Trong `features`: `denseblock3.denselayer{11,12,18..24}`, `transition3`, `denseblock4.denselayer{1..4,13..16}`, `norm5` |

Danh sách này cố định suốt stage fine-tune; các lớp còn lại đóng băng. BatchNorm giữ thống kê trung bình/phương sai đã lưu, nhưng hệ số nhân và độ lệch thuộc phần được mở vẫn có thể học.
Số tham số được đối chiếu config của run **`20261030T052121Z_cifar10_scheduled64_b895bfc1`**: [ResNet18](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/config.json) · [DenseNet121](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/finetune/config.json).

### 3.4. Đầu vào và đầu ra

Run scheduled dùng **45.000 ảnh train, 5.000 validation và 10.000 test**. Ảnh RGB gốc **32×32** được resize thành **64×64** và chuẩn hóa trước khi đưa vào model.

| Model | Input một batch | Feature trước classifier | Output |
| --- | --- | --- | --- |
| ResNet18 | `[128,3,64,64]` | `[128,512]` | `[128,10]` logits |
| DenseNet121 | `[128,3,64,64]` | `[128,1024]` | `[128,10]` logits |

Ảnh và output dùng `float32` trong đường chạy mặc định; batch cuối có thể ít hơn **128 ảnh**. Nhãn là số nguyên **0–9** (`int64 [B]`), được dùng riêng để tính loss.
- **Khi train:** đưa logits trực tiếp vào `CrossEntropyLoss`.
- **Khi dự đoán:** lấy `argmax` để chọn lớp có điểm cao nhất; dùng softmax khi cần xác suất theo lớp.

### 3.5. File mã nguồn và checkpoint

| Tệp mã nguồn | Ký hiệu / Hàm chính (Ctrl+Click) | Dòng code | Việc đảm nhiệm |
| :----------- | :------------------------------- | :-------- | :------------- |
| [models.py](models.py) | [`SPECS`](models.py#L6), [`build_model()`](models.py#L12) | [6–27](models.py#L6) | Chọn weights, tạo model, thay classifier 10 lớp và bọc dropout. |
| [models.py](models.py) | [`classifier()`](models.py#L30), [`features()`](models.py#L34) | [30–40](models.py#L30) | Lấy classifier hoặc vector 512/1.024 chiều trước classifier. |
| [models.py](models.py) | [`trainable_blocks()`](models.py#L43), [`configure_trainable()`](models.py#L63) | [43–78](models.py#L43) | Xác định lớp mở và bật/tắt gradient theo stage. |
| [models.py](models.py) | [`training_mode()`](models.py#L81), [`parameter_counts()`](models.py#L90) | [81–92](models.py#L81) | Giữ thống kê BatchNorm; đếm tổng tham số và tham số được học. |
| [models.py](models.py) | [`last_block()`](models.py#L95), [`CachedTail`](models.py#L99) | [95–121](models.py#L95) | Chạy phần cuối từ đặc trưng trung gian đã cache; scheduled không dùng prefix cache. |
| [__init__.py](__init__.py) | [Import và `__all__`](__init__.py#L2) | [2–26](__init__.py#L2) | Công khai API cho training và inference. |

**Điểm gọi trong code:**
- Tạo model/head-only: [`run_model()`](../training/experiment.py#L337); cấu hình fine-tune: [`search_finetune()`](../training/experiment.py#L259).
- Nạp model suy luận: [`load_model()`](../inference.py#L18).

**Checkpoint và kiến trúc:** dưới `outputs/runs/<run_id>/<model>/`:
- `architecture.txt`, `parameters.csv`: cây lớp và số tham số; cờ trainable trong CSV được xuất ở head-only.
- `head_selected/best.pt`: weights model đầy đủ sau head-only; `finetune/best.pt`: weights model đầy đủ được chọn sau fine-tune.

[Registry](../../outputs/selected_run.json) chỉ run đang chọn. Khôi phục scheduled cần đúng kiến trúc và dropout **0,2**; xem [hướng dẫn artifact](../../README.md#24-artifact-và-tái-lập). Lịch sử LR/batch và cơ chế resume nằm ở [training](../training/README.md).

## 4. Thảo luận và quyết định kỹ thuật

### 4.1. Vì sao dùng model pretrained?

Lab02 giữ backbone đã học trên ImageNet và thay classifier để tận dụng đặc trưng ảnh có sẵn cho CIFAR-10. Thiết kế này cho phép khảo sát hai mức thích nghi: chỉ học đầu phân loại và điều chỉnh thêm backbone. Run scheduled chưa so sánh trực tiếp với huấn luyện từ đầu, nên chưa kết luận được mức lợi ích riêng của pretrained.

### 4.2. Vì sao so sánh ResNet18 và DenseNet121?

Hai model đại diện cho **cộng nhánh residual** và **ghép kênh dense**. Lab02 đối chiếu chất lượng phân loại cùng chi phí thực thi của hai backbone. Chúng khác cả kiến trúc và số tham số, nên kết quả là so sánh tổng thể, không cô lập riêng tác động của phép cộng hay phép ghép.

### 4.3. Vì sao head-only trước rồi mới fine-tune?

Head-only tạo baseline từ backbone cố định với **5.130/10.250 tham số được học**. Fine-tune tiếp tục từ baseline đó, mở thêm backbone để đặc trưng thích nghi với CIFAR-10. Mở thêm lớp tăng khả năng điều chỉnh nhưng cũng tăng chi phí và nguy cơ overfit; kết quả phải được đối chiếu với baseline.

### 4.4. Giới hạn khi diễn giải kết quả

Run scheduled dùng dropout **0,2**, giữ thống kê BatchNorm cố định và chỉ mở các lớp đã liệt kê. Đây là cấu hình đã thực nghiệm, chưa chứng minh tối ưu. Head-only và fine-tune còn khác augmentation và ngân sách epoch, nên chênh lệch kết quả không thể quy hoàn toàn cho việc mở backbone.
