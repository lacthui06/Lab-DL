# Lab 02 — Đề bài gốc và phạm vi triển khai

Nguồn đối chiếu: ảnh `Screenshot 2026-10-25 201146.png`, slide trang 23 do người dùng cung cấp ngày 26/10/2026. Nội dung đề là dữ liệu đầu vào của bài tập; các lựa chọn triển khai nằm trong walkthrough.

## Practice 2 - Hands-on practice with pre-trained neural network architectures

### Exercises

1. Experiment with different pre-trained models (VGG, DenseNet, etc.).
2. Adjust the hyperparameters (learning rate, batch size, etc.).
3. Use TensorBoard to monitor the training process.

### Objective

- Become familiar with loading and using pre-trained models from torchvision.models.
- Understand how to adapt and fine-tune pre-trained models for specific tasks.

### Steps (chép lại và rút gọn từ ảnh)

1. Environment Setup: ensure PyTorch and torchvision are installed; if not, install using `pip install torch torchvision`.
2. Load a Pre-trained Model: torchvision.models provides models like ResNet, VGG, DenseNet; `pretrained=True` loads ImageNet weights.
3. Explore the Model Architecture: print the model to examine its structure; identify layers and parameters.
4. Adapt the Model for a Specific Task: replace the final classification layer to match the number of classes; freeze earlier layers for transfer learning or unfreeze final layers to fine-tune.
5. Prepare the Data: load the dataset; preprocess with torchvision.transforms; create batches with torch.utils.data.DataLoader.
6. Train the Model: define the loss and optimizer; write the training loop.
7. Evaluate the Model: evaluate performance on a test dataset.

## Phạm vi đã chọn trong source

CIFAR-10, ResNet18 và DenseNet121 là lựa chọn của bài làm. Ảnh đề không bắt buộc CIFAR-10, độ phân giải 224 hay phải chạy đủ VGG/ResNet/DenseNet. Source dùng API `weights=...` để tải weights ImageNet thay cho cờ `pretrained=True` cũ. Đặc tả, kiểm tra đáp ứng đề và giới hạn bằng chứng nằm trong [walkthrough_lab_02.md](walkthrough_lab_02.md).

## Thay đổi được xác nhận ngày 30/10/2026 — cấu hình scheduled

Đây là yêu cầu bổ sung của bài làm, không phải nội dung đề gốc hay kết quả đã chạy. Source hỗ trợ công thức này qua `config/scheduled.json`; full training do người dùng tự khởi chạy. Không gán kết quả của run cũ cho cấu hình dưới đây. Sau khi run hoàn tất, `scripts/execute_notebook.py` dựng phần Modeling theo artifact của run mới, lưu bản notebook trước và sau thực thi theo hash nội dung.

### 1. Dữ liệu và các cơ chế giữ lại

- CIFAR-10, RGB 64×64, train/validation/test 45.000/5.000/10.000; giữ split đã lưu và seed 42.
- Pretrained ImageNet; thay classifier thành 10 đầu ra. Resize → ToTensor → Normalize theo mean/std ImageNet.
- Head: dùng đặc trưng cache từ preprocessing cố định, không augmentation.
- Fine-tune train: RandomCrop(32, padding=4, padding_mode='reflect') → RandomHorizontalFlip(p=0.5) → preprocessing cố định. Validation/test không augmentation.
- Dropout p=0.2 ngay trước Linear của classifier; bật khi train, tắt khi validation/test.
- Optimizer AdamW, weight decay 1e-4; loss cross-entropy.
- BatchNorm: giữ cố định running_mean/running_var. Weight/bias của BatchNorm thuộc phần được mở vẫn được học; phần freeze không cập nhật weight/bias.

### 2. Phạm vi freeze/unfreeze và giới hạn tham số

Head-only freeze toàn bộ backbone, chỉ học classifier. Fine-tune mở đúng danh sách bên dưới; tất cả tham số ngoài danh sách giữ freeze. Giới hạn trainable không vượt 50% tổng số tham số của từng mô hình sau khi thay head; không phải giới hạn 50% số layer hoặc giới hạn cho từng module riêng lẻ. Running statistics BatchNorm không tính là tham số học.

| Model | Phần được học khi fine-tune | Phần freeze |
|---|---|---|
| ResNet18 | Toàn bộ `layer4.1` và `fc` | `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4.0` |
| DenseNet121 | `denseblock3.denselayer11–12`, `denseblock3.denselayer18–24`; toàn bộ `transition3` có tham số; `denseblock4.denselayer1–4`, `denseblock4.denselayer13–16`; `norm5`; `classifier` | Các phần trước block3; block3 layer 1–10, 13–17; block4 layer 5–12 |

Với DenseNet, tên đầy đủ trong model có tiền tố `features.` trừ `classifier`. Các khoảng layer gồm cả hai đầu. Block3 có 24 dense layer, block4 có 16. ResNet `layer4.1` là BasicBlock thứ hai của layer4, gồm cả hai convolution và hai BatchNorm; yêu cầu mở các phần đầu/giữa của ResNet trước đó đã được thay bằng lựa chọn này.

| Phần mở của DenseNet121 | Số tham số |
|---|---:|
| Block3: layer 11–12, 18–24 | 1.303.360 |
| Transition3: BatchNorm weight/bias + convolution | 526.336 |
| Block4: layer 1–4, 13–16 | 1.079.040 |
| Norm5 weight/bias | 2.048 |
| Classifier 10 lớp | 10.250 |
| Tổng trainable | **2.921.034 / 6.964.106 = 41,94%** |

ResNet18 mở `layer4.1` (4.720.640) + `fc` (5.130), tổng **4.725.770 / 11.181.642 = 42,26%**. Trước train phải đếm trực tiếp tham số thực tế, đối chiếu danh sách và dừng với lỗi rõ ràng nếu vượt 50% hoặc có module không tồn tại.

`transition3` nối block3 với block4: BatchNorm → ReLU → Conv 1×1 (1.024 → 512 kênh) → AveragePool 2×2. BatchNorm có 2.048 tham số, convolution có 524.288; ReLU/pooling không có tham số học. Mở toàn bộ transition3 ở đây là mở weight/bias BatchNorm và trọng số convolution, vẫn giữ running statistics cố định.

### 3. Thay Grid Search bằng một công thức có LR schedule

Không chạy tích Descartes LR × batch và không tìm nhiều candidate cho cấu hình mới. Mỗi model chạy một giai đoạn head rồi một giai đoạn fine-tune; fine-tune khởi tạo từ best head checkpoint của chính model đó. Batch size **128 cố định** trong cả hai giai đoạn.

| Thiết lập | Head-only | Fine-tune |
|---|---|---|
| Ngân sách | 12 epoch | Tối đa 20 epoch |
| LR đầu | Head 1e-3 | Backbone được mở 1e-4; head 1e-3 |
| LR cuối lịch | Head 1e-4 | Backbone 1e-5; head 1e-4 |
| Scheduler | Cosine giảm còn 10% LR đầu | Cosine giảm mỗi nhóm còn 10% LR đầu |
| Early stopping | Không | Validation loss; min_epochs=10, patience=7, min_delta=1e-4 |

Lịch cosine không restart, không warmup. Gọi bước scheduler sau mỗi epoch; ghi LR thực sự dùng trước khi bước scheduler. Với LR đầu của mỗi nhóm là `base_lr`, dùng hệ số chung `f(t) = 0.1 + 0.9 × (1 + cos(pi × t / T)) / 2`, trong đó t là số epoch đã hoàn tất và T là ngân sách của giai đoạn (12 hoặc 20). LR = base_lr × f(t). Như vậy ở t=0 dùng LR đầu; t=T/2 còn 55%; t=T đạt 10%. LR tại t=T là mức sau bước scheduler cuối, không gọi nhầm là LR đã dùng để train epoch cuối. Nếu dừng sớm thì lịch chưa nhất thiết đạt mức cuối.

Không dùng một giá trị eta_min tuyệt đối chung làm hai nhóm fine-tune mất tỷ lệ LR 10:1. Khi resume phải phục hồi optimizer, scheduler, epoch và trạng thái early stopping để lịch tiếp tục đúng.

Early stopping theo dõi loss từ epoch đầu: cải thiện khi loss mới thấp hơn best monitored loss ít nhất theo điều kiện `new_loss < best_loss - 1e-4`; nếu không thì tăng bộ đếm stale. Chỉ cho phép dừng từ epoch 10 khi stale >= 7. Không đợi đến epoch 10 mới bắt đầu đếm. Best checkpoint chọn độc lập bằng validation accuracy cao nhất, hòa ưu tiên validation loss thấp hơn, tiếp tục hòa giữ epoch sớm hơn.

### 4. Chi phí backward và giới hạn diễn giải

ResNet giữ phần đầu freeze liên tục, chỉ cần xây đồ thị gradient cho cụm cuối được mở. DenseNet **giữ yêu cầu mở xen kẽ block3/block4**, không chuyển sang chỉ mở toàn bộ block4. Layer được mở sớm nhất là block3 layer11; backward phải truyền về tới layer này qua các phép toán phía sau, kể cả layer có trọng số freeze. Block3 layer1–10 được freeze. Không dùng no_grad/detach giữa các phần được mở làm mất gradient về layer được học.

Có thể chạy phần prefix hoàn toàn freeze trước block3 của DenseNet, hoặc trước layer4.1 của ResNet, mà không ghi đồ thị gradient. Fine-tune vẫn lấy ảnh augmentation mới mỗi lần; không chuyển sang cache cố định làm mất augmentation. Dưới 50% tham số không bảo đảm dưới 50% FLOPs, thời gian hay bộ nhớ; cần đo trên run mới.

Learning curves cũ cho thấy fine-tune loss còn giảm ở cuối 10 epoch, nhưng không chứng minh lịch mới tối ưu. LR fine-tune mới cao gấp 10 lần run cũ; có rủi ro dao động hoặc làm thay đổi mạnh đặc trưng pretrained. Đồng thời đổi freeze và scheduler nên không quy chênh lệch kết quả cho riêng một cơ chế.

### 5. Nghiệm thu khi triển khai sau bước đặc tả này

1. Kiểm tra chính xác tên module được mở/freeze, số tham số và giới hạn 50% cho cả hai model; BatchNorm running statistics không đổi trong train.
2. Kiểm tra batch=128, head 12 epoch, fine-tune tối đa 20; LR hai nhóm theo đúng tỷ lệ, các mốc cosine và resume không lệch lịch.
3. Kiểm tra gradient tới các layer DenseNet được mở ở đầu block3, không có gradient trọng số ở phần freeze; train/eval bật tắt dropout và augmentation đúng phạm vi.
4. Kiểm tra early stopping và checkpoint selection theo hai tiêu chí độc lập đã mô tả; không dùng test để chọn lịch, cấu hình hay epoch.
5. Chạy kiểm tra nhỏ trước huấn luyện đầy đủ; ghi run mới riêng, giữ nguyên checkpoint/log/kết quả lịch sử. Không tự chạy full training CPU.
6. Cập nhật Modeling để phân biệt cấu hình mới chưa chạy với kết quả đã đo; chỉ thay bảng kết quả sau khi có artifact mới. EDA giữ thống kê dữ liệu gốc. Đồng bộ quyết định và kết quả vào walkthrough khi triển khai/đánh giá, không trình bày kế hoạch này thành kết quả thực nghiệm.

### 6. Lưu kết quả các lần chạy và output notebook

Mỗi run mới có thư mục riêng trong `outputs/runs/<run_id>/`, giữ nguyên config, checkpoint, history, prediction và metric của run trước. Figure trình bày tách theo `outputs/review/<run_id>/`. Khi đổi run được xem rồi thực thi lại cùng notebook, output của cell được thay bằng kết quả run đang chọn; một file ipynb không tự giữ lịch sử mọi phiên bản output. Lệnh train lưu bản notebook đang có vào `notebook_snapshots/<content_hash>/` của run được chọn trước khi tạo kết quả mới. Lệnh execute notebook tiếp tục lưu bản trước và sau thực thi trong thư mục run đang xem; nội dung notebook có thể đang thuộc run trước nên cần đọc run_id bên trong bản snapshot, không suy ra chỉ từ thư mục chứa. Không ghi đè snapshot đã có. Phải chạy lệnh train mới trước khi đổi registry bằng tay để bản notebook cũ được lưu gắn với run cũ.
