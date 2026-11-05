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

## Các quyết định đã ghi nhận và lý do chọn kiến trúc

Đây là bản tổng hợp từ source, cấu hình run scheduled đã chọn và các quyết định đã ghi ở tài liệu này; không phải trích nguyên văn lịch sử chat. Không coi mọi lựa chọn kỹ thuật là yêu cầu của giảng viên. Khi có khác biệt, config/metadata của từng run mô tả điều kiện thực thi của run đó.

| Chủ đề | Quyết định hiện hành và lý do | Giới hạn hoặc phương án đã thay thế |
|---|---|---|
| Dataset/split | CIFAR-10 RGB; 45.000 train, 5.000 validation, 10.000 test; lưu indices, seed 42 | Test không chọn cấu hình; một seed chưa đánh giá độ ổn định |
| Backbone | ResNet18 và DenseNet121, weights ImageNet1K_V1 tường minh; thay classifier 10 lớp | Đây là lựa chọn bài làm; đề không bắt buộc chạy VGG hoặc chứng minh hai mạng tối ưu |
| Preprocessing | Run scheduled resize 64×64, normalize theo pretrained weights; giữ RGB | 64 là điều kiện thực nghiệm tiết kiệm tính toán, không đồng nhất benchmark chuẩn 224; feature thủ công ở EDA không nối vào classifier |
| Hai giai đoạn | Học classifier từ pooled features trước; fine-tune từ best head checkpoint | Head cache không đồng nghĩa mọi fine-tune đều dùng cache |
| Freeze/BatchNorm | Scheduled mở các module đã liệt kê bên dưới, dưới 50% tham số; BN running statistics luôn cố định | Frozen weights không có nghĩa module không thực hiện forward; BN affine ở phần mở vẫn học |
| Regularization | Dropout 0,2; crop/flip chỉ fine-tune train; validation/test deterministic | Không ghép cache cố định với augmentation mới mỗi lượt |
| Optimizer/scheduler | Scheduled: AdamW, head 1e-3, fine backbone 1e-4/head 1e-3; cosine ratio xuống 0,1 | Run lịch sử có search/cấu hình khác; không gọi scheduled là optimizer ablation |
| Selection | Checkpoint: val accuracy, hòa xét val loss rồi epoch sớm; stopping theo val loss | Hai tiêu chí khác nhau; không chọn theo test |
| Monitoring/save-load | History, TensorBoard artifacts, checkpoint metadata và roundtrip đã lưu | Xem notebook không khởi động server hoặc thực hiện lại full-test sau load |
| Benchmark bổ sung | Training batch 1 riêng, checkpoint bất biến, process riêng; giữ inference một ảnh đã có | Số CPU training và T4 inference có phạm vi/thiết bị khác nhau, không ghép thành cùng phép đo |

### Vì sao ResNet18 và DenseNet121?

ResNet18 dùng residual connection: block học phần hiệu chỉnh rồi cộng với nhánh tắt. Mạng có bốn stage, mỗi stage hai BasicBlock, thuận tiện khảo sát thay head và mở các block cuối. DenseNet121 nối các feature map trước vào đầu vào layer sau trong cùng dense block; điều này cho phép tái sử dụng đặc trưng với số tham số khác ResNet. Cặp này phục vụ mục tiêu học hai kiểu kết nối và fine-tuning, không phải kết quả tìm kiếm kiến trúc tối ưu.

DenseNet ít tham số hơn trong cấu hình này nhưng concatenation, activation và đường truyền gradient có thể làm thời gian/bộ nhớ khác kỳ vọng. So sánh hiện tại thay đồng thời kiến trúc và recipe unfreeze; không quy toàn bộ chênh lệch cho residual hay dense connection. Resize 64 vẫn dùng stem ImageNet nguyên bản, không tự thay stem thành CNN riêng cho CIFAR.

### Kiến trúc theo layer/module của model thực tế

Các mô tả sau là cấu trúc của `build_model(..., dropout=0.2)` cho run scheduled; convolution mặc định không bias trừ classifier. Tên trong ngoặc là tên module để tra source và checkpoint. Bảng tensor tham số và toàn bộ `str(model)` được notebook mở rộng từ `parameters.csv` và `architecture.txt` của đúng run, không dựng một sơ đồ khác rồi gọi là kiến trúc đã chạy.

**ResNet18:**

| Thứ tự | Layer/module | Kênh và phép toán |
|---|---|---|
| 1 | `conv1`, `bn1`, `relu`, `maxpool` | Conv 7×7, 3→64, stride 2/pad 3; BN64; ReLU; MaxPool3 stride 2/pad 1 |
| 2 | `layer1.0`, `layer1.1` | Hai BasicBlock 64→64; mỗi block Conv3→BN→ReLU→Conv3→BN→cộng identity→ReLU |
| 3 | `layer2.0`, `layer2.1` | 64→128 rồi 128→128; block đầu stride 2 và nhánh tắt Conv1 stride 2 + BN |
| 4 | `layer3.0`, `layer3.1` | 128→256 rồi 256→256; block đầu downsample như trên |
| 5 | `layer4.0`, `layer4.1` | 256→512 rồi 512→512; block đầu downsample như trên |
| 6 | `avgpool`, flatten | AdaptiveAvgPool(1,1), mỗi ảnh thành vector 512 |
| 7 | `fc.0`, `fc.1` | Dropout(0,2) → Linear(512,10), bias=True; output logits |

Mỗi BasicBlock có các tên con `conv1`, `bn1`, `relu`, `conv2`, `bn2`; ba block đổi kênh có thêm `downsample.0` (Conv1) và `downsample.1` (BN). Conv3 dùng padding 1; stride 2 ở conv1 của block đầu stage 2–4. Đầu vào 64×64 cho các mốc spatial 32→16→16→8→4→2→1 trước classifier. Scheduled chỉ mở `layer4.1` và `fc`; phần còn lại freeze.

**DenseNet121:**

| Thứ tự | Layer/module | Kênh và phép toán |
|---|---|---|
| 1 | `features.conv0/norm0/relu0/pool0` | Conv7 3→64 stride 2/pad 3; BN64; ReLU; MaxPool3 stride 2/pad 1 |
| 2 | `features.denseblock1` | 6 DenseLayer; growth rate 32: 64→256 |
| 3 | `features.transition1` | BN256→ReLU→Conv1 256→128→AvgPool2 stride 2 |
| 4 | `features.denseblock2` | 12 DenseLayer: 128→512 |
| 5 | `features.transition2` | BN512→ReLU→Conv1 512→256→AvgPool2 stride 2 |
| 6 | `features.denseblock3` | 24 DenseLayer: 256→1024 |
| 7 | `features.transition3` | BN1024→ReLU→Conv1 1024→512→AvgPool2 stride 2 |
| 8 | `features.denseblock4`, `features.norm5` | 16 DenseLayer: 512→1024; BN1024 |
| 9 | Functional ReLU, adaptive average pool, flatten | ReLU→AdaptiveAvgPool(1,1)→vector 1024; các phép functional không hiện như module riêng trong str(model) |
| 10 | `classifier.0`, `classifier.1` | Dropout(0,2)→Linear(1024,10), bias=True; output logits |

Mỗi `denselayerN` gồm `norm1`→`relu1`→`conv1` (1×1, C→128)→`norm2`→`relu2`→`conv2` (3×3, 128→32, padding 1). Đầu vào layer thứ N có C=C_đầu_block+32×(N−1); nối 32 kênh mới với các feature trước. Các khoảng `denselayer1..6/12/24/16` gồm mọi layer theo quy tắc này. DenseLayer mặc định không có dropout nội bộ; dropout 0,2 ở classifier. Với ảnh 64, spatial sau stem là 16, qua ba transition là 8→4→2; cuối cùng pool về 1. Danh sách mở xen kẽ scheduled và số tham số được giữ ở mục cấu hình bên dưới.

### Hợp đồng benchmark training batch size 1 được duyệt

- Chọn tường minh run và `finetune/best.pt` của hai model; cùng index thuộc selected train split, kiểm checksum split trong checkpoint. Không dùng test để đo training.
- Process riêng/model; batch 1, FP32, full-image forward. Giữ freeze, dropout và BN theo metadata; dùng preprocessing deterministic, không augmentation để đầu vào giống nhau. Đây là microbenchmark khác pipeline train có augmentation.
- Tạo optimizer mới với loại, weight decay và LR ban đầu của recipe; không phục hồi optimizer state hoặc scheduler, không gọi đây là tiếp tục training tại best epoch.
- Một bước đầu đo memory có optimizer allocation; warmup 3, timing 10 bước riêng; CUDA đồng bộ nếu chọn GPU. Timing gồm zero_grad/forward/loss/backward/step; không gồm IO, preprocessing, transfer, validation, load model hoặc ghi file.
- RSS baseline/peak/tăng thêm tính bằng MiB, lấy mẫu 1 ms nên có thể bỏ lỡ đỉnh; gồm runtime và model. VRAM allocated/reserved chỉ từ CUDA allocator, CPU ghi null. Batch 1 không biến RSS thành bộ nhớ riêng của ảnh.
- Checkpoint chỉ đọc, kết quả ở thư mục mới và từ chối ghi đè. Test phải kiểm freeze/BN, checkpoint bất biến, cấu hình sai, đơn vị và notebook không tự benchmark. Benchmark không thay accuracy, history hoặc thời gian train cũ.

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
