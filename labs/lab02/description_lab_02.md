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

## Yêu cầu cập nhật đã thảo luận: tuning DenseNet121

### Thiết kế thay thế mới nhất: notebook chuẩn hai model

Phần này thay các lựa chọn DenseNet-only/shared-separate và 5+5 trial cũ bên dưới trong thí nghiệm mới; các phần cũ là lịch sử thảo luận, không cộng dồn ngân sách. Giữ bốn điều kiện ResNet18 head/fine-tune và DenseNet121 head/fine-tune. Cả hai train head mới từ ImageNet, cache train/val feature một lần, train classifier 10 epoch, AdamW LR 0.001/weight decay 1e-4/cosine. Fine-tune ResNet mở layer4+fc; DenseNet mở block4+norm5+classifier, BN statistics cố định. Mỗi model 3 Optuna trial tìm LR chung log [1e-6,1e-3], weight decay cố định 1e-4, tối đa 10 epoch/trial, AdamW/cosine; không augmentation/dropout/pruning. Tổng 6 trial, tối đa 60 epoch fine-tune; head và extraction là chi phí riêng. Objective/checkpoint/early stopping theo validation loss, min_delta 0.001/patience 3/min_epochs 4. Cả hai model khóa winner trước test. Notebook 04 dùng src.standard_suite, hiển thị và lưu plot loss/accuracy/LR mỗi epoch, trial table sau mỗi trial, đánh giá từng model và tổng hợp bốn điều kiện cuối cùng. Config standard.json là nguồn triển khai. Target dưới 5 giờ chưa được đo; 3 trial không bảo đảm tối ưu.

Phạm vi toàn bài vẫn gồm ResNet18 và DenseNet121. Notebook Kaggle chạy ResNet18 theo recipe cũ trong completed.json (6 head trial, 9 fine candidate), chỉ đổi DenseNet sang Optuna. ResNet được chạy bổ sung riêng, không làm chạy lại DenseNet đã hoàn tất; tổng thời gian gồm cả hai pipeline. So sánh kết quả được ghi rõ khác tiêu chí chọn checkpoint giữa hai recipe.

Phần này ghi yêu cầu mở rộng riêng của bài làm, không thay thế đề gốc phía trên. Đây là đặc tả trước triển khai; chưa có kết quả Optuna mới và chưa được coi là đã thay đổi code training. Chỉ triển khai/chạy khi người dùng chốt bước tiếp theo.

### 1. Nhánh và ranh giới thay đổi

- Phát triển trên nhánh `lab02`. Không merge hoặc lấy thay đổi từ `thuong-lab02` trong đợt này.
- Tập trung vào fine-tuning DenseNet121 và cách tìm/chọn cấu hình. EDA, dữ liệu, split và các phần ngoài phạm vi này tiếp tục dùng thiết kế hiện hành; không mở rộng lại thí nghiệm ResNet18 chỉ vì thêm Optuna cho DenseNet.
- Không bắt buộc duy trì thiết kế/bảng so sánh candidate cũ trong phần đang làm mới. Khi có kết quả mới hợp lệ, thay phần hiển thị đang dùng trong notebook, bảng/hình kết quả và walkthrough tương ứng; không trộn history, checkpoint hoặc metrics cũ vào thí nghiệm mới.
- Quyền thay kết quả đang trình bày không đồng nghĩa xóa run gốc. Lần chạy mới vẫn có run ID riêng, log và provenance; không ghi đè artifact nguồn hoặc tuyên bố số liệu mới trước khi chạy.
- Chưa thêm augmentation hoặc dropout. Chỉ xem xét trong bước sau nếu kết quả mới còn overfit; không đưa code/recipe từ nhánh kia vào trước.

### 2. Phạm vi mở tầng

- Giới hạn việc mở backbone trong `features.denseblock4`; `denseblock1–3`, `transition3` và các phần backbone trước đó giữ đóng băng.
- DenseNet121 có 16 **dense layer** trong block4; mỗi dense layer chứa nhiều module/phép toán. Mở cả 16 tương đương mở toàn bộ block4, không phải chỉ 16 module PyTorch đơn lẻ.
- Mở cố định toàn bộ 16 dense layer của `denseblock4` (`denselayer1–16`) trong mọi trial; không tuning số layer mở.
- Giữ đúng chính sách `last_block` hiện có trong `src.models.configure_trainable`: train toàn bộ `features.denseblock4`, affine weights/bias của `features.norm5` và `classifier`. `norm5` nằm sau block4, không phải một trong 16 dense layer. Cả hai phương án LR dùng cùng chính sách này.
- BatchNorm trong các phần được mở vẫn train affine weights/bias, nhưng running mean/variance giữ cố định bằng `src.models.training_mode`. Các BatchNorm thuộc backbone đóng băng không cập nhật cả affine parameters lẫn running statistics.
- Các lựa chọn mở 4/8/16 layer từng được thảo luận nhưng không thuộc thiết kế đã chốt.

### 3. Optuna và hai phương án learning rate

- Bổ sung Optuna để chọn hyperparameter dựa trên validation thay cho chỉ dựa vào danh sách candidate fine-tune cố định.
- So sánh hai phương án: **LR chung** cho phần backbone được mở và classifier; **LR riêng** cho hai nhóm tham số. Trong phương án chung, hai nhóm phải thực sự có cùng LR; trong phương án riêng, ghi đủ LR từng nhóm và không mặc định head luôn gấp 10 lần backbone.
- Các tham số đóng băng không cập nhật. Head là classifier cuối; backbone được mở là phần trích xuất đặc trưng bên trong block4 được phép cập nhật.
- Dùng cùng split, checkpoint head khởi tạo, seed, batch size, preprocessing, phạm vi mở tầng và ngân sách tối đa cho hai phương án để so sánh công bằng. Reset về cùng điểm khởi tạo trước mỗi trial.
- Mỗi phương án được tuning LR phù hợp riêng; so sánh hai recipe tốt nhất không được mô tả thành bằng chứng nhân quả riêng của LR grouping nếu còn khác độ mở hoặc tham số khác.
- Giữ AdamW, tìm LR và weight decay; cố định mở toàn bộ block4. Phạm vi số học của search space phải được ghi rõ trước khi chạy, không tự kế thừa mọi candidate ngoài block4.

### 4. Scheduler và ngân sách

- Giữ `CosineAnnealingLR` trong từng trial cho cả hai phương án. Scheduler là nhóm phương pháp điều chỉnh LR; cosine là loại cụ thể được đề xuất ở đây, không phải một bước riêng ngoài scheduler.
- Optuna tìm LR ban đầu; cosine giảm LR từng nhóm trong quá trình train, step sau epoch. Không tìm thêm loại scheduler trong cùng thí nghiệm này.
- Ngân sách đã chốt: **5 trial cho LR chung + 5 trial cho LR riêng = tổng 10 trial**, tối đa **20 epoch/trial**, có early stopping. Đây là khảo sát ban đầu với ngân sách nhỏ, chưa đủ khẳng định ưu thế thống kê.
- Nếu sau này được yêu cầu mở rộng, tiếp tục study đã lưu lên 10 trial mỗi phương án; không tự tăng ngân sách hoặc chạy lại từ đầu.
- Phương án hiện được đề xuất để dễ đọc kết quả: dùng early stopping, chưa bật pruning ở lượt đầu. Nếu thêm pruning phải chốt warm-up/startup và điều kiện loại trial trước khi chạy; không ghi mặc định đã được đồng ý.

### 5. Thống nhất objective, checkpoint và early stopping

- Cả ba dùng **validation cross-entropy loss**, theo chiều minimize. Accuracy, macro F1 và các metric theo lớp vẫn lưu để đánh giá; không dùng test hoặc accuracy để âm thầm thay checkpoint thắng theo loss.
- Người dùng giao lựa chọn `patience` và `min_delta` cho người triển khai. Thiết lập được đề xuất: `patience=4`, `min_delta=0.001`, không early-stop trước epoch 6, tối đa 20 epoch.
- Khởi tạo mốc tốt nhất bằng loss hợp lệ đầu tiên. Một epoch chỉ được công nhận là cải thiện nếu loss nhỏ hơn mốc đang lưu ít nhất `min_delta`; cùng phép so sánh này dùng để lưu checkpoint và reset patience. Objective trả loss tại checkpoint đã lưu theo quy tắc đó, không lấy một cực tiểu nhỏ hơn nhưng không có checkpoint tương ứng.
- Epoch không đạt mức cải thiện tăng bộ đếm; đủ patience và đã tới epoch tối thiểu thì dừng. Trial có loss NaN/Inf được ghi failed, không làm ứng viên thắng.
- Những ngưỡng này là lựa chọn vận hành khởi đầu, không phải giá trị đã chứng minh tối ưu. Cosine cần ghi đúng T_max và LR thực mỗi epoch; early stopping có thể kết thúc trước khi LR giảm hết lịch 20 epoch.

### 6. Đánh giá và artifact

- Dùng validation để chọn cấu hình/trial/checkpoint; chỉ đánh giá test sau khi khóa lựa chọn. Ghi giới hạn rằng official test đã từng được xem trong các thí nghiệm cũ; không tuyên bố đây là test hoàn toàn chưa từng tiếp xúc ở cấp độ thiết kế nghiên cứu.
- Mọi trial lưu cấu hình, nhóm LR, số dense layer mở, số tham số trainable, history train/validation loss và accuracy, epoch checkpoint, trạng thái dừng và log. Lưu study/trial table, best parameters và nguồn checkpoint khởi tạo.
- So sánh LR chung/riêng bằng bảng metrics và đường cong có nhãn rõ; giữ confusion matrix, ROC/PR OvR, mẫu dự đoán và kiểm chứng save/load trên test trong phần kết quả được cập nhật.
- Thí nghiệm mới thay nội dung đang hiển thị trong phạm vi được sửa sau khi có evidence; output ngoài phạm vi giữ nguyên. Không thêm một nhánh báo cáo trùng lặp chỉ để giữ thiết kế candidate cũ.
- Sửa code/lỗi môi trường/log kỹ thuật giữ local; walkthrough trình bày phương pháp, kết quả và giới hạn. Chưa chạy thì ghi chưa chạy, không tạo benchmark giả hoặc kết luận đã giảm overfit.

### 7. Những điểm còn cần chốt trước triển khai

Độ mở và ngân sách đã chốt: toàn bộ block4 cùng norm5/classifier theo chính sách hiện hành, 5 trial mỗi phương án, tổng 10 trial, tối đa 20 epoch/trial. Trước khi chạy cần ghi rõ search space LR/weight decay và batch size; đề xuất lượt đầu chưa dùng pruning. Những chi tiết thực thi này không được làm thay đổi ngân sách hoặc mở thêm block ngoài phạm vi đã chốt.

### 8. Bảng lộ trình triển khai (Pipeline Execution Roadmap)

Bảng mô tả luồng mục tiêu cho notebook modeling sau cập nhật, theo định dạng Lab01: bước, giai đoạn, nội dung, module và artifact. Các module/hàm ghi “cần bổ sung/cập nhật” chưa được coi là đã triển khai; tên artifact mới là hợp đồng dự kiến. Phần EDA và head preparation được tái sử dụng; không mặc định chạy lại mỗi khi đọc notebook hoặc tuning.

| Bước | Giai đoạn | Nội dung chi tiết | Module triển khai | Output / artifact |
|---|---|---|---|---|
| 1 | Setup & Import | Nạp cấu hình, seed, thiết bị, đường dẫn dữ liệu; tách chế độ xem và thực thi | [src/training/trainer.py](src/training/trainer.py): `seed_all`<br>[src/utils/artifacts.py](src/utils/artifacts.py): `resolve_data_dir` | Run riêng, config, environment snapshot, console log local và thông tin phiên bản source |
| 2 | EDA & Split Handoff | Dùng split stratified 45.000/5.000 và official test 10.000; xác minh không giao train/validation, thống kê và integrity | [src/datasets/eda.py](src/datasets/eda.py): `locked_split`, `duplicate_audit`, `export_summary`<br>[src/datasets/classification.py](src/datasets/classification.py): `prepare_indices` | `split_indices.npz`, `selected_indices.npz`, EDA tables/figures, data summary và hash split |
| 3 | Preprocessing & DataLoaders | RGB, transform ImageNet, input 224×224; cùng split/batch giữa hai phương án; chưa augmentation | [src/datasets/classification.py](src/datasets/classification.py): `image_transform`, `IndexedImages`, `image_loader` | Cấu hình preprocessing, kiểm tra shape/nhãn/membership; train/val/test loaders |
| 4 | Khóa điểm khởi tạo và phạm vi fine-tune | Train head mới một lần từ ImageNet pretrained rồi nạp checkpoint được chọn cho mọi trial; mở block4 + norm5 + classifier, đóng băng phần trước; kiểm tra gradient/BN | [src/models/backbones.py](src/models/backbones.py): `build_model`, `configure_trainable`, `training_mode`, `parameter_counts` | Hash checkpoint nguồn, kiến trúc, danh sách trainable/frozen, số tham số và kết quả sanity/gradient check |
| 5 | Định nghĩa hai study Optuna | LR chung: một LR cho cả hai nhóm; LR riêng: LR backbone và LR head; AdamW, weight decay và ngân sách bằng nhau | [src/training/optuna_dense.py](src/training/optuna_dense.py): `suggest`<br>[src/training/tuning.py](src/training/tuning.py) | Search-space config, seed/sampler, hai study lưu bền, 5 trial mỗi study và bảng trial |
| 6 | Train từng trial + cosine | Reset model về cùng head checkpoint; cập nhật các tham số mở; train tối đa 20 epoch, cosine step cuối epoch, chưa pruning ở phương án đề xuất | [src/training/trainer.py](src/training/trainer.py)<br>[src/training/trainer.py](src/training/trainer.py): `fit`, `pass_epoch` | Mỗi trial: config, TensorBoard, history loss/accuracy và LR từng nhóm, thời gian, log và trạng thái |
| 7 | Checkpoint & Early Stopping thống nhất | Validation CE loss, minimize; cùng min_delta 0.001 cho lưu checkpoint và reset bộ đếm; patience 4, dừng từ epoch 6 | [src/training/trainer.py](src/training/trainer.py)<br>[src/training/trainer.py](src/training/trainer.py): `fit` | Best checkpoint mỗi trial, loss/epoch tại checkpoint, số epoch thực và lý do dừng; objective đúng checkpoint |
| 8 | Khóa trial thắng mỗi phương án | Chọn bằng objective validation loss đã thống nhất; không dùng test để chọn; ghi cả accuracy/macro F1 khi đo | [src/training/tuning.py](src/training/tuning.py): `select_trial` | `best_parameters.json`, bảng so sánh validation của hai phương án, study history và checkpoint được khóa |
| 9 | Đánh giá test & Trực quan hóa | Đánh giá hai checkpoint đã khóa trên cùng official test; so loss/accuracy/F1, hội tụ, confusion có số, ROC/PR OvR và mẫu dự đoán | [src/training/trainer.py](src/training/trainer.py): `predict`<br>[src/utils/plots.py](src/utils/plots.py): `export_evaluation` | Metrics JSON, classification report, confusion CSV/PNG, test predictions NPZ, ROC/PR và ảnh dự đoán; đường cong train/val của hai phương án |
| 10 | Save/Load & Test lại | Tạo model mới từ kiến trúc đã lưu, nạp checkpoint, eval trên test và kiểm tra khớp dự đoán/metrics theo tolerance | [src/training/trainer.py](src/training/trainer.py): `run_model`<br>[src/models/backbones.py](src/models/backbones.py) | Model metadata/checkpoint, roundtrip verification, full-test-after-load metrics và lưới nhãn thật/dự đoán/confidence |
| 11 | Notebook, Walkthrough & Bàn giao | Thay phần kết quả trong phạm vi cập nhật bằng artifact mới; phần ngoài phạm vi giữ nguyên; cập nhật roadmap đúng hàm thật | [src/utils/notebook_views.py](src/utils/notebook_views.py)<br>[src/utils/artifacts.py](src/utils/artifacts.py) | Bảng/hình và nhận xét từ run mới, provenance, hướng dẫn xem/tái lập; log kỹ thuật giữ local, chưa tạo PDF |

Roadmap này không phải xác nhận đã chạy Optuna. Trước khi triển khai sẽ thay tên/hàm dự kiến bằng API thực tế và kiểm tra notebook không vô tình kích hoạt training trong chế độ xem.

Các phần trên chưa phải lệnh chạy huấn luyện; lượt cập nhật này chỉ ghi nhận phạm vi và thảo luận vào description.

### 9. Chạy lại toàn bộ trên Kaggle

Không dùng checkpoint head cũ. Mặc định head_checkpoint=null: tải DenseNet121 pretrained ImageNet, thay classifier CIFAR10 và train head mới 10 epoch, LR 0.001, AdamW weight_decay 1e-4. Chọn head bằng validation loss rồi reset mọi Optuna trial về cùng checkpoint head mới. Chi phí 10 epoch head là phần bổ sung ngoài ngân sách 5+5 trial fine-tune. Notebook 03 tự tải CIFAR10 khi chưa gắn input, ghi output và weights cache dưới /kaggle/working, đóng gói artifacts khi hoàn tất hoặc training lỗi. Chưa khẳng định GPU end-to-end khi chưa có run thực tế.

## Kiến trúc mô hình và lý do lựa chọn

Bài toán sử dụng phương pháp Chuyển giao học tập (Transfer Learning) từ tập dữ liệu quy mô lớn ImageNet sang CIFAR-10 (10 lớp, ảnh gốc 32×32 được chuẩn hóa và nâng tỷ lệ lên 224×224 theo ImageNet standard). Hai kiến trúc mạng nơ-ron tích chập (CNN) được lựa chọn khảo sát là **ResNet-18** và **DenseNet-121**.

Cả hai mô hình đều sử dụng trọng số khởi tạo chuẩn `IMAGENET1K_V1` từ `torchvision.models`. Lớp phân loại cuối cùng (classifier/fc) được thay thế bằng một lớp `nn.Linear` có kích thước đầu ra bằng 10 ứng với 10 lớp của CIFAR-10. Hàm mất mát là `nn.CrossEntropyLoss()` (nhận raw logits trực tiếp từ mô hình, không áp dụng hàm `Softmax` trước loss để đảm bảo độ ổn định số học qua log-sum-exp trick).

### 1. Bảng so sánh tham số và phạm vi huấn luyện

| Thuộc tính / Mô hình | ResNet-18 | DenseNet-121 |
|---|---|---|
| **Kích thước đầu vào** | $3 \times 224 \times 224$ (RGB) | $3 \times 224 \times 224$ (RGB) |
| **Kiến trúc phân loại gốc** | `fc`: Linear(512 $\rightarrow$ 1000) | `classifier`: Linear(1024 $\rightarrow$ 1000) |
| **Kiến trúc phân loại CIFAR-10** | `fc`: Linear(512 $\rightarrow$ 10) | `classifier`: Linear(1024 $\rightarrow$ 10) |
| **Tổng số tham số (Total parameters)** | 11.181.642 (100%) | 6.964.106 (100%) |
| **Giai đoạn 1: Head-only (Linear Probing)** | 5.130 tham số (0,046%) | 10.250 tham số (0,147%) |
| - Phạm vi trainable | Chỉ lớp `fc` | Chỉ lớp `classifier` |
| - Trạng thái backbone | Đóng băng hoàn toàn (frozen) | Đóng băng hoàn toàn (frozen) |
| **Giai đoạn 2: Fine-tuning (`last_block`)** | 8.398.858 tham số (75,11%) | 2.170.378 tham số (31,17%) |
| - Phạm vi trainable | `layer4` + `fc` | `features.denseblock4` + `features.norm5` + `classifier` |
| - Trạng thái backbone | Đóng băng `conv1`, `bn1`, `layer1–3` | Đóng băng `conv0`, `norm0`, `denseblock1–3`, `transition1–3` |
| **Giai đoạn 2 mở rộng (`last_two_blocks`)** | 10.498.570 tham số (93,89%) | 5.534.474 tham số (79,47%) |
| - Phạm vi trainable | `layer3` + `layer4` + `fc` | `denseblock3` + `transition3` + `denseblock4` + `norm5` + `classifier` |

### 2. Chi tiết kiến trúc ResNet-18 (Residual Network)

ResNet-18 giải quyết vấn đề tiêu biến gradient (vanishing gradient) khi tăng độ sâu mạng thông qua cơ chế kết nối tắt (residual shortcut / skip connection): $\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$.

```text
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
  (layer1): Sequential(2 BasicBlocks, 64 channels, stride 1)
  (layer2): Sequential(2 BasicBlocks, 128 channels, stride 2, downsample shortcut)
  (layer3): Sequential(2 BasicBlocks, 256 channels, stride 2, downsample shortcut)
  (layer4): Sequential(2 BasicBlocks, 512 channels, stride 2, downsample shortcut)
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=10, bias=True)
)
```

- **Đặc điểm**: Mỗi `BasicBlock` gồm 2 lớp tích chập $3\times 3$ kèm BatchNorm và ReLU. Ở `layer4`, không gian đặc trưng được giảm về $7\times 7$ với 512 kênh trước khi qua `AdaptiveAvgPool2d` tạo vector đặc trưng 512 chiều.
- **Khối mở (`layer4` + `fc`)**: Chiếm 8.398.858 tham số (~75,11% toàn mạng). Việc mở toàn bộ `layer4` cho phép tinh chỉnh các bộ lọc ngữ nghĩa cấp cao mà không làm xáo trộn các bộ lọc phát hiện cạnh, màu sắc sơ cấp ở các tầng dưới (`layer1–3`).

### 3. Chi tiết kiến trúc DenseNet-121 (Densely Connected Network)

DenseNet-121 mở rộng ý tưởng skip connection bằng cách nối tất cả các tầng trực tiếp với nhau (Dense Connectivity): Mỗi tầng nhận đầu vào là phép ghép nối kênh (channel concatenation) của đầu ra từ toàn bộ các tầng trước đó: $\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{\ell-1}])$.

```text
DenseNet(
  (features): Sequential(
    (conv0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (norm0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu0): ReLU(inplace=True)
    (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1)
    (denseblock1): 6 _DenseLayers (growth_rate k=32, out_channels=256)
    (transition1): Conv2d(256, 128, 1x1) + AvgPool2d(2, 2)
    (denseblock2): 12 _DenseLayers (growth_rate k=32, out_channels=512)
    (transition2): Conv2d(512, 256, 1x1) + AvgPool2d(2, 2)
    (denseblock3): 24 _DenseLayers (growth_rate k=32, out_channels=1024)
    (transition3): Conv2d(1024, 512, 1x1) + AvgPool2d(2, 2)
    (denseblock4): 16 _DenseLayers (growth_rate k=32, out_channels=1024)
    (norm5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (classifier): Linear(in_features=1024, out_features=10, bias=True)
)
```

- **Cấu trúc mỗi `_DenseLayer` (Bottleneck design)**:
  - `norm1` (BatchNorm) $\rightarrow$ `relu1` $\rightarrow$ `conv1` ($1\times 1$, bottleneck $4k=128$ kênh)
  - `norm2` (BatchNorm) $\rightarrow$ `relu2` $\rightarrow$ `conv2` ($3\times 3$, tạo $k=32$ kênh đặc trưng mới)
- **Tầng chuyển tiếp (`Transition Layer`)**: Thực hiện giảm số kênh (compression rate $\theta = 0,5$) bằng tích chập $1\times 1$ và giảm kích thước không gian bằng $2\times 2$ Average Pooling.
- **Khối mở (`features.denseblock4` + `features.norm5` + `classifier`)**: Gồm 16 dense layers cuối cùng và lớp chuẩn hóa cuối trước khi vào classifier. Chiếm 2.170.378 tham số (~31,17% toàn mạng).

### 4. Lý do lựa chọn và Chiến lược Transfer Learning

1. **Khả năng tái sử dụng đặc trưng thị giác (Feature Reusability)**:
   - CIFAR-10 có kích thước tập dữ liệu vừa phải (50.000 ảnh). Nếu huấn luyện từ đầu (scratch) trên mạng sâu 18 hay 121 tầng, mô hình rất dễ rơi vào tình trạng overfitting nghiêm trọng hoặc khó hội tụ.
   - Trọng số ImageNet-1K cung cấp bộ biểu diễn đặc trưng tổng quát xuất sắc, giúp rút ngắn thời gian hội tụ và đạt độ chính xác cao.
2. **Chiến lược huấn luyện 2 giai đoạn (Two-Stage Transfer Learning Protocol)**:
   - **Giai đoạn 1 (Linear Probing / Head Training)**: Đóng băng toàn bộ backbone, chỉ huấn luyện classifier mới với learning rate $10^{-3}$. Giai đoạn này cho phép trích xuất đặc trưng cố định (feature caching), tính toán cực nhanh (chỉ mất vài giây/epoch) để tìm ra trọng số đầu phân loại thích ứng ban đầu mà không làm biến dạng các trọng số tiền huấn luyện.
   - **Giai đoạn 2 (Fine-tuning)**: Mở khóa các tầng đặc trưng ngữ nghĩa cuối (`layer4` của ResNet-18, `denseblock4 + norm5` của DenseNet-121) và huấn luyện với learning rate rất nhỏ ($10^{-5} - 10^{-4}$). Điều này giúp tinh chỉnh các biểu diễn cấp cao phù hợp với 10 lớp của CIFAR-10, đồng thời hạn chế tối đa nguy cơ "quên kiến thức cũ" (Catastrophic Forgetting).
3. **Quy tắc kiểm soát BatchNorm nghiêm ngặt (`training_mode`)**:
   - Khi fine-tune, tất cả các lớp `BatchNorm2d` được ép hoạt động ở chế độ `eval()` (giữ cố định `running_mean` và `running_var` được học từ ImageNet, chỉ cho phép cập nhật các tham số affine $\gamma, \beta$ nếu tầng đó thuộc khối được mở).
   - Lý do: Batch size nhỏ trong quá trình fine-tune và sự dịch chuyển phân bố dữ liệu (Domain Shift giữa ImageNet và CIFAR-10) sẽ làm sai lệch nghiêm trọng các thống kê chuẩn hóa nếu bật cập nhật running statistics, dẫn đến suy giảm độ chính xác và mất ổn định gradient.
