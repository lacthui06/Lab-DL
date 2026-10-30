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
| 1 | Setup & Import | Nạp cấu hình, seed, thiết bị, đường dẫn dữ liệu; tách chế độ xem và thực thi | Hiện có: `src.experiment.seed_all`, `src.delivery.resolve_data_dir`; cần nối entry point Optuna | Run riêng, config, environment snapshot, console log local và thông tin phiên bản source |
| 2 | EDA & Split Handoff | Dùng split stratified 45.000/5.000 và official test 10.000; xác minh không giao train/validation, thống kê và integrity | Hiện có: `src.eda.locked_split`, `duplicate_audit`, `export_summary`; `src.data.prepare_indices` | `split_indices.npz`, `selected_indices.npz`, EDA tables/figures, data summary và hash split |
| 3 | Preprocessing & DataLoaders | RGB, transform ImageNet, input 224×224; cùng split/batch giữa hai phương án; chưa augmentation | Hiện có: `src.data.image_transform`, `IndexedImages`, `image_loader` | Cấu hình preprocessing, kiểm tra shape/nhãn/membership; train/val/test loaders |
| 4 | Khóa điểm khởi tạo và phạm vi fine-tune | Train head mới một lần từ ImageNet pretrained rồi nạp checkpoint được chọn cho mọi trial; mở block4 + norm5 + classifier, đóng băng phần trước; kiểm tra gradient/BN | Hiện có: `src.models.build_model`, `configure_trainable`, `training_mode`, `parameter_counts`; cần nối tải head được chọn | Hash checkpoint nguồn, kiến trúc, danh sách trainable/frozen, số tham số và kết quả sanity/gradient check |
| 5 | Định nghĩa hai study Optuna | LR chung: một LR cho cả hai nhóm; LR riêng: LR backbone và LR head; AdamW, weight decay và ngân sách bằng nhau | Cần bổ sung tích hợp Optuna trong `src.tuning` và điều phối trong `src.experiment`; không coi `fine_candidates` hiện tại là Optuna | Search-space config, seed/sampler, hai study lưu bền, 5 trial mỗi study và bảng trial |
| 6 | Train từng trial + cosine | Reset model về cùng head checkpoint; cập nhật các tham số mở; train tối đa 20 epoch, cosine step cuối epoch, chưa pruning ở phương án đề xuất | Cần cập nhật `src.experiment.fit`, `pass_epoch`, `src.tuning.make_optimizer`, `make_scheduler` cho search mới | Mỗi trial: config, TensorBoard, history loss/accuracy và LR từng nhóm, thời gian, log và trạng thái |
| 7 | Checkpoint & Early Stopping thống nhất | Validation CE loss, minimize; cùng min_delta 0.001 cho lưu checkpoint và reset bộ đếm; patience 4, dừng từ epoch 6 | Cần cập nhật `src.experiment.fit` và `src.tuning.stopping_policy`; logic checkpoint hiện tại còn dùng accuracy | Best checkpoint mỗi trial, loss/epoch tại checkpoint, số epoch thực và lý do dừng; objective đúng checkpoint |
| 8 | Khóa trial thắng mỗi phương án | Chọn bằng objective validation loss đã thống nhất; không dùng test để chọn; ghi cả accuracy/macro F1 khi đo | Cần cập nhật logic chọn trial trong `src.tuning`; `select_trial` hiện tại còn dùng accuracy/loss/time bands | `best_parameters.json`, bảng so sánh validation của hai phương án, study history và checkpoint được khóa |
| 9 | Đánh giá test & Trực quan hóa | Đánh giá hai checkpoint đã khóa trên cùng official test; so loss/accuracy/F1, hội tụ, confusion có số, ROC/PR OvR và mẫu dự đoán | Hiện có: `src.experiment.predict`, `src.reporting.export_evaluation`; cần cập nhật plotting/view theo LR chung/riêng | Metrics JSON, classification report, confusion CSV/PNG, test predictions NPZ, ROC/PR và ảnh dự đoán; đường cong train/val của hai phương án |
| 10 | Save/Load & Test lại | Tạo model mới từ kiến trúc đã lưu, nạp checkpoint, eval trên test và kiểm tra khớp dự đoán/metrics theo tolerance | Hiện có mẫu roundtrip trong `src.experiment.run_model`; cần nối cho hai kết quả Optuna | Model metadata/checkpoint, roundtrip verification, full-test-after-load metrics và lưới nhãn thật/dự đoán/confidence |
| 11 | Notebook, Walkthrough & Bàn giao | Thay phần kết quả trong phạm vi cập nhật bằng artifact mới; phần ngoài phạm vi giữ nguyên; cập nhật roadmap đúng hàm thật | Cần cập nhật `src.notebook_views`, notebook 02 và walkthrough; tái sử dụng artifact IO trong `src.delivery` | Bảng/hình và nhận xét từ run mới, provenance, hướng dẫn xem/tái lập; log kỹ thuật giữ local, chưa tạo PDF |

Roadmap này không phải xác nhận đã chạy Optuna. Trước khi triển khai sẽ thay tên/hàm dự kiến bằng API thực tế và kiểm tra notebook không vô tình kích hoạt training trong chế độ xem.

Các phần trên chưa phải lệnh chạy huấn luyện; lượt cập nhật này chỉ ghi nhận phạm vi và thảo luận vào description.

### 9. Chạy lại toàn bộ trên Kaggle

Không dùng checkpoint head cũ. Mặc định head_checkpoint=null: tải DenseNet121 pretrained ImageNet, thay classifier CIFAR10 và train head mới 10 epoch, LR 0.001, AdamW weight_decay 1e-4. Chọn head bằng validation loss rồi reset mọi Optuna trial về cùng checkpoint head mới. Chi phí 10 epoch head là phần bổ sung ngoài ngân sách 5+5 trial fine-tune. Notebook 03 tự tải CIFAR10 khi chưa gắn input, ghi output và weights cache dưới /kaggle/working, đóng gói artifacts khi hoàn tất hoặc training lỗi. Chưa khẳng định GPU end-to-end khi chưa có run thực tế.
