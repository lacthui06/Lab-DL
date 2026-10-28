# Lab 02 — Transfer Learning trên CIFAR10

## 1. Mục tiêu
Khảo sát ResNet18 và DenseNet121 pre-trained ImageNet, thay classifier thành 10 lớp, so sánh đóng băng backbone và fine-tune một/hai block cuối. Điều chỉnh learning rate/batch size, theo dõi bằng TensorBoard, đánh giá và lưu/tải model.

## 2. Dữ liệu và phạm vi thực nghiệm
Toàn bộ split CIFAR-10; ngân sách epoch và độ phân giải được ghi rõ bên dưới.
Profile: `cifar10_gpu_tuning_group_02`. Train/validation/test thực tế: 45,000/5,000/10,000. Seed: 42.
Split gốc 45.000/5.000 được tạo stratified; test giữ nguồn test chính thức. Khi chạy pilot, mỗi split được lấy tập con stratified riêng. Mỗi lớp có 4.500 train / 500 validation / 1.000 test khi dùng toàn bộ dữ liệu.
Ảnh RGB gốc 32×32 được giữ màu. Đầu vào thực nghiệm là 224×224; normalization dùng mean/std của weights IMAGENET1K_V1. Với 224×224, dùng trọn transforms chính thức; với 64×64, resize trực tiếp để phù hợp ngân sách CPU, không phải preprocessing mặc định của weights. Không augmentation trong thí nghiệm này để tái sử dụng feature đóng băng và giữ tiền xử lý nhất quán.

## 3. Phương pháp
Mỗi mô hình thay classifier bằng Linear 10 đầu ra. Head-only đóng băng backbone và cache feature train/val ở eval mode. Cache tương đương forward backbone cho đầu vào cố định; thời gian trích xuất được ghi riêng.
Tìm learning rate [0.0001, 0.001, 0.01] và batch size [64, 128], 10 epoch/candidate với adamw, weight_decay=0.0001. Các candidate dùng cùng initial head và shuffle seed. Policy chọn candidate: {'accuracy_tolerance': 0.001, 'loss_tolerance': 0.005}; accuracy band → validation loss band → fit time. Checkpoint trong mỗi trial chọn accuracy cao nhất, hòa chọn loss thấp hơn rồi epoch sớm hơn.
Fine-tune: mọi candidate khởi tạo lại từ cùng selected head; cấu hình optimizer/LR/depth/scheduler được chọn chỉ bằng validation. Ngân sách tối đa 30 epoch; stopping policy={'min_epochs': 10, 'patience': 7, 'min_delta': 0.0001}. Số epoch thực chạy và stop_reason nằm trong training_summary.json; không suy ra hội tụ từ ngân sách. Trong mỗi trial, BatchNorm running statistics giữ cố định. Mở hai block DenseNet gồm denseblock3+transition3+denseblock4+norm5. Cache prefix chỉ hỗ trợ last block; cấu hình tuned tắt cache cho cả hai kiến trúc.
- resnet18: adamw; depth=last_two_blocks; backbone_lr=3e-05; head_lr=0.0003; scheduler=cosine
- densenet121: adamw; depth=last_two_blocks; backbone_lr=3e-05; head_lr=0.0003; scheduler=cosine
Toàn bộ fine-search: `finetune_search_results.csv`; xem `cost_comparison.csv` để phân biệt chi phí toàn search và selected recipe.

CrossEntropyLoss nhận logits. Checkpoint chọn theo validation. Test không tham gia search hoặc lựa chọn checkpoint; hai điều kiện head/fine-tune được xác định trước và đánh giá sau khi khóa lựa chọn. Mỗi condition được tải vào model instance mới trước đánh giá.

## 4. Kết quả đo được
| Model | Stage | Val acc | Test acc | Macro F1 | Trainable | Train (s) |
|---|---|---:|---:|---:|---:|---:|
| resnet18 | head | 87.04% | 85.86% | 0.8590 | 5,130 | 9.5 |
| resnet18 | finetune | 94.06% | 93.71% | 0.9371 | 10,498,570 | 1723.9 |
| densenet121 | head | 88.90% | 87.90% | 0.8791 | 10,250 | 10.2 |
| densenet121 | finetune | 94.78% | 94.61% | 0.9459 | 5,534,474 | 3439.4 |

- resnet18: fine-tune thay đổi test accuracy +7.85 điểm phần trăm và Macro F1 +0.0781 so với head-only.
- densenet121: fine-tune thay đổi test accuracy +6.71 điểm phần trăm và Macro F1 +0.0668 so với head-only.

![So sánh](comparison.png)

Train (s) chỉ là thời gian fit candidate được chọn hoặc fine-tune, chưa bao gồm feature extraction/search. `training_summary.json` của head_selected ghi riêng thời gian trích xuất và tổng search. Vì fine-tune tiếp tục từ head, chi phí toàn pipeline phải cộng các giai đoạn.

## 5. Phân tích và giới hạn
Kết quả chỉ đại diện cho seed và phạm vi dữ liệu đã ghi. Fine-tune có thêm epoch và được khởi tạo từ head đã học; chênh lệch phản ánh chiến lược hai giai đoạn, chưa cô lập tác động unfreeze ở cùng ngân sách tính toán.
Không mặc định pre-trained hoặc fine-tune phải tốt hơn. Tập nguồn ImageNet khác phân phối ảnh CIFAR-10 độ phân giải thấp. Phóng ảnh không bổ sung thông tin chi tiết mới.
Không so sánh trực tiếp với Lab1 vì FashionMNIST và CIFAR-10 là hai bài toán khác nhau. Cấu hình CPU 64×64 và GPU 224×224 cũng phải được báo cáo riêng; không suy diễn kết quả của cấu hình chưa chạy.

## 6. Bằng chứng và tái lập
`config.json`, `environment.json`, `source_hashes.json`, hai NPZ indices, `console.log`, `status.json` và SHA256 manifest ghi điều kiện chạy. Từng kiến trúc có architecture, parameters, preprocessing, search_results, selection, TensorBoard event và learning curves. Mỗi condition có best.pt, history, classification report, confusion counts, probabilities, ảnh đúng/sai và roundtrip.json.
Lưới ảnh chọn tối đa 8 đúng/8 sai để minh họa; xác suất softmax không bảo đảm dự đoán đúng. Số trong confusion matrix là count, màu chuẩn hóa theo lớp thật.

## 7. Tài liệu tham khảo
- [Torchvision models và weights](https://docs.pytorch.org/vision/main/models)
- [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial)
- [TensorBoard với PyTorch](https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial)

## 8. Khảo sát lỗi và hình bổ sung

### Dữ liệu RGB và phân bố lớp

![Ảnh train theo lớp](eda/training_examples.png)

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| airplane | 4500 | 500 | 1000 |
| automobile | 4500 | 500 | 1000 |
| bird | 4500 | 500 | 1000 |
| cat | 4500 | 500 | 1000 |
| deer | 4500 | 500 | 1000 |
| dog | 4500 | 500 | 1000 |
| frog | 4500 | 500 | 1000 |
| horse | 4500 | 500 | 1000 |
| ship | 4500 | 500 | 1000 |
| truck | 4500 | 500 | 1000 |

- resnet18/head_selected: cặp lỗi nhiều nhất dog → cat (140 ảnh trong tập test đã chọn).

**Ba lớp F1 thấp nhất — resnet18/head_selected:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| cat | 0.7279 | 0.7890 | 0.7572 |
| bird | 0.8022 | 0.8110 | 0.8066 |
| dog | 0.8792 | 0.7500 | 0.8095 |

- resnet18/finetune: cặp lỗi nhiều nhất cat → dog (82 ảnh trong tập test đã chọn).

**Ba lớp F1 thấp nhất — resnet18/finetune:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| cat | 0.8636 | 0.8610 | 0.8623 |
| dog | 0.8795 | 0.8980 | 0.8887 |
| bird | 0.9374 | 0.9130 | 0.9250 |


### resnet18

![Learning curves](resnet18/learning_curves.png)

![Hyperparameter search](resnet18/hyperparameter_search.png)

![Confusion matrix](resnet18/finetune/confusion_matrix.png)

![Prediction examples](resnet18/finetune/prediction_examples.png)

![ROC và PR](resnet18/finetune/roc_pr.png)

- densenet121/head_selected: cặp lỗi nhiều nhất dog → cat (110 ảnh trong tập test đã chọn).

**Ba lớp F1 thấp nhất — densenet121/head_selected:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| cat | 0.7566 | 0.8020 | 0.7786 |
| dog | 0.8588 | 0.8210 | 0.8395 |
| deer | 0.8684 | 0.8380 | 0.8529 |

- densenet121/finetune: cặp lỗi nhiều nhất cat → dog (54 ảnh trong tập test đã chọn).

**Ba lớp F1 thấp nhất — densenet121/finetune:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| cat | 0.9205 | 0.8680 | 0.8935 |
| dog | 0.9270 | 0.9010 | 0.9138 |
| deer | 0.9369 | 0.9500 | 0.9434 |


### densenet121

![Learning curves](densenet121/learning_curves.png)

![Hyperparameter search](densenet121/hyperparameter_search.png)

![Confusion matrix](densenet121/finetune/confusion_matrix.png)

![Prediction examples](densenet121/finetune/prediction_examples.png)

![ROC và PR](densenet121/finetune/roc_pr.png)

### TensorBoard và chi phí

![TensorBoard scalars](tensorboard_evidence.png)

Hình này dựng từ TensorBoard events thật; không phải ảnh chụp giao diện. Mở TensorBoard theo README để xem tương tác.

Điều kiện được đề xuất bằng validation: **densenet121/finetune**, validation accuracy **94.78%**.

Chi phí pipeline gồm feature extraction, toàn bộ head search và toàn bộ fine-tune search nếu có; chưa gồm EDA, ghi báo cáo và inference test. Xem cost_comparison.csv.

| Model | Stage | Feature extraction (s) | All head search (s) | Pipeline search train (s) | Test (s) |
|---|---|---:|---:|---:|---:|
| resnet18 | head | 124.4 | 69.9 | 194.2 | 26.3 |
| resnet18 | finetune | 124.4 | 69.9 | 8396.9 | 26.6 |
| densenet121 | head | 243.3 | 75.9 | 319.2 | 49.4 |
| densenet121 | finetune | 243.3 | 75.9 | 16738.7 | 49.5 |

### Cấu hình head được chọn bằng validation

| Model | LR | Batch | Best epoch | Validation accuracy |
|---|---:|---:|---:|---:|
| resnet18 | 0.001 | 128 | 10 | 87.04% |
| densenet121 | 0.001 | 128 | 7 | 88.90% |

### Diễn giải chỉ số và thiết kế

Accuracy là số ảnh dự đoán đúng chia tổng số ảnh. Macro F1 tính F1 riêng cho từng lớp rồi lấy trung bình không trọng số; vì vậy mỗi lớp đóng góp ngang nhau. Macro Precision/Recall và cross-entropy test được lưu trong metrics.json. ROC-AUC và average precision tính one-vs-rest từng lớp, không phải cùng một đại lượng.

ResNet dùng residual connections để cộng tín hiệu qua các block; DenseNet nối các feature maps từ nhiều lớp trong dense block. Thực nghiệm thay head và giữ cấu trúc backbone gốc, không thiết kế lại CNN từ đầu.

Profile cifar10_gpu_tuning_group_02: ngân sách tối đa 30 epoch fine-tune, ảnh 224x224, seed 42. Kết quả chỉ đại diện cấu hình và ngân sách này; một seed chưa chứng minh ưu thế thống kê.