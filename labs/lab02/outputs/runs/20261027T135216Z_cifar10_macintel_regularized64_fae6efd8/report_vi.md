# Lab 02 — Transfer Learning trên CIFAR10

## 1. Mục tiêu
Khảo sát ResNet18 và DenseNet121 pre-trained ImageNet, thay classifier thành 10 lớp, so sánh đóng băng backbone và fine-tune một/hai block cuối. Điều chỉnh learning rate/batch size, theo dõi bằng TensorBoard, đánh giá và lưu/tải model.

## 2. Dữ liệu và phạm vi thực nghiệm
Toàn bộ split CIFAR-10; ngân sách epoch và độ phân giải được ghi rõ bên dưới.
Profile: `cifar10_macintel_regularized64`. Train/validation/test thực tế: 45,000/5,000/10,000. Seed: 42.
Split gốc 45.000/5.000 được tạo stratified; test giữ nguồn test chính thức. Khi chạy pilot, mỗi split được lấy tập con stratified riêng. Mỗi lớp có 4.500 train / 500 validation / 1.000 test khi dùng toàn bộ dữ liệu.
Ảnh RGB gốc 32×32 được giữ màu. Đầu vào thực nghiệm là 64×64; normalization dùng mean/std của weights IMAGENET1K_V1. Với 224×224, dùng trọn transforms chính thức; với 64×64, resize trực tiếp để phù hợp ngân sách CPU, không phải preprocessing mặc định của weights. Không augmentation trong thí nghiệm này để tái sử dụng feature đóng băng và giữ tiền xử lý nhất quán.

## 3. Phương pháp
Mỗi mô hình thay classifier bằng Linear 10 đầu ra. Head-only đóng băng backbone và cache feature train/val ở eval mode. Cache tương đương forward backbone cho đầu vào cố định; thời gian trích xuất được ghi riêng.
Tìm learning rate [0.001, 0.01] và batch size [64, 128], 10 epoch/candidate với adamw, weight_decay=0.0001. Các candidate dùng cùng initial head và shuffle seed. Policy chọn candidate: {'accuracy_tolerance': 0, 'loss_tolerance': 0}; accuracy band → validation loss band → fit time. Checkpoint trong mỗi trial chọn accuracy cao nhất, hòa chọn loss thấp hơn rồi epoch sớm hơn.
Fine-tune: mọi candidate khởi tạo lại từ cùng selected head; cấu hình optimizer/LR/depth/scheduler được chọn chỉ bằng validation. Ngân sách tối đa 10 epoch; stopping policy=fixed epoch budget. Số epoch thực chạy và stop_reason nằm trong training_summary.json; không suy ra hội tụ từ ngân sách. Trong mỗi trial, BatchNorm running statistics giữ cố định. Mở hai block DenseNet gồm denseblock3+transition3+denseblock4+norm5. Cache prefix chỉ hỗ trợ last block; cấu hình tuned tắt cache cho cả hai kiến trúc.
- resnet18: adamw; depth=last_units; backbone_lr=1e-05; head_lr=0.0001; scheduler=none
- densenet121: adamw; depth=last_units; backbone_lr=1e-05; head_lr=0.0001; scheduler=none
Toàn bộ fine-search: `finetune_search_results.csv`; xem `cost_comparison.csv` để phân biệt chi phí toàn search và selected recipe.

CrossEntropyLoss nhận logits. Checkpoint chọn theo validation. Test không tham gia search hoặc lựa chọn checkpoint; hai điều kiện head/fine-tune được xác định trước và đánh giá sau khi khóa lựa chọn. Mỗi condition được tải vào model instance mới trước đánh giá.

## 4. Kết quả đo được
| Model | Stage | Val acc | Test acc | Macro F1 | Trainable | Train (s) |
|---|---|---:|---:|---:|---:|---:|
| resnet18 | head | 76.40% | 75.22% | 0.7499 | 5,130 | 14.1 |
| resnet18 | finetune | 80.68% | 80.49% | 0.8043 | 4,725,770 | 3104.7 |
| densenet121 | head | 79.04% | 78.24% | 0.7822 | 10,250 | 14.7 |
| densenet121 | finetune | 80.66% | 80.10% | 0.8000 | 651,658 | 5557.6 |

- resnet18: fine-tune thay đổi test accuracy +5.27 điểm phần trăm và Macro F1 +0.0544 so với head-only.
- densenet121: fine-tune thay đổi test accuracy +1.86 điểm phần trăm và Macro F1 +0.0178 so với head-only.

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
