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
