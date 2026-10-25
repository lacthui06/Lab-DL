# Lab 02 — CIFAR-10 with Pretrained Neural Networks

Phân loại 10 lớp **CIFAR-10 RGB** bằng **ResNet18 và DenseNet121 pretrained ImageNet**, so sánh head-only và fine-tune block cuối, khảo sát learning rate/batch size, theo dõi TensorBoard. Kết quả FashionMNIST trong các run trước là lịch sử riêng, không được trộn với CIFAR-10.

## Mở bài làm

- `notebooks/01_cifar10_eda_architecture.ipynb`: dữ liệu, split, ảnh màu, thống kê và kiến trúc.
- `notebooks/lab_02_transfer_learning.ipynb`: notebook chính, huấn luyện, TensorBoard, kết quả, phân tích lỗi và save/load.
- `RESULTS.md`: báo cáo và bằng chứng của run được chọn.
- `outputs/selected_run.json`: chọn run rõ ràng, không tự lấy thư mục mới nhất.

Notebook mặc định `MODE='view'`: hiển thị kết quả đã lưu. Đặt `MODE='train'` để tạo run mới. Có thể chọn interpreter đang có tại `D:\DL\lab2\.venv\Scripts\python.exe`, hoặc tạo `.venv` riêng trong `lab2` theo hướng dẫn dưới đây. Chạy notebook từ đầu. Khi chuyển máy, đặt `LAB02_ROOT` đến source và `LAB02_RUN` đến run đã lưu.

## Chạy lại

Thực hiện từ thư mục `lab2`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Tải CIFAR-10 và pretrained weights chính thức
.\.venv\Scripts\python.exe scripts/download_assets.py
# Máy chủ CIFAR-10 chậm: chia phần, kiểm tra MD5 chính thức
.\.venv\Scripts\python.exe scripts/download_cifar10.py

# Toàn bộ 45.000 train / 5.000 validation / 10.000 test, CPU, ảnh 64x64
.\.venv\Scripts\python.exe scripts/run.py --config config/cpu_full.json

# Toàn bộ dữ liệu, preprocessing 224x224 chính thức, 10 epoch/giai đoạn, nên dùng GPU
.\.venv\Scripts\python.exe scripts/run.py --config config/full.json

# Chạy ngắn kiểm tra pipeline, không phải kết quả full
.\.venv\Scripts\python.exe scripts/run.py --config config/cpu_pilot.json

# TensorBoard: http://localhost:6006
.\.venv\Scripts\tensorboard.exe --logdir outputs/runs --port 6006

# Kiểm thử và đóng gói sau khi run hoàn thành
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/build_notebook.py
.\.venv\Scripts\python.exe scripts/finalize.py
.\.venv\Scripts\python.exe scripts/execute_notebook.py
.\.venv\Scripts\python.exe scripts/execute_notebook.py 01_cifar10_eda_architecture.ipynb
```

## Thiết kế thí nghiệm

1. Từ 50.000 train chính thức, chia stratified **45.000/5.000**, seed 42. Test chính thức **10.000**. Lưu chỉ số, kiểm tra train/validation không trùng. Mỗi lớp có 4.500/500/1.000 ảnh ở profile toàn bộ.
2. Giữ RGB gốc 32×32. Normalization theo weights `IMAGENET1K_V1`. CPU resize trực tiếp 64×64; GPU dùng đầy đủ resize/crop/normalize 224×224 chính thức. Đây là hai điều kiện thí nghiệm khác nhau.
3. Không augmentation để cache feature cố định. Thống kê EDA chỉ lấy từ train, không thay normalization pretrained. Augmentation có thể là thí nghiệm mở rộng.
4. Head-only: thay lớp cuối bằng Linear 10 logits, backbone đóng băng ở eval mode, cache train/validation features. Các trial cùng khởi tạo classifier, seed và epoch. AdamW, CrossEntropyLoss, weight decay 1e-4.
5. CPU thử LR `{0.001, 0.01}` × batch `{32, 64}`, 5 epoch/trial cho **mỗi kiến trúc**. Chọn best validation accuracy; hòa chọn trial/epoch sớm nhất. GPU thêm LR 0.0001 và dùng 10 epoch/trial.
6. Fine-tune từ head được chọn: mở `layer4` của ResNet18 hoặc `denseblock4 + norm5` của DenseNet121, cùng classifier. LR backbone 1e-5, head 1e-4; CPU 2 epoch, GPU 10 epoch. BatchNorm giữ running statistics; affine weights trong block mở vẫn học.
   Profile CPU hiện hỗ trợ cache đầu ra của prefix đóng băng để giảm tính lặp. Kiểm thử đối chiếu logits, gradients và schema checkpoint; mỗi model dùng cache còn đối chiếu trên ảnh thật. Xem `prefix_equivalence.json` và cột `execution` trong bảng chi phí để biết run thực tế dùng cách nào. Đánh giá test luôn dùng toàn bộ model trên ảnh.
7. Khóa checkpoint bằng validation rồi đánh giá các điều kiện head/fine-tune đã định trước trên test. Test không dùng để search hoặc quyết định cấu hình tiếp theo. Fine-tune có thêm ngân sách nên không xem là so sánh ngang tổng epoch.

## Đáp ứng đề bài

| Yêu cầu | Triển khai | Bằng chứng |
|---|---|---|
| Nhiều pretrained models | ResNet18, DenseNet121; ImageNet weights rõ phiên bản | preprocessing.json, checkpoint metadata |
| Khảo sát kiến trúc | Model, named parameters, tổng số/tham số được học | architecture.txt, parameters.csv, notebook |
| Adapt, freeze, fine-tune | Linear 10 lớp, head-only, block cuối | models.py, gradient tests |
| Chuẩn bị CIFAR-10 | Split, RGB transforms, DataLoader, EDA | NPZ indices, EDA, data_summary |
| Điều chỉnh LR/batch | Grid độc lập, chọn bằng validation | search_results.csv, selection.json |
| TensorBoard | Scalars, LR, hparams, ảnh, graph, histogram | Event files |
| Evaluate | Accuracy/Macro F1/Precision/Recall/loss, confusion, ROC/PR, ảnh đúng/sai | CSV, NPZ, PNG và báo cáo |
| Save/load | Instance mới, eval, so xác suất và CLI inference | best.pt, roundtrip.json |

VGG trong đề là ví dụ. Hai họ kiến trúc được chọn đáp ứng yêu cầu thử nhiều pretrained models và giữ chi phí CPU trong phạm vi thực nghiệm.

## Artifact và tái lập

Mỗi run có tên riêng: config, environment, console log, status, snapshot source lúc bắt đầu, SHA256 manifest, indices, EDA, search, history, checkpoints, TensorBoard và predictions. Chỉ run hoàn tất mới cập nhật registry. Giữ toàn bộ run cũ.

Thời gian fit candidate không gồm feature extraction hoặc toàn bộ search. `head_selected/training_summary.json` tách các khoản chi phí. Fine-tune tiếp tục từ head; chi phí toàn pipeline phải cộng cả hai giai đoạn. Một seed không đủ kết luận ý nghĩa thống kê. Chưa có baseline train-from-scratch nên không kết luận định lượng lợi ích ImageNet pretraining so với khởi tạo ngẫu nhiên.

`requirements-lock-windows.txt` ghi môi trường local; không dùng lock Windows để cài Linux. Trên Kaggle bật GPU/Internet, đưa source vào thư mục có thể ghi, cài requirements và chạy `run_suite('config/full.json', work_dir='/kaggle/working/lab2')`. Luồng Kaggle chưa được kiểm chứng end-to-end tại máy local.

## Nguồn

- [CIFAR-10 chính thức](https://cave.cs.toronto.edu/kriz/cifar.html)
- [Torchvision CIFAR10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
- [Torchvision pretrained models](https://docs.pytorch.org/vision/stable/models.html)
- [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [TensorBoard với PyTorch](https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
