# Lab 02 — Yêu cầu và thiết kế hiện hành

Đề gốc, Chapter 3 trang 23: khảo sát nhiều pretrained architectures, thay classifier, freeze/fine-tune, chuẩn bị dữ liệu, train/evaluate; điều chỉnh LR/batch size và theo dõi TensorBoard. CIFAR-10 là ví dụ ở trang 24. VGG, DenseNet và ResNet là các ví dụ kiến trúc, không phải danh sách bắt buộc chạy tất cả.

Yêu cầu người dùng: hoàn thành Lab 02 bằng **CIFAR-10**. Dùng ResNet18 và DenseNet121 ImageNet weights, head-only và fine-tune block cuối; grid LR × batch chỉ chọn bằng validation. Không dùng kết quả FashionMNIST làm bằng chứng CIFAR-10.

Split stratified seed 42: 45.000 train / 5.000 validation / 10.000 test. Giữ RGB. Profile cpu_full chạy toàn bộ ảnh, resize 64×64, 5 head epoch/trial, 2 fine-tune epoch. Profile full dùng 224×224, 10 epoch/giai đoạn, dành cho GPU. Profile cpu_pilot chỉ kiểm tra tập con. Ghi đúng profile thực tế; không suy diễn số của cấu hình chưa chạy.

Sản phẩm: hai notebook có output, source/CLI, EDA, kiến trúc, search và learning curves, TensorBoard, báo cáo tiếng Việt Markdown/HTML, bảng/hình đánh giá, checkpoint kèm metadata, roundtrip và inference mẫu, manifest/log/snapshot, kiểm thử và README. Không yêu cầu PDF ở lượt này.
