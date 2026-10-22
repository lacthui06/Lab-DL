# Description — Lab01: FashionMNIST với PyTorch

## Đề bài gốc

Learn: Study PyTorch Tensors, Datasets/Loaders, Transforms, Model Building, Autograd, Optimization, and Model Saving/Loading.

Implement: Load FashionMNIST, apply transforms, build a neural network, create a training loop, evaluate accuracy, save/load the model. Experiment with network/hyperparameters, visualize loss and predicted-vs-actual images. Deliver Python code, report, loss graphs and image displays. Use PyTorch documentation/tutorials.

## Phạm vi hiện hành đã thống nhất

1. Hai notebook: EDA/preprocessing/feature engineering và modeling/optimizer/tuning. Roadmap dạng bảng 5 cột ở đầu mỗi notebook. Chế độ view/train dùng cùng cell hiển thị, tránh phần đọc lại lặp cuối notebook.
2. FashionMNIST 28×28 grayscale, split stratified 54.000 train/6.000 validation, test gốc 10.000; giữ indices. Rescale /255, đo thống kê pixel và fit feature scaler chỉ trên train.
3. Trích xuất 10 feature từ ảnh [0,1]: mean, std, foreground ratio, bbox width/height, aspect ratio, centroid x/y, symmetry, edge density. Pixel và feature được standardize riêng. Không augmentation trong comparison này.
4. SmallMLP 784-128-10; MainMLP 784-256-128-10; MainMLP_FE 794-256-128-10. Cùng split/seed/config để so dung lượng và tác động FE. Cùng class FashionMLP.
5. Thực nghiệm optimizer riêng trên MainMLP pixel: SGD/RMSprop/Adam cùng LR=0.001; tuning độc lập 10 trial/optimizer, tối đa 10 epoch. Tune LR, weight_decay và momentum SGD. Chọn bằng validation; test chỉ đánh giá sau lựa chọn. Với 10 startup trials, lượt hiện tại chưa vào pha TPE thích nghi. Không mặc định tuned tốt hơn baseline.
6. Hình: learning curves, validation comparison, Accuracy/Macro F1, chi phí, confusion matrices, prediction grids; ROC+PR OvR một ảnh hai panel cho mỗi model. Không F1 heatmap hoặc biểu đồ cột TPR/FPR. Hiển thị chọn lọc nhưng lưu đủ artifact.
7. Hỗ trợ local CPU/CUDA và Kaggle; cấu hình nguồn ở config/workflow.json, seed=42, epoch=10, batch=128 cho kết quả hiện hành. Mọi run lưu config, environment, log, status, manifest, bảng/hình/checkpoint/prediction; run mới không ghi đè cũ.
8. Walkthrough ghi phương pháp, quyết định và kết quả có bằng chứng. Nhật ký chỉnh sửa và phân tích lỗi nội bộ tách khỏi nội dung dùng cho báo cáo. Chưa tạo PDF; mục tiêu bản xuất sau này >=25 trang bằng nội dung có bằng chứng.

## Quy ước tài liệu và lịch sử

README hướng dẫn chạy; description chứa yêu cầu và quyết định hiện hành; walkthrough trình bày bài làm/kết quả; internal_changes là nhật ký nội bộ. Bản trước hợp nhất, đề bài đã chú giải và các lựa chọn cũ được giữ nguyên trong archive/pre_cleanup_20261022. Không xóa outputs/runs. Danh mục run đang dùng nằm trong config/workflow.json, không tự chọn run mới nhất.

## Rà soát tính nhất quán sau triển khai

Confusion matrix phải ghi số ảnh ở mọi ô, màu biểu diễn tỷ lệ theo lớp thật và có colorbar. Roadmap phải liên kết file/hàm đang thực thi, phân biệt artifact sinh khi train và dữ liệu đọc khi view. Chỉ loại bỏ file không còn tác dụng sau khi kiểm tra tham chiếu; giữ module tương thích, notebook, bản lưu lịch sử và toàn bộ bằng chứng thí nghiệm.

## Bổ sung trình bày sau tìm siêu tham số

Notebook modeling hiển thị loss và accuracy train/validation của lượt refit best study, cùng history theo epoch. Cuối notebook phải có bước save/load model: lưu state_dict và kiến trúc/config, tải vào instance mới ở eval mode và kiểm tra dự đoán trước/sau, lưu toàn bộ bằng chứng và log vào run riêng.

Bổ sung làm rõ: loss/accuracy sau search phải so sánh đủ SGD, RMSprop, Adam với best parameters riêng; dùng history best completed trial và ghi rõ nguồn, không chỉ winner. Model tải lại phải đánh giá toàn bộ test và hiển thị mẫu đúng/sai với nhãn thật, nhãn dự đoán, xác suất. Mọi hình/bảng/metric có artifact và log.

Notebook EDA phải trình bày trực tiếp code trích xuất 10 feature và bảng giá trị từng ảnh; dùng source hiện hành, không duy trì bản sao công thức trong notebook.

Notebook EDA phải trình bày từng mục 1–7 đúng roadmap, output nằm ngay sau bước tương ứng. Phải phân biệt backend prepare_data chạy gộp và cell trình bày/kiểm chứng; bước FE phải gọi extractor trên ảnh thật rồi đối chiếu CSV/scaler/NPZ, không chỉ in source.
