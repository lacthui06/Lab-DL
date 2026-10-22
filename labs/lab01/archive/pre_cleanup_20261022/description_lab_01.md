# Description — Lab 01: PyTorch FashionMNIST Classification

> **Lab:** Lab 01

---

## Ghi chú về file này

Đây là bản ghi **đề bài gốc** (đầu vào) do giảng viên giao kèm phần **làm rõ và thống nhất phạm vi thực hiện** sau khi thảo luận chi tiết.

Bài làm và toàn bộ kết quả thực nghiệm nằm ở `walkthrough_lab_01.md` (đầu ra). Hai file bổ sung cho nhau:
- `description_lab_01.md`: đề bài nhận vào và phạm vi đã chốt.
- `walkthrough_lab_01.md`: bài làm và báo cáo kỹ thuật hoàn chỉnh.

---

## 1. Bối cảnh và Mục tiêu

### 1.1. Đề bài gốc (Original Brief)
Learn: Study PyTorch Tensors, Datasets/Loaders, Transforms, Model Building, Autograd, Optimization, and Model Saving/Loading.

### 1.2. Thống nhất thực hiện (Agreed Scope & Objectives)
- **Chuẩn hóa dữ liệu theo EDA/FE:** Sử dụng dữ liệu đã qua phân tầng nghiêm ngặt (Stratified Split 54,000 train / 6,000 val) và bộ tham số chuẩn hóa đo đạc thực nghiệm (Global Mean = 0.286209, Global Std = 0.353160).
- **Kiến trúc mô hình thuần PyTorch (FashionMLP):** Sử dụng một kiến trúc mạng MLP duy nhất được xây dựng từ đúng 3 loại layer cơ bản: `nn.Flatten`, `nn.Linear`, `nn.ReLU` (không BatchNorm, không Dropout) với cấu hình 3 tầng Linear (`784 -> 256 -> 128 -> 10`) vừa vặn, đạt dung lượng biểu diễn tốt.
- **Tự động dò tìm siêu tham số bằng Optuna:** Áp dụng framework Optuna (thuật toán Bayesian Optimization TPE) để tự động tìm ra bộ learning rate và momentum tối ưu nhất cho từng thuật toán (SGD + Momentum, RMSprop, Adam) theo cùng một tiêu chí công bằng (Validation Accuracy).
- **So sánh 3 thuật toán Optimizer toàn diện (Fixed LR vs Optuna):** Đánh giá 3 bộ tối ưu (SGD + Momentum, RMSprop, Adam) qua 2 kịch bản: dùng chung Learning Rate `1e-3` và tự động dò tìm siêu tham số đỉnh cao bằng Optuna (Peak-to-Peak kết hợp `CosineAnnealingLR`).
- **Đánh giá đường cong ROC-AUC đa lớp (One-vs-Rest) & Kiểm chứng Model Roundtrip:** Tính xác suất phân lớp softmax trên 10 lớp, vẽ đường cong ROC đa lớp độc lập với ngưỡng quyết định, lưu/tải lại checkpoint mô hình chiến thắng và đánh giá kiểm chứng trên 10,000 ảnh tập Test kèm lưới 16 ảnh dự đoán trực quan.

---

## 2. Yêu cầu (Requirements & Tasks)

### 2.1. Đề bài gốc (Original Brief)
Implement:
- Load FashionMNIST dataset.
- Apply transforms.
- Build a neural network.
- Create a training loop (forward, loss, backward, optimize).
- Evaluate model accuracy.
- Save and load the model.

Tasks:
- Experiment with network/hyperparameters.
- Visualize loss.
- Display predicted vs actual images.

### 2.2. Thống nhất thực hiện (Agreed Tasks Breakdown)
- **Task 1 — 9-Step Core EDA, Feature Engineering & Pipeline Guard:**
  - Thực hiện đầy đủ 9 bước Core EDA theo `docs/eda-image.md` & `docs/feature-image.md`, xuất 7 bảng thống kê CSV vào `outputs/eda/tables/` và 7 biểu đồ PNG vào `outputs/eda/figures/`.
  - Nạp dữ liệu qua `src.datasets.classification.get_dataloaders()`, kết nối cấu hình `config/augmentations.yaml`, tuân thủ thứ tự transform: Augmentation -> ToTensor -> Normalize.
  - Chốt chặn Guard (`verify_pipeline_handoff`) kiểm tra tính toàn vẹn của 4 tiêu chí bàn giao từ EDA và Feature Engineering trước khi cho phép huấn luyện.
- **Task 2 — Huấn luyện & Đánh giá Kiến trúc (FashionMLP):**
  - Kiến trúc: `Input(784) -> Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)` (`235,146` tham số). Thuần túy 3 loại layer cơ bản (`Flatten`, `Linear`, `ReLU`).
  - Chốt chặn Sanity check 1-batch overfit kéo loss về sát 0 (`0.0005 < 0.15`) trước khi huấn luyện chính thức.
  - Huấn luyện mô hình 10 epochs với `Adam` + `CosineAnnealingLR`, vẽ đường cong Loss & Accuracy, chẩn đoán Confusion Matrix đếm số nguyên 10x10 và hiển thị lưới dự đoán 16 ảnh thực tế.
- **Task 3 — So sánh 3 Optimizer (Cùng LR 1e-3 vs. Dò tìm siêu tham số bằng Optuna):**
  - *Bước 3.1 (Kịch bản 1 — Cùng Learning Rate `1e-3`):* So sánh trực diện SGD (`momentum=0.9`), RMSprop (`momentum=0.0`), và Adam ở cùng bước nhảy `1e-3` trong 10 epochs để quan sát tốc độ hội tụ tự nhiên.
  - *Bước 3.2 (Kịch bản 2 — Tối ưu hóa bằng Optuna & Đua Peak-to-Peak):* Thiết lập hàm mục tiêu `objective(trial)` với `TPESampler(seed=42)` và `MedianPruner`, chạy 12 trials để tự động tìm Learning Rate (và Momentum cho SGD) tối ưu nhất cho từng thuật toán, sau đó huấn luyện đối đầu 10 epochs kèm `CosineAnnealingLR` để chọn ra `winner_model`.
- **Task 4 — Khảo sát ROC-AUC Đa lớp (One-vs-Rest) & Kiểm chứng Model Roundtrip:**
  - Đánh giá năng lực phân biệt của `winner_model` qua đường cong One-vs-Rest ROC và điểm AUC cho từng lớp (kèm Micro-average và Macro-average AUC).
  - Lưu trữ checkpoint `outputs/checkpoints/fashion_mlp_final.pt`, khởi tạo mô hình sạch, nạp lại trọng số qua `load_checkpoint()`, đánh giá độc lập trên 10,000 ảnh tập Test (`Test Accuracy = 89.30%`), hiển thị lưới 16 ảnh dự đoán (`test_prediction_grid.png`) và đóng gói toàn bộ `outputs/` thành `all_outputs.zip`.

---

## 3. Ràng buộc kỹ thuật (Constraints)

### 3.1. Đề bài gốc (Original Brief)
Use: PyTorch docs/tutorials.

### 3.2. Thống nhất thực hiện (Agreed Constraints)
- **Tự động hóa với Optuna có kiểm soát:** Giới hạn 12 trials nhanh (3 epochs/trial) và kích hoạt cơ chế `MedianPruner` để tiết kiệm tối đa GPU/CPU quota.
- **Kiến trúc Modular Clean Architecture:** Phân tách rõ ràng 4 phân hệ trong `src/` (`datasets/`, `models/`, `training/`, `utils/`), kết nối đồng bộ với `config/*.yaml` và `scripts/*.py`.
- **Quy chuẩn Notebook tinh gọn (Zero Waste):** Notebook `notebooks/lab01.ipynb` gồm các cell súc tích gọi trực tiếp hàm từ `src/`, không chứa các khối định nghĩa hàm/class cồng kềnh và hoàn toàn không chứa emoji/icon.
- **Tính tái lập nghiêm ngặt (Strict Reproducibility) & Không dùng số liệu giả:** Cố định seed (`seed=42`) cho Optuna sampler, PyTorch, NumPy, Python random; tuyệt đối không dùng hằng số chuẩn hóa giả lập (mọi thống kê phải đọc từ `preprocessing_config.json`).

---

## 4. Sản phẩm cần nộp (Deliverables)

### 4.1. Đề bài gốc (Original Brief)
Deliver: Python code, brief report, loss graphs, image displays.

### 4.2. Thống nhất thực hiện (Agreed Deliverables)
- **Mã nguồn và Notebook:**
  - File notebook tương tác hoàn chỉnh: `labs/lab01/notebooks/lab01.ipynb` có đầy đủ output thực thi và hình ảnh trực quan (bao gồm đủ 7 biểu đồ EDA, biểu đồ huấn luyện, so sánh optimizer, ROC-AUC, và lưới kiểm chứng tập Test).
  - Gói thư viện chuẩn hóa trong `labs/lab01/src/` (`datasets/`, `models/`, `training/`, `utils/`), bộ CLI trong `labs/lab01/scripts/` (`train.py`, `eval.py`, `infer.py`) và 22 bài test trong `labs/lab01/tests/`.
- **Bộ hình ảnh và biểu đồ chẩn đoán (thư mục `outputs/`):**
  - 7 bảng CSV thống kê (`outputs/eda/tables/*.csv`) và 7 biểu đồ EDA (`outputs/eda/figures/*.png`).
  - Đồ thị huấn luyện Loss & Accuracy (`outputs/predictions/learning_curves.png`).
  - Ma trận nhầm lẫn (`outputs/predictions/confusion_matrix.png`) và lưới 16 ảnh dự đoán tập Validation (`outputs/predictions/prediction_grid.png`) + tập Test (`outputs/predictions/test_prediction_grid.png`).
  - Đồ thị so sánh 3 Optimizer ở cùng LR `1e-3` (`optimizer_fixed_lr_comparison.png`) và sau khi tinh chỉnh bằng Optuna (`optimizer_comparison.png`).
  - Đồ thị đường cong ROC-AUC đa lớp One-vs-Rest (`outputs/predictions/roc_curves.png`).
- **Checkpoints & Báo cáo:**
  - Checkpoint mô hình tốt nhất (`outputs/checkpoints/fashion_mlp_best.pt`, `outputs/checkpoints/fashion_mlp_final.pt`).
  - Toàn bộ kết quả định lượng được ghi nhận và phân tích trong tài liệu sống `walkthrough_lab_01.md`.


## Phạm vi cập nhật đã thống nhất ngày 22/10/2026

Mục này thay thế các mô tả phạm vi cũ mâu thuẫn ở trên; đề bài gốc được giữ để truy vết.

- Hai notebook: `01_eda_preprocessing_features.ipynb` và `02_modeling_comparison.ipynb`. Notebook `lab01.ipynb` giữ làm lịch sử, không phải entry point cho pipeline mới.
- EDA -> rescaling [0,255] thành [0,1] -> trích xuất feature từ ảnh [0,1]. Pixel được standardize theo mean/std train; 10 feature được scale bằng thống kê chỉ fit train.
- Ba cấu hình: SmallMLP 784-128-10; MainMLP 784-256-128-10; MainMLP_FE 794-256-128-10. So sánh dung lượng (Small/Main) và tác động FE (Main/Main_FE).
- Feature: mean, std, foreground ratio, bbox width/height, aspect ratio, centroid x/y, bilateral symmetry, edge density. Ngưỡng foreground cố định 0.1; mọi công thức và trường hợp ảnh rỗng phải xác định rõ.
- Cùng split và quy trình chọn best validation checkpoint. Test không dùng để chỉnh hyperparameter. Không augmentation ở comparison đầu tiên để tránh feature không khớp ảnh biến đổi.
- Output: learning curves, validation comparison, Accuracy/Macro F1, F1 heatmap theo lớp, confusion matrix, prediction grid chung 16 ảnh, ROC, PR micro và PR theo lớp, chi phí huấn luyện. Không thêm biểu đồ TPR/FPR.
- Mỗi run có thư mục riêng, config, môi trường, log stdout/stderr, trạng thái, manifest SHA256 và artifact bảng/hình/checkpoint/prediction. Run thất bại cũng phải lưu log.
- Optimizer comparison là thực nghiệm riêng trên MainMLP; Optuna chỉ báo cáo kết quả thực chạy.
- Walkthrough ghi quyết định, triển khai, artifact và kết quả đã xác minh. Chưa train thì ghi chưa có kết quả. Nhật ký sửa đổi và phân tích lỗi chi tiết giữ nội bộ; không đưa vào báo cáo xuất sau này.
- Chưa tạo PDF. Mục tiêu báo cáo sau này >=25 trang, không dùng nội dung chưa xác minh để tăng độ dài.


### Cập nhật môi trường theo yêu cầu người dùng
Full training được phép trên cả local CPU/CUDA và Kaggle. Tự chọn CUDA nếu có, CPU nếu không; không giới hạn số epoch theo thiết bị. GPU là lựa chọn tăng tốc, không phải điều kiện bắt buộc. Mỗi run ghi thiết bị thực tế để so sánh thời gian đúng ngữ cảnh.


### Điều chỉnh hiển thị modeling
Bỏ F1 heatmap. ROC-AUC và Precision–Recall one-vs-rest được ghép thành một ảnh hai panel cho mỗi model, tổng ba ảnh. Mỗi panel có 10 đường theo lớp. Notebook không tự hiển thị mọi PNG hoặc lặp learning curves; bảng/hình còn lại vẫn lưu artifact. Confusion matrix giữ vai trò đánh giá riêng.


### Optimizer và hyperparameter tuning đã thống nhất

Giữ nguyên ba model comparison. Thực nghiệm riêng trên MainMLP pixel: (1) SGD momentum 0.9/RMSprop/Adam cùng lr=0.001, weight_decay=0; (2) ba study độc lập, mỗi optimizer 10 trials, tối đa 10 epoch/trial, cùng batch=128/seed=42/CosineAnnealingLR và MedianPruner. LR log-uniform SGD [1e-3,1e-1], RMSprop/Adam [1e-4,1e-2]; weight_decay [1e-7,1e-2] log; momentum SGD [0.8,0.95]. Ngân sách bằng số trial và epoch tối đa, không giả định cùng thời gian thực vì pruning. Chọn theo best validation accuracy; test không tham gia objective. Refit cấu hình thắng theo validation và đánh giá test một lần. Mỗi candidate có config/log/history/checkpoint; study SQLite lưu trạng thái trials/pruning. Không vẽ PR/ROC cho từng trial. Hình: fixed-LR validation curves, best score theo trial, LR-score; bảng best config và thời gian. Hỗ trợ local CPU/CUDA và Kaggle.


### Roadmap và confusion matrix
Bổ sung Pipeline Execution Roadmap đầu hai notebook, ghi bước, mục tiêu và artifact. Hiển thị confusion matrix của ba model và bản refit, thay vì chỉ lưu ảnh trong artifact. Giữ nguyên việc bỏ F1 heatmap; confusion matrix là output đánh giá riêng. Không thay đổi cấu hình/search budget của lượt tuning đã chạy.


Roadmap đầu hai notebook đã chuyển sang bảng 5 cột theo mẫu người dùng: Bước, Giai đoạn, Nội dung chi tiết, Module triển khai, Output/Artifact. Nội dung và đường dẫn khớp pipeline run-based hiện tại; không đổi code hoặc chạy lại thí nghiệm.
