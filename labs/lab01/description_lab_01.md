# Description — Lab 01: FashionMNIST với PyTorch

## 1. Đề bài gốc và Mục tiêu bài làm

### 1.1. Đề bài gốc

**Learn:** Study PyTorch Tensors, Datasets/Loaders, Transforms, Model Building, Autograd, Optimization, and Model Saving/Loading.

**Implement:** Load FashionMNIST, apply transforms, build a neural network, create a training loop, evaluate accuracy, save/load the model. Experiment with network/hyperparameters, visualize loss and predicted-vs-actual images. Deliver Python code, report, loss graphs and image displays. Use PyTorch documentation/tutorials.

### 1.2. Mục tiêu bài làm & Ranh giới mở rộng

- Giữ đề bài gốc làm đầu vào cốt lõi; mở rộng bài làm thành một pipeline hoàn chỉnh gồm: EDA, preprocessing, feature engineering, ba cấu hình MLP, training/validation, thí nghiệm optimizer riêng, đánh giá test đa chiều và save/load roundtrip.
- Trả lời 3 câu hỏi thực nghiệm trọng tâm:
  1. Một MLP nhỏ hơn (SmallMLP) có tiết kiệm tham số trong khi giữ hiệu năng hợp lý so với MainMLP không?
  2. Mười đặc trưng hình học/cường độ thủ công có cải thiện biểu diễn khi ghép nối cùng 784 pixel không (MainMLP vs MainMLP_FE)?
  3. Hiệu quả của 3 thuật toán tối ưu (SGD, RMSprop, Adam) thay đổi thế nào giữa điều kiện cùng learning rate và khi tìm siêu tham số riêng?

---

<a id="kiến-trúc-mô-hình-và-lý-do-lựa-chọn"></a>
## 2. Kiến trúc Hệ thống & Danh mục Mô-đun (System Architecture & Module Mapping)

Để đảm bảo nguyên tắc Clean Architecture và tính mô-đun hóa cao, các thành phần kỹ thuật chi tiết đã được tách rời thành các tài liệu đặc tả độc lập theo đúng tên từng module:

```text
labs/lab01/
├── models/            # Thư mục checkpoint bàn giao chính thức (best_model.pt, manifest.json)
│   └── README.md      # Hướng dẫn nạp nhanh mô hình phục vụ chấm bài và suy luận
├── src/
│   ├── datasets/      # Module xử lý dữ liệu, stratified split, trích xuất 10 feature
│   │   └── README.md  # Đặc tả chi tiết module datasets
│   ├── models/        # Module kiến trúc FashionMLP, khởi tạo Kaiming, 3 biến thể
│   │   └── README.md  # Đặc tả chi tiết module models
│   ├── training/      # Module huấn luyện, 1-batch sanity check, tuning Optuna
│   │   └── README.md  # Đặc tả chi tiết module training
│   ├── utils/         # Module quản lý run, seed, snapshot môi trường, vẽ đồ thị
│   │   └── README.md  # Đặc tả chi tiết module utils
│   ├── inference.py   # Module suy luận, nạp checkpoint alias, roundtrip check
│   └── workflow.py    # Kịch bản thực thi pipeline tự động end-to-end
```

### Tóm tắt các Module cốt lõi:

0. [**Models Checkpoint Handoff (`models/README.md`)**](models/README.md):
   - Nơi lưu trữ tập trung các checkpoint mô hình đã huấn luyện hoàn chỉnh để nộp bài (`best_model.pt`, `MainMLP.pt`, `SmallMLP.pt`, `MainMLP_FE.pt`).
   - Kèm file sổ cái `manifest.json` ghi vết nguồn gốc run, điểm validation, test accuracy và tham số.
   - Hỗ trợ nạp nhanh chỉ bằng 2 dòng code `load_model("final")`.

1. [**Models Module (`src/models/README.md`)**](src/models/README.md):
   - Định nghĩa lớp mạng `FashionMLP` thuần PyTorch (Linear -> ReLU -> Dropout).
   - Ba cấu hình đối sánh: **`SmallMLP`** (101.770 tham số), **`MainMLP`** (235.146 tham số), và **`MainMLP_FE`** (237.706 tham số).
   - Khởi tạo trọng số Kaiming Normal, cố định seed 42, đầu vào `(Batch, 784/794)` và đầu ra Logits `(Batch, 10)`.

2. [**Datasets Module (`src/datasets/README.md`)**](src/datasets/README.md):
   - Phân chia phân tầng 54.000 train / 6.000 val / 10.000 test (seed 42).
   - Chuẩn hóa Z-score pixel (`mean=0.2860, std=0.3530`) và fit `StandardScaler` cho 10 đặc trưng hình học duy nhất trên train (chống data leakage).
   - Đóng gói dữ liệu tiền xử lý vào file nén `data/prepared.npz`.

3. [**Training Module (`src/training/README.md`)**](src/training/README.md):
   - Chốt chặn 1-Batch Sanity Overfit Probe (8 mẫu, 120 bước, tiêu chí `loss < 0.2`, `train_acc > 80%`).
   - Huấn luyện đối sánh Fixed-LR trên 3 mô hình qua 10 epoch.
   - Tối ưu hóa siêu tham số bằng Optuna (10 trial độc lập cho SGD, RMSprop, Adam) và refit mô hình chiến thắng.

4. [**Utils Module (`src/utils/README.md`)**](src/utils/README.md):
   - Quản lý vòng đời thư mục thực nghiệm khép kín `outputs/runs/<run_id>/`.
   - Cố định seed toàn bộ môi trường, snapshot phiên bản phần cứng/thư viện.
   - Trực quan hóa chẩn đoán lỗi (Learning curves, Confusion matrix, ROC/PR OvR curves).

5. [**Inference & Serving (`src/inference.py` & `src/README.md`)**](src/README.md):
   - Giao diện nạp mô hình nhanh: `load_model("final")` hoặc `load_model("MainMLP")`.
   - Suy luận đơn mẫu `infer_sample()` và kiểm chứng độ trôi roundtrip `save_load_model()`.

---

## 3. Bảng Lộ Trình Triển Khai Toàn Diện (End-to-End Pipeline Execution Roadmap)

| Bước | Giai đoạn (Phase) | Dữ liệu đầu vào (Input & Input Path) | Xử lý gì (Nội dung chi tiết & Thuật toán) | Module & Hàm thực thi | Sản phẩm đầu ra (Output & Artifact Path) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Data Preparation & Feature Extraction** | - 70.000 ảnh xám `(28, 28)` uint8 `[0, 255]`.<br>- 70.000 nhãn `int64 0..9`.<br>*(Nguồn: `data/FashionMNIST/raw/`)* | 1. Phân tầng Stratified split 9:1 (54k train, 6k val, 10k test, seed 42).<br>2. Rescale `[0, 1]`, tính `mean=0.2860`, `std=0.3530` trên train, chuẩn hóa Z-score pixel.<br>3. Trích xuất 10 đặc trưng hình học/cường độ.<br>4. Fit `StandardScaler` chỉ trên train, transform val và test. | [`src/datasets/base.py`](src/datasets/base.py): `make_stratified_split`<br>[`src/datasets/features.py`](src/datasets/features.py): `extract_features`<br>[`src/datasets/preparation.py`](src/datasets/preparation.py): `prepare_data` | Mảng nén NumPy chứa:<br>- `train_pixels (54000, 784)`, `val_pixels (6000, 784)`, `test_pixels (10000, 784)` float32.<br>- `train_features (54000, 10)`, `val_features (6000, 10)`, `test_features (10000, 10)` float32.<br>- Labels `int64` và split indices.<br>*(File: `data/prepared.npz`)* |
| **2** | **1-Batch Sanity Overfit Probe** | - Instance mô hình `FashionMLP` chưa train.<br>- Mini-batch 8 mẫu từ train: `inputs (8, 784/794)` float32, `targets (8,)` int64.<br>*(Nguồn: `data/prepared.npz`)* | 1. Huấn luyện cục bộ 120 bước với Adam/SGD trên đúng 8 mẫu (không reset gradient).<br>2. Xác thực đường truyền gradient backward và khả năng kéo loss về sát 0.<br>3. Kiểm tra điều kiện: `loss < 0.2` và `train_acc > 80%`. | [`src/training/trainer.py`](src/training/trainer.py): `sanity_overfit_check` | File JSON ghi nhận kết quả kiểm chứng kiến trúc:<br>- `passed: True / False`<br>- `final_loss: float`<br>- `final_acc: float`<br>- `steps: int`<br>*(File: `outputs/runs/<run_id>/sanity.json`)* |
| **3** | **Model Training & Baseline Comparison** | - `DataLoader` train/val `(Batch=128, 784/794)`.<br>- 3 cấu hình: `SmallMLP` (101.770p), `MainMLP` (235.146p), `MainMLP_FE` (237.706p).<br>- Hyperparameters: Adam, LR=0.001, Epochs=10, CrossEntropyLoss. | 1. Huấn luyện 10 epoch: forward pass qua Linear -> ReLU -> Linear.<br>2. Tính `CrossEntropyLoss`, backward pass, Adam step.<br>3. Đánh giá loss và accuracy trên validation sau mỗi epoch.<br>4. Lưu checkpoint có `val_acc` cao nhất. | [`src/training/comparison.py`](src/training/comparison.py): `run_model`<br>[`src/training/trainer.py`](src/training/trainer.py): `Trainer.train_one_epoch`, `evaluate` | 1. `best.pt`: checkpoint trọng số tốt nhất (`state_dict` + metadata).<br>2. `history.json`: mảng loss/acc qua 10 epoch.<br>3. `config.json`: cấu hình siêu tham số.<br>4. `environment_snapshot.json`: snapshot môi trường phần cứng và thư viện.<br>*(Thư mục: `outputs/runs/<run_id>/`)* |
| **4** | **Hyperparameter Tuning (Optuna)** | - Pipeline huấn luyện `MainMLP` trên train/val split.<br>- Không gian tìm kiếm: LR [1e-4, 1e-1], weight_decay [1e-6, 1e-2], momentum [0, 0.95].<br>- 3 study độc lập cho SGD, RMSprop, Adam (10 trial/study). | 1. Khởi tạo Optuna study với sampler TPE cho từng optimizer.<br>2. Thực thi 10 trial lấy mẫu siêu tham số và đo `val_acc`.<br>3. Chọn cấu hình có `val_acc` cao nhất (và `val_loss` thấp nhất).<br>4. Refit mô hình chiến thắng trên toàn bộ tập train. | [`src/training/tuning.py`](src/training/tuning.py): `run_tuning`, `refit_winner` | 1. `best.pt`: checkpoint mô hình sau refit.<br>2. `best_params.json`: bộ siêu tham số tối ưu.<br>3. `config.json`: cấu hình run tối ưu.<br>4. `history.json`: tiến trình huấn luyện refit.<br>*(Thư mục: `outputs/runs/<run_id>/`)* |
| **5** | **Independent Evaluation & Diagnostics** | - Checkpoint `best.pt` ở chế độ `eval()`.<br>- Test DataLoader 10.000 mẫu `(Batch=128, 784/794)` và `test_labels (10000,)`. | 1. Feedforward suy luận `torch.no_grad()` trên 10.000 mẫu test.<br>2. Tính Logits -> Softmax probabilities `(10000, 10)` -> Argmax `(10000,)`.<br>3. Tính toán Test Loss, Accuracy, Precision, Recall/TPR, FPR cho 10 lớp.<br>4. Vẽ 3 hình trực quan hóa: Learning Curves, Confusion Matrix, Đường cong ROC/PR One-vs-Rest. | [`src/inference.py`](src/inference.py): `evaluate_run`<br>[`src/utils/plots.py`](src/utils/plots.py): `plot_loss_and_acc_curves`, `draw_confusion_counts`<br>[`src/utils/comparison_plots.py`](src/utils/comparison_plots.py): `build_comparison` | 1. Bảng số liệu: `metrics.json`, `classification_report.json` và `.csv`.<br>2. Mảng xác suất test: `test_predictions.npz`.<br>3. 3 đồ thị chẩn đoán lỗi: `learning_curves.png`, `confusion_comparison.png`, `*_ovr_curves.png`.<br>*(Thư mục: `outputs/runs/<run_id>/`)* |
| **6** | **Save/Load Verification & Inference Serving** | - Checkpoint `best.pt` hoặc alias `'final' / 'MainMLP' / 'SmallMLP'`.<br>- Ảnh đầu vào đơn mẫu: `(1, 28, 28)` uint8 hoặc Tensor batch 16 mẫu test đầu tiên. | 1. Nạp weights vào instance độc lập qua `load_model()`.<br>2. So sánh Softmax probability giữa trước và sau lưu trên 16 mẫu test (`torch.allclose` với `atol=1e-6`).<br>3. Suy luận đơn mẫu (`infer_sample`): chuẩn hóa, forward, Softmax, Argmax.<br>4. Đo benchmark latency (batch=1) và throughput (batch=128).<br>5. Xuất lưới 16 ảnh test (8 đúng viền xanh, 8 sai viền đỏ) kèm xác suất. | [`src/inference.py`](src/inference.py): `save_load_model`, `load_model`, `infer_sample`<br>[`src/utils/performance.py`](src/utils/performance.py): `benchmark_models` | 1. Trạng thái kiểm chứng roundtrip: `True`.<br>2. Đầu ra suy luận đơn mẫu: `predicted_class_id`, `class_name`, `confidence`.<br>3. Lưới 16 ảnh chẩn đoán: `loaded_model_test_samples.png`.<br>4. Báo cáo SLA tốc độ suy luận (ms/sample và samples/sec).<br>*(Thư mục: `outputs/runs/<run_id>/`)* |

## 4. Hợp đồng Dữ liệu Tổng thể & Danh mục Sản phẩm (Data Schemas & Deliverables)

### 4.1. Hợp đồng dữ liệu đầu vào tổng thể (Input Schema)
| Thành phần dữ liệu | Định dạng & Kiểu dữ liệu | Kích thước (Shape) | Giá trị / Miền giá trị | Ghi chú xử lý |
|---|---|---|---|---|
| **Ảnh thô (Raw Images)** | NumPy / Torch `uint8` | `(N, 28, 28)` | `[0, 255]` | Ảnh xám FashionMNIST gốc qua `torchvision.datasets`. |
| **Nhãn thô (Raw Labels)** | NumPy / Torch `int64` | `(N,)` | `0 .. 9` | 10 lớp thời trang rời rạc, mỗi ảnh đúng 1 nhãn. |
| **Tập chia (Data Splits)** | Stratified split (seed 42) | Train: 54.000<br>Val: 6.000<br>Test: 10.000 | Tỷ lệ 9:1 trên train-pool 60.000 | Lưu index cố định vào `data/prepared.npz` để tái lập. |
| **Pixel chuẩn hóa (Model Input - Pixel)** | Tensor `float32` | `(Batch, 784)` | Chuẩn hóa Z-score | Rescale `[0, 1]` rồi chuẩn hóa theo mean=0.2860, std=0.3530 đo từ tập train. |
| **Đặc trưng thủ công (Extracted Features)** | Tensor `float32` | `(Batch, 10)` | Chuẩn hóa `StandardScaler` | 10 đặc trưng hình học/cường độ (mean, std, fg_ratio, bbox_w, bbox_h, aspect, cx, cy, symmetry, edge). Fit scaler chỉ trên train. |
| **Đầu vào MainMLP_FE (Hybrid Input)** | Tensor `float32` | `(Batch, 794)` | Ghép nối tensor (Concat) | Ghép 784 pixel chuẩn hóa + 10 feature chuẩn hóa theo chiều ngang (`dim=1`). |

### 4.2. Hợp đồng đầu ra suy luận tổng thể (Output Schema)
| Thành phần đầu ra | Kiểu dữ liệu | Kích thước (Shape) | Miền giá trị | Mục đích sử dụng |
|---|---|---|---|---|
| **Raw Logits** | Tensor `float32` | `(Batch, 10)` | $(-\infty, +\infty)$ | Đầu ra trực tiếp của tầng Linear cuối; truyền thẳng vào `nn.CrossEntropyLoss`. |
| **Lớp dự đoán (Argmax)** | Tensor `int64` | `(Batch,)` | `0 .. 9` | Nhãn phân loại cuối cùng (`predicted_label = logits.argmax(dim=-1)`). |
| **Xác suất Softmax** | Tensor `float32` | `(Batch, 10)` | `[0.0, 1.0]`, $\sum = 1$ | Tính toán độ tin cậy mẫu, vẽ đường cong ROC/PR OvR và phân tích mẫu đúng/sai. |
| **Đánh giá Đa lớp** | Metric Dict / JSON | Scalar float | Accuracy, Macro F1, Log Loss, Class Report | Đánh giá hiệu năng tổng thể và chi tiết từng lớp theo Precision, Recall, Support. |

### 4.3. Danh mục sản phẩm giao nộp (Delivered Products & Artifacts)
| Loại sản phẩm | Đường dẫn / Tên file | Mô tả nội dung kỹ thuật |
|---|---|---|
| **Mã nguồn module** | `src/datasets/`, `src/models/`, `src/training/`, `src/utils/`, `src/inference.py` | Toàn bộ mã nguồn Python chuẩn module, không dùng biến toàn cục, có type hint và docstring. |
| **Notebook trình bày** | `notebooks/01_eda_preprocessing_features.ipynb`<br>`notebooks/02_modeling_comparison.ipynb` | Hai notebook đã thực thi đầy đủ kèm output: Roadmap 11 bước, hiển thị hình/bảng ngay sau bước thực thi. |
| **Bộ dữ liệu chuẩn hóa** | `data/prepared.npz` | Chứa toàn bộ mảng `train_pixels`, `val_pixels`, `test_pixels`, feature, nhãn và indices. |
| **Checkpoints mô hình bàn giao** | `models/best_model.pt`, `models/MainMLP.pt`, `models/manifest.json` | Bộ trọng số chính thức sẵn sàng nạp ngay để chấm bài và suy luận ([`models/README.md`](models/README.md)). |
| **Checkpoints thực nghiệm** | `outputs/runs/<run_id>/best.pt` | Trọng số mô hình tốt nhất (`model_state_dict`) và siêu dữ liệu validation (`metadata`) của từng run. |
| **Hồ sơ môi trường & Config** | `outputs/runs/<run_id>/environment_snapshot.json`<br>`outputs/runs/<run_id>/config.json` | Phiên bản Python, PyTorch, CUDA, platform và toàn bộ hyperparameter của từng lần chạy. |
| **Kết quả đánh giá** | `test_predictions.npz`, `metrics.json`, `classification_report.json/csv` | Lưu toàn bộ xác suất dự đoán test và bảng metric phân loại đầy đủ theo từng lớp. |
| **Minh họa trực quan** | `learning_curves.png`, `confusion_comparison.png`, `*_ovr_curves.png`, `loaded_model_test_samples.png` | Ảnh trực quan hóa chất lượng cao: ma trận nhầm lẫn có số đếm, đường cong ROC/PR OvR, lưới dự đoán 16 ảnh. |
| **Tài liệu học thuật** | `reports/walkthrough_lab_01.md`<br>`reports/walkthrough_lab_01.html` | Báo cáo kỹ thuật toàn diện, dẫn chiếu trực tiếp đến artifacts, bản HTML trực quan và chuẩn bị dàn trang PDF >= 25 trang. |

---

## 5. Tiêu chí nghiệm thu (Success Criteria)

Quy trình và sản phẩm của Lab 01 phải vượt qua toàn bộ 6 tiêu chí kiểm chứng kỹ thuật sau trước khi nghiệm thu, trong đó các chỉ số hiệu năng định lượng chỉ tập trung vào **Loss** và **Accuracy (Accu)**:

1. **Cổng kiểm tra 1-Batch Sanity Overfit (Pre-training Gate):**
   - Trước khi huấn luyện thật, mô hình bắt buộc chạy sanity overfit trên mini-batch 8 mẫu trong 120 bước.
   - Điều kiện đạt: `loss < 0.2` và `train_acc > 80%`. Nếu không đạt, pipeline tự động dừng và ghi log lỗi, không thực hiện full training.
2. **Kỷ luật chống rò rỉ dữ liệu (Data Leakage Protocol):**
   - Phân chia tập train (54.000) và validation (6.000) theo phương pháp phân tầng (stratified) với seed cố định 42.
   - Thống kê chuẩn hóa pixel (`mean=0.2860`, `std=0.3530`) và `StandardScaler` của 10 feature chỉ được `fit` duy nhất trên tập train; tập validation và test chỉ áp dụng phép biến đổi (`transform`).
   - Tập test (10.000) hoàn toàn bị cô lập: không tham gia vào tuning hyperparameter, không dùng để chọn epoch hoặc chọn optimizer thắng cuộc.
3. **Tiêu chí nghiệm thu sau khi huấn luyện toàn bộ (Post-Training Acceptance Criteria - Loss & Accu):**
   - **Mô hình Baseline (MainMLP - Fixed LR):**
     - `val_acc >= 87.0%` và `val_loss < 0.38`.
     - `test_acc >= 87.0%` và `test_loss < 0.38`.
   - **Mô hình Tối ưu (Final Model sau tuning / so sánh):**
     - `test_acc >= 88.0%` (kết quả thực tế đạt ~89.1% - 89.6%).
     - `test_loss < 0.35` (kết quả thực tế đạt ~0.30 - 0.31).
   - **Kiểm soát Overfitting (Generalization Gap):**
     - Độ chênh lệch giữa tập train và validation không vượt quá ngưỡng: `abs(train_acc - val_acc) <= 5.0%`.
4. **Tiêu chí tìm siêu tham số (Hyperparameter Tuning):**
   - 3 study độc lập cho SGD, RMSprop, Adam với 10 trial mỗi optimizer, không gian tìm kiếm LR, weight_decay, momentum theo đúng miền log-uniform đã định nghĩa.
   - Cấu hình chiến thắng được chọn dựa trên `val_acc` cao nhất (và `val_loss` thấp nhất), sau đó refit và đánh giá lại trên toàn bộ tập test để kiểm chứng tính tổng quát hóa.
5. **Kiểm chứng nạp/lưu mô hình (Save/Load Verification):**
   - Nạp checkpoint `best.pt` vào một instance mô hình mới độc lập ở chế độ `eval()`.
   - Kiểm tra sai số dự đoán (roundtrip verification): sai khác xác suất Softmax giữa trước và sau khi lưu phải đạt chuẩn `torch.allclose(atol=1e-6, rtol=1e-5)`.
   - Đánh giá lại mô hình đã tải trên toàn bộ 10.000 mẫu test: `test_acc` và `test_loss` phục hồi phải khớp 100% so với số liệu ghi nhận lúc huấn luyện.
6. **Tính toàn vẹn mã nguồn và khả năng tái lập (Reproducibility & Test Suite):**
   - Toàn bộ test suite trong `labs/lab01/tests/` phải pass 100% (49/49 tests).
   - Mọi số liệu `test_acc`, `test_loss`, `val_acc`, `val_loss` trong `reports/walkthrough_lab_01.md` phải truy vết được về file artifact JSON/CSV thực tế trong `outputs/runs/`, tuyệt đối không có số liệu giả lập hoặc ước tính cảm tính.

---

## 6. Phạm vi hiện hành đã thống nhất

1. Hai notebook: EDA/preprocessing/feature engineering và modeling/optimizer/tuning. Roadmap dạng bảng 5 cột ở đầu mỗi notebook. Chế độ view/train dùng cùng cell hiển thị, tránh phần đọc lại lặp cuối notebook.
2. FashionMNIST 28×28 grayscale, split stratified 54.000 train/6.000 validation, test gốc 10.000; giữ indices. Rescale /255, đo thống kê pixel và fit feature scaler chỉ trên train.
3. Trích xuất 10 feature từ ảnh [0,1]: mean, std, foreground ratio, bbox width/height, aspect ratio, centroid x/y, symmetry, edge density. Pixel và feature được standardize riêng. Không augmentation trong comparison này.
4. SmallMLP 784-128-10; MainMLP 784-256-128-10; MainMLP_FE 794-256-128-10. Cùng split/seed/config để so dung lượng và tác động FE. Cùng class FashionMLP.
5. Thực nghiệm optimizer riêng trên MainMLP pixel: SGD/RMSprop/Adam cùng LR=0.001; tuning độc lập 10 trial/optimizer, tối đa 10 epoch. Tune LR, weight_decay và momentum SGD. Chọn bằng validation; test chỉ đánh giá sau lựa chọn. Với 10 startup trials, lượt hiện tại chưa vào pha TPE thích nghi. Không mặc định tuned tốt hơn baseline.
6. Hình: learning curves, validation comparison, Accuracy/Macro F1, chi phí, confusion matrices, prediction grids; ROC+PR OvR một ảnh hai panel cho mỗi model. Không F1 heatmap hoặc biểu đồ cột TPR/FPR. Hiển thị chọn lọc nhưng lưu đủ artifact.
7. Hỗ trợ local CPU/CUDA và Kaggle; cấu hình nguồn ở config/workflow.json, seed=42, epoch=10, batch=128 cho kết quả hiện hành. Mọi run lưu config, environment, log, status, manifest, bảng/hình/checkpoint/prediction; run mới không ghi đè cũ.
8. Walkthrough ghi phương pháp, quyết định và kết quả có bằng chứng. Nhật ký chỉnh sửa và phân tích lỗi nội bộ tách khỏi nội dung dùng cho báo cáo. Chưa tạo PDF; mục tiêu bản xuất sau này >=25 trang bằng nội dung có bằng chứng.

---

## 7. Thảo luận và quyết định trước triển khai (Lịch sử truy vết)

Mục này phục hồi các vấn đề đã được ghi trong những bản description trước khi hợp nhất, đồng thời nối tiếp các yêu cầu bổ sung đã chốt trong cuộc trao đổi hiện tại. Đây là bản đối chiếu quyết định theo tài liệu còn lưu, không phải bản chép nguyên văn toàn bộ lịch sử chat. Các câu hỏi bên dưới diễn đạt lại chủ đề thảo luận; không coi là trích dẫn lời người dùng. Phương án cũ được ghi rõ trạng thái để không áp dụng nhầm vào pipeline hiện hành.

### 1. Mục tiêu bài làm và ranh giới với đề bài gốc
**Vấn đề đã làm rõ:** cần chỉ hoàn thành các thao tác PyTorch cơ bản hay mở rộng thành quy trình có EDA, so sánh kiến trúc và optimizer?
**Đã chốt:** giữ đề bài gốc làm đầu vào; mở rộng bài làm thành EDA/preprocessing/feature engineering, ba cấu hình MLP, training/validation, thí nghiệm optimizer riêng, đánh giá test và save/load. Description ghi yêu cầu, lựa chọn và tiêu chí nghiệm thu; walkthrough trình bày phương pháp/kết quả; README hướng dẫn chạy. Không trộn nhật ký sửa phần mềm vào báo cáo.
**Ghi chú nguồn đề:** bản description đầu tiên ghi tiêu đề “Exercise: PyTorch FashionMNIST Classification” và một danh sách 1–5 thiếu nội dung trong bản sao đề. Không tự đặt nội dung cho danh sách trống hoặc biến yêu cầu mở rộng thành lời nguyên văn của giảng viên.

### 2. Dữ liệu, split và chuẩn hóa lấy từ đâu?
**Đã chốt:** FashionMNIST thông qua torchvision; tập train gốc 60.000 ảnh chia stratified thành 54.000 train và 6.000 validation; test gốc 10.000 ảnh giữ riêng. Lưu indices để tái dùng đúng mẫu giữa các model. Kiểm tra dữ liệu và đo thống kê trước modeling; không dùng thống kê validation/test để fit preprocessing.
**Điều chỉnh so với bản cũ:** bản trước từng ghi cố định mean/std vào nội dung phạm vi. Pipeline hiện hành phải đọc thống kê train từ artifact, không lấy một hằng số trong tài liệu làm nguồn chuẩn hóa cho mọi run. Pixel rescale [0,255] → [0,1] trước khi trích feature; pixel và feature được standardize riêng.

### 3. Có augmentation và feature engineering không?
**Phương án từng ghi:** pipeline transform nối cấu hình augmentation YAML theo thứ tự Augmentation → ToTensor → Normalize.
**Quyết định hiện hành:** không augmentation trong comparison ba model. Trích đúng 10 feature từ ảnh [0,1]: mean, std, foreground ratio, bbox width/height, aspect ratio, centroid x/y, symmetry và edge density; ngưỡng foreground 0,1, định nghĩa rõ ảnh rỗng. Scaler feature chỉ fit train. Không giữ cấu hình augmentation cũ như một bước đang chạy. Quyết định này tránh việc feature được tính từ ảnh gốc nhưng pixel đầu vào đã bị biến đổi khác.

### 4. Vì sao MLP và vì sao chuyển từ một model sang ba cấu hình?
**Phương án ban đầu:** một FashionMLP 784 → 256 → 128 → 10, chỉ dùng Flatten, Linear và ReLU; không BatchNorm, không Dropout.
**Đã chốt sau đó:** SmallMLP 784 → 128 → 10 làm baseline gọn; MainMLP 784 → 256 → 128 → 10 khảo sát tăng dung lượng; MainMLP_FE 794 → 256 → 128 → 10 khảo sát thêm 10 feature. Cùng class FashionMLP, cùng split/seed và điều kiện so sánh. Small/Main thay đổi cả độ sâu và độ rộng; Main/Main_FE thay đổi đầu vào và số tham số lớp đầu. Không tuyên bố đã cô lập từng yếu tố hoặc đã tìm được kiến trúc tối ưu.

### 5. Kiểm tra gì trước khi huấn luyện chính thức?
**Đã chốt:** có kiểm tra hợp đồng dữ liệu/split, kích thước đầu vào 784/794, forward/backward và sanity overfit một mini-batch. Trong runner hiện hành, sanity học 8 mẫu trong 120 bước, đạt khi loss < 0,2 và accuracy > 80%; sau đó reset seed, model và loader trước training chính thức. Sanity thất bại thì dừng và lưu log. Đây là kiểm tra khả năng học/gradient, không phải kết quả đánh giá tổng quát hóa.
**Điều chỉnh tên gọi:** các tên guard/loader từng có trong source cũ chỉ là tham chiếu lịch sử; không coi `verify_pipeline_handoff` hoặc `src.dataset.get_dataloaders` là entry point bắt buộc của source hiện hành.

### 6. So sánh optimizer như thế nào để không đổi nhiều yếu tố cùng lúc?
**Đã chốt:** thí nghiệm optimizer tách riêng trên MainMLP nhận pixel. Gồm hai kịch bản: SGD momentum 0,9, RMSprop momentum 0 và Adam cùng LR=0,001/weight_decay=0; sau đó mỗi optimizer có study tuning riêng. Giữ seed, split, batch size và ngân sách epoch để đối chiếu; không giả định cùng số trial có nghĩa cùng thời gian thực khi có pruning.
**Phương án cũ đã thay thế:** bản đầu đề xuất đối đầu “peak-to-peak”, đo giây/epoch, thời gian đạt 87% validation accuracy và vẽ validation accuracy theo wall-clock. Mục tiêu diễn đạt sẵn ưu thế của Adam trong bản cũ không phải kết luận được chấp nhận trước khi có dữ liệu. Hiện giữ thời gian thực đã ghi và yêu cầu đo tài nguyên training bổ sung; chưa có bằng chứng xác nhận biểu đồ time-to-87% cho pipeline hiện hành. Không dùng benchmark ms/sample mới để thay thế phép đo hội tụ theo thời gian đó.

### 7. Tuning bao nhiêu trial, tìm tham số gì và có scheduler không?
**Phương án từng ghi:** 10–12 hoặc 12 trial nhanh, có bản ghi 3 epoch/trial; LR SGD từng bắt đầu từ 1e-2. Giai đoạn cuối từng đề xuất bỏ scheduler để phục vụ threshold scan. Đây là các thiết kế lịch sử, không phải cấu hình hiện hành.
**Đã chốt hiện hành:** 10 trial độc lập cho mỗi SGD/RMSprop/Adam, tối đa 10 epoch/trial, batch 128, seed 42, CosineAnnealingLR, MedianPruner. LR log-uniform: SGD [1e-3,1e-1], RMSprop/Adam [1e-4,1e-2]; weight_decay log [1e-7,1e-2]; momentum SGD [0,8;0,95]. Chọn best validation accuracy; test không tham gia objective. Refit cấu hình thắng theo validation rồi đánh giá test.
**Giới hạn phải nói rõ:** 10 trial hiện tại bằng 10 startup trials nên không mô tả là đã khai thác pha TPE thích nghi. Baseline weight_decay=0 chưa nằm trong miền log search, chưa enqueue baseline; không mặc định tuning phải thắng fixed-LR. Mọi thay đổi ngân sách sau này phải được ghi thành thí nghiệm mới.

### 8. Threshold scan hay argmax, ROC và Precision–Recall?
**Phương án cũ đã ghi:** quét 100 ngưỡng xác suất 0–1, theo dõi Precision/Recall/F1/Accuracy, đánh dấu ngưỡng tối đa F1 và xuất bảng bước 0,05. Không phục hồi yêu cầu này thành một bước đã thực hiện ở pipeline hiện hành.
**Quyết định hiện hành:** dự đoán 10 lớp bằng argmax; đánh giá ROC AUC và Average Precision theo one-vs-rest. ROC và PR ghép một ảnh hai panel/model, mỗi panel có 10 đường theo lớp. Không tự thêm threshold scan, pAUC, micro/macro AUC lịch sử hoặc dùng test để chọn ngưỡng.

### 9. Notebook phải trình bày những gì và tránh lặp ra sao?
**Phương án ban đầu:** một notebook gọn, có bản đặt giới hạn 12–15 cell, mỗi cell gọi module; đường dẫn notebook cũ thay đổi giữa các phiên bản.
**Đã chốt hiện hành:** hai notebook `01_eda_preprocessing_features.ipynb` và `02_modeling_comparison.ipynb`. Roadmap đầu mỗi notebook có 5 cột: Bước, Giai đoạn, Nội dung chi tiết, Module triển khai, Output/Artifact. Cell gọi code trong src; output nằm ngay sau bước tương ứng, không có phần đọc lại trùng ở cuối. EDA trình bày các mục 1–7 theo roadmap; phân biệt backend chạy gộp với các cell giải thích/kiểm chứng. Feature extraction phải chạy trên ảnh thật, hiển thị giá trị và đối chiếu CSV/scaler/NPZ, không chỉ in source.
**Chốt visualization:** giữ loss/accuracy train–validation, so sánh validation, Accuracy/Macro F1, chi phí, confusion matrix và prediction grid. F1 heatmap và biểu đồ cột TPR/FPR được bỏ; không bỏ các metric TPR/FPR trong báo cáo theo lớp. Confusion matrix có số ở mọi ô, màu chuẩn hóa theo lớp thật và colorbar. Bỏ yêu cầu cũ tự hiển thị toàn bộ PNG và PR/ROC từng trial; artifact vẫn lưu đủ.

### 10. Sau tuning và save/load phải kiểm chứng gì?
**Đã chốt:** hiển thị history và loss/accuracy train–validation của refit; đồng thời so sánh history best completed trial của đủ SGD/RMSprop/Adam với best parameters riêng, ghi rõ trial hay refit. Không chỉ hiển thị optimizer thắng.
Lưu state_dict và cấu hình kiến trúc, khởi tạo instance mới, load ở eval mode, kiểm tra dự đoán/xác suất trước và sau tải. Đánh giá model đã tải trên toàn bộ test, có nhãn thật, nhãn dự đoán và xác suất cho mẫu đúng/sai. Các phiên bản từng yêu cầu lưới 16 ảnh; yêu cầu hiện hành làm rõ tối đa 8 đúng và 8 sai, có bổ sung khi một nhóm thiếu. Hình minh họa không thay thế metric của toàn bộ test.

### 11. Môi trường chạy, quản lý artifact và báo cáo
**Đã chốt theo yêu cầu người dùng:** hỗ trợ local CPU/CUDA và Kaggle; GPU tăng tốc nhưng không là điều kiện bắt buộc của API Lab 1. Ghi thiết bị thực của từng run để diễn giải thời gian; không coi kết quả CPU và GPU là cùng điều kiện. Mục tiêu cũ hoàn thành toàn pipeline trên Kaggle trong 3–5 phút chưa được xác minh và không dùng làm cam kết thời gian hiện hành.
Giữ cấu trúc `src/datasets`, `src/models`, `src/training`, `src/utils`, notebook ngắn gọi module; config hiện hành là `config/workflow.json`, không buộc khôi phục YAML cũ. Mỗi run có config, môi trường, log stdout/stderr, status, manifest và artifact; không ghi đè run cũ, thất bại/pruned vẫn có log. Không chọn run mới nhất tự động. Walkthrough là báo cáo có bằng chứng, không chèn metric của K-fold/sweep hoặc kiến trúc lịch sử chưa xác minh. Chưa xuất PDF; mục tiêu bản dàn trang sau này ít nhất 25 trang, số trang chỉ xác nhận khi render thật. Các đường dẫn checkpoint/ZIP cố định trong bản cũ được thay bằng run riêng và publication chọn lọc; không coi ZIP cũ là artifact phải sinh lại ở mọi lần chạy.

### 12. Yêu cầu bổ sung về kiến trúc, performance và monitoring
**Đã chốt trong trao đổi bổ sung:** description in nguyên kiến trúc dạng text từng layer của cả ba model và lý do chọn, không chỉ ghi chuỗi kích thước. Notebook modeling in cùng kiểu kiến trúc, giữ train/validation accuracy và loss theo epoch, thêm time/sample và RAM–VRAM/sample.
Người dùng đã chọn **chỉ đo training**, không đo inference. Phép đo bổ sung dùng vài batch trên checkpoint bản sao trong process riêng; nêu rõ batch size, thiết bị, warmup, số lần lặp, cách chuẩn hóa theo số mẫu và phạm vi bộ nhớ. Không giả dữ liệu VRAM từ run CPU hoặc gán số đo mới cho lịch sử training cũ.
Các trao đổi về TensorBoard/timm là làm rõ công cụ hiện có, chưa phải yêu cầu chuyển sang TensorBoard hoặc thêm timm. Monitoring hiện dùng console.log, history CSV/JSON và hình; việc thêm dashboard cần được chốt riêng. Tương tự, kiểm tra duplicate hiện hành là ảnh trùng chính xác, không đồng nghĩa đã kiểm tra near-duplicate.

### Nguồn khôi phục và cách cập nhật tiếp
- `fa42fa2` và `6fbe29f`: description ghi đề bài gốc, vai trò tài liệu và ghi chú danh sách 1–5 trống trong bản sao đề.
- `fb20fa8`, file `labs/lab01/description_lab_01.md`: phạm vi ban đầu, threshold scan, so optimizer theo thời gian thực, ngân sách và phương án scheduler cũ.
- `9a0bd1a`, cùng file: phạm vi một FashionMLP, Core EDA/FE, optimizer fixed-LR/Optuna và ROC/save-load.
- `e0fd041`, file `labs/lab01/archive/pre_cleanup_20261022/description_lab_01.md`: các cập nhật đã chốt ngày 22/10/2026 về hai notebook, ba model, FE, môi trường, visualization, tuning và roadmap.
- Các phần “Bổ sung trình bày sau tìm siêu tham số”, “Viết lại báo cáo đầy đủ” và trao đổi hiện tại: yêu cầu bổ sung về trình bày, save/load, kiến trúc và benchmark training.

---

## 8. Quy ước tài liệu và lịch sử

README hướng dẫn chạy; description chứa yêu cầu và quyết định hiện hành; walkthrough trình bày bài làm/kết quả; internal_changes là nhật ký nội bộ. Bản trước hợp nhất, đề bài đã chú giải và các lựa chọn cũ được truy vết qua lịch sử Git (commit e0fd041). Không xóa outputs/runs. Danh mục run đang dùng nằm trong config/workflow.json, không tự chọn run mới nhất.
