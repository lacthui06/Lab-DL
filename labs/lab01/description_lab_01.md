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
- **So sánh 3 thuật toán Optimizer toàn diện:** Đánh giá 3 bộ tối ưu mà Optuna tìm được trên cả 2 tiêu chí cốt lõi: độ chính xác (Validation Accuracy) và chi phí thời gian huấn luyện thực tế (Wall-clock time in seconds / tốc độ bứt phá).
- **Sàng ngưỡng quyết định phân loại (Threshold Scan):** Tối ưu hóa điểm cắt xác suất để cực đại hóa F1-score cho bài toán phân loại, vẽ đồ thị và xuất bảng theo dõi trực quan.

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
- **Task 1 — Data Pipeline & Guard:**
  - Nạp dữ liệu qua `src.dataset.get_dataloaders()`, tuân thủ thứ tự transform: Augmentation -> ToTensor -> Normalize.
  - Chốt chặn Guard (assert) kiểm tra tính toàn vẹn của artifacts từ EDA và Feature Engineering trước khi cho phép huấn luyện.
- **Task 2 — Huấn luyện & Đánh giá Kiến trúc (FashionMLP):**
  - Kiến trúc: `Input(784) -> Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)`. Thuần túy 3 loại layer cơ bản, không BatchNorm, không Dropout.
  - Chốt chặn Sanity check 1-batch overfit kéo loss về sát 0 trước khi huấn luyện chính thức.
  - Huấn luyện mô hình, vẽ đường cong Loss & Accuracy, chẩn đoán Confusion Matrix và hiển thị lưới dự đoán 16 ảnh thực tế.
- **Task 3 — Dò tìm siêu tham số bằng Optuna & So sánh đối đầu 3 Optimizer:**
  - *Bước 3.1 (Tối ưu hóa bằng Optuna):* Thiết lập hàm mục tiêu `objective(trial)` chung trên tập validation 6.000 mẫu, chạy 12 trials có cơ chế Pruning (cắt tỉa sớm) để tự động tìm ra:
    + SGD: learning rate tối ưu trong $[10^{-2}, 10^{-1}]$ và momentum trong $[0.8, 0.95]$.
    + RMSprop: learning rate tối ưu trong $[10^{-4}, 10^{-2}]$.
    + Adam: learning rate tối ưu trong $[10^{-4}, 10^{-2}]$.
  - *Bước 3.2 (So sánh đối đầu Peak-to-Peak & Thời gian thực):* Lấy 3 cấu hình đỉnh cao nhất do Optuna tìm được đem so tài trực diện:
    + Đo đạc độ chính xác (Validation Accuracy).
    + Đo thời gian huấn luyện thực tế (số giây trên mỗi epoch, thời gian bứt phá chạm ngưỡng 87% accuracy).
    + Vẽ biểu đồ Val Accuracy theo thời gian thực (Wall-clock time in seconds) để thấy rõ ưu thế tốc độ hội tụ của Adam so với SGD.
    + Chọn ra Optimizer chiến thắng.
- **Task 4 — Sàng ngưỡng phân loại (Classification Decision Threshold Scan):**
  - Huấn luyện mô hình tối ưu với Optimizer thắng cuộc (bỏ Learning Rate Scheduler để giữ pipeline đơn giản, đánh giá năng lực tự thân của optimizer).
  - Quét 100 ngưỡng xác suất từ 0.0 đến 1.0.
  - Đo đạc đồng thời 4 chỉ số: Precision, Recall, F1-Score, Accuracy.
  - Vẽ đồ thị 4 đường kèm vạch đỏ đánh dấu ngưỡng tối ưu t* (nơi F1-Score đạt đỉnh cực đại).
  - Xuất bảng theo dõi quá trình tối ưu hóa ngưỡng (bước nhảy 0.05) y hệt hình mẫu thực tế.

---

## 3. Ràng buộc kỹ thuật (Constraints)

### 3.1. Đề bài gốc (Original Brief)
Use: PyTorch docs/tutorials.

### 3.2. Thống nhất thực hiện (Agreed Constraints)
- **Tự động hóa với Optuna có kiểm soát:** Giới hạn 10-12 trials nhanh và kích hoạt cơ chế `MedianPruner` (cắt tỉa ngay sau 3 epoch nếu trial hoạt động kém) để tiết kiệm tối đa GPU quota.
- **Bỏ Learning Rate Scheduler ở giai đoạn cuối:** Giữ learning rate cố định trong suốt quá trình train để đơn giản hóa pipeline, không làm phức tạp hóa mô hình, dành trọn trọng tâm cho bài toán Threshold Scan.
- **Quy chuẩn Notebook tinh gọn (Zero Waste):** Notebook mới giới hạn tối đa 12-15 cells, mỗi cell chỉ 3-6 dòng code gọi module từ `src/`, không chứa các khối logic nặng hay định nghĩa hàm cồng kềnh.
- **Tối ưu thời gian chạy GPU:** Hoàn thành toàn bộ quá trình (từ Baseline, Optuna, So sánh Optimizer đến Threshold Scan) trên Kaggle GPU trong **dưới 3-5 phút**.
- **Tính tái lập nghiêm ngặt (Strict Reproducibility):** Cố định seed cho Optuna sampler (`seed=42`), PyTorch, NumPy, Python random để loại trừ nhiễu phần cứng.

---

## 4. Sản phẩm cần nộp (Deliverables)

### 4.1. Đề bài gốc (Original Brief)
Deliver: Python code, brief report, loss graphs, image displays.

### 4.2. Thống nhất thực hiện (Agreed Deliverables)
- **Mã nguồn và Notebook:**
  - File notebook huấn luyện mới tinh gọn: `labs/lab01/notebooks/lab01_fashionmnist_walkthrough.ipynb` có đầy đủ output và hình ảnh trực quan.
  - Các module tiện ích bổ sung trong `labs/lab01/src/`: hàm tích hợp Optuna objective, hàm đo throughput thời gian, hàm quét ngưỡng và xuất đồ thị/bảng theo dõi Threshold Scan.
- **Bộ hình ảnh và biểu đồ chẩn đoán (thư mục `outputs/`):**
  - Đồ thị huấn luyện Loss & Accuracy của mô hình FashionMLP.
  - Bảng/Lịch sử thử nghiệm tham số của Optuna.
  - Đồ thị so sánh 3 Optimizer theo thời gian thực (Seconds).
  - Đồ thị quét ngưỡng Threshold Scan (Precision, Recall, F1, Accuracy) kèm vạch đỏ ngưỡng tối ưu và bảng theo dõi.
  - Ma trận nhầm lẫn (Confusion Matrix) và lưới 16 ảnh dự đoán đúng (xanh) / sai (đỏ).
- **Checkpoints & Báo cáo:**
  - Checkpoint mô hình tốt nhất (`outputs/checkpoints/`).
  - Toàn bộ kết quả định lượng được ghi nhận và phân tích trong tài liệu sống `walkthrough_lab_01.md`.
