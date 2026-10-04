# Đặc Tả Kỹ Thuật (ML/DL Specification) — Lab 01: FashionMNIST Classification

> **Dự án:** Phân loại trang phục FashionMNIST với PyTorch  
> **Người thực hiện:** AI Assistant (Antigravity) & Student  
> **Mã bài tập:** Lab 01 — Deep Learning Labs  
> **Tài liệu tham chiếu:** [`dl_workflows/DL_flow.md`](../../dl_workflows/DL_flow.md), [`dl_workflows/dl_data_and_augmentation_guide.md`](../../dl_workflows/dl_data_and_augmentation_guide.md), [`dl_workflows/dl_training_and_optimization_guide.md`](../../dl_workflows/dl_training_and_optimization_guide.md)

---

## 1. Phân loại Bài toán & Phương thái (Problem & Modality)
- **Phương thái (Modality):** Computer Vision (CV) — Ảnh đơn kênh (Grayscale Image Classification).
- **Dạng bài toán:** Phân loại đa lớp (Multi-class Classification) với 10 lớp thời trang rời rạc:
  - 0: T-shirt/top (Áo thun)
  - 1: Trouser (Quần dài)
  - 2: Pullover (Áo len chui đầu)
  - 3: Dress (Đầm / Váy liền)
  - 4: Coat (Áo khoác)
  - 5: Sandal (Giày sandal)
  - 6: Shirt (Áo sơ mi)
  - 7: Sneaker (Giày thể thao)
  - 8: Bag (Túi xách)
  - 9: Ankle boot (Ủng / Bốt cổ thấp)
- **Định dạng dữ liệu đầu vào:** Tensor ảnh 2D `[Batch_Size, 1, 28, 28]` với kiểu dữ liệu `torch.float32`.
- **Định dạng đầu ra:** Vector Logits `[Batch_Size, 10]`, dự đoán lớp $\hat{y} = \arg\max(\text{logits})$.

---

## 2. Hợp đồng Dữ liệu & Phân tách (Data Contract & Split Strategy)
- **Kích thước tập dữ liệu:**
  - Tập thô chính thức: 60,000 ảnh huấn luyện, 10,000 ảnh kiểm thử (Test).
  - Kích thước ảnh: $28 \times 28$ pixel, 1 kênh màu (Grayscale), pixel ban đầu $[0, 255]$.
- **Phân bố nhãn (Class Balance):**
  - Cân bằng hoàn hảo: 6,000 mẫu/lớp trong tập huấn luyện, 1,000 mẫu/lớp trong tập kiểm thử. Không gặp bẫy mất cân bằng nhãn (Class Imbalance).
- **Chiến lược phân tách (Data Split):**
  - **Train set:** 54,000 mẫu ($90\%$ của tập train gốc).
  - **Validation set:** 6,000 mẫu ($10\%$ của tập train gốc) — dùng để theo dõi loss, chọn hyperparameter và checkpointing.
  - **Test set:** 10,000 mẫu độc lập — chỉ đánh giá 1 lần duy nhất ở khâu nghiệm thu cuối cùng.
  - Cố định `seed = 42` khi thực hiện `torch.utils.data.random_split` để đảm bảo $Train \cap Val = \emptyset$ và hoàn toàn tái lập.
- **Tiền xử lý & Chuẩn hóa:**
  - Chuyển đổi sang Tensor: giá trị pixel quy về $[0.0, 1.0]$.
  - Chuẩn hóa theo Mean và Std thực nghiệm của FashionMNIST: $\mu = 0.2860, \sigma = 0.3205$.
  - Tăng cường dữ liệu (cho các Run nâng cao): `RandomHorizontalFlip(p=0.5)` và nhẹ nhàng `RandomRotation(degrees=10)`. Không lật dọc vì quần áo có phương hướng trên-dưới cố định.

---

## 3. Mô hình Cơ sở & Chỉ số Mục tiêu (Baseline & Target Metrics)
- **Baseline Model (Run #1):**
  - Kiến trúc: Simple Multi-Layer Perceptron (MLP) 2 tầng.
  - Cấu trúc: `Flatten` $\to$ `Linear(784, 128)` $\to$ `ReLU()` $\to$ `Linear(128, 10)`.
  - Mục đích: Thiết lập mốc điểm số tối thiểu nhanh chóng.
- **Mô hình Cải tiến (Run #2 & Run #3):**
  - **Run #2 (Tuned MLP):** Deep MLP 3 tầng có `BatchNorm1d`, `Dropout(0.2)`, `AdamW`, và `CosineAnnealingLR`.
  - **Run #3 (FashionCNN):** Kiến trúc Mạng Nơ-ron Tích chập (CNN) gồm 2 khối Convolutional (`Conv2d` $\to$ `BatchNorm2d` $\to$ `ReLU` $\to$ `MaxPool2d`) + Head phân loại với `Dropout(0.3)`.
- **Chỉ số đo lường chính (Primary Metrics):**
  - **Top-1 Accuracy:** Tỷ lệ phân loại chính xác trên tập Validation và Test.
  - **Macro F1-Score:** Đo lường độ chuẩn xác trung bình trên 10 lớp nhằm phát hiện xem có lớp nào bị bỏ sót hay không.
  - **Cross-Entropy Loss:** Đo lường độ hội tụ và rủi ro Overfitting.
- **Ngưỡng nghiệm thu (Acceptance Thresholds):**
  - Baseline MLP: Accuracy $\ge 85\%$.
  - Tuned MLP: Accuracy $\ge 88\%$.
  - FashionCNN: Accuracy $\ge 91\%$, vượt Baseline tối thiểu $+4\%$.

---

## 4. Hàm Mất Mát & Tối Ưu Hóa (Loss & Optimization Design)
- **Hàm mất mát (Loss Function):**
  - `nn.CrossEntropyLoss()`: Kết hợp `LogSoftmax` và `NLLLoss` đảm bảo tính ổn định số học (Numerical Stability).
- **Chiến lược Tối ưu (Optimization Strategy):**
  - Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)` giúp điều hòa trọng số hiệu quả hơn Adam tiêu chuẩn.
  - Bộ điều chỉnh Learning Rate: `CosineAnnealingLR` hạ dần tốc độ học theo hàm Cosine tới $10^{-5}$ ở epoch cuối cùng.
  - Vòng lặp tối ưu: Sử dụng `optimizer.zero_grad(set_to_none=True)` để tiết kiệm bộ nhớ và tăng tốc xử lý gradient.

---

## 5. Ràng buộc Môi Trường & Bộ Kiểm Thử Bắt Buộc
- **Môi trường:**
  - Python 3.13, PyTorch 2.14.0, Torchvision 0.29.1, Scikit-Learn 1.9.1, Matplotlib 3.11.2.
  - Hỗ trợ cả thiết bị CPU và CUDA GPU (tự động nhận diện thiết bị).
- **Mã nguồn tái lập:**
  - `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`.
- **5 Bài Kiểm thử Bắt buộc (TDD):**
  1. Kiểm thử kích thước và tính toàn vẹn dữ liệu.
  2. Kiểm thử cơ chế nạp Tensor (`torch.float32`, shape `[1, 28, 28]`).
  3. Kiểm thử truyền thuận (Forward pass) khớp shape đầu ra `[B, 10]`.
  4. **Sanity Overfit 1 Batch Test:** Bắt buộc mô hình ép được Loss $\to 0.0$ và Accuracy $= 100\%$ trên 8 mẫu trước khi train toàn bộ.
  5. Kiểm thử Lưu & Nạp mô hình (Save/Load Checkpoint equivalence).
