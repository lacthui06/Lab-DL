# Module models — FashionMLP Architectures & Factory

## 1. Thông Tin Mô-đun (Name & Metadata)
- **Tên mô-đun:** `src/models` (FashionMLP Architectures, Parameter Management & Model Factory)
- **Ngày tạo (Created):** 2026-10-22 · **Tác giả (Author):** Thái Anh Lạc
- **Ngày cập nhật (Updated):** 2026-11-10 · **Người cập nhật (Author):** Thái Anh Lạc
- **Tệp tài liệu chính thức:** `labs/lab01/src/models/README.md`

---

## 2. Mục Đích (Purpose)
Module này là bộ khung nơ-ron chính của bài Lab 01, giải quyết 2 bài toán thực nghiệm cốt lõi được đặc tả trong [Description Mục 1.2](../../description_lab_01.md#12-mục-tiêu-bài-làm--ranh-giới-mở-rộng):
1. **Khảo sát dung lượng mạng (Capacity Study):** Thiết kế mạng nhỏ 1 tầng ẩn (`SmallMLP` ~101k tham số) để đối sánh với mạng chuẩn 2 tầng ẩn (`MainMLP` ~235k tham số), nhằm đánh giá sự đánh đổi giữa kích thước mô hình, thời gian huấn luyện và độ chính xác phân loại.
2. **Khảo sát giá trị đặc trưng thủ công (Feature Engineering Study):** Thiết kế biến thể mạng (`MainMLP_FE` ~237k tham số) nhận đầu vào là 784 điểm ảnh kết hợp cùng 10 đặc trưng hình học/độ sáng, nhằm kiểm chứng giả thuyết liệu tri thức thủ công có hỗ trợ mạng nơ-ron phân biệt các loại trang phục tốt hơn hay không.

Module cung cấp:
- Lớp mạng nơ-ron thuần PyTorch [`FashionMLP`](mlp.py#L8) dùng chung cho cả 3 cấu hình.
- Hàm khởi tạo [`build_model()`](__init__.py#L8) linh hoạt điều chỉnh số chiều đầu vào (`in_features`) và danh sách tầng ẩn (`hidden_dims`).
- Hàm kiểm tra tham số [`count_parameters()`](__init__.py#L20) xác thực 100% tham số đều tham gia tính toán gradient (`requires_grad=True`).

---

## 3. Khởi Tạo Mô Hình (Init Model)

### 3.1. Kiến Trúc Mô Hình (Model Architecture)
Cả 3 mô hình đều được xếp tầng theo dạng hình phễu (thu nhỏ dần từ số điểm ảnh về 10 lớp phân loại):

```text
1. SmallMLP (Baseline nhỏ gọn - 1 tầng ẩn - 101.770 tham số):
   Ảnh vào (Batch, 784) ──> Flatten
                        ──> Linear(784, 128) ──> ReLU 
                        ──> Linear(128, 10)  ──> Logits (Batch, 10)

2. MainMLP (Baseline tiêu chuẩn - 2 tầng ẩn - 235.146 tham số):
   Ảnh vào (Batch, 784) ──> Flatten
                        ──> Linear(784, 256) ──> ReLU 
                        ──> Linear(256, 128) ──> ReLU 
                        ──> Linear(128, 10)  ──> Logits (Batch, 10)

3. MainMLP_FE (Biến thể kết hợp 10 đặc trưng - 2 tầng ẩn - 237.706 tham số):
   Ảnh + 10 FE (Batch, 794) ──> Flatten
                            ──> Linear(794, 256) ──> ReLU 
                            ──> Linear(256, 128) ──> ReLU 
                            ──> Linear(128, 10)  ──> Logits (Batch, 10)
```

- **Quy mô tham số:**
  - `SmallMLP`: **101.770** tham số.
  - `MainMLP`: **235.146** tham số.
  - `MainMLP_FE`: **237.706** tham số (chỉ tăng đúng 2.560 trọng số ở tầng đầu tiên, tức tăng vỏn vẹn **+1.09%** so với `MainMLP`, tạo điều kiện đối sánh rất công bằng).
- **Khởi tạo trọng số (Weight Initialization):**
  - Mặc định PyTorch tự động áp dụng thuật toán **Kaiming Uniform** cho các tầng `nn.Linear` khi đi kèm hàm kích hoạt `ReLU`, giúp giữ phương sai tín hiệu ổn định qua các tầng và chống tiêu biến đạo hàm.

### 3.2. Đầu Vào và Đầu Ra của Mô Hình (Input / Output Model)

| Tên mô hình | Đầu vào (Input Tensor) | Đầu ra (Output Tensor) | Ý nghĩa nghiệp vụ |
| :--- | :--- | :--- | :--- |
| **`SmallMLP`** | `(Batch, 784)` · `torch.float32` | `(Batch, 10)` · `torch.float32` | Điểm ảnh Z-score chuẩn hóa $\rightarrow$ 10 Logits thô. |
| **`MainMLP`** | `(Batch, 784)` · `torch.float32` | `(Batch, 10)` · `torch.float32` | Điểm ảnh Z-score chuẩn hóa $\rightarrow$ 10 Logits thô. |
| **`MainMLP_FE`** | `(Batch, 794)` · `torch.float32` | `(Batch, 10)` · `torch.float32` | 784 pixel Z-score ghép nối 10 feature $\rightarrow$ 10 Logits thô. |

- **Tại sao không đặt Softmax trong mô hình?**
  - Khi huấn luyện: Đầu ra là Logits thô đưa thẳng vào [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). PyTorch tự động gộp Log-Softmax và NLLLoss ở tầng C++ giúp tính toán ổn định và chống tràn số học.
  - Khi suy luận: Chỉ khi cần xem phần trăm độ tự tin hoặc vẽ đường cong ROC/PR, ta mới gọi [`torch.softmax()`](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) và lấy [`torch.argmax()`](https://pytorch.org/docs/stable/generated/torch.argmax.html) để chọn nhãn có xác suất cao nhất.

### 3.3. File Mã Nguồn & Điểm Gọi (Source Files)

Giữ phím **`Ctrl + Click`** để mở thẳng đến dòng code:

| Tệp mã nguồn | Ký hiệu / Hàm chính (Ctrl+Click) | Dòng code | Việc đảm nhiệm |
| :--- | :--- | :--- | :--- |
| [`mlp.py`](mlp.py) | [`class FashionMLP`](mlp.py#L8) | [`mlp.py#L8-L36`](mlp.py#L8-L36) | Định nghĩa mạng nơ-ron kế thừa `nn.Module`, lắp ghép `nn.Sequential` gồm `Flatten`, `Linear` và `ReLU`, chạy hàm `forward()`. |
| [`__init__.py`](__init__.py) | [`build_model()`](__init__.py#L8)<br>[`count_parameters()`](__init__.py#L20) | [`__init__.py#L8-L28`](__init__.py#L8-L28) | Factory function tạo instance theo tên và hàm kiểm tra toàn bộ tham số có bật học (`requires_grad=True`) không. |

- **Điểm gọi trong code:**
  - Sanity check 8 mẫu: [`trainer.py#L136`](../training/trainer.py#L136)
  - Huấn luyện 3 mô hình so sánh: [`comparison.py#L49`](../training/comparison.py#L49)
  - Khảo sát Optimizer: [`tuning.py#L104`](../training/tuning.py#L104)
  - Nạp mô hình suy luận: [`inference.py#L100`](../inference.py#L100)
- **Checkpoint bàn giao tại thư mục `models/`:**
  - [`labs/lab01/models/best_model.pt`](../../models/best_model.pt): Checkpoint tối ưu nhất sau tuning và refit.
  - [`labs/lab01/models/MainMLP.pt`](../../models/MainMLP.pt): Checkpoint cấu hình chuẩn 235k tham số.
  - [`labs/lab01/models/SmallMLP.pt`](../../models/SmallMLP.pt): Checkpoint cấu hình nhỏ 101k tham số.
  - [`labs/lab01/models/MainMLP_FE.pt`](../../models/MainMLP_FE.pt): Checkpoint cấu hình kết hợp 10 đặc trưng.
  - [`labs/lab01/models/manifest.json`](../../models/manifest.json): Sổ cái JSON lưu vết xuất xứ từng checkpoint.

---

## 4. Các Thảo Luận & Quyết Định Kỹ Thuật (Discussions from Description)

### 4.1. Vì sao dùng MLP thuần mà không dùng CNN? ([Description Mục 7.4](../../description_lab_01.md#4-vì-sao-mlp-và-vì-sao-chuyển-từ-một-model-sang-ba-cấu-hình))
- Mục tiêu trọng tâm của Lab 01 là nắm chắc các thao tác nền tảng của PyTorch (Tensor, Autograd, Module, vòng lặp Optimizer).
- Dùng MLP phẳng giúp đo lường năng lực học thô của mạng nơ-ron truyền thẳng khi không có tích chập không gian, tạo mốc đối chiếu (baseline) chuẩn xác để làm nổi bật sự vượt trội của kiến trúc CNN ở Lab 02.

### 4.2. Vì sao kiên quyết không dùng BatchNorm và Dropout? ([Description Mục 7.4](../../description_lab_01.md#4-vì-sao-mlp-và-vì-sao-chuyển-từ-một-model-sang-ba-cấu-hình))
- Quyết định kỹ thuật đã chốt là loại bỏ hoàn toàn các kỹ thuật này để đảm bảo tính công bằng khi so sánh dung lượng mạng.
- Nếu thêm Dropout hoặc BatchNorm, ta sẽ không thể phân biệt được sự cải thiện là do bản thân dung lượng nơ-ron hay do cơ chế regularization ngẫu nhiên làm giảm hiện tượng học vẹt. Mô hình được giữ thuần khiết nhất: chỉ dùng `Flatten`, `Linear` và `ReLU`.

### 4.3. So sánh SmallMLP vs MainMLP — Khảo sát dung lượng tổng thể ([Description Mục 7.4](../../description_lab_01.md#4-vì-sao-mlp-và-vì-sao-chuyển-từ-một-model-sang-ba-cấu-hình))
- Cần làm rõ về mặt lý thuyết: `SmallMLP` (784 $\rightarrow$ 128 $\rightarrow$ 10) khác `MainMLP` (784 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 10) ở **cả độ sâu** (1 tầng ẩn vs 2 tầng ẩn) **lẫn độ rộng** (128 vs 384 nơ-ron ẩn).
- Do đó, thiết kế này đóng vai trò là một cuộc khảo sát dung lượng mạng tổng thể (composite capacity study), chứ không tuyên bố là đã cô lập riêng rẽ yếu tố độ sâu mạng.

### 4.4. Giả thuyết kỹ thuật của biến thể MainMLP_FE ([Description Mục 7.3, 7.4](../../description_lab_01.md#3-có-augmentation-và-feature-engineering-không))
- Biến thể `MainMLP_FE` chỉ tăng thêm 1.09% số tham số ($10 \times 256 = 2.560$ trọng số ở tầng đầu), đảm bảo sự đối đầu diễn ra rất sòng phẳng.
- Quyết định này nhằm kiểm chứng một câu hỏi kỹ thuật lớn: Liệu việc bổ sung tri thức chuyên gia (10 đặc trưng hình học/độ sáng tính tay) có giúp mạng nơ-ron tăng cường khả năng biểu diễn so với việc chỉ học từ 784 điểm ảnh thô hay không, và sự chênh lệch đó có bù đắp được chi phí tính toán trích xuất đặc trưng hay không.
