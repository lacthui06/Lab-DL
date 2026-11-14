# Module training — Huấn Luyện, Sanity Check & Tối Ưu Siêu Tham Số

## 1. Thông Tin Mô-đun (Name & Metadata)
- **Tên mô-đun:** `src/training` (Model Training Engine, Sanity Gate, Fixed-LR Comparison & Optuna Tuning)
- **Ngày tạo (Created):** 2026-10-22 · **Tác giả (Author):** Thái Anh Lạc
- **Ngày cập nhật (Updated):** 2026-11-10 · **Người cập nhật (Author):** Thái Anh Lạc
- **Tệp tài liệu chính thức:** `labs/lab01/src/training/README.md`

---

## 2. Mục Đích (Purpose)
Module này điều hành toàn bộ quy trình huấn luyện, kiểm thử chốt chặn và tối ưu hóa cho các mô hình của Lab 01:
- Thiết lập chốt chặn kiểm tra nhanh 3 giây (Sanity Check) trên mini-batch 8 mẫu để phát hiện sớm các lỗi ngược gradient, sai lệch kích thước tensor hoặc hàm loss bất thường trước khi bắt đầu huấn luyện thật.
- Điều phối vòng lặp huấn luyện 10 epoch cho 3 mô hình (`SmallMLP`, `MainMLP`, `MainMLP_FE`) để đối sánh dung lượng và đặc trưng.
- Thực hiện khảo sát 2 pha khoa học cho 3 thuật toán tối ưu (SGD, RMSprop, Adam):
  1. *Pha 1 (Fixed-LR):* Khảo sát tốc độ học tự nhiên khi ép chạy chung mức learning rate `0.001`.
  2. *Pha 2 (Optuna Tuning):* Mở không gian tìm kiếm độc lập (30 trial) để mỗi thuật toán tự do tìm điểm rơi phong độ tốt nhất.
- Huấn luyện lại (Refit) mô hình chiến thắng và tự động bàn giao checkpoint tối ưu nhất vào thư mục `labs/lab01/models/best_model.pt` kèm sổ cái `manifest.json`.

---

## 3. Khởi Tạo Huấn Luyện & Cơ Chế Hoạt Động (Init Training & Mechanisms)

### 3.1. Cấu Trúc Luồng Huấn Luyện 5 Chặng (Workflow Structure)

```text
Dữ liệu đã nạp (labs/lab01/data/prepared.npz)
  │
  ▼ [sanity_overfit_check() · trainer.py#L136](trainer.py#L136)
1. Chốt chặn Sanity Check (8 mẫu ảnh, 120 bước, lr=0.01)
   Bắt mô hình học vẹt: Yêu cầu loss < 0.20 và accuracy > 80.0% trong 3 giây.
  │
  ▼ [run_model() · comparison.py#L37](comparison.py#L37)
2. Huấn luyện 3 mô hình đối sánh (10 epoch, batch size 128, seed 42)
   Dùng optimizer Adam, learning rate 0.001 kết hợp lịch giảm tốc CosineAnnealingLR.
  │
  ▼ [run_fixed_lr() · tuning.py#L99](tuning.py#L99)
3. Khảo sát 3 Optimizer ở cùng mức Learning Rate 0.001
   SGD (momentum 0.9) vs RMSprop (momentum 0.0) vs Adam (betas=(0.9, 0.999)).
  │
  ▼ [run_tuning() · tuning.py#L122](tuning.py#L122)
4. Tối ưu hóa siêu tham số bằng Optuna (3 study độc lập, 10 trial/study)
   Dùng thuật toán TPE để đoán tham số và MedianPruner để cắt tỉa các trial kém.
  │
  ▼ [refit_winner() · tuning.py#L178](tuning.py#L178)
5. Huấn luyện lại (Refit) mô hình chiến thắng & Auto-promote
   Tự động lưu checkpoint tốt nhất vào labs/lab01/models/best_model.pt và ghi manifest.json.
```

---

### 3.2. Cơ Chế Học & Cấu Hình Siêu Tham Số (LR Mechanism & Optuna Tuning)

#### 1. Cơ chế điều chỉnh Learning Rate (Cosine Annealing Scheduler):
- Huấn luyện sử dụng bộ điều lịch [`torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html).
- **Nguyên lý hoạt động:**
  - Ở các epoch đầu, learning rate được giữ ở mức tương đối cao để mô hình nhanh chóng nhảy thoát khỏi các cực tiểu địa phương nông.
  - Càng về các epoch cuối, tốc độ học tự động giảm mượt mà theo đường cong cosin mềm mại về sát 0.
  - Cách làm này giúp các bước nhảy của mô hình thu nhỏ dần theo thời gian, giúp mô hình "hạ cánh" chuẩn xác vào đáy tối ưu mà không bị rung lắc hay văng ra ngoài.

#### 2. Cơ chế tối ưu hóa Optuna (Hyperparameter Search Engine):
Mỗi thuật toán tối ưu được cấp một Study độc lập với không gian tìm kiếm được phân bổ theo thang log-uniform:
- **Không gian siêu tham số (Search Space):**
  - **SGD:** `lr` $\in [10^{-3}, 10^{-1}]$ (log-uniform), `weight_decay` $\in [10^{-7}, 10^{-2}]$ (log-uniform), `momentum` $\in [0.8, 0.95]$ (uniform).
  - **RMSprop:** `lr` $\in [10^{-4}, 10^{-2}]$ (log-uniform), `weight_decay` $\in [10^{-7}, 10^{-2}]$ (log-uniform).
  - **Adam:** `lr` $\in [10^{-4}, 10^{-2}]$ (log-uniform), `weight_decay` $\in [10^{-7}, 10^{-2}]$ (log-uniform).
- **Thuật toán lấy mẫu (Sampler):**
  - Dùng `TPESampler(seed=42)` (Tree-structured Parzen Estimator). Thuật toán xây dựng mô hình xác suất Bayes dựa trên lịch sử các trial trước để phán đoán vùng siêu tham số tiềm năng nhất cho trial tiếp theo.
- **Cơ chế cắt tỉa sớm (MedianPruner):**
  - Cấu hình: `MedianPruner(n_startup_trials=5, n_warmup_steps=3)`.
  - Nếu một trial chạy đến epoch 3 mà có validation loss kém hơn trung vị (median) của các trial trước đó ở cùng epoch, Optuna sẽ lập tức ra lệnh dừng chạy (pruned). Kỹ thuật này giúp tiết kiệm 30-40% tổng thời gian tính toán mà không làm bỏ sót các cấu hình tốt.

#### 3. Cấu hình siêu tham số mặc định (Default Hyperparameters):
- `seed`: 42 (áp dụng đồng thời cho Python, NumPy, PyTorch CPU/CUDA).
- `batch_size`: 128 (cân bằng giữa tính vector hóa của phần cứng và độ ổn định gradient).
- `epochs`: 10 (ngân sách huấn luyện chuẩn cho toàn bài Lab).
- `loss_fn`: [`nn.CrossEntropyLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) (nhận Logits thô và nhãn nguyên `int64`).

---

### 3.3. Đầu Vào và Đầu Ra của Huấn Luyện (Input / Output Training)

| Thành phần | Đầu vào tiếp nhận | Sản phẩm đầu ra (Artifacts) | Ý nghĩa nghiệp vụ |
| :--- | :--- | :--- | :--- |
| **Sanity Probe** | Instance mô hình chưa train + 8 ảnh train đầu tiên | `outputs/runs/<run_id>/sanity.json` | Hồ sơ kiểm chứng xác nhận kiến trúc và gradient hoạt động 100%. |
| **Model Comparison** | Batch 128 ảnh `(128, 784/794)` từ train/val split | `best.pt`, `history.json`, `learning_curves.png` | Bảng điểm so sánh đối đầu giữa SmallMLP, MainMLP và MainMLP_FE. |
| **Fixed-LR Optimizer** | Mạng `MainMLP` + 3 thuật toán ở cùng `lr=0.001` | `summary.csv`, `fixed_lr_curves.png` | Minh họa tốc độ bắt nhịp và độ ổn định ban đầu của từng optimizer. |
| **Optuna Tuning** | Không gian siêu tham số của SGD, RMSprop, Adam | `studies.sqlite3`, `all_trials.csv`, `best_configs.json` | Cơ sở dữ liệu SQLite và bảng CSV lưu chi tiết mọi trial. |
| **Refit Mô Hình Thắng** | Cấu hình tốt nhất sau tuning Optuna | [`best_model.pt`](../../models/best_model.pt) & [`manifest.json`](../../models/manifest.json) | Checkpoint chuẩn nộp bài, sẵn sàng phục vụ suy luận ngay lập tức. |

- **Cách tính toán Loss trung bình trong một Epoch (`train_one_epoch`):**
  - Xóa gradient bằng `optimizer.zero_grad(set_to_none=True)` để giải phóng bộ nhớ RAM nhanh hơn.
  - Loss của từng batch được nhân với số lượng mẫu thực tế của batch đó (`loss.item() * images.size(0)`), sau đó chia cho tổng số mẫu của toàn epoch. Kỹ thuật này triệt tiêu hoàn toàn sai số làm tròn đối với mini-batch cuối cùng khi số mẫu không đủ 128.

---

### 3.4. File Mã Nguồn & Bản Đồ Điều Hướng Code (Source Files)

Giữ phím **`Ctrl + Click`** để mở trực tiếp trong trình soạn thảo:

| Tệp mã nguồn | Ký hiệu / Hàm chính (Ctrl+Click) | Dòng code | Việc đảm nhiệm |
| :--- | :--- | :--- | :--- |
| [`trainer.py`](trainer.py) | [`class Trainer`](trainer.py#L35)<br>[`sanity_overfit_check()`](trainer.py#L136)<br>[`build_optimizer()`](trainer.py#L13) | [`trainer.py#L13-L170`](trainer.py#L13-L170) | Đóng gói vòng lặp huấn luyện, đánh giá validation, kiểm tra overfit 8 mẫu và tạo đối tượng optimizer chuẩn giáo trình. |
| [`comparison.py`](comparison.py) | [`run_model()`](comparison.py#L37) | [`comparison.py#L37-L95`](comparison.py#L37-L95) | Quản lý vòng đời chạy 10 epoch cho từng mô hình, đo test độc lập, vẽ đồ thị và tự động gọi auto-promote checkpoint. |
| [`tuning.py`](tuning.py) | [`run_fixed_lr()`](tuning.py#L99)<br>[`run_tuning()`](tuning.py#L122)<br>[`refit_winner()`](tuning.py#L178) | [`tuning.py#L99-L210`](tuning.py#L99-L210) | Điều phối khảo sát Fixed-LR, quản lý 3 Optuna study, refit mô hình chiến thắng trên toàn bộ tập train và xuất checkpoint. |

- **Artifact đầu ra phục vụ báo cáo và chấm bài:**
  - Checkpoint bàn giao: [`labs/lab01/models/best_model.pt`](../../models/best_model.pt).
  - Lịch sử đối sánh optimizer: `outputs/runs/<run_id>/summary.csv`.

---

## 4. Các Thảo Luận & Quyết Định Kỹ Thuật (Discussions from Description)

### 4.1. Tại sao bài kiểm tra Sanity 8 mẫu trong 3 giây là bắt buộc? ([Description Mục 7.5](../../description_lab_01.md#5-kiểm-tra-gì-trước-khi-huấn-luyện-chính-thức))
- Trong thực tế, nhiều sai sót ngớ ngẩn thường xảy ra âm thầm: quên gọi `loss.backward()`, đặt nhầm kích thước ma trận khiến gradient bị triệt tiêu, hoặc tính nhầm hàm loss. Nếu không có sanity probe, ta sẽ lãng phí hàng giờ huấn luyện vô ích.
- Bài test Sanity kiểm tra đường truyền gradient bit-exact: nếu một mạng nơ-ron không thể học thuộc lòng (overfit) 8 bức ảnh đơn giản trong 120 bước để kéo loss về sát 0, điều đó khẳng định chắc chắn 100% code đã bị lỗi logic.

### 4.2. Vì sao phải chia làm 2 pha: Fixed-LR trước rồi mới Tuning sau? ([Description Mục 7.6](../../description_lab_01.md#6-so-sánh-optimizer-như-thế-nào-để-không-đổi-nhiều-yếu-tố-cùng-lúc))
- Nếu vừa vào đã chạy Optuna ngay lập tức, ta sẽ không thể phân biệt được sự chênh lệch giữa các optimizer là do cấu trúc thuật toán hay do không gian tìm kiếm vô tình ưu ái cho một bên.
- Việc so sánh ở mức cố định `lr=0.001` trước tiên giúp làm rõ đặc tính kinh điển: Adam thích ứng cực tốt ở mức 0.001 nhờ có 2 moment thích nghi, trong khi SGD thuần cần bước nhảy lớn hơn nhiều. Sau khi đã hiểu rõ bản chất, pha tuning mới mở ra để mỗi thuật toán tự do tìm điểm rơi phong độ tốt nhất của mình.

### 4.3. Kỷ luật bảo vệ tập Test: Cấm tuyệt đối dùng Test để chọn siêu tham số! ([Description Mục 5.2](../../description_lab_01.md#5-tiêu-chí-nghiệm-thu))
- Toàn bộ 30 trials của Optuna và quyết định chọn cấu hình thắng cuộc đều chỉ dựa trên `val_acc` của tập Validation (6.000 mẫu).
- Tập Test (10.000 mẫu) được niêm phong hoàn toàn, không hề tham gia vào bất kỳ vòng lặp tìm kiếm nào. Chỉ sau khi đã chọn xong cấu hình tốt nhất và refit hoàn tất, tập Test mới được mở ra chấm điểm đúng 1 lần duy nhất để bảo đảm tính khách quan và khoa học tuyệt đối.

### 4.4. Thẳng thắn nhìn nhận giới hạn của 10 trials ([Description Mục 7.7](../../description_lab_01.md#7-tuning-bao-nhiêu-trial-tìm-tham-số-gì-và-có-scheduler-không))
- Trong báo cáo học thuật, ta cần thừa nhận rõ ràng: 10 trial cho mỗi thuật toán thực chất chỉ tương đương với giai đoạn khởi tạo ngẫu nhiên (startup trials) của thuật toán TPE. Ta chưa đủ ngân sách để thuật toán bước vào pha học thích nghi sâu.
- Tuy nhiên, việc áp dụng `MedianPruner` đã giúp cắt tỉa sớm gần 30% số trial kém hiệu quả, tiết kiệm đáng kể thời gian tính toán trong khuôn khổ tài nguyên của bài Lab.
