# Lab 01 — FashionMNIST Image Classification with PyTorch

Bài lab xây dựng pipeline Deep Learning end-to-end để phân loại 10 nhóm trang phục trong bộ dữ liệu FashionMNIST (60.000 mẫu train, 10.000 mẫu test). Bài lab so sánh 2 kiến trúc (Baseline MLP, Tuned MLP), áp dụng quy trình TDD với bài test 1-batch sanity overfit cục bộ, và huấn luyện đầy đủ trên Kaggle GPU.

---

## Structure

- `description_lab_01.md`: đề bài gốc do giảng viên giao (đầu vào, giữ nguyên văn).
- `walkthrough_lab_01.md`: tài liệu sống duy nhất hợp nhất 3 Rule (Spec -> Changelog -> Report) — bài làm và kết quả (đầu ra).
- Quy trình chung cho bài toán ảnh: `docs/eda-image.md` và `docs/feature-image.md`.
- `notebooks/`:
  - `lab01_eda.ipynb`: tạo stratified split một lần, chạy EDA, ghi artifact vào `outputs/eda/`.
  - `lab01_feature_engineering.ipynb`: đo và chốt thống kê normalization, kiểm tra augmentation, ghi `preprocessing_config.json`.
  - `lab01_fashionmnist_walkthrough.ipynb`: huấn luyện trên Kaggle, có guard cell kiểm tra artifact trước khi train.
- `src/`: các module Python tái sử dụng (`config.py`, `split.py`, `dataset.py`, `eda.py`, `model.py`, `engine.py`, `utils.py`).
- `tests/`: các bài test TDD bắt buộc.
- `data/`: dataset đầu vào cục bộ (được git ignore).
- `outputs/`: checkpoint, log đánh giá và các hình trực quan. `outputs/eda/` chứa split artifact, thống kê và bảng/biểu đồ EDA.

---

## Pipeline versions

Hai version dưới đây phải được ghi kèm mọi kết quả, vì số liệu đo trên version khác nhau không so sánh được với nhau.

| Version | Giá trị | Ghi chú |
|---|---|---|
| `split_v2_stratified` | 54,000 train / 6,000 validation, `random_state=42`, `stratify=y` | Thay `random_split` cũ; cho đúng 5,400 và 600 ảnh mỗi lớp. Split lưu tại `outputs/eda/split_indices.npz`, `src/dataset.py` chỉ đọc lại. |
| `preproc_v2_train_global_std` | `mean=0.286209`, `std=0.353160` (population, `ddof=0`) | Đo trên 54,000 ảnh train đã scale `[0,1]`, trước augmentation. Đọc từ `outputs/eda/preprocessing_config.json`. |

**Vì sao đổi:** `std=0.3205` cũ là trung bình std của từng ảnh, không phải global std. Dùng nó làm mẫu số khiến dữ liệu sau Normalize có std xấp xỉ 1.1019 thay vì 1.0. Split cũ (`random_split`) cho validation lệch lớp từ 537 đến 658 ảnh thay vì 600.

**Cảnh báo so sánh:** kết quả Phase 1 và Phase 2 trong mục Results Summary bên dưới được đo dưới `split_v1` + `preproc_v1`. Chưa có lần chạy lại nào trên `split_v2` + `preproc_v2`.

---

## Reproduce

**Phiên bản môi trường:** yêu cầu Python `>=3.12, !=3.14.1` (suy ra từ Requires-Python của các thư viện đã pin), ghi dạng comment trong `requirements.txt` (không dùng dòng `python==3.13.15` vì không phải cú pháp pip hợp lệ). Bản đã kiểm chứng ở local là 3.13.15. Mỗi lần chạy ghi lại phiên bản thực tế vào `outputs/environment_snapshot.json`.

1. Cài đặt môi trường theo `requirements.txt` ở thư mục gốc repository. File này ghi cả yêu cầu Python (`>=3.12, !=3.14.1`) và bản đã kiểm chứng (local: 3.13.15) ở dạng comment, dùng để đối chiếu với Kaggle.
2. Chạy `notebooks/lab01_eda.ipynb` để tạo split artifact và EDA outputs (local CPU, khoảng 1 phút).
3. Chạy `notebooks/lab01_feature_engineering.ipynb` để chốt normalization và ghi `preprocessing_config.json` (local CPU).
4. Chạy test TDD cục bộ để xác thực pipeline (bao gồm bài test 1-batch sanity overfit):
   `python -m pytest tests/ -q`
5. Chạy huấn luyện đầy đủ trên Kaggle GPU. Notebook `lab01_fashionmnist_walkthrough.ipynb` sẽ dừng ngay ở PHẦN 1B nếu thiếu artifact EDA hoặc Feature Engineering, thay vì âm thầm dùng mean/std dự phòng.
6. Tải artifact trở về `outputs/` (bao gồm `environment_snapshot.json`) và cập nhật `walkthrough_lab_01.md`.

**Thứ tự bắt buộc:** EDA -> Feature Engineering -> Training. `src/dataset.py` gọi `load_split()` và raise `FileNotFoundError` kèm hướng dẫn nếu artifact vắng mặt; nó không bao giờ tự chia lại, vì một split khác sẽ làm mọi thống kê đã đo trở nên sai.

---

## Results Summary

Kết quả dưới đây lấy từ lần chạy Kaggle GPU đã commit (Phase 1 optimizer là `SGD + Momentum`, lr=0.05 cho cả hai run), đo dưới `split_v1` + `preproc_v1`. Xem cảnh báo so sánh ở mục Pipeline versions.

| Run | Model | Test Accuracy | Test Loss | Macro F1 |
|---|---|---|---|---|
| Run #1 Baseline | Baseline MLP (2 layers) | 85.14% | 0.4443 | 84.93% |
| Run #2 Tuned | Tuned MLP (3 layers, BN, Dropout) | **87.81%** | **0.3312** | **87.71%** |

Mô hình được chọn: **Tuned MLP (Run #2)** — cải thiện +2.67% Test Accuracy, +2.78% Macro F1 và giảm 25.5% Test Loss so với baseline. Lớp yếu nhất ở cả hai run là Shirt (Recall 0.514 ở baseline, 0.636 ở run tuned).

**Phase 2 — so sánh optimizer (Stratified 5-fold CV, backbone `TunedMLP`, 15 epoch):**

| Kịch bản | SGD + Momentum | RMSprop | Adam |
|---|---|---|---|
| 2a (tất cả lr = 1e-3) | 87.7417 | 88.7650 | 88.8083 |
| 2b (SGD lr = 0.05) | 88.5533 | 88.7650 | 88.8083 |

Kết luận theo rule đã đăng ký trước ở Section 1.6.9: SGD ở 1e-3 tụt sau RMSprop/Adam (chênh 1.0233 điểm, khoảng 8.8 lần độ lệch chuẩn theo fold), và khi chỉ nâng lr của SGD lên 0.05 thì phần lớn khoảng cách được thu hẹp (+0.8116, khoảng 79%). Vì vậy khoảng cách ở 2a là do bước nhảy (step size) chứ không phải do bản chất update rule. RMSprop và Adam chênh nhau 0.0433 — nhỏ hơn nhiều so với độ lệch chuẩn theo fold (0.1167) — nên được báo cáo là nằm trong nhiễu, không kết luận optimizer nào hơn. RMSprop và Adam cho ra số liệu y hệt nhau ở cả 2a và 2b, chứng tỏ pipeline deterministic và chênh lệch nhỏ không phải do phần cứng.

**Phase 2 — Scenario 2c: learning-rate sweep (Stratified 5-fold CV, backbone `TunedMLP`, 15 epoch):**

Vì accuracy cuối của 3 optimizer quá sát nhau ở 2a/2b, kịch bản 2c sweep learning rate để tìm lr tối ưu và độ nhạy của từng optimizer: 3 optimizer x 5 lr x 5 fold = 75 run, mỗi run 15 epoch (giữ đúng bằng 2a/2b để không nhánh nào bị đo ở độ chín hội tụ khác nhau). Lưới lr: SGD + Momentum `1e-3, 5e-3, 1e-2, 5e-2, 1e-1`; RMSprop và Adam `1e-4, 3e-4, 1e-3, 3e-3, 1e-2`. Chính sách momentum/beta chốt trước khi chạy: SGD `momentum=0.9`, RMSprop giữ mặc định PyTorch `momentum=0.0`, Adam giữ mặc định `betas=(0.9, 0.999)` (Adam không có tham số `momentum`; `beta1` là thành phần momentum-like của nó). Cell có lr quá lớn gây NaN/Inf được ghi `diverged` và loại khỏi mean/std, không nội suy (lần chạy này không cell nào diverged). Vì cùng budget/fold/seed với 2a/2b, cột `1e-3` tái tạo chính xác số của 2a — dùng làm phép kiểm tra determinism. Spec ở Section 1.6.10, kết quả và bảng đầy đủ ở Section 3.8 của `walkthrough_lab_01.md`.

**Kết quả 2c (lr tối ưu và mean val acc tại đó, 15 epoch, 5 fold):**

| Optimizer | lr tối ưu | Mean Val Acc | Std | Độ nhạy (tụt tới bước kế tiếp) |
|---|---|---|---|---|
| SGD + Momentum | 1e-2 | 88.5950 | 0.2269 | 0.0417 (lên 5e-2); 0.1800 (xuống 5e-3) |
| RMSprop | 3e-3 | 88.7967 | 0.2123 | 0.0317 (xuống 1e-3); 0.3367 (lên 1e-2) |
| Adam | 1e-3 | 88.8083 | 0.2888 | 0.0883 (xuống 3e-4); 0.1366 (lên 3e-3) |

Kết luận theo rule đã đăng ký trước ở Section 1.6.10: mỗi optimizer có một lr tối ưu nằm trong lưới (accuracy tăng rồi giảm), nhưng so peak-với-peak thì ba điểm tối ưu tương đương trong nhiễu: chênh lớn nhất (Adam − SGD = 0.2133) vẫn nhỏ hơn độ lệch chuẩn theo fold của chính SGD (0.2269), và chênh RMSprop − Adam (0.0116) nhỏ hơn nhiều so với std của RMSprop (0.2123) — nên không kết luận optimizer nào hơn. Điều tách bạch được là cấu trúc: SGD cần bước nhảy lớn nhất (tối ưu ở 1e-2, lớn hơn RMSprop 3e-3 và Adam 1e-3), RMSprop nhạy nhất khi vượt bước (mất 0.3367 khi lên 1e-2), Adam kém ổn định nhất giữa các fold (std 0.2888). Không cell nào diverged. Cột `1e-3` tái tạo 2a và ô SGD `5e-2` tái tạo 2b với delta `+0.0000`, xác nhận pipeline deterministic. Vì mọi đường cong vẫn đang tăng ở epoch 15 và SGD tăng nhiều nhất giai đoạn cuối, budget 15 epoch được ghi nhận là có thể còn đánh giá thấp đỉnh của SGD (bảo thủ cho SGD, không phải bất lợi cho nó).

Chi tiết đầy đủ (bảng per-class, phân tích confusion matrix, số liệu convergence theo epoch) xem tại `walkthrough_lab_01.md`.
