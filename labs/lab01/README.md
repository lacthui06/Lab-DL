# Lab 01 — FashionMNIST Image Classification with PyTorch

Bài lab xây dựng pipeline Deep Learning end-to-end để phân loại 10 nhóm trang phục trong bộ dữ liệu FashionMNIST (60.000 mẫu train, 10.000 mẫu test). Bài lab so sánh 2 kiến trúc (Baseline MLP, Tuned MLP), áp dụng quy trình TDD với bài test 1-batch sanity overfit cục bộ, và huấn luyện đầy đủ trên Kaggle GPU.

---

## Structure

- `description_lab_01.md`: đề bài gốc do giảng viên giao (đầu vào, giữ nguyên văn).
- `walkthrough_lab_01.md`: tài liệu sống duy nhất hợp nhất 3 Rule (Spec -> Changelog -> Report) — bài làm và kết quả (đầu ra).
- `notebooks/`: notebook Jupyter tương tác cho phân tích và trực quan hóa.
- `src/`: các module Python tái sử dụng (`config.py`, `dataset.py`, `model.py`, `engine.py`, `utils.py`).
- `tests/`: các bài test TDD bắt buộc.
- `data/`: dataset đầu vào cục bộ (được git ignore).
- `outputs/`: checkpoint, log đánh giá và các hình trực quan.

---

## Reproduce

**Phiên bản môi trường:** yêu cầu Python `>=3.12, !=3.14.1` (suy ra từ Requires-Python của các thư viện đã pin), ghi dạng comment trong `requirements.txt` (không dùng dòng `python==3.13.15` vì không phải cú pháp pip hợp lệ). Bản đã kiểm chứng ở local là 3.13.15. Mỗi lần chạy ghi lại phiên bản thực tế vào `outputs/environment_snapshot.json`.

1. Cài đặt môi trường theo `requirements.txt` ở thư mục gốc repository. File này ghi cả yêu cầu Python (`>=3.12, !=3.14.1`) và bản đã kiểm chứng (local: 3.13.15) ở dạng comment, dùng để đối chiếu với Kaggle.
2. Chạy test TDD cục bộ để xác thực pipeline (bao gồm bài test 1-batch sanity overfit).
3. Chạy huấn luyện đầy đủ trên Kaggle GPU.
4. Tải artifact trở về `outputs/` (bao gồm `environment_snapshot.json`) và cập nhật `walkthrough_lab_01.md`.

---

## Results Summary

Kết quả dưới đây lấy từ lần chạy Kaggle GPU đã commit (Phase 1 optimizer là `SGD + Momentum`, lr=0.05 cho cả hai run).

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

Chi tiết đầy đủ (bảng per-class, phân tích confusion matrix, số liệu convergence theo epoch) xem tại `walkthrough_lab_01.md`.
