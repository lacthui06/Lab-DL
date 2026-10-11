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

1. Cài đặt môi trường theo `requirements.txt` ở thư mục gốc repository.
2. Chạy test TDD cục bộ để xác thực pipeline (bao gồm bài test 1-batch sanity overfit).
3. Chạy huấn luyện đầy đủ trên Kaggle GPU.
4. Tải artifact trở về `outputs/` và cập nhật `walkthrough_lab_01.md`.

---

## Results Summary

| Run | Model | Test Accuracy | Macro F1 |
|---|---|---|---|
| Run #1 Baseline | Baseline MLP (2 layers) | 87.33% | không ghi nhận |
| Run #2 Tuned | Tuned MLP (3 layers, BN, Dropout) | **88.01%** | không ghi nhận |

Mô hình được chọn: **Tuned MLP (Run #2)** — cải thiện +0.68% Test Accuracy và giảm 7.0% Test Loss so với baseline.

Ghi chú: Run #1 và Run #2 không ghi nhận Macro F1 trong lần chạy Kaggle đã commit, nên ô này để trống thay vì suy diễn số.

Chi tiết đầy đủ xem tại `walkthrough_lab_01.md`.
