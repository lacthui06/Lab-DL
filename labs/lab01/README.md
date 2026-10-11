# Lab 01 — FashionMNIST Image Classification with PyTorch

Bài lab xây dựng pipeline Deep Learning end-to-end để phân loại 10 nhóm trang phục trong bộ dữ liệu FashionMNIST (60.000 mẫu train, 10.000 mẫu test). Bài lab so sánh 3 kiến trúc (Baseline MLP, Tuned MLP, FashionCNN), áp dụng quy trình TDD với bài test 1-batch sanity overfit cục bộ, và huấn luyện đầy đủ trên Kaggle GPU.

---

## Structure

- `walkthrough_lab_01.md`: tài liệu sống duy nhất hợp nhất 3 Rule (Spec -> Changelog -> Report).
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
| Run #1 Baseline | Baseline MLP (2 layers) | 87.39% | 87.29% |
| Run #2 Tuned | Tuned MLP (3 layers, BN, Dropout) | 87.86% | 87.85% |
| Run #3 Final | FashionCNN (2 Conv blocks + Head) | **92.65%** | **92.60%** |

Mô hình được chọn: **FashionCNN (Run #3)** — cải thiện +5.26% Test Accuracy và giảm 40.9% Test Loss so với baseline.

Chi tiết đầy đủ xem tại `walkthrough_lab_01.md`.
