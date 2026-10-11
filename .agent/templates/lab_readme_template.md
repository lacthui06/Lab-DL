# Lab 0X — <tên bài lab>

Một đoạn ngắn mô tả bài lab: mục tiêu, dataset, và mô hình chính.

---

## Structure

- `walkthrough_lab_0X.md`: tài liệu sống duy nhất hợp nhất 3 Rule (Spec -> Changelog -> Report).
- `notebooks/`: notebook Jupyter tương tác cho phân tích và trực quan hóa.
- `src/`: các module Python tái sử dụng (`config.py`, `dataset.py`, `model.py`, `engine.py`, `utils.py`).
- `tests/`: các bài test TDD bắt buộc.
- `data/`: dataset đầu vào cục bộ (được git ignore).
- `outputs/`: checkpoint, log đánh giá và các hình trực quan.

---

## Reproduce

1. Cài đặt môi trường theo `requirements.txt` ở thư mục gốc repository.
2. Chạy test TDD cục bộ để xác thực pipeline.
3. Chạy huấn luyện đầy đủ trên Kaggle GPU.
4. Tải artifact trở về `outputs/` và cập nhật `walkthrough_lab_0X.md`.

---

## Results Summary

| Run | Model | Test Accuracy | Macro F1 |
|---|---|---|---|
| Run #1 Baseline | | | |
| Run #2 Tuned | | | |
| Run #3 Final | | | |

Chi tiết đầy đủ xem tại `walkthrough_lab_0X.md`.
