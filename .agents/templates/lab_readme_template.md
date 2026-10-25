# Lab 0X — <tên bài lab>

Một đoạn ngắn mô tả bài lab: mục tiêu, dataset, và mô hình chính.

---

## Structure

- `description_lab_0X.md`: đề bài gốc do giảng viên giao (đầu vào, giữ nguyên văn).
- `walkthrough_lab_0X.md`: tài liệu sống duy nhất hợp nhất 3 Rule (Spec -> Changelog -> Report) — bài làm và kết quả (đầu ra).
- `config/`: cấu hình YAML tách biệt (`config.yaml`, `augmentations.yaml`, `experiments.yaml`).
- `notebooks/`: notebook Jupyter tương tác cho phân tích và trực quan hóa (`lab0X.ipynb`).
- `src/`: các phân hệ Python tái sử dụng (`datasets/`, `models/`, `training/`, `utils/`).
- `scripts/`: điểm vào thực thi dòng lệnh (`train.py`, `eval.py`, `infer.py`).
- `tests/`: các bài test TDD bắt buộc (`test_datasets.py`, `test_models.py`, `test_training.py`).
- `data/`: dataset đầu vào cục bộ (được git ignore).
- `outputs/`: checkpoint (`checkpoints/`), log đánh giá (`logs/`), bảng/hình EDA (`eda/`) và hình dự đoán (`predictions/`).

---

## Reproduce

**Phiên bản môi trường:** ghi phiên bản trình thông dịch Python dạng comment trong `requirements.txt` (không dùng dòng `python==x.y.z` vì không phải cú pháp pip hợp lệ), và để mỗi lần chạy ghi lại phiên bản thực tế vào `outputs/logs/environment_snapshot.json`.

1. Cài đặt môi trường theo `requirements.txt` ở thư mục gốc repository.
2. Chạy test TDD cục bộ để xác thực pipeline (`pytest tests/ -v`).
3. Chạy huấn luyện đầy đủ trên Kaggle GPU hoặc qua notebook/CLI.
4. Tải artifact trở về `outputs/` (bao gồm `outputs/logs/environment_snapshot.json`) và cập nhật `walkthrough_lab_0X.md`.

---

## Results Summary

| Run | Model | Test Accuracy | Macro F1 |
|---|---|---|---|
| Run #1 Baseline | | | |
| Run #2 Tuned | | | |
| Run #3 Final | | | |

Chi tiết đầy đủ xem tại `walkthrough_lab_0X.md`.
