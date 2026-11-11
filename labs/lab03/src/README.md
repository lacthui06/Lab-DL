# Gói Mã Nguồn Chính (labs/lab03/src)

## 1. Thông tin chung
- **Tác giả:** Thái Anh Lạc
- **Ngày cập nhật:** 2026-11-11
- **Vị trí:** `labs/lab03/src/`

## 2. Kiến trúc & Bản đồ các mô-đun
Toàn bộ mã nguồn thực thi bài Lab 03 được tổ chức theo kiến trúc phân tầng rõ ràng (Clean Modular Architecture):

```text
src/
├── datasets/                 # Nạp dữ liệu Rotten Tomatoes và tăng cường văn bản
│   ├── loader.py             # Nạp official splits và mã hóa DistilBertTokenizer
│   └── augmentation.py       # Tăng cường từ đồng nghĩa có kiểm soát cực tính
│
├── models/                   # Định nghĩa kiến trúc mô hình
│   └── classifier.py         # DistilBERT classifier head, dropout overrides và kiểm tra suy luận
│
├── training/                 # Quy trình huấn luyện và tối ưu
│   ├── trainer.py            # Hugging Face Trainer với Early Stopping theo validation loss
│   ├── thresholds.py         # Quét 19 ngưỡng quyết định và khóa ngưỡng tối ưu
│   ├── metrics.py            # Đánh giá Accuracy, Binary F1, Precision, Recall
│   └── frozen.py             # Thí nghiệm đóng băng backbone transformer
│
├── utils/                    # Bộ công cụ hỗ trợ và giám sát
│   ├── runs.py               # Quản lý run nguyên tử, snapshot môi trường, băm SHA-256
│   └── plots.py              # Vẽ đường cong học tập, ma trận nhầm lẫn, calibration curve
│
├── data_preparation.py       # Pipeline chuẩn bị dữ liệu đầu vào cho notebook
├── study.py                  # Baseline TF-IDF + LogisticRegression đối chứng
├── performance.py            # Đo đạc chi phí thời gian (ms/mẫu) và bộ nhớ RAM/VRAM
├── workflow.py               # Điểm vào điều phối chung giữa chế độ VIEW và TRAIN
└── final_review.py           # Tổng hợp kết quả, rà soát ngưỡng và kiểm toán bài nộp
```

## 3. Nguyên tắc vận hành
1. **Chế độ kép (Dual Mode):**
   - `MODE="view"`: Nạp nhanh các kết quả, biểu đồ và số liệu đã được kiểm chứng mà không cần train lại.
   - `MODE="train"`: Huấn luyện mới trọn vẹn từ đầu trên GPU (Kaggle/CUDA), xuất kết quả và checkpoint mới.
2. **Bảo toàn dữ liệu khoa học:** Tập Validation và Test giữ nguyên văn bản gốc, ngưỡng quyết định chỉ được tối ưu hóa từ tập Validation.
