# Mô-đun Huấn Luyện & Tối Ưu Hóa (src/training)

## 1. Thông tin chung
- **Tác giả:** Thái Anh Lạc
- **Ngày cập nhật:** 2026-11-11
- **Vị trí:** `labs/lab03/src/training/`
- **Tệp mã nguồn:** [`trainer.py`](trainer.py), [`thresholds.py`](thresholds.py), [`metrics.py`](metrics.py), [`frozen.py`](frozen.py)

## 2. Mục đích & Vai trò
- Điều phối toàn bộ quá trình fine-tuning mô hình ngôn ngữ DistilBERT thông qua Hugging Face `Trainer`.
- Áp dụng cơ chế dừng sớm (Early Stopping) có kiểm soát dựa trên độ mất mát tập kiểm định (`eval_loss`) nhằm ngăn chặn tình trạng học vẹt.
- Triển khai cơ chế quét ngưỡng quyết định (Threshold Sweep) tối ưu hóa điểm số Binary F1 và cân bằng giữa Precision và Recall.
- Cung cấp tùy chọn đóng băng các tầng Transformer (Frozen Backbone) để so sánh chi phí tính toán và hiệu năng giữa phân loại tầng đầu và fine-tuning toàn diện.

## 3. Khởi tạo & Cơ chế hoạt động

### 3.1. Cấu Hình Trainer & Dừng Sớm (`trainer.py`)
- `make_trainer(model, tokenizer, ds, config, run_path)`: Thiết lập đối tượng `TrainingArguments` với các siêu tham số chính:
  - Tối đa 5 epoch (`num_train_epochs=5`).
  - Tối ưu hóa bằng thuật toán **AdamW** với hệ số triệt tiêu trọng số (`weight_decay=0.01`).
  - Đánh giá theo chu kỳ từng epoch (`eval_strategy="epoch"`) và tự động lưu lại checkpoint có kết quả tốt nhất (`load_best_model_at_end=True`).
- **EarlyStoppingCallback:** Giám sát `eval_loss` với cấu hình nghiêm ngặt (`patience=2`, `min_delta=0.001`). Nếu sau 2 epoch liên tiếp mà loss trên tập validation không giảm thêm ít nhất 0.001, quá trình huấn luyện sẽ dừng lại ngay lập tức để bảo vệ mô hình.

### 3.2. Quét Ngưỡng Quyết Định Nhị Phân (`thresholds.py`)
- `scan_thresholds(labels, positive_probabilities, thresholds)`:
  - Đầu vào bắt buộc phải là vector xác suất của lớp dương tính: **`P(POSITIVE)`**, không được lấy giá trị confidence của nhãn chiến thắng.
  - Quét qua lưới 19 ngưỡng quyết định từ `0.05` đến `0.95` với bước nhảy `0.05`.
  - Quy tắc phân loại: Dự đoán nhãn `1` (Positive) khi và chỉ khi `P(POSITIVE) >= threshold`.
  - Tìm ra ngưỡng tối ưu trên tập Validation (ngưỡng đạt Binary F1 cao nhất), sau đó "khóa" ngưỡng này lại để áp dụng nguyên trạng lên tập Test độc lập.

### 3.3. Đo Lường Chỉ Số Đánh Giá (`metrics.py`)
- `compute_metrics(eval_pred)`: Tính toán bộ 4 chỉ số cốt lõi cho bài toán phân loại nhị phân: **Accuracy**, **Binary F1**, **Precision**, và **Recall** tại ngưỡng mặc định 0.5.

### 3.4. Thí Nghiệm Đóng Băng Tầng (`frozen.py`)
- Đóng băng toàn bộ trọng số của các khối DistilBert Transformer, chỉ cho phép cập nhật gradient ở tầng phân loại cuối cùng (`pre_classifier` và `classifier`).

## 4. Thảo luận kỹ thuật & Quyết định thiết kế
- **Tại sao chọn dừng sớm theo `eval_loss` thay vì `eval_accuracy`:**
  - Accuracy chỉ đo tỷ lệ đoán đúng nhãn dựa trên ngưỡng cắt, trong khi Validation Loss đo lường trực tiếp mức độ tự tin và khoảng cách xác suất của mô hình.
  - Một mô hình có thể duy trì Accuracy đi ngang nhưng Validation Loss bắt đầu tăng vọt, báo hiệu các dự đoán sai đang ngày càng nghiêm trọng (hiện tượng overconfident overfitting). Do đó, `eval_loss` là tín hiệu cảnh báo nhạy bén và đáng tin cậy nhất.
- **Tại sao phải quét ngưỡng (Threshold Sweep):**
  - Ngưỡng mặc định `0.5` chỉ tối ưu khi chi phí của việc đoán nhầm False Positive và False Negative là hoàn toàn tương đương và phân bố xác suất cân bằng hoàn hảo.
  - Trong thực tế phân tích sắc thái cảm xúc, việc điều chỉnh ngưỡng sang vùng lân cận (ví dụ `0.45`) có thể giúp vớt lại nhiều mẫu đánh giá tích cực nhẹ mà không làm suy giảm độ chuẩn xác chung, giúp nâng cao đáng kể điểm số F1-score.
- **Quy tắc bất biến:** Ngưỡng quyết định **chỉ được phép chọn từ tập Validation**, tuyệt đối không được nhìn trước tập Test để tránh nhiễm bẩn dữ liệu (data leakage).
