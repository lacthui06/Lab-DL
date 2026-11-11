# Mô-đun Dữ Liệu & Tiền Xử Lý (src/datasets)

## 1. Thông tin chung
- **Tác giả:** Thái Anh Lạc
- **Ngày cập nhật:** 2026-11-11
- **Vị trí:** `labs/lab03/src/datasets/`
- **Tệp mã nguồn:** [`loader.py`](loader.py), [`augmentation.py`](augmentation.py), liên kết chặt chẽ với [`data_preparation.py`](../data_preparation.py)

## 2. Mục đích & Vai trò
- Quản lý việc nạp tập dữ liệu văn bản phân loại cảm xúc Rotten Tomatoes chính thức từ Hugging Face Hub.
- Đảm bảo tính toàn vẹn của dữ liệu: kiểm tra đúng số lượng mẫu các phân vùng (train: 8.530, validation: 1.066, test: 1.066) và nhãn nhị phân chuẩn `{0, 1}`.
- Thực hiện mã hóa văn bản thành tensor (`input_ids`, `attention_mask`) thông qua `DistilBertTokenizerFast`.
- Cung cấp cơ chế tăng cường dữ liệu văn bản (Text Augmentation) có kiểm soát ngữ nghĩa và kiểm toán rò rỉ dữ liệu (leakage audit).
- Hỗ trợ toàn diện 6 bước EDA (`docs/eda.md`) và 6 bước Feature Engineering (`docs/feature.md`) thông qua `data_preparation.py`.

## 3. Khởi tạo & Cơ chế hoạt động

### 3.1. Nạp và Mã Hóa Dữ Liệu (`loader.py`)
- `load_data(config)`: Tải tập dữ liệu theo tên và mã revision được chỉ định trong cấu hình. Kiểm tra kích thước chính xác tuyệt đối; từ chối thực thi nếu phát hiện văn bản rỗng hoặc nhãn sai lệch.
- `tokenize_data(ds, tokenizer, max_length)`: Áp dụng tokenizer hàng loạt (`batched=True`), cắt ngắn văn bản vượt quá `max_length` (`truncation=True`) và loại bỏ cột văn bản thô để sẵn sàng đưa vào PyTorch DataLoader.

### 3.2. Tăng Cường Văn Bản Có Kiểm Soát (`augmentation.py`)
- `augment_train(raw, settings, seed)`:
  - Sử dụng từ điển phân cực cảm xúc (`LEXICON`) để thay thế từ đồng nghĩa có cùng cực tính (chỉ thay 1 từ trên mỗi mẫu được chọn).
  - Áp dụng bộ lọc biểu thức chính quy (`PROTECTED`) để bảo vệ nghiêm ngặt các từ phủ định hoặc chuyển ý (như *not, no, never, but, however, although*), tránh làm đảo ngược cảm xúc gốc của câu.
  - Khử trùng lặp và kiểm toán rò rỉ: Mọi câu văn sinh ra đều được đối chiếu với tập Validation và Test (`heldout set`), loại bỏ ngay lập tức nếu xuất hiện trùng lặp để bảo vệ tính độc lập khoa học của tập đánh giá.

### 3.3. Khám Phá Dữ Liệu & Kỹ Nghệ Đặc Trưng (`data_preparation.py`)
- `compute_summary_statistics(values)`: Tính toán 5-number summary (Min, Q1, Median, Q3, Max), Mean, Std và phát hiện ngoại lệ (IQR Outliers) cho độ dài từ, ký tự và tokens.
- `extract_linguistic_features(texts)`: Trích xuất các đặc trưng ngôn ngữ học (Handcrafted features): tỷ lệ chữ hoa (`uppercase_ratio`), tỷ lệ dấu câu (`punctuation_ratio`), độ dài trung bình từ, và từ ngữ chỉ thị phủ định / tương phản.
- `select_chi2_features(matrix, labels, feature_names, top_k=25)`: Thực hiện kiểm định thống kê Chi-Square ($\chi^2$) lựa chọn các n-grams có năng lực phân loại mạnh nhất từ ma trận TF-IDF.
- `export_data_review(raw, tokenizer, settings, output)`: Điều phối xuất toàn bộ bảng số liệu và 14 hình ảnh trực quan hóa cho báo cáo và 3 notebooks.

## 4. Thảo luận kỹ thuật & Quyết định thiết kế
- **Độ dài cắt ngắn (`max_length=128`):** Dựa trên phân tích phân bố độ dài từ trong tập Rotten Tomatoes, 99.8% câu review có độ dài dưới 60 từ. Việc đặt `max_length=128` bao phủ toàn bộ nội dung mà không gây lãng phí bộ nhớ VRAM khi tính toán cơ chế Attention.
- **Tại sao không dùng Back-Translation hay mô hình sinh để Augmentation:** Trên tập dữ liệu nhỏ (~8.5k mẫu), việc dùng mô hình sinh tự do dễ đưa vào các từ ngữ sai lệch nhãn gốc. Việc thay thế từ điển đồng nghĩa có kiểm soát cực tính và bảo vệ từ phủ định giúp duy trì nhãn chính xác 100%.
- **Chính sách không chạm vào Validation/Test:** Mọi phép biến đổi và tăng cường dữ liệu chỉ thực hiện trên tập Train. Tập Validation và Test giữ nguyên văn bản gốc từ ban tổ chức để đảm bảo kết quả đo lường khách quan.
