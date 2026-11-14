# Mô-đun Dữ Liệu & Tiền Xử Lý (src/datasets)

## 1. Thông tin chung
- **Tác giả:** Thái Anh Lạc
- **Ngày cập nhật:** 2026-11-14
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

---

## 4. Thảo luận kỹ thuật & Quyết định thiết kế (Design Decisions & Discussion)

### 4.1. Thảo luận về quy trình EDA (Tuân thủ 6 bước theo `docs/eda.md`)
1. **Bước 1 - Data Overview & Schema:** Tập dữ liệu Rotten Tomatoes gồm 10.662 mẫu đánh giá phim ngắn (Train: 8.530, Validation: 1.066, Test: 1.066). Schema tinh gọn gồm 2 trường: `text` (chuỗi văn bản tiếng Anh) và `label` (số nguyên `0` = Negative, `1` = Positive). 
2. **Bước 2 - Summary Statistics & 5-Number Summary:** 
   - Độ dài từ (Word counts): $Min = 1$, $Q1 = 13$, $Median = 20$, $Mean = 21.0$, $Q3 = 28$, $Max = 56$, $IQR = 15$.
   - Độ dài ký tự: $Mean = 115$ ký tự, $Min = 4$, $Max = 266$ ký tự.
   - Thống kê cho thấy các đánh giá phim đều là các câu đơn/ngắn, không xuất hiện các văn bản dài kiểu bài tiểu luận (như IMDb 500 từ).
3. **Bước 3 - Phân tích đơn biến & Cân bằng nhãn:**
   - Tập dữ liệu cân bằng tuyệt đối 50/50 giữa hai lớp trên cả 3 tập: Train có đúng 4.265 mẫu/lớp; Validation và Test mỗi tập có đúng 533 mẫu/lớp.
   - Nhờ sự cân bằng hoàn hảo này, độ đo Accuracy phản ánh chính xác hiệu năng mô hình, đồng thời Macro F1 và Binary F1 đóng vai trò kiểm chứng chéo mà không lo bị lệch về lớp đa số.
4. **Bước 4 - Phân tích song biến & Độc lập độ dài câu:**
   - Biểu đồ phân bố độ dài theo nhãn (`lengths_by_class.png`) cho thấy đường cong mật độ của lớp Positive và Negative gần như trùng khít hoàn toàn.
   - Quyết định: Loại bỏ nguy cơ độ dài câu trở thành đặc trưng rò rỉ giả tạo (spurious feature). Mô hình không thể dựa vào câu ngắn hay câu dài để đoán nhãn.
5. **Bước 5 - Kiểm toán dữ liệu thiếu, trùng lặp & Rò rỉ dữ liệu (Leakage Audit):**
   - Đạt 0 mẫu khuyết thiếu (0 NaN, 0 Null, 0 chuỗi rỗng).
   - Kiểm toán băm MD5 và chuẩn hóa chuỗi xác nhận 0 mẫu trùng lặp chéo giữa Train với Validation/Test, bảo toàn tính khách quan tuyệt đối cho tập đánh giá độc lập.
6. **Bước 6 - Đánh giá chất lượng dữ liệu:**
   - Toàn bộ 10.662 mẫu đều nguyên vẹn, nhãn sạch, sẵn sàng chuyển giao cho giai đoạn tiền xử lý và kỹ nghệ đặc trưng.

### 4.2. Thảo luận về quy trình Feature Engineering (Tuân thủ 6 bước theo `docs/feature.md`)
1. **Bước 1 & 2 - Xử lý ngoại lệ & Chiến lược Truncation Cutoff:**
   - Dò ngoại lệ theo khoảng $1.5 \times IQR$ ($Upper Bound = 50.5$ từ): Chỉ có 0.6% số mẫu có độ dài > 50 từ. Thay vì xóa bỏ gây mất mát ngữ nghĩa, chúng tôi áp dụng cơ chế cắt ngắn (Truncation) có kiểm soát.
   - Phân tích độ dài token qua DistilBERT WordPiece Tokenizer (`token_length_distribution.png`): 99.4% câu có độ dài dưới 64 tokens, và 100% câu dưới 80 tokens.
   - **Quyết định:** Chọn `max_length = 64` cho quy trình huấn luyện Transformer. So với `max_length = 128`, việc chọn 64 giúp giảm 50% chiều dài chuỗi, tương ứng giảm 75% chi phí tính toán cơ chế tự chú ý (Self-Attention có độ phức tạp $O(N^2)$), tăng thông lượng xử lý lên gấp đôi mà bảo toàn được 99.4% ngữ cảnh câu.
2. **Bước 3 - Tokenization & Dynamic Padding:**
   - Sử dụng `DistilBertTokenizerFast` với bộ từ vựng 30.522 tokens.
   - Mã hóa hai tensor thiết yếu: `input_ids` (chỉ số token kèm cặp token ranh giới `[CLS]` và `[SEP]`) và `attention_mask` (gán giá trị `1` cho token văn bản thật, `0` cho token padding đệm thêm).
   - Attention Mask triệt tiêu hoàn toàn trọng số chú ý tại các vị trí padding, đảm bảo đệm không gây méo mó biểu diễn vector ngữ cảnh.
3. **Bước 4 & 5 - Tạo đặc trưng ngôn ngữ học (Linguistic Handcrafted Features) & TF-IDF:**
   - Trích xuất 8 đặc trưng ngôn ngữ: số lượng từ, số ký tự, độ dài từ trung bình, tỷ lệ chữ hoa (cảm xúc nhấn mạnh), tỷ lệ dấu câu (dấu chấm than/hỏi), và số lượng từ ngữ phủ định/tương phản.
   - **Phát hiện thú vị:** Từ ngữ tương phản (*but, however, although, despite, yet*) xuất hiện với tần suất cao hơn ở các đánh giá tiêu cực (14.2% so với 8.9% ở tích cực). Điều này phản ánh văn phong phê bình phim điển hình: các nhà phê bình thường khen một điểm nhỏ trước (ví dụ khen diễn xuất hoặc hình ảnh) rồi dùng liên từ tương phản để chốt hạ lời chê tổng thể bộ phim.
   - Xây dựng ma trận TF-IDF 5.000 unigrams/bigrams với `sublinear_tf=True` để làm baseline truyền thống đối chiếu với DistilBERT.
4. **Bước 6 - Lựa chọn đặc trưng (Feature Selection qua Chi-Square & Delta TF-IDF):**
   - Áp dụng kiểm định thống kê Chi-Square ($\chi^2$) trên ma trận TF-IDF: Các từ *bad, movie, film, dull, funny, best, boring, flat* đạt giá trị $\chi^2$ vượt trội với $p < 0.001$, khẳng định mối liên hệ thống kê chặt chẽ với nhãn cảm xúc.
   - Biểu đồ phổ chênh lệch trọng số (Delta TF-IDF Spectrum): Lọc ra các từ có sức nặng phân tách cực mạnh (*bad* lệch âm mạnh nhất, *best/funny* lệch dương cao nhất).
5. **Kỹ nghệ mở rộng: Tăng cường dữ liệu có kiểm soát (Text Augmentation & Guardrails):**
   - *Tại sao không dùng Back-Translation hay mô hình sinh (LLM)*: Trên tập dữ liệu nhỏ (~8.5k câu), mô hình sinh tự do dễ làm trôi dạt ngữ nghĩa (Semantic Drift) hoặc sinh ra câu trái với nhãn ban đầu.
   - *Quy tắc bảo vệ (Semantic Guardrails)*: Sử dụng thay thế từ đồng nghĩa theo ngữ cảnh với ràng buộc nghiêm ngặt: bảo vệ 100% các từ phủ định (`not, no, never, without`) và liên từ tương phản (`but, however`). Nhờ đó, 96.8% mẫu tăng cường giữ nguyên vẹn sắc thái cảm xúc gốc.
   - *Chính sách không chạm vào Validation/Test*: Mọi phép biến đổi và tăng cường dữ liệu chỉ thực hiện trên tập Train. Tập Validation và Test giữ nguyên văn bản gốc từ ban tổ chức để đảm bảo kết quả đo lường khách quan.
