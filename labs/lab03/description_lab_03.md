# Description — Lab 03: Sentiment Analysis and Text Classification Finetuning with Hugging Face

> **Lab:** Lab 03

---

## Ghi chú về file này

Đây là bản ghi **đề bài gốc** (đầu vào) do giảng viên giao kèm phần **làm rõ và thống nhất phạm vi thực hiện** sau khi thảo luận chi tiết

Bài làm và toàn bộ kết quả thực nghiệm nằm ở `walkthrough_lab_03.md` (đầu ra). Hai file bổ sung cho nhau:
- `description_lab_03.md`: đề bài nhận vào và phạm vi đã chốt.
- `walkthrough_lab_03.md`: bài làm và báo cáo kỹ thuật hoàn chỉnh.

---

## 1. Bối cảnh và Mục tiêu

### 1.1. Đề bài gốc (Original Brief)
Exercise1: Sentiment Analysis with Hugging Face. 
Exercise2: Finetuning a Pretrained Model for Binary Text Classification

### 1.2. Thống nhất thực hiện (Agreed Scope & Objectives)
- **Sử dụng hệ sinh thái Hugging Face:** Làm quen và khai thác sức mạnh của thư viện `transformers` để tải và sử dụng các mô hình ngôn ngữ lớn đã được huấn luyện sẵn (Pre-trained models) từ Hugging Face Hub.
- **Phân tích cảm xúc văn bản (Inference):** Ứng dụng mô hình có sẵn để mã hóa và phân tích cảm xúc (Sentiment Analysis) trên các mẫu câu văn bản cơ bản mà không cần huấn luyện lại.
- **Tinh chỉnh mô hình (Finetuning):** Xây dựng quy trình huấn luyện chuyển giao (Transfer Learning) thông qua việc tinh chỉnh (finetuning) một mô hình pre-trained cho bài toán phân loại văn bản nhị phân (Binary Text Classification) sử dụng `Trainer` API và bộ dữ liệu từ thư viện `datasets`.

---

## 2. Yêu cầu (Requirements & Tasks)

### 2.1. Đề bài gốc (Original Brief)
**Exercise 1: Sentiment Analysis with Hugging Face**
1. Install the Hugging Face transformers library.
2. Use a pre-trained sentiment analysis model from the Hugging Face Hub.
3. Tokenize a sample sentence.
4. Perform sentiment analysis on the sentence.

**Exercise 2: Finetuning a Pretrained Model for Binary Text Classification**
1. Install the necessary Hugging Face libraries (transformers, datasets, evaluate).
2. Load a simple dataset for binary text classification.
3. Load a pretrained model and its tokenizer.
4. Preprocess the dataset to be suitable for the model.
5. Define training arguments.
6. Create a Trainer object and finetune the model.
7. Evaluate the finetuned model.

### 2.2. Thống nhất thực hiện (Agreed Tasks Breakdown)
- **Task 1 — Sentiment Analysis Inference (Exercise 1):**
  - Cài đặt thư viện `transformers`.
  - Tải một mô hình phân tích cảm xúc và tokenizer tương ứng (ví dụ: `pipeline` hoặc tải rời pre-trained model) từ Hugging Face Hub.
  - Chuẩn bị một câu văn mẫu, thực hiện quá trình Tokenization để chuyển văn bản thành input tensors.
  - Đưa input qua mô hình để lấy kết quả phân tích cảm xúc dự đoán.
- **Task 2 — Dataset Preparation & Preprocessing (Exercise 2 - Steps 1-4):**
  - Cài đặt đầy đủ bộ 3 thư viện: `transformers`, `datasets`, `evaluate`.
  - Sử dụng thư viện `datasets` để tải một bộ dữ liệu phân loại nhị phân cơ bản (ví dụ: `imdb`, `glue`...).
  - Tải một mô hình ngôn ngữ pre-trained (như BERT, DistilBERT, v.v.) kèm Tokenizer phù hợp với bài toán phân loại chuỗi.
  - Xây dựng hàm tiền xử lý (preprocess) áp dụng Tokenizer lên toàn bộ tập dữ liệu (padding và truncation hợp lý).
- **Task 3 — Model Finetuning & Evaluation (Exercise 2 - Steps 5-7):**
  - Thiết lập các tham số huấn luyện qua lớp `TrainingArguments` (bao gồm learning rate, batch size, số epochs, v.v.).
  - Khởi tạo đối tượng `Trainer`, truyền vào mô hình, training arguments, tập dữ liệu train/eval, tokenizer và hàm tính toán metric (compute_metrics) từ thư viện `evaluate`.
  - Chạy tiến trình huấn luyện (`trainer.train()`) để finetune mô hình.
  - Chạy tiến trình đánh giá (`trainer.evaluate()`) trên tập Validation/Test và xuất báo cáo kết quả metric (Accuracy, F1...).

---

## 3. Ràng buộc kỹ thuật (Constraints)

### 3.1. Đề bài gốc (Original Brief)
Use: Hugging Face libraries (transformers, datasets, evaluate).

### 3.2. Thống nhất thực hiện (Agreed Constraints)
- **Tuân thủ chuẩn Hugging Face:** Bắt buộc sử dụng các API tiêu chuẩn của Hugging Face (`AutoModelForSequenceClassification`, `AutoTokenizer`, `Trainer`, `TrainingArguments`) để đảm bảo tính đồng bộ của hệ sinh thái.
- **Quản lý tài nguyên huấn luyện:** Thiết lập batch size và số epochs hợp lý trong `TrainingArguments` để đảm bảo thời gian finetuning không quá dài, phù hợp với môi trường Lab.
- **Cấu trúc mã nguồn tinh gọn:** Viết mã rõ ràng trong Jupyter Notebook, có comment giải thích cho từng bước khai báo mô hình, tokenization, và vòng lặp Trainer.

---

## 4. Sản phẩm cần nộp (Deliverables)

### 4.1. Đề bài gốc (Original Brief)
Deliver: Python code/Notebook implementing Exercise 1 and Exercise 2, along with output metrics and evaluation results.

### 4.2. Thống nhất thực hiện (Agreed Deliverables)
- **Mã nguồn và Notebook:** File tương tác hoàn chỉnh `notebooks/lab03.ipynb` chứa toàn bộ code thực thi cho cả Exercise 1 và Exercise 2, kèm theo các output hiển thị rõ ràng trên cell.
- **Kết quả hiển thị:**
  - Kết quả in ra text (Output) của mô hình Sentiment Analysis đối với câu văn mẫu (Exercise 1).
  - Bảng log quá trình huấn luyện từ đối tượng `Trainer` (Loss, Epochs, v.v.).
  - Kết quả các metric (Accuracy/F1) của mô hình sau khi finetune ở bước đánh giá (Exercise 2).
- **Báo cáo kỹ thuật:** Toàn bộ log huấn luyện, kết quả đánh giá và quy trình thực hiện được ghi nhận vào tài liệu sống `walkthrough_lab_03.md`.