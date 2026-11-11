# Mô-đun Kiến Trúc Mô Hình (src/models)

## 1. Thông tin chung
- **Tác giả:** Thái Anh Lạc
- **Ngày cập nhật:** 2026-11-11
- **Vị trí:** `labs/lab03/src/models/`
- **Tệp mã nguồn:** [`classifier.py`](classifier.py)

## 2. Mục đích & Vai trò
- Khởi tạo kiến trúc mô hình phân loại cảm xúc nhị phân dựa trên nền tảng Transformer nhẹ: **DistilBERT** (`distilbert-base-uncased`).
- Cung cấp cơ chế tinh chỉnh lớp phân loại (Classification Head) với các tham số Dropout tùy biến để chống hiện tượng học vẹt (overfitting).
- Cung cấp hai đường dẫn suy luận độc lập: qua Hugging Face High-level Pipeline và qua Low-level PyTorch Forward, đồng thời thực hiện kiểm tra chéo (Cross-verification) tính nhất quán số học.

## 3. Khởi tạo & Cơ chế hoạt động

### 3.1. Nạp Mô Hình và Tokenizer (`load_model`)
- Khởi tạo `AutoTokenizer` và `AutoModelForSequenceClassification` từ Hugging Face Hub theo `name` và `revision`.
- Gán nhãn nhị phân chuẩn xác: `0: "NEGATIVE"`, `1: "POSITIVE"`.
- Cho phép ghi đè cấu hình Dropout trong quá trình fine-tuning (`dropout`, `attention_dropout`, `seq_classif_dropout`) với điều kiện giá trị xác suất hợp lệ thuộc nửa khoảng `[0, 1)`.

### 3.2. Suy Luận Thủ Công và Đối Chiếu Pipeline
- `manual_predict(model, tokenizer, texts, max_length)`: Đưa mô hình về chế độ đánh giá (`model.eval()`), mã hóa tensor kèm padding/truncation, thực thi suy luận trong ngữ cảnh không tính đạo hàm (`torch.inference_mode()`), và áp dụng Softmax trên Logits để thu được xác suất hai lớp.
- `compare_inference(model, tokenizer, texts, max_length)`: Chạy song song cả hai cơ chế suy luận (Hugging Face Pipeline vs Manual PyTorch). Xác nhận nhãn đầu ra trùng khớp và độ sai lệch xác suất không vượt quá ngưỡng sai số số học `1e-5`.
- `verify_saved_model(reference_model, model_path, texts)`: Kiểm tra độ đồng nhất giữa mô hình tham chiếu và checkpoint lưu trên đĩa (trùng khớp tên tham số, độ lệch trọng số trong phạm vi dung sai và kết quả suy luận giống nhau trên cùng một môi trường CPU FP32).

## 4. Thảo luận kỹ thuật & Quyết định thiết kế
- **Lý do lựa chọn DistilBERT thay vì BERT-base hoặc RoBERTa:**
  - DistilBERT giữ lại 97% năng lực hiểu ngôn ngữ tự nhiên của BERT-base thông qua kỹ thuật chưng cất tri thức (Knowledge Distillation), nhưng giảm 40% số lượng tham số (~66M tham số so với ~110M của BERT).
  - Tốc độ suy luận nhanh hơn 60%, giúp việc huấn luyện và thử nghiệm trên GPU miễn phí (Kaggle T4) hoặc CPU cục bộ diễn ra nhanh chóng, tiết kiệm chi phí tính toán.
- **Thiết kế Classification Head:** Đầu ra của DistilBERT trích xuất vector biểu diễn của token đầu tiên `[CLS]`, đi qua tầng Dropout chống overfit, sau đó chiếu qua tầng Linear `(768 -> 2)` để sinh ra 2 logits tương ứng với hai lớp cảm xúc.
- **Ranh giới an toàn suy luận:** Luôn kiểm tra tính tương đương số học giữa Pipeline cấp cao và Forward thủ công để đảm bảo mô hình khi đưa vào sản phẩm thực tế không bị ảnh hưởng bởi các lớp bọc (wrapper/autocast).
