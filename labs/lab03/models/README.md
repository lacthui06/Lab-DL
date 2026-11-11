# Lab03 · Thư Mục Mô Hình Bàn Giao (models/)

Thư mục này quản lý siêu dữ liệu và hướng dẫn sử dụng mô hình phân loại cảm xúc nhị phân (Binary Sentiment Classification) của Lab 03 trên tập dữ liệu Rotten Tomatoes:

| Mô hình | Kiến trúc / Cơ chế | Ngưỡng quyết định | Test Accuracy | Binary F1 | Ghi chú kỹ thuật |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **TF-IDF Baseline** | `TfidfVectorizer + LogisticRegression` | 0.50 | 78.99% | 78.91% | Baseline đối chứng truyền thống |
| **DistilBERT (Ngưỡng mặc định)** | `DistilBERT + Classification Head` | 0.50 | 82.93% | 82.89% | Phân loại theo Argmax xác suất chuẩn |
| **DistilBERT (Ngưỡng tối ưu)** | `DistilBERT + Optimal Validation Threshold` | **0.45** | **83.11%** | **83.30%** | **Mô hình chiến thắng đạt kết quả cao nhất** |

---

## Cách nạp mô hình chạy suy luận nhanh

### Cách 1: Sử dụng mã nguồn của Lab 03 (`src.models.classifier`)
```python
from src.models.classifier import load_model, manual_predict

# 1. Nạp mô hình DistilBERT và Tokenizer
model, tokenizer = load_model("distilbert-base-uncased", finetuning=True)

# 2. Dự đoán cảm xúc câu review
texts = [
    "A remarkably charming, witty, and deeply emotional masterpiece.",
    "Utterly boring, pretentious, and an absolute waste of two hours."
]
results = manual_predict(model, tokenizer, texts)

# 3. Áp dụng ngưỡng tối ưu 0.45
for r in results:
    p_positive = r["probabilities"][1]
    pred_label = "POSITIVE" if p_positive >= 0.45 else "NEGATIVE"
    print(f"Câu: '{r['text']}' -> P(Positive): {p_positive:.4f} => {pred_label}")
```

### Cách 2: Sử dụng trực tiếp Hugging Face Pipeline
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
predictions = classifier(["An entertaining and delightful movie!"])
print(predictions)
```

---

## Truy nguyên nguồn gốc (Provenance)
Thông tin chi tiết về các siêu tham số, phân phối xác suất và báo cáo kiểm định được lưu vết tại [`manifest.json`](manifest.json) dẫn nguồn từ run Kaggle `20261103T110437Z_frozen_trial_b021d6eb`.
