# Lab01 · Thư Mục Mô Hình Bàn Giao (models/)

Thư mục này chứa toàn bộ các file trọng số mô hình đã được huấn luyện hoàn chỉnh của Lab 01, sẵn sàng để nạp và suy luận ngay:

| Tên file checkpoint | Kiến trúc mô hình | Số tham số | Điểm Val Acc | Điểm Test Acc | Ghi chú nghiệp vụ |
| --- | --- | ---: | ---: | ---: | --- |
| [SmallMLP.pt](SmallMLP.pt) | `FashionMLP(784 -> 128 -> 10)` | 101.770 | 90.07% | 88.52% | Bản rút gọn, ít tham số nhất, tốc độ chạy nhanh nhất |
| [MainMLP.pt](MainMLP.pt) | `FashionMLP(784 -> 256 -> 128 -> 10)` | 235.146 | 90.78% | 89.41% | Bản chuẩn huấn luyện trên 784 điểm ảnh Z-score |
| [MainMLP_FE.pt](MainMLP_FE.pt) | `FashionMLP(794 -> 256 -> 128 -> 10)` | 237.706 | 90.88% | 89.60% | Bản chuẩn kết hợp thêm 10 đặc trưng hình học thủ công |
| [best_model.pt](best_model.pt) | `FashionMLP(784 -> 256 -> 128 -> 10)` | 235.146 | 90.85% | 89.14% | **Mô hình tối ưu nhất sau khi tuning Optuna và refit lại** |
| [manifest.json](manifest.json) | File JSON siêu dữ liệu | — | — | — | Bảng kê truy nguyên chi tiết (thời gian lưu, nguồn gốc run) |

---

## Cách nạp mô hình chạy thử nhanh

Trong code hoặc Jupyter Notebook của bạn, chỉ cần gọi đúng 2 dòng:

```python
from src import load_model, infer_sample

# 1. Nạp mô hình tốt nhất từ best_model.pt
model = load_model("final")

# 2. Hoặc nạp kiến trúc cụ thể từ MainMLP.pt
model_main = load_model("MainMLP")

# 3. Chạy thử suy luận mẫu test số 0
ket_qua = infer_sample("final", index=0)
print(f"Nhãn thật: {ket_qua['true_label']} - Dự đoán: {ket_qua['predicted_label']}")
```
