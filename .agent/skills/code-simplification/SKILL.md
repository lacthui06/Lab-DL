---
name: code-simplification
description: Refactor và tinh gọn codebase Data Science và Deep Learning. Chuyển code Jupyter Notebook khám phá thành gói Python module sẵn sàng production, loại bỏ biến toàn cục, hằng số hard-code và ký tự phi chuẩn.
---

# code-simplification (Bản DS & AI)

## Tổng quan

Skill này refactor các script notebook khám phá (exploratory), biến toàn cục rải rác, và logic tùy hứng thành **code Python sạch, module hóa, chuẩn production**. Nó chuẩn bị codebase cho việc nộp bài môn học hoặc tích hợp kỹ thuật, đồng thời đảm bảo tái lập chính xác kết quả thí nghiệm.

Điểm then chốt: refactor không được làm thay đổi kết quả. Một refactor thành công phải cho ra cùng loss và metric với code gốc khi chạy lại với seed cố định.

## Khi nào sử dụng

- Dọn dẹp và tổ chức codebase sau khi huấn luyện mô hình thành công, trước khi nộp bài.
- Tách các thành phần tái sử dụng được từ notebook `.ipynb` sang các file `.py` có cấu trúc dưới `src/`.
- Loại bỏ các hyperparameter hard-code và các hằng số phân mảnh.

## 6 bước refactor

### 1. Loại bỏ hằng số và đường dẫn file hard-code
- Tập trung mọi hyperparameter, đường dẫn, và cài đặt thực thi vào các lớp cấu hình có cấu trúc hoặc file YAML config. Ví dụ một dataclass cấu hình huấn luyện nên chứa các trường: đường dẫn dữ liệu, batch size, learning rate, số epoch, và random seed.

### 2. Loại bỏ biến toàn cục (Global Variables)
- Thay thế trạng thái toàn cục kiểu notebook bằng các hàm thuần (pure function) với chữ ký tường minh:
  - `load_data(path: str) -> pd.DataFrame`
  - `preprocess_features(df: pd.DataFrame, is_train: bool) -> np.ndarray`
  - `train_one_epoch(model: nn.Module, loader: DataLoader, ...) -> float`

### 3. Duy trì nguồn chân lý duy nhất cho tiền xử lý (Single Source of Truth)
- Không duy trì logic tiền xử lý trùng lặp cho huấn luyện và đánh giá. Đóng gói các biến đổi trong các pipeline class thống nhất hoặc Torchvision transforms để đảm bảo hành vi giống hệt nhau trong lúc inference.
- Đây là nguồn gốc của lỗi train/serve skew - một trong những lỗi production nguy hiểm và khó phát hiện nhất.

### 4. Thêm type hint và docstring chính xác
- Bao gồm chú thích kiểu (type annotation) và docstring mô tả chiều tensor để loại bỏ nhập nhằng. Docstring của hàm `forward` cần nêu rõ shape của tensor đầu vào và shape của logits đầu ra.

### 5. Loại bỏ dead code và import không dùng
- Dọn sạch các import không dùng, câu lệnh `print()` nháp, và các cell thí nghiệm đã deprecated.

### 6. Áp đặt chính sách không Emoji nghiêm ngặt
- Nghiêm cấm emoji và icon trang trí trong:
  - File source code (`.py`, `.sh`).
  - Câu lệnh xuất ra terminal (`print()`, logger, exception message).
  - Comment code và docstring.
  - Markdown và code cell của Jupyter Notebook (`.ipynb`).
- Thể hiện mọi trạng thái bằng text kỹ thuật chuẩn (ví dụ `[INFO]`, `[SUCCESS]`, `[ERROR]`, `Section 1: ...`). Điều này ngăn ngừa sự cố `UnicodeEncodeError` trên các hệ điều hành khác nhau (như Windows CP1252) và duy trì chuẩn học thuật chuyên nghiệp.

## Tiêu chí xác thực

- Việc chạy lại các module đã refactor với random seed cố định cho ra loss và metric đánh giá giống hệt nhau.
- Quét repository xác nhận không còn emoji hay biểu tượng đồ họa phi chuẩn trong code và notebook.
