---
name: code-simplification
description: Tinh gọn, tái cấu trúc và làm sạch mã nguồn Data Science và AI. Chuyển đổi mã nguồn nháp từ Jupyter Notebook thành các module Python tái sử dụng, loại bỏ biến toàn cục và số hard-code.
---

# code-simplification (DS & AI Edition)

## Tổng quan
Kỹ năng này chịu trách nhiệm biến các đoạn mã nháp lộn xộn, biến toàn cục và vòng lặp phức tạp (vốn thường thấy trong Jupyter Notebook) thành **mã nguồn Python chuẩn kỹ thuật (Production Clean Code)**, sẵn sàng để nộp bài hoặc tích hợp vào hệ thống lớn mà vẫn giữ nguyên 100% kết quả huấn luyện.

## Khi nào sử dụng
- Khi mô hình đã chạy thành công và muốn dọn dẹp để nộp bài hoặc bàn giao.
- Khi cần chuyển đổi từ file `.ipynb` (Jupyter Notebook) sang các file script `.py` có cấu trúc module.
- Khi mã nguồn có quá nhiều tham số viết thẳng vào code (Hard-coded constants).

## 5 Bước làm sạch mã nguồn DS & AI

### 1. Trục xuất số cứng và đường dẫn cứng (No Hard-coded Values)
- Gom toàn bộ siêu tham số (Hyperparameters) và đường dẫn file vào một lớp cấu hình `Config` hoặc file `config.yaml`:
  ```python
  from dataclasses import dataclass

  @dataclass
  class TrainConfig:
      data_path: str = "data/raw_dataset.csv"
      batch_size: int = 32
      learning_rate: float = 1e-4
      epochs: int = 20
      random_seed: int = 42
  ```

### 2. Triệt tiêu biến toàn cục (Kill Global Variables)
- Mã nguồn trong Notebook thường dùng biến toàn cục chạy xuyên suốt các cell. Khi đơn giản hóa, bắt buộc phải đóng gói thành các hàm thuần túy (Pure Functions) nhận tham số đầu vào và trả về đầu ra rõ ràng:
  - `load_data(path: str) -> pd.DataFrame`
  - `preprocess_features(df: pd.DataFrame, is_train: bool) -> np.ndarray`
  - `train_one_epoch(model: nn.Module, loader: DataLoader, ...) -> float`

### 3. Tái sử dụng logic tiền xử lý (Single Source of Truth)
- Tuyệt đối không viết 2 hàm tiền xử lý riêng cho tập Train và tập Inference. Đóng gói quy trình xử lý thành một class hoặc Pipeline thống nhất (dùng `sklearn.pipeline.Pipeline` hoặc class kế thừa) để đảm bảo dữ liệu đưa vào dự đoán được xử lý đúng 100% như lúc huấn luyện.

### 4. Bổ sung Type Hints và Docstrings ngắn gọn
- Thêm chú thích kiểu dữ liệu (Type Hints) cho các tham số dạng Tensor hoặc DataFrame để người đọc code không bị mơ hồ về kích thước shape:
  ```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      """
      Args:
          x: Input tensor có shape (batch_size, channels, height, width)
      Returns:
          Logits tensor có shape (batch_size, num_classes)
      """
  ```

### 5. Dọn dẹp mã chết (Dead Code & Unused Imports)
- Xóa toàn bộ các dòng `import` thừa, các dòng `print()` nháp trong vòng lặp và các cell thử nghiệm thất bại bị bỏ xó.

## Tiêu chí nghiệm thu (Verification)
- Chạy lại toàn bộ script đã được làm sạch với cùng random seed và khẳng định kết quả đo lường (Metrics/Loss) hoàn toàn trùng khớp với phiên bản ban đầu.
