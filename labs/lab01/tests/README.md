# Bộ Kiểm Thử Tự Động — Lab 01 Tests (`labs/lab01/tests/`)

## 1. Thông Tin Mô-đun (Name & Metadata)
- **Tên mô-đun:** `tests/` (Automated Test Suite & Regression Verification)
- **Ngày tạo (Created):** 2026-10-22 · **Tác giả (Author):** Thái Anh Lạc
- **Ngày cập nhật (Updated):** 2026-11-10 · **Người cập nhật (Author):** Thái Anh Lạc
- **Tệp tài liệu chính thức:** `labs/lab01/tests/README.md`

---

## 2. Mục Đích (Purpose)
Thư mục này chứa toàn bộ hệ thống kiểm thử tự động gồm **49 bài test (Test Cases)** độc lập nhằm bảo đảm tính toàn vẹn và độ tin cậy của mã nguồn Lab 01:
- Tự động kiểm tra từng mắt xích nhỏ nhất trong hệ thống: từ việc nạp dữ liệu, chia phân tầng, tính toán 10 đặc trưng hình học, dựng khung mạng nơ-ron, vòng lặp huấn luyện, tối ưu hóa siêu tham số Optuna cho đến kịch bản thực thi end-to-end.
- Đóng vai trò là "chốt chặn an toàn" (Safety Gate): Bất kỳ một chỉnh sửa mã nguồn nào làm lệch kích thước tensor, làm rò rỉ dữ liệu hoặc làm hỏng logic lưu checkpoint đều sẽ bị bộ test phát hiện ngay lập tức.
- Giúp bạn chỉ cần gõ đúng 1 dòng lệnh duy nhất để tự tin nghiệm thu toàn bộ bài làm trước khi nộp bài cho giảng viên:
  ```bash
  pytest labs/lab01/tests
  ```

---

## 3. Danh Mục Các Tệp Kiểm Thử (Test Suite Inventory)

### 3.1. Phân Loại & Bản Đồ Điều Hướng Code
Bộ test được phân chia thành 9 tệp kiểm thử chuyên biệt, tương ứng với từng giai đoạn trong pipeline:

| Tệp kiểm thử (Source File) | Số bài test | Mục tiêu kiểm chứng kỹ thuật | Liên kết xem code (Ctrl+Click) |
| :--- | :---: | :--- | :--- |
| [`test_bootstrap.py`](test_bootstrap.py) | 2 | Kiểm tra khởi tạo môi trường, đường dẫn thư mục gốc `DATA_DIR`, `OUTPUTS_DIR` và khả năng nạp các gói thư viện cơ bản. | [`test_bootstrap.py#L1`](test_bootstrap.py#L1) |
| [`test_datasets.py`](test_datasets.py) | 16 | Kiểm tra thuật toán phân chia phân tầng Stratified Split (đúng tỷ lệ 9:1, đúng 54k train/6k val/10k test), kiểm tra tính chuẩn hóa Z-score pixel và chống rò rỉ dữ liệu giữa các tập. | [`test_datasets.py#L1`](test_datasets.py#L1) |
| [`test_features_comparison.py`](test_features_comparison.py) | 8 | Kiểm tra hàm trích xuất 10 đặc trưng hình học/độ sáng, kiểm tra tính đúng đắn của kích thước đầu ra `(N, 10)` và xử lý an toàn các trường hợp biên (ảnh hoàn toàn đen, ảnh rỗng). | [`test_features_comparison.py#L1`](test_features_comparison.py#L1) |
| [`test_models.py`](test_models.py) | 2 | Kiểm tra lớp mạng `FashionMLP`, hàm khởi tạo `build_model()`, số chiều đầu ra Logits `(Batch, 10)` và kiểm tra hàm đếm tham số `count_parameters()`. | [`test_models.py#L1`](test_models.py#L1) |
| [`test_training.py`](test_training.py) | 4 | Kiểm tra vòng lặp lớp `Trainer`, hàm `train_one_epoch()`, hàm `evaluate()` và bài test chốt chặn Sanity check 8 mẫu overfit (yêu cầu loss < 0.2 và accuracy > 80%). | [`test_training.py#L1`](test_training.py#L1) |
| [`test_tuning.py`](test_tuning.py) | 2 | Kiểm tra hàm tạo optimizer `build_optimizer()` (SGD, RMSprop, Adam), cấu hình Optuna Study và cơ chế cắt tỉa sớm `MedianPruner`. | [`test_tuning.py#L1`](test_tuning.py#L1) |
| [`test_performance.py`](test_performance.py) | 6 | Kiểm tra hàm đo đạc tài nguyên huấn luyện `benchmark_models()` và `single_sample_training_benchmark()` trong tiến trình con (subprocess), kiểm tra số đo thời gian và RAM. | [`test_performance.py#L1`](test_performance.py#L1) |
| [`test_publication.py`](test_publication.py) | 3 | Kiểm tra bộ công cụ xuất bản artifact, xác thực tính toàn vẹn của các bảng số liệu CSV, JSON và các biểu đồ PNG. | [`test_publication.py#L1`](test_publication.py#L1) |
| [`test_workflow.py`](test_workflow.py) | 6 | Kiểm tra kịch bản chạy tự động end-to-end, kiểm tra độ ổn định lưu/nạp mô hình (roundtrip save/load) và cơ chế tự động bàn giao mô hình `promote_model_checkpoint()`. | [`test_workflow.py#L1`](test_workflow.py#L1) |

- **Tổng cộng:** **49 bài test (All Passed 100%)**.

---

### 3.2. Cấu Hình Kiểm Thử Chung (`conftest.py`)
Tệp [`conftest.py`](conftest.py#L1) thiết lập fixture dùng chung cho toàn bộ pytest:
- Tự động thêm đường dẫn gốc của repository vào `sys.path` để mọi file test có thể import trực tiếp từ `src`.
- Khóa chặt seed ngẫu nhiên mặc định trước khi mỗi fixture chạy để các ca kiểm tra tính toán số học luôn có tính lặp lại tuyệt đối (deterministic).

---

## 4. Các Thảo Luận & Quyết Định Kỹ Thuật (Discussions from Description)

### 4.1. Vì sao một bài Lab Deep Learning lại cần có bộ Test Suite toàn diện? ([Description Mục 5.6](../../description_lab_01.md#5-tiêu-chí-nghiệm-thu))
- Trong phát triển phần mềm truyền thống, kiểm thử giúp bắt lỗi cú pháp. Nhưng trong Deep Learning, phần lớn các lỗi nghiêm trọng lại **không hề làm văng lỗi chương trình**:
  - Viết nhầm code khiến dữ liệu train bị rò rỉ sang test $\rightarrow$ Code vẫn chạy mượt, kết quả cao bất thường nhưng thực chất là sai hoàn toàn về mặt khoa học.
  - Quên gọi `optimizer.zero_grad()` hoặc quên `loss.backward()` $\rightarrow$ Mô hình vẫn chạy qua các epoch nhưng không hề học được gì.
- Bộ test suite 49 bài này hoạt động như một hệ thống "kiểm toán độc lập", bảo vệ toàn bộ các hợp đồng dữ liệu, kích thước tensor và tính toàn vẹn khoa học của bài Lab.

### 4.2. Bài test Sanity Check trong code khác gì so với Test Suite của Pytest? ([Description Mục 7.5](../../description_lab_01.md#5-kiểm-tra-gì-trước-khi-huấn-luyện-chính-thức))
- **Pytest (Unit Tests):** Chạy trước khi bắt đầu dự án, kiểm tra xem các hàm, class và pipeline có bị lỗi cú pháp, sai shape hay lỗi logic hay không.
- **Sanity Check Probe (In-training Gate):** Chạy ngay trước thời điểm bấm nút huấn luyện thật. Nó đưa đúng 8 mẫu ảnh thật vào mô hình chưa train và ép học vẹt trong 120 bước. Chỉ khi mô hình chứng minh được nó có khả năng kéo loss về sát 0 trên dữ liệu thực tế thì quá trình huấn luyện 10 epoch mới được phép bắt đầu.

### 4.3. Kỷ luật tái lập kết quả bit-exact giữa các lần chạy test ([Description Mục 7.11](../../description_lab_01.md#11-môi-trường-chạy-quản-lý-artifact-và-báo-cáo))
- Toàn bộ các bài test liên quan đến việc chia dữ liệu (`make_stratified_split`), khởi tạo trọng số (`FashionMLP`) và trích xuất đặc trưng (`extract_features`) đều sử dụng `seed=42`.
- Điều này đảm bảo rằng dù chạy kiểm thử trên máy Windows, Linux hay macOS, kết quả kiểm tra từng mảng số học đều phải khớp chính xác 100%, không bị phụ thuộc vào tính ngẫu nhiên của phần cứng.
