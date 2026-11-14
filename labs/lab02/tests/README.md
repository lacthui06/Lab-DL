# Lab02 · Bản Đồ Test Gates

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-11 | 0.1.0 | Thường FEAI, cùng Codex | Mô tả phạm vi các test gate, lệnh chạy và phụ thuộc artifact |
| 2026-11-11 | 0.1.1 | Thường FEAI, cùng Codex | Bổ sung lệnh chạy không có giao diện và phạm vi 33 test đã kiểm chứng |

## 1. Mục đích

Bộ test kiểm tra các hợp đồng quan trọng của Lab02 trước khi sử dụng kết quả: ảnh/nhãn/index đi cùng nhau, phần backbone đóng băng không bị cập nhật, lựa chọn không dùng test và benchmark giữ đúng phạm vi đo.

Gate ở đây là nhóm kiểm tra cho một trách nhiệm kỹ thuật. Bảng này mô tả các test hiện có; không đồng nghĩa mọi gate tự động chạy trước mỗi CLI hoặc toàn bộ test vừa được xác nhận đạt.

## 2. Bản đồ kiểm tra

Phần chuẩn hóa notebook và HTML có thêm [`test_notebook_workflow.py`](test_notebook_workflow.py) (view không import training, mode sai, config thiếu, run mới và benchmark khác run) và [`test_html_report.py`](test_html_report.py) (nguồn run, ảnh, điều hướng và escape code). Chạy cùng `test_notebook_review.py` sau khi build báo cáo HTML.

| Gate | Tệp kiểm thử | Điều cần bảo vệ | Giới hạn hoặc điều kiện |
| --- | --- | --- | --- |
| Dữ liệu và EDA | [test_eda.py](test_eda.py), [test_contracts.py](test_contracts.py) | Split không giao nhau, phát hiện split bị sửa, công thức đặc trưng, bảo toàn RGB và normalization. | Một số ca đối chiếu cần dữ liệu CIFAR-10 hoặc artifact của run được chọn. |
| Model và freeze | [test_contracts.py](test_contracts.py), [test_tuning.py](test_tuning.py) | Shape logits, số tham số, gradient ở phần mở và thống kê BatchNorm cố định. | Kiểm tra trên input được dựng cho test không chứng minh chất lượng phân loại. |
| Feature và prefix cache | [test_feature_cache.py](test_feature_cache.py), [test_cached_prefix.py](test_cached_prefix.py) | Feature/nhãn/index đúng thứ tự, shuffle tái lập, kết quả đường cache tương đương và hook được dọn khi lỗi. | Prefix cache chỉ hợp lệ trong điều kiện đóng băng/transform mà source hỗ trợ. |
| Training và lựa chọn | [test_tuning.py](test_tuning.py) | Optimizer, scheduler, early stopping, chọn theo validation và khởi tạo lại cùng head cho các candidate. | Smoke test dùng fixture không thay thế full training trên CIFAR-10. |
| Sanity overfit và resume | [test_tuning.py](test_tuning.py) | Học được một batch nhỏ; khôi phục trạng thái epoch, optimizer, scheduler và RNG đúng. | Test có thể thực hiện các bước cập nhật trên model thử nghiệm; không phải lệnh train run bàn giao. |
| Inference và benchmark | [test_inference_benchmark.py](test_inference_benchmark.py), [test_training_benchmark.py](test_training_benchmark.py) | Load đúng metadata, tách các lượt đo, kiểm tra input lỗi, phân biệt RSS/VRAM và bảo toàn checkpoint nguồn. | Có kiểm tra CUDA bằng mock; đạt test không chứng minh benchmark GPU đã chạy trên phần cứng thật. |
| Notebook và trình bày | [test_notebook_review.py](test_notebook_review.py) | Đọc đúng artifact, từ chối bằng chứng sai/hỏng, xử lý đường dẫn và giữ notebook xem kết quả không tự train/benchmark. | Output notebook vẫn cần kiểm tra trực quan khi sửa bố cục. |
| Ranh giới import và bàn giao | [test_import_boundaries.py](test_import_boundaries.py), [test_cleanup.py](test_cleanup.py) | Lazy import, CLI không tự thực thi khi import/help, inventory source, đường resume và orchestration bàn giao. | Một số ca dùng mock; các ca liên quan dữ liệu thật cần đủ artifact cục bộ. |

## 3. Lệnh chạy

Từ thư mục `labs/lab02/`, với môi trường dependencies của Lab02 đã được kích hoạt:

```bash
python -m pytest tests -q
```

Muốn chạy riêng phần trình bày notebook và hợp đồng inference:

```bash
python -m pytest tests/test_notebook_review.py tests/test_inference_benchmark.py -q
```

Nếu chạy trong terminal không có giao diện hoặc backend Matplotlib trên macOS bị lỗi, dùng backend `Agg` để vẽ hình không cần mở cửa sổ. Với shell zsh/bash:

```bash
MPLBACKEND=Agg python -m pytest tests/test_notebook_review.py tests/test_inference_benchmark.py -q
```

Ngày 2026-11-11, lệnh với backend `Agg` đã chạy trên môi trường local của dự án: **33 passed, 0 failed, 0 skipped**. Kết quả chỉ áp dụng cho hai tệp test trong lệnh, chưa xác nhận toàn bộ suite Lab02 đạt.

Chạy kiểm thử có thể cập nhật trọng số model thử nghiệm trong bộ nhớ và tạo artifact trong thư mục tạm của pytest. Nó không tương đương Run All của notebook xem kết quả.

## 4. Dữ liệu, artifact và cách đọc lỗi

Các ca dùng fixture hoặc model khởi tạo không pretrained giúp kiểm tra logic mà không cần tải weights ImageNet. Những ca đọc CIFAR-10 thật hoặc đối chiếu run lịch sử vẫn cần dữ liệu và artifact tương ứng.

Một số ca có điều kiện `skip` khi thiếu CIFAR-10; các ca khác đọc trực tiếp registry, split hoặc summary và có thể lỗi khi thiếu tệp. Không được diễn giải “skipped” thành “passed”, hoặc mọi lỗi thiếu artifact thành lỗi thuật toán.

Khi gặp lỗi, đọc tên test, input và assertion trước khi sửa source. Phân biệt lỗi hợp đồng tính toán, lỗi môi trường/phụ thuộc và lỗi thiếu bằng chứng.

## 5. Quan hệ với kiểm chứng run

Unit test kiểm tra hành vi của hàm và tình huống được dựng. Kiểm chứng run đối chiếu kết quả thực tế qua split, predictions, metrics và manifest; xem [`finalize_run()`](../src/utils/delivery.py#L71).

Sanity overfit cho thấy model có thể học một batch nhỏ trong điều kiện test, chưa chứng minh không có leakage hoặc đạt accuracy mong muốn trên dữ liệu chưa thấy. Kết quả nghiệm thu phải ghi lệnh chạy, số passed/failed/skipped và các điều kiện chưa kiểm chứng.
