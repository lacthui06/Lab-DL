# Lab 02 — Transfer learning trên CIFAR-10


So sánh ResNet18 và DenseNet121 pretrained, head-only và fine-tune. Kết quả hiện tại là CPU 64×64, seed 42; kế hoạch tinh chỉnh mới chưa chạy.

## Structure

- [description_lab_02.md](description_lab_02.md): bản diễn giải đề đang có trong repo, giữ nguyên nội dung; không phải bản chụp nguyên văn đề giảng viên.
- [walkthrough_lab_02.md](walkthrough_lab_02.md): tài liệu sống duy nhất, theo [template](../../.agents/templates/walkthrough_template.md): Spec → Changelog → Report.
- [01_lab02_eda.ipynb](notebooks/01_lab02_eda.ipynb): raw → preprocessing → feature engineering, không nạp mạng.
- [02_lab02_modeling.ipynb](notebooks/02_lab02_modeling.ipynb): kiến trúc tóm tắt, LR search, đường cong, chất lượng/chi phí, phân tích lỗi và đề xuất.
- [config/](config/): [full.json](config/full.json) cho GPU 224×224; [cpu_pilot.json](config/cpu_pilot.json) kiểm tra nhỏ; [cpu_full.json](config/cpu_full.json) mô tả run CPU lịch sử; [tuning_plan.json](config/tuning_plan.json) là kế hoạch chưa thực thi, không đưa trực tiếp cho run.py.
- [requirements.txt](requirements.txt): phiên bản cố định từ Windows; cài trên Linux/Kaggle chưa kiểm chứng.
- [outputs/selected_run.json](outputs/selected_run.json): run được chọn; [outputs/label_update.json](outputs/label_update.json) lưu lịch sử sửa nhãn tài liệu. `outputs/runs/` là bằng chứng gốc; `outputs/review/` chứa biểu đồ trình bày mới, không ghi đè run gốc.

### Tham chiếu source và CLI

| File | Nơi sử dụng / mục đích |
|---|---|
| [src/data.py](src/data.py) | experiment.py và infer.py: split, biến đổi và loader |
| [src/models.py](src/models.py) | experiment.py và infer.py: khởi tạo mạng, freeze, cached tail |
| [src/experiment.py](src/experiment.py) | run.py, resume_interrupted.py: train/validation, checkpoint |
| [src/reporting.py](src/reporting.py) | experiment.py: xuất metric/hình từ dự đoán |
| [src/notebook_views.py](src/notebook_views.py) | Cả hai notebook: đọc bảng/hình đã lưu, không import torch hoặc models |
| [src/delivery.py](src/delivery.py) | finalize.py và complete_delivery.py: đóng gói và kiểm chứng |
| [src/__init__.py](src/__init__.py) | Khai báo package src |
| [scripts/run.py](scripts/run.py) | Điểm vào huấn luyện, dùng config/*.json |
| [scripts/infer.py](scripts/infer.py) | Nạp checkpoint và dự đoán một ảnh |
| [scripts/download_assets.py](scripts/download_assets.py) | CIFAR-10 và weights chính thức |
| [scripts/download_cifar10.py](scripts/download_cifar10.py) | Tải CIFAR-10 chia phần khi máy chủ chậm; không tải weights |
| [scripts/resume_interrupted.py](scripts/resume_interrupted.py) | Phục hồi tại ranh giới kiến trúc, không phải resume giữa epoch |
| [scripts/execute_notebook.py](scripts/execute_notebook.py) | Thực thi notebook với interpreter hiện tại |
| [scripts/finalize.py](scripts/finalize.py) | CLI ngắn gọi delivery.finalize_run; cần dataset, checkpoint, events đầy đủ |
| [scripts/complete_delivery.py](scripts/complete_delivery.py) | CLI ngắn chờ run, chạy test/finalize/notebook/infer |
| [tests/test_contracts.py](tests/test_contracts.py) | Shape, freeze, split và một batch overfit |
| [tests/test_cached_prefix.py](tests/test_cached_prefix.py) | Đối chiếu logits/gradient cache và forward đầy đủ |
| [tests/test_notebook_review.py](tests/test_notebook_review.py) | Đối chiếu metric, ranh giới EDA và liên kết tài liệu; không cần torch |

Các CLI còn lại có chức năng riêng nên được giữ theo phương án đã trao đổi. `build_notebook.py` đã bỏ. Ba source snapshot trong run là chứng cứ lịch sử được manifest/hash tham chiếu; không phải code đang thực thi và không xóa như file rác.

## Reproduce

Chạy từ `labs/lab02`. Để **xem kết quả**, không cần dataset/weights: mở hai notebook đã có output, hoặc chạy lại bằng interpreter có NumPy, pandas, matplotlib, IPython, nbformat, nbclient và ipykernel:

```powershell
python -X utf8 scripts/execute_notebook.py 01_lab02_eda.ipynb
python -X utf8 scripts/execute_notebook.py 02_lab02_modeling.ipynb
python -X utf8 -m unittest discover -s tests -p test_notebook_review.py -v
```

Để **huấn luyện**, tạo môi trường từ requirements, rồi chạy trên GPU. Local chỉ chạy test/sanity nhỏ theo [rule Zero Waste](../../.agents/AGENTS.md):

```powershell
python -m pip install -r requirements.txt
python scripts/download_assets.py
python -m pytest tests -q
python scripts/run.py --config config/full.json --device cuda
python scripts/complete_delivery.py outputs/runs/<run_id>
tensorboard --logdir outputs/runs
```

Chạy `download_cifar10.py` trước `download_assets.py` nếu cần phương án chia phần. `finalize.py` có thể gọi riêng sau run; nó lưu `requirements-freeze.txt` vào run, không ghi đè requirements dự án. Đổi run xem bằng `LAB02_RUN`; đổi vị trí source bằng `LAB02_ROOT`.

Python của run lịch sử: CPython 3.13.3; môi trường thực tế lưu trong `environment.json` của run. Ghim phiên bản không bảo đảm cài được trên mọi OS; chưa chạy lại full training trong đợt chỉnh trình bày này.

## Results Summary

| Fine-tune | Test accuracy | Test loss | Pipeline train (s) |
|---|---:|---:|---:|
| ResNet18 | 81,49% | 0,547387 | 1414,15 |
| DenseNet121 | 81,53% | 0,547715 | 741,18 |

DenseNet dùng cache prefix, ResNet không dùng: thời gian chỉ mô tả hai run thực tế, chưa xếp hạng tốc độ kiến trúc. Kết luận và kế hoạch LR/optimizer/unfreeze nằm trong [walkthrough](walkthrough_lab_02.md).
