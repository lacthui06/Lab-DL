# Lab 02 — Transfer learning trên CIFAR-10


So sánh ResNet18 và DenseNet121 pretrained, head-only và fine-tune. Kết quả hiện tại là CPU 64×64, seed 42; pipeline tinh chỉnh LR/optimizer/unfreeze đã triển khai và kiểm thử; benchmark tuning đầy đủ chưa chạy.

## Structure

- [description_lab_02.md](description_lab_02.md): nội dung đối chiếu trực tiếp ảnh đề trang 23; phân biệt yêu cầu của đề với lựa chọn của bài làm.
- [walkthrough_lab_02.md](walkthrough_lab_02.md): tài liệu sống duy nhất, theo [template](../../.agents/templates/walkthrough_template.md): Spec → Changelog → Report.
- [01_lab02_eda.ipynb](notebooks/01_lab02_eda.ipynb): raw → integrity → pixel/spatial EDA → preprocessing → handcrafted features; chỉ dùng `src/eda.py`, không nạp mạng.
- [02_lab02_modeling.ipynb](notebooks/02_lab02_modeling.ipynb): kiến trúc tóm tắt, LR search, đường cong, chất lượng/chi phí, phân tích lỗi và đề xuất.
- [config/](config/): [full.json](config/full.json) cho GPU 224×224; [cpu_pilot.json](config/cpu_pilot.json) kiểm tra nhỏ; [cpu_full.json](config/cpu_full.json) mô tả run CPU lịch sử; [tuned.json](config/tuned.json) là cấu hình tuning chạy được; thiết kế và lý do nằm trong walkthrough.
- [requirements.txt](requirements.txt): phiên bản cố định từ Windows; cài trên Linux/Kaggle chưa kiểm chứng.
- [outputs/selected_run.json](outputs/selected_run.json): run được chọn; [outputs/label_update.json](outputs/label_update.json) lưu lịch sử sửa nhãn tài liệu. `outputs/runs/` là bằng chứng gốc; `outputs/review/` chứa biểu đồ trình bày mới, không ghi đè run gốc.

### Tham chiếu source và CLI

| File | Nơi sử dụng / mục đích |
|---|---|
| [src/data.py](src/data.py) | experiment.py/infer.py: loader/transform; gọi eda.locked_split để tái sử dụng split |
| [src/models.py](src/models.py) | experiment.py và infer.py: khởi tạo mạng, freeze, cached tail |
| [src/tuning.py](src/tuning.py) | Validate grid, AdamW/SGD momentum, cosine và chọn bằng validation/loss/time |
| [src/costs.py](src/costs.py) | Nguồn tính chi phí search/selected recipe và khuyến nghị validation |
| [src/experiment.py](src/experiment.py) | run.py, resume_interrupted.py: train/validation, checkpoint |
| [src/reporting.py](src/reporting.py) | experiment.py và delivery.finalize_run: xuất metric/hình, một writer Markdown/HTML |
| [src/eda.py](src/eda.py) | analyze_data.py/notebook 01 và data/experiment: split, RGB statistics, per-run EDA; không import torch/model |
| [src/notebook_views.py](src/notebook_views.py) | Chỉ notebook modeling: đọc bằng chứng thí nghiệm, hiển thị biểu đồ và bảng ngắn |
| [src/delivery.py](src/delivery.py) | finalize.py [--all]; IO/path/hash dùng chung cho experiment, EDA, infer và resume; gọi reporting.write_final_report |
| [src/__init__.py](src/__init__.py) | Khai báo package src |
| [scripts/analyze_data.py](scripts/analyze_data.py) | Tái đo EDA raw; `--data-dir` trỏ thư mục chứa `cifar-10-batches-py` |
| [scripts/run.py](scripts/run.py) | Điểm vào huấn luyện, dùng config/*.json |
| [scripts/infer.py](scripts/infer.py) | Nạp checkpoint và dự đoán một ảnh |
| [scripts/download_assets.py](scripts/download_assets.py) | README: CIFAR-10/weights; --chunked là fallback tải chia phần; --dataset-only bỏ weights |
| [scripts/resume_interrupted.py](scripts/resume_interrupted.py) | Phục hồi tại ranh giới kiến trúc, không phải resume giữa epoch |
| [scripts/execute_notebook.py](scripts/execute_notebook.py) | Thực thi notebook với interpreter hiện tại |
| [scripts/finalize.py](scripts/finalize.py) | README: finalize run; --all còn chạy test/notebook/inference theo model được validation chọn |
| [tests/test_contracts.py](tests/test_contracts.py) | Shape, freeze, split và một batch overfit |
| [tests/test_cached_prefix.py](tests/test_cached_prefix.py) | Đối chiếu logits/gradient cache và forward đầy đủ |
| [tests/test_tuning.py](tests/test_tuning.py) | LR groups, scheduler, gradient hai block, reset trial, không dùng test khi search, timing và smoke pipeline |
| [tests/test_eda.py](tests/test_eda.py) | Chặn import training, kiểm tra feature/duplicate/split, đối chiếu hash và thống kê run cũ |
| [tests/test_cleanup.py](tests/test_cleanup.py) | Graph tham chiếu production, CLI --help/import an toàn, split chung, CSV schema mới, đúng run/checkpoint và tải chia phần có checksum |
| [tests/test_notebook_review.py](tests/test_notebook_review.py) | Đối chiếu metric, ranh giới EDA và liên kết tài liệu; không cần torch |

Có 10 module `src/` và 7 CLI. Các file đang thực thi được nối bằng import/call thực hoặc lệnh CLI trong bảng trên. Snapshot trong `outputs/runs/` được manifest/hash tham chiếu, giữ riêng khỏi graph code hiện hành.

Luồng chính: `run.py → experiment → data/models/tuning/reporting`; `data → eda.locked_split`; `reporting → costs`; `notebook 01 → eda`; `notebook 02 → notebook_views → costs`; `finalize.py → delivery → reporting`; `infer.py → data/models`.

Test graph bắt module production bị bỏ rơi, public function không có tham chiếu tĩnh và file thiếu link README. Tham chiếu trong test hoặc snapshot không được tính làm caller production. Docstring giải thích vai trò; không thêm import giả để giữ file.

## Reproduce

Chạy từ `labs/lab02`. Để **xem kết quả**, không cần dataset/weights: mở hai notebook đã có output, hoặc chạy lại bằng interpreter có NumPy, pandas, matplotlib, Pillow, scikit-learn, IPython, nbformat, nbclient và ipykernel:

```powershell
python -X utf8 scripts/execute_notebook.py 01_lab02_eda.ipynb
python -X utf8 scripts/execute_notebook.py 02_lab02_modeling.ipynb
python -X utf8 -m pytest tests -q
```

Để **huấn luyện**, tạo môi trường từ requirements, rồi chạy trên GPU. Local chỉ chạy test/sanity nhỏ theo [rule Zero Waste](../../.agents/AGENTS.md):

```powershell
python -m pip install -r requirements.txt
python scripts/download_assets.py
python -m pytest tests -q
python scripts/run.py --config config/full.json --device cuda
python scripts/finalize.py --run outputs/runs/<run_id> --all
tensorboard --logdir outputs/runs
```

EDA đã đo nằm trong `outputs/eda/`, tách khỏi `outputs/runs/`. Tái tạo từ raw bằng:

```powershell
python scripts/analyze_data.py --data-dir "<thư mục chứa cifar-10-batches-py>"
```

Lệnh không tải model, không huấn luyện và không đổi dữ liệu. EDA và training dùng chung `outputs/eda/split_indices.npz`, có kiểm tra hash/seed; cấu hình EDA tại [config/eda.json](config/eda.json). Thống kê/feature chỉ đo trên train; test chỉ tham gia kiểm tra integrity. Split tái dựng đã được đối chiếu hash với run lịch sử.

Tên tài liệu theo template: `description_lab_02.md`, `walkthrough_lab_02.md`, `README.md`; hai notebook giữ tiền tố thứ tự `01_`/`02_`. `Lab02_CIFAR10.md` không tồn tại trong checkout đầu vào và không tạo lại tài liệu kỹ thuật song song.

Tải chia phần: `python scripts/download_assets.py --chunked`. Chỉ tải dữ liệu: thêm `--dataset-only`. Mọi CLI dùng cùng chính sách đường dẫn raw: `--data-dir` (khi CLI hỗ trợ) → `LAB02_DATA` → `labs/lab02/data`. Weights vẫn lưu riêng tại `data/pretrained` của lab.

`finalize.py --run <run>` kiểm chứng/xuất report; `--all` còn chờ run, chạy test và hai notebook cho đúng run đó, rồi inference checkpoint được validation chọn. Muốn xem thủ công, dùng `LAB02_RUN`; vị trí source dùng `LAB02_ROOT`.

Resume: `python scripts/resume_interrupted.py outputs/runs/<run_id> --device cuda`. Chỉ hỗ trợ ranh giới kiến trúc; từ chối kiến trúc đang có trial/checkpoint dở dang hoặc thiết bị khác run gốc, không ghi đè kết quả. Không resume giữa epoch/trial.

Python của run lịch sử: CPython 3.13.3; môi trường thực tế lưu trong `environment.json` của run. Ghim phiên bản không bảo đảm cài được trên mọi OS; chưa chạy lại full training trong đợt chỉnh trình bày này.

## Results Summary

| Fine-tune | Test accuracy | Test loss | Pipeline train (s) |
|---|---:|---:|---:|
| ResNet18 | 81,49% | 0,547387 | 1414,15 |
| DenseNet121 | 81,53% | 0,547715 | 741,18 |

DenseNet dùng cache prefix, ResNet không dùng: thời gian chỉ mô tả hai run thực tế, chưa xếp hạng tốc độ kiến trúc. Kết luận và kế hoạch LR/optimizer/unfreeze nằm trong [walkthrough](walkthrough_lab_02.md).

## Tuning LR, optimizer và độ sâu

Chạy từ `labs/lab02`, sau khi cài môi trường có CUDA:

```powershell
python scripts/run.py --config config/tuned.json --dry-run
python scripts/run.py --config config/tuned.json --device cuda
python scripts/run.py --config config/tuned.json --device cuda --seed 43
python scripts/run.py --config config/tuned.json --device cuda --seed 44
```

Mỗi seed có 6 head trials và 13 fine-tune trials **mỗi model**, 10 epoch/fine trial: đây là grid đầy đủ, cần dự trù thời gian GPU. `split_seed=42` giữ tập dữ liệu cố định khi đổi seed khởi tạo/shuffle. `LAB02_DATA` có thể trỏ tới raw CIFAR-10 đã tải. Không thay sang flower_photos trong lượt này.

- AdamW: backbone LR `{3e-6,1e-5,3e-5}`; SGD momentum 0,9: `{1e-4,3e-4,1e-3}`. Head LR = 10× backbone LR.
- ResNet mở `layer4` hoặc `layer3+layer4`; DenseNet mở `denseblock4+norm5` hoặc `denseblock3+transition3+denseblock4+norm5`; luôn học classifier. BatchNorm running statistics giữ cố định.
- Có control LR cố định; các candidate grid dùng cosine. Mọi candidate bắt đầu lại từ cùng selected head. Mở hai block buộc tắt cache prefix; tuned tắt cho cả hai.
- Chọn trong dải accuracy cách mức tốt nhất ≤0,1 điểm phần trăm, rồi loss cách mức thấp nhất trong dải ≤0,005, sau đó ưu tiên fit time nhỏ hơn. Đây là ngưỡng đánh đổi cấu hình, không phải kiểm định hai model tương đương thống kê. Test chỉ chạy sau khi khóa candidate.

`finetune_search_results.csv` lưu toàn bộ grid; `selection.json` ghi winner; `finetune/` chứa checkpoint/history được chọn, không train lại thêm lần nữa. TensorBoard lưu LR từng nhóm, optimizer/depth/scheduler, train/val loss/accuracy. `cost_comparison.csv` tách tổng search với selected recipe, không cộng lặp selected trial. Full wall time của quá trình chọn model, throughput và peak CUDA memory cũng được ghi.

Khi chất lượng gần nhau, xem thêm thời gian của selected recipe và test pass. Run cũ khác cache policy nên chưa xếp hạng tốc độ kiến trúc bằng train time. Xem [walkthrough](walkthrough_lab_02.md#15-loss-function-and-optimization-design) để biết giới hạn và kiểm thử.
