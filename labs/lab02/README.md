# Lab 02 — Transfer learning trên CIFAR-10

## Cấu trúc module hiện hành

```text
src/
├── datasets/
│   ├── classification.py       # CIFAR10, transforms, DataLoader
│   └── eda.py                  # Split, integrity, thống kê và hình EDA
├── models/
│   ├── __init__.py             # Public API: src.models
│   └── backbones.py            # ResNet/DenseNet, freeze và feature extraction
├── training/
│   ├── trainer.py              # Loop chung, fit, predict; recipe lịch sử
│   ├── tuning.py               # Optimizer, scheduler và selection utilities
│   ├── standard_suite.py       # Bài chuẩn mới: 2 model, 3 trial/model, LR chung
│   └── optuna_dense.py         # Thí nghiệm DenseNet shared/separate trước đó
└── utils/
    ├── artifacts.py            # IO, checksum, resolve path và handoff lịch sử
    ├── plots.py                # Metrics, confusion, prediction và biểu đồ
    ├── costs.py                # Bảng chi phí tính toán
    ├── notebook_views.py       # Hiển thị notebook lịch sử
    └── merged_results.py       # Gộp các nhóm GPU lịch sử
```

Các file `.py` tên cũ ngay dưới `src/` là alias tương thích vài dòng, không giữ bản sao logic. Ví dụ `src.standard_suite` và `src.training.standard_suite` trỏ cùng một module; notebook/CLI cũ vẫn import được. Module `src.models` nay là package nhưng public API giữ nguyên. Script trong `scripts/` vẫn là entry point, không di chuyển hoặc thay tên.

Notebook chuẩn để chạy mới: [04_lab02_standard_kaggle.ipynb](notebooks/04_lab02_standard_kaggle.ipynb), dùng [standard.json](config/standard.json). Notebook 03 là thí nghiệm shared/separate trước đó; notebook 01/02 vẫn phục vụ EDA và kết quả GPU lịch sử. Xem [hướng dẫn outputs](outputs/README.md) để phân biệt các nhóm dữ liệu.

Source ZIP đã upload là snapshot độc lập: sửa local không làm đổi phiên Kaggle đang chạy. Bộ ZIP đóng gói sau refactor có cả package mới và alias cũ. Không thay source giữa chừng một trial; dùng ZIP mới cho phiên mới. Bootstrap vẫn nhận các file alias `src/standard_suite.py`, `src/optuna_dense.py`, `src/notebook_views.py` để không gãy phép tìm source.

**Đã hoàn tất 18 fine-tune trial (9/model), seed 42**, từ Kaggle Versions 5, 6, 8. So sánh ResNet18 và DenseNet121 pretrained ImageNet; head-only và fine-tune. Kết quả và nhận xét nằm trực tiếp trong hai notebook, không cần train lại để xem.

| Model | Test accuracy | Test loss | Macro F1 | Fine epoch thực chạy / tối đa | Fit trial được chọn |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 93,71% | 0,305124 | 0,937085 | 10/30 | 28,73 phút |
| DenseNet121 | 94,61% | 0,204466 | 0,945915 | 10/30 | 57,32 phút |

Winner được chọn bằng validation trên cả 9 candidate/model: AdamW, LR backbone 3e-5/head 3e-4, cosine, mở hai block cuối. DenseNet checkpoint epoch 6, ResNet epoch 10; cả hai dừng theo validation-loss patience. Có dấu hiệu overfit; một seed chưa chứng minh ưu thế thống kê. Fit trial được chọn không phải tổng chi phí search. Profile GPU đều full-image forward, tách khỏi run CPU 64×64 lịch sử.

## Đọc bài và kết quả

- [description_lab_02.md](description_lab_02.md): đề bài gốc và phạm vi lựa chọn.
- [walkthrough_lab_02.md](walkthrough_lab_02.md): tài liệu sống Spec → Changelog → Report, giữ lịch sử riêng và kết luận GPU hiện hành ở Section 3.
- [01_lab02_eda.ipynb](notebooks/01_lab02_eda.ipynb): raw → integrity → preprocess → handcrafted features; không modeling. Preview EDA 64×64 được ghi rõ; GPU dùng 224×224.
- [02_lab02_modeling.ipynb](notebooks/02_lab02_modeling.ipynb): bảng, PNG, head/fine curves, actual epochs/stop reason, metric theo lớp, lỗi và TensorBoard nhúng inline.
- [outputs/selected_run.json](outputs/selected_run.json): run GPU mặc định, các đường dẫn nhóm tường minh và registry CPU lịch sử.
- [tuning_submissions.json](outputs/review/tuning_submissions.json): ba nhóm hoàn tất; nhóm 3/4 không chạy theo phạm vi cuối. Kế hoạch 26 trial đã được thay thế, không phải phần còn chờ chạy.

Chạy từ `labs/lab02` với môi trường báo cáo đã cài:

```powershell
python -X utf8 scripts/execute_notebook.py 01_lab02_eda.ipynb
python -X utf8 scripts/execute_notebook.py 02_lab02_modeling.ipynb
python -X utf8 -m pytest tests -q
```

Notebook mặc định gộp các run trong registry, chọn theo validation accuracy tolerance 0,001, loss tolerance 0,005, rồi fit time. Không chọn bằng test. Khi yêu cầu một run riêng, notebook chỉ đọc run đó và nhận xét được tính từ chính history/metric của run:

```powershell
$env:LAB02_RUN = "outputs/kaggle/group_00/extracted/runs/20261027T042305Z_cifar10_gpu_tuning_group_00_c444e2bc"
python -X utf8 scripts/execute_notebook.py 02_lab02_modeling.ipynb
Remove-Item Env:LAB02_RUN
```

`LAB02_ROOT` chọn thư mục lab; `LAB02_RUN` chọn run tuyệt đối hoặc tương đối với lab. `finalize.py --run <run> --all` truyền đúng run cho notebook và inference. Thiếu artifact phải báo lỗi, không tự lấy run đầu tiên trong thư mục. Hai notebook có output đã lưu vẫn xem được khi không có dataset/checkpoint.

## Môi trường và tái lập

- [requirements.txt](requirements.txt): lock **Windows/CPU**, Python yêu cầu `>=3.12, !=3.14.1`, interpreter đã kiểm chứng 3.13.3; Torch 2.14.1, torchvision 0.29.1.
- [requirements-gpu.txt](requirements-gpu.txt): dependency trực tiếp lấy từ **Kaggle Version 8**, Linux Python 3.13.15, Torch 2.11.0+cu128/torchvision 0.26.0+cu128. Không dùng lock CPU để tuyên bố tái lập môi trường GPU. Lower-bound Requires-Python đầy đủ của profile GPU chưa được audit; 3.13.15 là bản đã chạy, không phải ngưỡng tối thiểu.
- [environment_snapshot.json](outputs/logs/environment_snapshot.json): metadata local và môi trường GPU gốc. Full `requirements-freeze.txt` và `environment.json` được giữ trong từng run. Chưa cài thử một môi trường GPU mới trong đợt sửa báo cáo này.

Để huấn luyện lại đúng **phạm vi 18 trial** trên GPU trong môi trường mới:

```powershell
python -m pip install -r requirements-gpu.txt
python scripts/download_assets.py
python -m pytest tests -q
python scripts/run.py --config config/completed.json --dry-run
python scripts/run.py --config config/completed.json --device cuda
```

[completed.json](config/completed.json) chứa đúng 9 candidate/model đã chạy. Head search 3 LR × batch 64/128, 10 epoch/trial; fine min 10/max 30, patience 7, min_delta 1e-4. Các nhóm cũ lặp lại head search; chạy config hợp nhất chỉ cần một head search/model. Split_seed=42 giữ nguyên tập khi đổi seed huấn luyện. Local chỉ test/sanity, không full train CPU theo quy tắc repo.

Inference từ checkpoint tốt nhất (checkpoint phải có local, không đi cùng Git):

```powershell
python scripts/infer.py --checkpoint outputs/kaggle/group_02/extracted/runs/20261027T140405Z_cifar10_gpu_tuning_group_02_ec77fa72/densenet121/finetune/best.pt --image "<anh_RGB>"
```

Mọi CLI dùng raw path: `--data-dir` nếu hỗ trợ → `LAB02_DATA` → `labs/lab02/data`. `python scripts/download_assets.py --chunked` là fallback có checksum; `--dataset-only` bỏ tải weights. EDA tái đo bằng `python scripts/analyze_data.py --data-dir <data>`; dùng chung split khóa seed 42. TensorBoard: `tensorboard --logdir outputs/kaggle`; events có local, được loại khỏi Git.

`finalize.py --run <run>` kiểm chứng báo cáo; `--all` còn chạy test/notebook/inference, cần raw/checkpoint/events. `resume_interrupted.py <run> --device cuda` chỉ hỗ trợ ranh giới kiến trúc, từ chối trial dở dang hoặc đổi thiết bị; không resume giữa epoch.

## Bản đồ code và CLI

| File | Caller và mục đích |
|---|---|
| [src/data.py](src/data.py) | experiment/infer: transforms và DataLoader, dùng eda.locked_split |
| [src/models/backbones.py](src/models/backbones.py) | experiment/infer: ImageNet weights, classifier, freeze và cached tail |
| [src/tuning.py](src/tuning.py) | experiment/merged_results: grid, optimizer, scheduler, validation-only selection |
| [src/costs.py](src/costs.py) | reporting/notebook_views/merged_results: chi phí và khuyến nghị validation |
| [src/experiment.py](src/experiment.py) | run/resume: train/validation, checkpoint và test sau lựa chọn |
| [src/reporting.py](src/reporting.py) | experiment/delivery/notebook_views: metric và hình |
| [src/eda.py](src/eda.py) | analyze_data/notebook 01/data: split, raw EDA, không import model |
| [src/notebook_views.py](src/notebook_views.py) | modeling/merged_results: view run, quality_cost/efficiency dùng khi LAB02_RUN chọn run riêng |
| [src/merged_results.py](src/merged_results.py) | modeling: đọc registry, gộp candidate, chọn winner động và nhận xét theo đúng run |
| [src/delivery.py](src/delivery.py) | finalize/experiment/EDA/infer/resume: IO, đường dẫn, hash và kiểm chứng |
| [src/__init__.py](src/__init__.py) | Khai báo package |
| [scripts/analyze_data.py](scripts/analyze_data.py) | Tái đo EDA từ raw |
| [scripts/run.py](scripts/run.py) | CLI huấn luyện/dry-run |
| [scripts/infer.py](scripts/infer.py) | Dự đoán từ checkpoint |
| [scripts/download_assets.py](scripts/download_assets.py) | Tải CIFAR-10/weights có checksum |
| [scripts/resume_interrupted.py](scripts/resume_interrupted.py) | Tiếp tục tại ranh giới kiến trúc |
| [scripts/execute_notebook.py](scripts/execute_notebook.py) | Thực thi notebook bằng interpreter hiện tại |
| [scripts/finalize.py](scripts/finalize.py) | Kiểm chứng và hoàn tất run |

Có 11 file trong `src/` gồm `__init__.py`, 7 CLI. Snapshot huấn luyện được bảo toàn trong ZIP local, không được tính làm caller production. Các view dùng thật trong báo cáo mặc định hoặc chế độ một run, không thêm import giả để giữ code.

## Config lịch sử và phạm vi

[full.json](config/full.json) là control GPU; [cpu_pilot.json](config/cpu_pilot.json) là sanity nhỏ; [cpu_full.json](config/cpu_full.json) mô tả run CPU lịch sử; [eda.json](config/eda.json) là profile EDA. [tuned.json](config/tuned.json) là kế hoạch gốc 26 trial, không phải phạm vi hoàn tất. [local_remaining.json](config/local_remaining.json) thuộc run local đã dừng. [tuned_16.json](config/tuned_16.json) và [tuned_16_10epochs.json](config/tuned_16_10epochs.json) là đề xuất đã rút lại, không áp dụng vào kết quả; giữ để truy vết, không dùng làm lệnh tái lập mặc định.

## Kiểm thử

- [test_contracts.py](tests/test_contracts.py): shape, freeze, split, one-batch overfit.
- [test_cached_prefix.py](tests/test_cached_prefix.py): logits/gradient tương đương cache.
- [test_tuning.py](tests/test_tuning.py): optimizer/LR/depth, reset initialization, validation-only search, stopping/time và smoke pipeline.
- [test_eda.py](tests/test_eda.py): EDA không import training, split và integrity.
- [test_cleanup.py](tests/test_cleanup.py): module/caller graph, CLI, shared paths và finalize.
- [test_notebook_review.py](tests/test_notebook_review.py): metric/counts, notebook output và link; test snapshot CPU lịch sử dùng registry riêng, không phụ thuộc run hiện hành.
- [test_merged_results.py](tests/test_merged_results.py): LAB02_RUN, đường dẫn registry chính xác, winner không cố định và phạm vi completed config.
- [conftest.py](tests/conftest.py): Matplotlib Agg cho Windows/CI, không yêu cầu Tcl/Tk.
# DenseNet Optuna update

Kaggle entry point: `notebooks/03_lab02_optuna_kaggle.ipynb`. Upload the generated `outputs/kaggle_upload/lab02_optuna_source.zip` as a private Kaggle Dataset (no previous model output is required). Enable GPU and Internet. The notebook checks inputs, runs small tests, writes under `/kaggle/working`, executes the 10 trials and packages full/partial outputs. No credentials are embedded. Download the output ZIP and the executed notebook after saving the version. This entry point has been syntax-checked locally; end-to-end GPU execution is pending.

The new fine-tuning scope compares shared versus separate backbone/head learning rates: 5 trials each, maximum 20 epochs, AdamW, cosine, full denseblock4 + norm5 + classifier. BatchNorm running statistics stay fixed. Selection, checkpoint saving and early stopping use validation loss with min_delta=0.001, patience=4, minimum 6 epochs. No augmentation, dropout or pruning is added.

From the repository root:

```powershell
python -m pip install -r labs/lab02/requirements-optuna.txt
python labs/lab02/scripts/run_optuna.py --dry-run
python labs/lab02/scripts/run_optuna.py --device cuda
```

Install the base local/GPU requirements first. The default config trains a fresh head for 10 epochs at LR 0.001 from ImageNet pretrained weights, selecting by validation loss; provide CIFAR10 under data/ or LAB02_DATA. head_checkpoint=null enables this full pipeline. Execution fails before training when these inputs are missing. New results are not yet measured; the older results below must not be presented as Optuna evidence. Completed new runs register outputs/selected_optuna.json. Trials save SQLite studies, histories, checkpoints, TensorBoard and local logs; both winners are locked before test. Shared/separate evaluation includes OvR curves, count confusion matrices, prediction grids and full-test reload verification.
# Canonical two-model notebook

Use `notebooks/04_lab02_standard_kaggle.ipynb` for the latest agreed complete experiment: ResNet18 and DenseNet121 head/fine-tune (four conditions). Config is `config/standard.json`: cached-feature head training 10 epochs; 3 Optuna trials per model, shared LR only, fixed weight decay 1e-4, max 10 fine epochs, cosine, loss selection and patience=3/min_delta=0.001/min_epochs=4. Per-epoch live plots and per-trial tables are persisted. No augmentation/dropout/pruning. Source archive: `outputs/kaggle_upload/lab02_standard_source.zip`. Completed stages can be reused through `outputs/selected_standard.json`; interrupted trials do not support exact resume. Notebook 03 is the earlier DenseNet shared/separate experiment; do not restart an active run to switch recipes. Under-five-hour runtime is a target, not a measured guarantee.
