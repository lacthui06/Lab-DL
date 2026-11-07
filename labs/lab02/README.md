# Lab 02: CIFAR-10 Transfer Learning with PyTorch

## Bằng chứng thực nghiệm và cách đọc kết quả

Mở [walkthrough](walkthrough_lab_02.md), [notebook EDA](notebooks/01_lab02_eda.ipynb) hoặc [notebook Modeling](notebooks/02_lab02_modeling.ipynb) để đọc phương pháp và output đã lưu. Không cần huấn luyện hoặc tải checkpoint chỉ để xem output có sẵn trong notebook.

Run được chọn trong [outputs/selected_run.json](outputs/selected_run.json) là `20261030T052121Z_cifar10_scheduled64_b895bfc1`: CPU, ảnh 64×64, seed 42. Kết quả head-only và fine-tune nằm trong [comparison.csv](outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/comparison.csv). Benchmark inference Tesla T4 và benchmark training CPU batch 1 có evidence riêng dưới `outputs/published/`; không dùng chúng thay cho thời gian full training.

Chạy lại các cell hiển thị cần artifact EDA, run đã chọn và evidence benchmark tương ứng. Bộ benchmark published không thay thế toàn bộ run training. Khi chuyển máy, đồng bộ artifact cần đọc; `git clone` không bảo đảm có checkpoint, cache hoặc event files nằm ngoài Git.

Dự án phân loại 10 lớp CIFAR-10 bằng transfer learning với ResNet18 và DenseNet121 pretrained ImageNet. Pipeline bao gồm EDA, preprocessing, khảo sát đặc trưng, huấn luyện classifier, fine-tune một phần backbone, đánh giá và so sánh chất lượng/chi phí. Source được tách theo chức năng để notebook tập trung vào phương pháp và output thực nghiệm.

Hai notebook chính hiện hành chỉ đọc artifact khi `Run All`: notebook 01 trình bày dữ liệu; notebook 02 trình bày kiến trúc, training, evaluation và benchmark. Huấn luyện và đo benchmark dùng CLI riêng.

---

## 1. Cấu trúc thư mục dự án

```text
labs/lab02/
├── config/
│   ├── eda.json                       # Cấu hình khảo sát dữ liệu
│   ├── scheduled.json                 # Recipe 64×64, batch 128, head rồi fine-tune
│   ├── full.json                      # Profile 224×224
│   ├── tuned.json                     # Grid LR/optimizer/unfreeze, yêu cầu CUDA
│   ├── cpu_pilot.json                 # Kiểm tra tập nhỏ
│   ├── cpu_full.json                  # Profile CPU lịch sử
│   └── training_benchmark.json        # Chọn checkpoint và phép đo training batch 1
├── data/                              # CIFAR-10 và pretrained weights local
├── outputs/
│   ├── selected_run.json              # Registry chọn run tường minh
│   ├── eda/                           # Bảng/hình, split và preprocessing config
│   ├── runs/
│   │   └── <run_id>/
│   │       ├── config.json / environment_snapshot.json
│   │       ├── split_indices.npz / selected_indices.npz
│   │       ├── comparison.csv / cost_comparison.csv
│   │       ├── resnet18/               # Trials, head_selected và finetune
│   │       ├── densenet121/            # Checkpoint, history, metrics và predictions
│   │       ├── source_snapshot/        # Source/config tại thời điểm chạy
│   │       └── notebook_snapshots/     # Bản notebook theo hash
│   ├── review/<run_id>/               # Hình trình bày, giữ riêng run gốc
│   ├── published/
│   │   ├── single_image_benchmark/    # Evidence inference T4 batch 1
│   │   └── training_benchmark/        # Evidence training CPU batch 1
│   └── logs/                          # Snapshot môi trường dùng chung
├── src/
│   ├── inference.py                   # Load checkpoint và dự đoán một ảnh
│   ├── datasets/
│   │   ├── eda.py                     # Raw, integrity, split và feature EDA
│   │   └── data.py                    # Dataset, transform và DataLoader
│   ├── models/
│   │   └── models.py                  # Weights, classifier, freeze và cached tail
│   ├── training/
│   │   ├── experiment.py              # Train/validation, checkpoint và suite
│   │   ├── feature_cache.py           # Cache feature/prefix và loader tensor
│   │   └── tuning.py                  # Grid và selection theo validation
│   └── utils/
│       ├── constants.py               # Hằng số dùng chung
│       ├── costs.py                   # Chi phí đọc từ artifact
│       ├── reporting.py               # Metric, bảng và biểu đồ
│       ├── notebook_views.py          # Hiển thị evidence trong notebook
│       ├── delivery.py                # Artifact IO, hash, finalize và resume
│       ├── inference_benchmark.py     # Latency, RAM/VRAM inference
│       └── training_benchmark.py      # Training step batch 1
├── scripts/
│   ├── analyze_data.py                # Tạo EDA từ raw
│   ├── download_assets.py             # Tải CIFAR-10 và pretrained weights
│   ├── run.py                         # Huấn luyện theo config
│   ├── infer.py                       # Dự đoán từ checkpoint
│   ├── finalize.py                    # Kiểm chứng/xuất kết quả
│   ├── resume_interrupted.py          # Khôi phục suite hoặc fit một trial
│   ├── execute_notebook.py            # Thực thi notebook
│   ├── benchmark_inference.py         # Chuẩn bị/đóng gói/benchmark inference
│   └── benchmark_training.py          # Benchmark training batch 1
├── tests/                             # Data/model/training, cache, view và benchmark
├── notebooks/
│   ├── 01_lab02_eda.ipynb
│   ├── 02_lab02_modeling.ipynb
│   └── 02_lab02_modeling-kaggle.ipynb  # Bản Modeling có output Kaggle
├── description_lab_02.md              # Đề bài và phạm vi đã thống nhất
├── walkthrough_lab_02.md              # Phương pháp và kết quả có bằng chứng
├── requirements.txt                   # Pins của môi trường Windows lịch sử
└── README.md                          # Tổng quan và hướng dẫn sử dụng
```

Cây thư mục lược bớt `__init__.py`, cache và artifact chi tiết. Không phải mọi run đều có đủ checkpoint/events trong checkout. Snapshot dùng để truy vết; source hiện hành nằm trong `src/` và `scripts/`.

### Tham chiếu source và CLI

| Nhóm | Module / điểm vào | Vai trò |
| :--- | :--- | :--- |
| Package | [src/__init__.py](src/__init__.py), [datasets](src/datasets/), [models](src/models/), [training](src/training/), [utils](src/utils/) | Namespace, dependency được nạp theo chức năng |
| Dữ liệu | [eda.py](src/datasets/eda.py), [data.py](src/datasets/data.py) | EDA, split, transform và loader |
| Mô hình | [models.py](src/models/models.py) | Khởi tạo, classifier, freeze/unfreeze |
| Training | [experiment.py](src/training/experiment.py), [feature_cache.py](src/training/feature_cache.py), [tuning.py](src/training/tuning.py) | Train/evaluate, cache và selection |
| Báo cáo | [reporting.py](src/utils/reporting.py), [notebook_views.py](src/utils/notebook_views.py), [costs.py](src/utils/costs.py) | Metric, hình và chi phí từ artifact |
| Artifact | [delivery.py](src/utils/delivery.py), [constants.py](src/utils/constants.py) | IO, hash, snapshot và giá trị dùng chung |
| Inference | [inference.py](src/inference.py), [inference_benchmark.py](src/utils/inference_benchmark.py) | Load model, predict và đo inference |
| Training benchmark | [training_benchmark.py](src/utils/training_benchmark.py) | Forward/loss/backward/optimizer step |
| Chuẩn bị | [analyze_data.py](scripts/analyze_data.py), [download_assets.py](scripts/download_assets.py) | EDA và tải dữ liệu/weights |
| Train/infer | [run.py](scripts/run.py), [infer.py](scripts/infer.py) | CLI huấn luyện và dự đoán |
| Bàn giao | [finalize.py](scripts/finalize.py), [resume_interrupted.py](scripts/resume_interrupted.py), [execute_notebook.py](scripts/execute_notebook.py) | Kiểm chứng, khôi phục, notebook |
| Benchmark | [benchmark_inference.py](scripts/benchmark_inference.py), [benchmark_training.py](scripts/benchmark_training.py) | CLI đo độc lập với full training |

---

## 2. Pipeline và điểm nhấn triển khai

### 2.1. EDA → preprocessing → feature engineering → modeling

```text
CIFAR-10 RGB 32×32, uint8 [0,255]
    → Integrity: checksum, nhãn, pixel, duplicate và split
    → Stratified train/validation; giữ test gốc độc lập
    → EDA pixel/spatial, ảnh trung bình, feature thủ công và tương quan
    → Preprocessing theo ImageNet weights và image_size của run
         ├─ Head-only: backbone đóng băng → pooled features → classifier 10 lớp
         └─ Fine-tune: selected head → mở một phần backbone → 10 logits
    → Validation chọn checkpoint/candidate
    → Test sau lựa chọn → metric, confusion matrix, prediction và ROC/PR
```

- Run hiện hành có 45.000 train, 5.000 validation và 10.000 test; split seed 42. EDA/training dùng chung split đã khóa; run lưu indices để đối chiếu.
- Pixel statistics và handcrafted features chỉ tính trên train chưa augmentation; validation/test tham gia kiểm tra integrity theo phạm vi ghi trong notebook.
- Ảnh được chuyển sang float tensor CHW, scale về [0,1], normalize bằng mean/std của ImageNet weights. Profile 64×64 resize trực tiếp; profile 224×224 dùng transform của weights. Không fit preprocessing trên test.
- Handcrafted features phục vụ EDA, không nối vào classifier CNN. Head-only dùng pooled features từ backbone pretrained.
- Scheduled bật RandomCrop/HorizontalFlip cho loader ảnh train ở fine-tune. Head dùng cached features với preprocessing xác định; cache cố định không tạo augmentation mới mỗi epoch. Validation/test không augmentation.

### 2.2. Hai kiến trúc và hai giai đoạn học

| Mô hình | Cách kết nối | Classifier của recipe scheduled | Số tham số tổng |
| :--- | :--- | :--- | ---: |
| ResNet18 | Residual addition | Dropout(0,2) → Linear(512, 10) | 11.181.642 |
| DenseNet121 | Nối feature maps qua dense block | Dropout(0,2) → Linear(1024, 10) | 6.964.106 |

Cả hai dùng `IMAGENET1K_V1`; dropout phụ thuộc config. Head-only chỉ học classifier. Fine-tune khởi đầu từ selected head, mở các module theo `unfreeze`. BatchNorm running statistics giữ cố định; affine weights trong phần mở vẫn có thể học.

`scheduled_tail` mở `layer4.1` của ResNet18 và một tập dense layer/transition/norm của DenseNet121; danh sách chính xác nằm trong `trainable_blocks` ở [models.py](src/models/models.py). Không diễn giải recipe này thành mở toàn bộ block cuối. Fine-tune của run hiện hành có 4.725.770/2.921.034 tham số trainable tương ứng, dưới giới hạn 50% tổng tham số.

Mục 2 notebook Modeling đọc kiến trúc đầy đủ và bảng tensor tham số. Lý do chọn model nằm trong [description](description_lab_02.md). Cột `head_stage_trainable` chỉ mô tả stage head; phần mở fine-tune phải đọc metadata/comparison.

### 2.3. Optimizer và hyperparameter

Recipe trong [scheduled.json](config/scheduled.json):

1. **Head:** AdamW, LR=0,001, tối đa 12 epoch, batch 128, pooled feature cache.
2. **Fine-tune:** AdamW, backbone LR=0,0001, head LR=0,001, tối đa 20 epoch; full image forward cho cả hai model, không cache frozen prefix.
3. **Scheduler/stopping:** cosine với tỷ lệ LR tối thiểu 0,1; weight decay=0,0001; early stopping theo validation loss, min 10 epoch, patience 7, min_delta=0,0001.

Đây là một recipe/model, không phải grid search LR/batch. Số epoch thực chạy đọc từ history, không đồng nhất với ngân sách tối đa.

[tuned.json](config/tuned.json) là thí nghiệm mở rộng riêng: ảnh 224×224, 6 head trials và 13 fine-tune candidates mỗi model/seed, batch grid 64/128; fine-tune tối đa 30 epoch. Grid khảo sát AdamW/SGD momentum 0,9, LR backbone, scheduler và độ sâu mở một/hai block; head LR bằng 10 lần backbone LR. Candidate bắt đầu từ cùng selected head; test không tham gia search.

Selection của grid giữ candidate trong dải validation accuracy cách mức tốt nhất ≤0,1 điểm phần trăm, rồi loss cách mức thấp nhất trong dải ≤0,005, sau đó ưu tiên fit time thấp hơn. Đây là chính sách đánh đổi, không phải kiểm định tương đương thống kê. Chưa có kết quả full grid được dùng trong README để khẳng định cải thiện.

### 2.4. Artifact và tái lập

Mỗi run có ID riêng, config, môi trường, indices, log, source snapshot và manifest. `head_selected/` và `finetune/` giữ checkpoint được chọn, history và kết quả. Tuning còn có `finetune_search_results.csv`/`selection.json` khi áp dụng; không refit thêm chỉ để báo cáo selected checkpoint.

Registry chọn run tường minh, không tự lấy run mới nhất. Notebook lưu hình trình bày dưới `outputs/review/<run_id>/`; thực thi có thể tạo lại hình/snapshot nhưng không train hoặc benchmark. Run mới lưu `best.pt` cho inference và `last.pt` cho resume cuối epoch; checkpoint lịch sử có thể thiếu schema resume mới.

---

## 3. Hướng dẫn cài đặt và thực thi

Các lệnh bên dưới chạy từ **`labs/lab02/`**, trừ khi ghi rõ khác. Từ repository root, chạy `cd labs/lab02` trước. Thay `<run_id>`, `<model>` và `<...>` bằng giá trị thật; model CLI dùng `resnet18` hoặc `densenet121`.

### 3.1. Chuẩn bị môi trường local

```powershell
python -m pip install -r requirements.txt
python -c "import sys, torch, torchvision; print(sys.version); print(torch.__version__); print(torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

[requirements.txt](requirements.txt) lưu pins từ Windows lịch sử; chưa xác nhận toàn bộ pins cài được trên Linux/Kaggle. Run scheduled được chọn thực tế dùng Python 3.12.7, PyTorch 2.2.2, torchvision 0.17.2 trên macOS CPU; xem [environment.json](outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/environment.json). Requirements lịch sử không đại diện môi trường thực tế của mọi run hoặc lockfile đã kiểm chứng đa nền tảng.

Notebook cần kernel đúng interpreter, NumPy, pandas, matplotlib, Pillow, scikit-learn, IPython, nbformat, nbclient và ipykernel để đọc/hiển thị evidence. Training cần thêm PyTorch/torchvision và dependency tương ứng. Chọn local CPU/GPU hoặc Kaggle theo workload/ngân sách; chạy sanity/smoke trước full training. Riêng `tuned.json` yêu cầu CUDA.

### 3.2. Notebook 01 — EDA, preprocessing và feature engineering

Mở [01_lab02_eda.ipynb](notebooks/01_lab02_eda.ipynb). Roadmap đầu notebook ghi từng bước, module/hàm tạo artifact và hàm hiển thị.

- `Run All` đọc artifact đã lưu; không tái tính raw, không nạp model hoặc train.
- Output gồm integrity, phân bố lớp, ảnh mẫu, pixel/spatial statistics, preprocessing, handcrafted features và data handoff.
- Muốn tạo lại EDA, dùng CLI; `--data-dir` trỏ thư mục chứa `cifar-10-batches-py`:

```powershell
python scripts/analyze_data.py --data-dir "<thư mục chứa cifar-10-batches-py>"
```

EDA lưu dưới `outputs/eda/`, gồm split indices, metadata và preprocessing config. Cấu hình tại [eda.json](config/eda.json); lệnh không tải model hoặc huấn luyện.

### 3.3. Notebook 02 — Modeling và evaluation

Mở [02_lab02_modeling.ipynb](notebooks/02_lab02_modeling.ipynb); bản có output Kaggle là [02_lab02_modeling-kaggle.ipynb](notebooks/02_lab02_modeling-kaggle.ipynb).

- Thứ tự: data contract → kiến trúc/freeze → learning curves → evaluation → selection → chi phí → save/load evidence → inference batch 1 → training batch 1 → bàn giao.
- `Run All` chỉ đọc run/evidence đã chọn. Lab02 không có công tắc `MODE='train'` như Lab01; muốn train dùng `scripts/run.py`.
- `LAB02_ROOT` chọn đường dẫn tuyệt đối tới lab; `LAB02_RUN` chọn run thay registry. Nếu thiếu artifact, cần đồng bộ đúng run.
- Mục 8 đọc inference evidence chọn trong biến `BENCHMARK`, kiểm checksum và training run; mục 8b đọc training batch 1 độc lập.
- Sau khi sửa source trong lúc kernel đang chạy, restart kernel rồi chạy setup lại để tránh dùng module đã cache.

```powershell
python -X utf8 scripts/execute_notebook.py 01_lab02_eda.ipynb
python -X utf8 scripts/execute_notebook.py 02_lab02_modeling.ipynb
```

### 3.4. Chạy từ CLI

```powershell
# Tải CIFAR-10 và weights; --dataset-only để chỉ tải dữ liệu
python scripts/download_assets.py

# Kiểm tra ngân sách/config, không huấn luyện
python scripts/run.py --config config/scheduled.json --dry-run

# Train scheduled trên CPU; dùng --device cuda khi có CUDA thực
python scripts/run.py --config config/scheduled.json --device cpu

# Sau khi suite hoàn tất, kiểm chứng/xuất kết quả đúng run
python scripts/finalize.py --run "outputs/runs/<run_id>"

# Dự đoán ảnh test hoặc ảnh riêng từ checkpoint
python scripts/infer.py --checkpoint "outputs/runs/<run_id>/<model>/finetune/best.pt" --sample-index 0
python scripts/infer.py --checkpoint "outputs/runs/<run_id>/<model>/finetune/best.pt" --image "<image.png>"
```

`download_assets.py --chunked` hỗ trợ tải chia phần có checksum. Raw-data path của CLI có hỗ trợ `--data-dir`: tham số tường minh → `LAB02_DATA` → `labs/lab02/data`; weights lưu riêng ở `data/pretrained`.

Lưu run ID in đầu log. Registry chỉ đổi khi suite hoàn tất; sau finalize, chạy lại notebook để cập nhật output. Không truyền `--run` thì finalize dùng registry. `--all` còn chờ run, chạy tests/notebooks và kiểm tra inference checkpoint được validation chọn; không cần dùng chỉ để xem kết quả có sẵn.

Grid tuning mở rộng khi đã có CUDA và ngân sách:

```powershell
python scripts/run.py --config config/tuned.json --dry-run
python scripts/run.py --config config/tuned.json --device cuda
python scripts/run.py --config config/tuned.json --device cuda --seed 43
python scripts/run.py --config config/tuned.json --device cuda --seed 44
```

`split_seed=42` giữ split khi đổi seed khởi tạo/shuffle. Hai lệnh seed 43/44 là hướng dẫn chạy bổ sung, không phải bằng chứng đã chạy nhiều seed.

### 3.5. Local, Kaggle và TensorBoard

Khi chuyển môi trường, mang source/config cùng data/checkpoint/artifact cần dùng; cập nhật `LAB02_ROOT`, `LAB02_DATA`, `LAB02_RUN` theo vị trí thật. Notebook Modeling Kaggle hiện tại là view. Evidence inference T4 không chứng minh toàn bộ training/pins đã được kiểm chứng trên Kaggle.

Training ghi events dưới `<model>/<trial>/tensorboard/`; `input_preview/` giữ ảnh đầu vào/model graph. Mở terminal thứ hai trong cùng môi trường:

```powershell
tensorboard --logdir outputs/runs --port 6006
```

Mở dashboard tại `http://localhost:6006` sau khi server khởi động. Có thể xem trong lúc train; để tập trung một run dùng `--logdir outputs/runs/<run_id>`. Máy khác cần server của máy đó hoặc đồng bộ events. CSV/PNG và ảnh TensorBoard lịch sử không thay events gốc, không chứng nhận server hiện đang chạy.

### 3.6. Khôi phục khi gián đoạn

```powershell
# Tiếp tục suite tại ranh giới kiến trúc
python scripts/resume_interrupted.py "outputs/runs/<run_id>" --device cpu

# Tiếp tục fit một trial mới từ last.pt cuối epoch
python scripts/resume_interrupted.py "outputs/runs/<run_id>" --trial resnet18/head_trial_00 --device cpu
```

Dùng cùng thiết bị của run gốc; đổi sang `cuda` khi run gốc dùng CUDA. Resume suite từ chối kiến trúc có trial/checkpoint dở dang. `--trial` chỉ tiếp tục fit, không tự chọn trial, test hoặc hoàn tất suite/report. Epoch dở chạy lại; chưa hỗ trợ giữa batch hoặc checkpoint cũ thiếu schema.

Giữ nguyên source/config, môi trường, indices và cache; CLI đối chiếu hash/metadata. `last.pt` giữ model/optimizer/scheduler, history, early-stopping và RNG/loader state. Timing resume cộng các đoạn fit, loại thời gian chờ; peak memory chỉ thuộc process hiện tại, không dùng xếp hạng benchmark mới.

---

## 4. Visualization và sản phẩm đầu ra

- **EDA:** integrity, phân bố lớp, ảnh RGB, pixel/spatial statistics, ảnh trung bình, preprocessing, handcrafted features, tương quan và tương đồng lớp.
- **Kiến trúc:** toàn bộ model, bảng tensor/số tham số và freeze/unfreeze theo metadata stage.
- **Training:** loss/accuracy train-validation, epoch thực chạy, LR và TensorBoard khi có events.
- **Evaluation:** loss, Accuracy, Macro Precision/Recall/F1; heatmap theo lớp Precision, Recall/TPR, F1, FPR; confusion matrix và ảnh đúng/sai.
- **ROC/PR:** fine-tune One-vs-Rest từ artifact; legend ROC dùng AUC, PR dùng Average Precision (AP).
- **Selection/chi phí:** trial, validation selection, tổng search/selected recipe, train/test timing và throughput theo phạm vi ghi nhận.
- **Save/load và benchmark:** roundtrip evidence đã lưu; inference latency/RSS/VRAM và training batch 1 độc lập.

Recall là TPR. ROC dùng FPR–TPR qua nhiều ngưỡng, khác bảng metric argmax. Macro curve score là trung bình score theo lớp từ CSV; không chạy inference mới để tạo đường cong. Không đánh đồng AP, ROC-AUC và accuracy. Hình phụ vẫn có thể nằm trong artifact/review dù không hiển thị hết trong notebook.

---

## 5. Kiểm thử và kiểm chứng

```powershell
python -X utf8 -m pytest tests -q
```

| Phạm vi | Bộ kiểm thử |
| :--- | :--- |
| Shape, freeze, split, one-batch overfit | [test_contracts.py](tests/test_contracts.py) |
| Logits/gradient cached tail và full forward | [test_cached_prefix.py](tests/test_cached_prefix.py) |
| Cache, thứ tự mẫu/nhãn/indices và shuffle | [test_feature_cache.py](tests/test_feature_cache.py) |
| LR groups, scheduler, reset candidate và test isolation | [test_tuning.py](tests/test_tuning.py) |
| Integrity, feature, split và ranh giới EDA | [test_eda.py](tests/test_eda.py) |
| Graph source, CLI, checksum và finalize/resume | [test_cleanup.py](tests/test_cleanup.py) |
| Notebook evidence, metric và liên kết | [test_notebook_review.py](tests/test_notebook_review.py) |
| Import nhẹ, lazy exports và chi phí không nạp model | [test_import_boundaries.py](tests/test_import_boundaries.py) |
| Inference timing và RAM/CUDA sampler | [test_inference_benchmark.py](tests/test_inference_benchmark.py) |
| Training batch 1, checkpoint và view | [test_training_benchmark.py](tests/test_training_benchmark.py) |

Lệnh chạy suite hiện hành; kết quả cần đọc từ lần thực thi tương ứng, không dùng số test lịch sử làm bảo đảm. Synthetic/smoke tests kiểm tra hợp đồng phần mềm, không phải kết quả chất lượng CIFAR-10. Chỉnh README không yêu cầu train lại hoặc ghi lại output notebook.

---

## 6. Kết quả thực nghiệm hiện hành

### 6.1. Head-only và fine-tune

Nguồn: [comparison.csv của run đã chọn](outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/comparison.csv), CPU, 64×64, seed 42, 10.000 ảnh test. Không trộn với profile CPU lịch sử hoặc grid 224×224.

| Model | Stage | Test Accuracy | Macro F1 | Test loss | Tham số trainable |
| :--- | :--- | ---: | ---: | ---: | ---: |
| ResNet18 | Head-only | 76,35% | 0,763008 | 0,696569 | 5.130 |
| ResNet18 | Fine-tune | 84,14% | 0,840593 | 0,484747 | 4.725.770 |
| DenseNet121 | Head-only | 78,79% | 0,787870 | 0,604426 | 10.250 |
| DenseNet121 | Fine-tune | 91,80% | 0,917978 | 0,301038 | 2.921.034 |

Fine-tune tăng test accuracy **7,79 điểm phần trăm** với ResNet18 và **13,01 điểm phần trăm** với DenseNet121 so với head-only tương ứng. DenseNet121 fine-tune cao hơn ResNet18 **7,66 điểm phần trăm** trong run này; một seed chưa chứng minh ưu thế kiến trúc nói chung. Validation quyết định selection, không tiếp tục chọn cấu hình theo test.

Thời gian fit fine-tune ghi trong cùng CSV khoảng 6.540,54 s/20.323,32 s tương ứng. Đây là `train_seconds` của stage, không phải tổng pipeline head/extract/search. Cả hai fine-tune dùng full image forward, head dùng pooled feature cache. Chi phí đầy đủ nằm trong [cost_comparison.csv](outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/cost_comparison.csv) và [walkthrough](walkthrough_lab_02.md).

### 6.2. Phạm vi tuning và giới hạn

Scheduled đã hoàn tất một recipe/model; không gọi đây là kết quả tối ưu từ grid LR/optimizer/unfreeze. [full.json](config/full.json), [cpu_full.json](config/cpu_full.json), [cpu_pilot.json](config/cpu_pilot.json) và [tuned.json](config/tuned.json) có phạm vi khác nhau; đọc config của đúng run khi so sánh.

Grid tuned có source/tests nhưng chưa có kết quả full grid dùng ở bảng trên. Image size, augmentation, cache, độ sâu mở và ngân sách đều ảnh hưởng chất lượng/timing; không suy ra lợi ích riêng một yếu tố từ các run khác nhiều cấu hình.

---

## 7. Tài liệu, lịch sử và lưu trữ

- [Description](description_lab_02.md): đề gốc và phạm vi đã thống nhất.
- [Walkthrough](walkthrough_lab_02.md): nguồn báo cáo duy nhất, gồm đặc tả, phương pháp và kết quả có evidence.
- [Registry](outputs/selected_run.json): run dùng trình bày; giữ riêng run cũ thay vì ghi đè.
- `outputs/runs/`: artifact gốc; `outputs/review/`: hình trình bày; `outputs/published/`: benchmark evidence có checksum/provenance.

Chuyển máy cần đồng bộ run và raw data/cache nếu train hoặc resume. Không commit weights nặng, dữ liệu hoặc log riêng tư vào bộ báo cáo. README chưa cung cấp đường dẫn tải gói đầy đủ thay cho artifact local; bundle benchmark không phải bản sao toàn bộ thí nghiệm.

Nhật ký sửa code/debug giữ local theo quy định, ngoài báo cáo. Notebook snapshots giữ nội dung/output theo hash; liên kết tương đối có thể vẫn theo vị trí notebook gốc, không mặc định chạy trực tiếp từ thư mục snapshot.

---

## 8. Kiến trúc và hiệu năng training theo mẫu

Mục 2 notebook Modeling đọc kiến trúc/bảng tham số của run. Mục **8b — Training batch 1** đọc [training_performance.csv](outputs/published/training_benchmark/scheduled64_cpu_batch1/training_performance.csv), [biểu đồ tài nguyên](outputs/published/training_benchmark/scheduled64_cpu_batch1/training_resources.png) và [phương pháp đo](outputs/published/training_benchmark/scheduled64_cpu_batch1/methodology.json).

[Config benchmark](config/training_benchmark.json) chọn tường minh run/checkpoint, CPU, ảnh 64×64, batch 1, warmup 3, 10 lượt đo, 2 thread. Đây là vài training step trên một mẫu lặp lại, không full training và không quy đổi từ batch 128.

```powershell
# Từ labs/lab02, sau khi có dữ liệu và checkpoint
python scripts/benchmark_training.py
```

CLI không tải data/weights, không sửa checkpoint và từ chối output đã tồn tại. Muốn đo lại, chọn output mới trong config; muốn CUDA phải chọn `device=cuda` và chạy CUDA thực.

Timing gồm forward + CrossEntropyLoss + backward + optimizer.step khi batch đã trên thiết bị; loại loading, IO, preprocessing, transfer, validation và ghi artifact. Optimizer mới dùng LR đầu recipe, không resume optimizer cuối training; không scheduler/augmentation. Freeze/dropout/BatchNorm theo checkpoint.

RAM là sampled process RSS mỗi 1 ms; timing/memory đo riêng. RSS gồm runtime, không phải bộ nhớ riêng của ảnh; CPU VRAM không khả dụng. Memory đo bước đầu gồm cấp phát optimizer; timing dùng warmed steps. `Run All` chỉ kiểm manifest/nguồn checkpoint rồi đọc evidence.

## 9. Benchmark inference một ảnh với model đã nạp sẵn

Mục **8 — Loaded-model inference** notebook Modeling đọc [evidence T4](outputs/published/single_image_benchmark/20261102T184034Z_kaggle/) và [provenance](outputs/published/single_image_benchmark/20261102T184034Z_kaggle/provenance.json). Không cần notebook benchmark 03 riêng. Khi chuyển máy mang cả thư mục evidence; sai checksum/training run sẽ bị từ chối.

Hai process riêng cho ResNet18/DenseNet121, batch 1, ảnh 64×64, FP32, eval và inference_mode. Không tính nạp checkpoint/khởi tạo model; request gồm đọc/giải mã ảnh, preprocessing theo metadata, transfer, forward, softmax/kết quả về CPU và tạo JSON. Không bật AMP/TF32/compile.

- Lượt đầu sau load ghi riêng; 20 warmup rồi 100 lượt đo mean/median/p95, ms/ảnh.
- Stage diagnostics đo 20 lượt riêng có CUDA synchronization; không cộng stage time thay tổng latency.
- RAM/VRAM đo riêng 20 lượt. RSS là cả process, sampler 1 ms có thể bỏ lỡ peak ngắn; allocated/reserved thuộc PyTorch, không phải toàn bộ CUDA context/VRAM process.
- Đọc lặp ảnh có thể hưởng OS cache; tài nguyên quan sát không chứng minh cấu hình phần cứng tối thiểu.

Chuẩn bị từ terminal Lab02 khi có hai checkpoint fine-tune:

```powershell
python scripts/benchmark_inference.py prepare-image --data-dir data --index 0 --output ../../.agents/single-image-benchmark/input.png
python scripts/benchmark_inference.py package --run outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1 --image ../../.agents/single-image-benchmark/input.png --output ../../.agents/single-image-benchmark/lab02-single-image-benchmark.zip
```

Ảnh được xuất ở process riêng; process benchmark không nạp toàn bộ test batch. ZIP gồm source, ảnh/metadata, hai checkpoint đã chọn bằng validation và manifest. Đưa lên Kaggle Dataset private, giải nén vào thư mục mới dưới `/kaggle/working/`, đối chiếu manifest; không chạy lại training hoặc cài lại toàn bộ pins Windows.

Từ thư mục `lab02` của gói, mỗi lệnh là process độc lập:

```powershell
python scripts/benchmark_inference.py run --checkpoint checkpoints/resnet18.pt --image input.png --device cuda --output-dir /kaggle/working/single-image-results
python scripts/benchmark_inference.py run --checkpoint checkpoints/densenet121.pt --image input.png --device cuda --output-dir /kaggle/working/single-image-results
```

CLI yêu cầu CUDA thực khi chọn `cuda`; `--device cpu` tạo phép đo CPU riêng. Mỗi model có benchmark ID, config/environment, samples, summary/prediction và manifest. Không đổi registry/metric training; thu hồi kết quả và notebook đã chạy trước khi kết thúc Kaggle. Inference T4 và training CPU batch 1 khác thiết bị/phạm vi, không so sánh trực tiếp như cùng workload.
