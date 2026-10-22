# Lab 01: FashionMNIST Image Classification with PyTorch

Dự án phân loại 10 lớp trang phục FashionMNIST bằng PyTorch, triển khai đầy đủ từ EDA, preprocessing và feature engineering đến huấn luyện, đánh giá, so sánh kiến trúc, optimizer và hyperparameter. Mã nguồn được phân tách theo chức năng để notebook tập trung trình bày phương pháp và output thực nghiệm.

Hai notebook hiện hành dùng chung cấu hình và artifact: notebook 01 chuẩn bị dữ liệu; notebook 02 so sánh ba cấu hình MLP và thực nghiệm optimizer. Có thể chạy trên local CPU/CUDA hoặc Kaggle. Mỗi lần thực thi tạo run riêng, lưu log và kết quả để truy vết.

---

## 1. Cấu trúc thư mục dự án

```text
labs/lab01/
├── config/
│   ├── workflow.json                  # Cấu hình hiện hành, kiến trúc và danh mục run được chọn
│   ├── config.yaml                    # Cấu hình CLI cũ; giữ tương thích lịch sử
│   ├── augmentations.yaml             # Cấu hình augmentation của pipeline cũ
│   └── experiments.yaml               # Cấu hình optimizer cũ; không điều khiển notebook mới
├── data/                              # FashionMNIST thô, tải bằng torchvision khi cần
├── outputs/
│   ├── runs/                          # Artifact hiện hành, mỗi lần chạy có thư mục riêng
│   │   ├── <preparation-run>/
│   │   │   ├── eda/
│   │   │   │   ├── tables/            # Structural/integrity/distribution/pixel/spatial/feature tables
│   │   │   │   ├── figures/           # Các hình khảo sát dữ liệu
│   │   │   │   ├── split_indices.npz  # Chỉ số train/validation cố định
│   │   │   │   ├── split_metadata.json
│   │   │   │   └── preprocessing_config.json
│   │   │   ├── prepared.npz           # Pixel, feature, label và original indices theo split
│   │   │   ├── feature_scaler.json    # Mean/scale feature chỉ fit trên train
│   │   │   ├── train_features.csv     # Một dòng feature cho mỗi ảnh train
│   │   │   ├── val_features.csv
│   │   │   ├── test_features.csv
│   │   │   └── feature_*.png          # Hình học, phân bố và tương quan feature
│   │   ├── <model-run>/
│   │   │   ├── best.pt                # Weights tại best validation epoch
│   │   │   ├── sanity.json            # Kết quả one-batch sanity trước huấn luyện
│   │   │   ├── history.json / history.csv
│   │   │   ├── learning_curves.png
│   │   │   ├── test_predictions.npz   # Xác suất, nhãn và chỉ số test
│   │   │   ├── classification_report.json / .csv
│   │   │   └── metrics.json           # Accuracy, Macro F1/Precision/Recall, loss, chi phí
│   │   ├── <comparison-run>/          # Bảng và hình so sánh ba model
│   │   ├── <optimizer-fixed-run>/     # Fixed LR, history/checkpoint từng optimizer
│   │   ├── <optimizer-tuning-run>/
│   │   │   ├── studies.sqlite3        # Study/trial/intermediate values của Optuna
│   │   │   ├── trials/                # Config, partial history, checkpoint từng trial
│   │   │   ├── *_trials.csv / all_trials.csv
│   │   │   ├── best_configs.json / .csv
│   │   │   └── tuning_progress.png
│   │   └── <verification-run>/        # Log kiểm thử, kiểm tra dữ liệu và notebook đã thực thi
│   ├── eda/                           # Output lịch sử; không tự coi là preparation run mới
│   ├── checkpoints/                   # Checkpoint lịch sử
│   ├── logs/                          # Log lịch sử
│   └── predictions/                   # Hình đánh giá lịch sử
├── src/
│   ├── workflow.py                    # Load cấu hình, chọn train/view và điều phối các giai đoạn
│   ├── inference.py                   # Eval/infer theo metadata và prepared data của run
│   ├── datasets/
│   │   ├── base.py                    # Stratified split, indices và dataset helpers
│   │   ├── eda.py                     # Core EDA, integrity, phân bố và thống kê dữ liệu
│   │   ├── features.py                # Định nghĩa chung cho 10 feature, bbox và cường độ/cạnh
│   │   ├── preparation.py             # Chạy EDA, fit scaler và tạo prepared.npz
│   │   └── classification.py          # Transforms/DataLoader của luồng lịch sử
│   ├── models/
│   │   └── mlp.py                     # Class FashionMLP dùng chung cho ba cấu hình
│   ├── training/
│   │   ├── trainer.py                 # Train/evaluate loop, optimizer factory, sanity overfit
│   │   ├── comparison.py              # Ba cấu hình model, best checkpoint và test evaluation
│   │   ├── tuning.py                  # Fixed LR, Optuna, refit và tổng hợp optimizer
│   │   └── experiments.py             # API thí nghiệm lịch sử, giữ để tương thích
│   └── utils/
│       ├── helpers.py                 # Seed, device, environment, đường dẫn và checkpoint
│       ├── runs.py                    # Run ID, stdout/stderr log, trạng thái và SHA256 manifest
│       ├── metrics.py                 # Metric theo lớp cho API lịch sử
│       ├── plots.py                   # Hình EDA và các hàm visualization dùng chung
│       └── comparison_plots.py        # Model comparison, confusion matrix, ROC/PR OvR
├── scripts/
│   ├── train.py                       # Train một cấu hình hiện hành
│   ├── eval.py                        # Đánh giá lại best checkpoint theo run
│   ├── infer.py                       # Dự đoán một ảnh test theo run
│   └── run_optimizer_experiments.py   # Fixed LR -> tuning -> refit -> tổng hợp
├── tests/
│   ├── test_datasets.py               # Integrity, split, bbox và preprocessing
│   ├── test_models.py                 # Kiến trúc, số tham số và tensor shape
│   ├── test_training.py               # Training/checkpoint/sanity và API lịch sử
│   ├── test_bootstrap.py              # Fresh runtime và thiếu Optuna
│   ├── test_features_comparison.py    # Feature, ba model, artifact và visualization
│   ├── test_tuning.py                 # Study budgets, pruning, refit và test isolation
│   └── test_workflow.py               # View không train, feature chung và inference contract
├── notebooks/
│   ├── 01_eda_preprocessing_features.ipynb
│   ├── 02_modeling_comparison.ipynb
│   └── lab01.ipynb                     # Trang chỉ dẫn đến notebook hiện hành và bản lịch sử
├── archive/
│   └── pre_cleanup_20261022/           # Bản tài liệu/notebook/config/CLI trước hợp nhất
├── description_lab_01.md              # Đề bài gốc và phạm vi đã thống nhất
├── walkthrough_lab_01.md              # Phương pháp và kết quả có bằng chứng
├── internal_changes.md                # Nhật ký nội bộ, không đưa vào báo cáo PDF
├── requirements.txt                   # Dependencies cho lab
└── README.md                          # Hướng dẫn tổng quan và sử dụng
```

Cây thư mục mô tả các nhóm file chính; những file `__init__.py`, metadata và artifact chi tiết được lược bớt để dễ đọc. Mỗi run hiện hành còn có `config.json`, `environment_snapshot.json`, `console.log`, `status.json` và `manifest.json`. Không phải mọi loại run đều có checkpoint hoặc prediction.

---

## 2. Pipeline và điểm nhấn triển khai

### 2.1. EDA -> preprocessing -> feature engineering -> modeling

```text
FashionMNIST gốc [0,255]
    -> EDA và integrity checks
    -> Stratified train/validation split, giữ test gốc
    -> Rescaling /255 thành [0,1]
         ├─ Standardization pixel bằng mean/std train -> 784 pixel
         └─ Trích xuất 10 feature -> StandardScaler fit train -> 10 feature
                ├─ SmallMLP và MainMLP nhận 784 pixel
                └─ MainMLP_FE nhận 784 pixel + 10 feature
```

- **Rescaling** đưa pixel về [0,1]. **Standardization** tính `(x - mean_train) / std_train`. Feature hình dáng/cường độ được đo trước standardization pixel.
- Train/validation gồm 54.000/6.000 ảnh; test gốc có 10.000 ảnh. Mean/std của preparation run đã chọn xấp xỉ 0.2862/0.3532; giá trị thực đọc từ artifact thay vì nhập lại vào code.
- Feature scaler chỉ fit train rồi áp dụng validation/test. Bảng tổng hợp theo lớp phục vụ EDA; bảng theo từng ảnh mới là dữ liệu dùng cho modeling.
- 10 feature gồm mean, std, foreground ratio, bbox width/height, aspect ratio, centroid x/y, symmetry và edge density. Foreground dùng ngưỡng 0.1 trên ảnh [0,1]. Công thức chi tiết nằm trong walkthrough và source.
- Lượt comparison hiện tại không augmentation để giữ pixel và handcrafted feature tương ứng cùng một ảnh. Cấu hình augmentation cũ vẫn được giữ cho lịch sử, không tự bật trong pipeline mới.
- Bảng EDA lịch sử có định nghĩa edge density khác. Không đổi cách diễn giải artifact cũ; EDA mới dùng helper đo cường độ/cạnh chung với modeling.

### 2.2. Ba cấu hình model

| Cấu hình | Đầu vào | Kiến trúc | Mục đích |
| :--- | :--- | :--- | :--- |
| SmallMLP | 784 pixel | 784 -> 128 -> 10 | Baseline ít tham số |
| MainMLP | 784 pixel | 784 -> 256 -> 128 -> 10 | So tác động tăng dung lượng |
| MainMLP_FE | 784 pixel + 10 feature | 794 -> 256 -> 128 -> 10 | So tác động feature engineering |

Cả ba dùng class `FashionMLP` với Flatten, Linear và ReLU. Comparison dùng chung split/seed, Adam, LR=0.001, 10 epoch, batch=128 và CosineAnnealingLR. Trước train có one-batch overfit; sau đó reset seed/model/loader để sanity không làm thay đổi điểm khởi đầu của lượt train chính.

### 2.3. Optimizer và hyperparameter

Thực nghiệm riêng cố định MainMLP pixel:

1. **Fixed LR:** SGD momentum=0.9, RMSprop và Adam cùng LR=0.001, weight_decay=0. RMSprop không dùng momentum trong comparison này; đó là lựa chọn cấu hình thí nghiệm.
2. **Search:** ba study Optuna độc lập, 10 trial/optimizer, tối đa 10 epoch/trial; tune LR, weight_decay và thêm momentum cho SGD. MedianPruner dừng trial ít triển vọng.
3. **Selection/refit:** chọn best bằng validation accuracy, refit từ đầu và đánh giá test. Không sử dụng test trong objective; không mặc định best study tốt hơn cấu hình fixed.

TPESampler được cấu hình 10 startup trials. Với ngân sách 10 trial/optimizer đã chạy, study còn ở pha lấy mẫu khởi động, chưa kiểm chứng pha TPE thích nghi. Ngân sách trial/epoch bằng nhau không đồng nghĩa thời gian bằng nhau vì pruning.

### 2.4. Artifact và tái lập

Run ID gồm thời gian UTC, tên giai đoạn và mã định danh. Mỗi run giữ config/môi trường, stdout/stderr, trạng thái và SHA256 manifest. Trial pruned vẫn giữ partial history và checkpoint nếu có; không bị coi là lỗi phần mềm. History model được ghi sau mỗi epoch, checkpoint được chọn theo validation.

`config/workflow.json` chỉ rõ run được dùng để trình bày; không tự chọn run mới nhất. Archive giữ lịch sử source/tài liệu; `outputs/runs/` giữ bằng chứng thực nghiệm. Việc dọn tài liệu không sửa CSV, prediction hoặc weights của run cũ.

---

## 3. Hướng dẫn cài đặt và thực thi

Các lệnh bên dưới chạy từ **repository root**, thư mục có `labs/`.

### 3.1. Chuẩn bị môi trường local

```powershell
python -m pip install -r labs/lab01/requirements.txt
python -c "import sys, torch; print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Notebook cần môi trường Jupyter và Python kernel phù hợp, chẳng hạn extension Jupyter của VS Code. Chọn đúng interpreter đã cài dependencies. Không cần GPU để train: CUDA được chọn khi có, CPU được dùng khi không có. Phiên bản thực tế của từng run nằm trong environment snapshot; requirements hiện dùng các khoảng phiên bản, không phải lockfile môi trường tuyệt đối.

### 3.2. Notebook 01 — EDA, preprocessing và feature engineering

Mở [01_eda_preprocessing_features.ipynb](notebooks/01_eda_preprocessing_features.ipynb). Roadmap ở đầu notebook ghi các bước, module và artifact.

- `MODE = 'view'`: đọc preparation run đã đăng ký, hiển thị bảng/hình có sẵn.
- `MODE = 'train'`: chạy lại pipeline chuẩn bị dữ liệu và tạo preparation run mới. Tên chế độ dùng thống nhất với notebook modeling; notebook 01 không huấn luyện neural network.
- Sau khi tạo preparation run muốn sử dụng, cập nhật `runs.preparation` trong `config/workflow.json` đến thư mục đó. Không trỏ trực tiếp tới CSV tổng hợp theo lớp.
- Giữ cả run directory, gồm split/scaler/config/NPZ, khi chuyển máy.

### 3.3. Notebook 02 — Modeling và optimizer

Mở [02_modeling_comparison.ipynb](notebooks/02_modeling_comparison.ipynb).

- Đặt `MODE = 'view'` để xem các run đã chọn mà không train lại.
- Đặt `MODE = 'train'` để tạo lượt chạy mới; thực hiện các cell theo thứ tự roadmap.
- Các cell kết quả dùng chung cho hai chế độ, mỗi hình xuất hiện ở vị trí tương ứng; không có khối đọc lại lặp cuối notebook.
- Các đường dẫn trong `runs.models`, `runs.comparison`, `runs.fixed`, `runs.tuning`, `runs.final` và `runs.optimizer_results` là danh mục kết quả được chọn. Chạy mới không tự ghi đè danh mục lịch sử này.
- Khi sửa source trong lúc kernel đang chạy, restart kernel rồi chạy setup lại để tránh dùng module Python đã cache.

### 3.4. Chạy từ CLI

```powershell
# Train một cấu hình; lấy tham số và preparation run từ workflow.json
python labs/lab01/scripts/train.py --model MainMLP
python labs/lab01/scripts/train.py --model MainMLP_FE

# Đánh giá hoặc infer từ thư mục run chứa config.json và best.pt
python labs/lab01/scripts/eval.py --run "labs/lab01/outputs/runs/<model-run>"
python labs/lab01/scripts/infer.py --run "labs/lab01/outputs/runs/<model-run>" --sample-idx 0

# Ghi đè vị trí prepared.npz khi chuyển máy
python labs/lab01/scripts/eval.py --run "<model-run>" --data "<preparation-run>/prepared.npz"

# Chạy fixed LR -> tuning -> refit -> tổng hợp; 10 trial cho MỖI optimizer
python labs/lab01/scripts/run_optimizer_experiments.py "<preparation-run>/prepared.npz" --epochs 10 --trials 10 --threads 2
```

Thay các phần `<...>` bằng đường dẫn thật. Giao diện CLI hiện hành dùng run directory; tùy chọn `--checkpoint`/YAML và `--sample_idx` của bản cũ không còn là giao diện chính. Script cũ được giữ trong archive để truy vết.

### 3.5. Local và Kaggle

Trên Kaggle cần có cả source Lab01 lẫn artifact cần dùng; notebook không tự mount source từ repository. Thiết lập working directory để cell setup tìm được `src/models/mlp.py`. Source/config đọc từ thư mục source; output mặc định ghi dưới `/kaggle/working/outputs/` khi runtime có `/kaggle/working`.

Có thể đặt `LAB01_WORK_DIR` trước khi import `src` để chọn thư mục làm việc khác. `MODE='view'` cần các run đã đăng ký tồn tại tại vị trí tương ứng; nếu artifact nằm chỗ khác, cập nhật đường dẫn trong workflow.json (có hỗ trợ đường dẫn tuyệt đối). Local đã kiểm chứng; chưa xác nhận một lượt Kaggle end-to-end mới.

---

## 4. Visualization và sản phẩm đầu ra

- **EDA:** ảnh mẫu, phân phối lớp, thống kê pixel, ảnh trung bình/phương sai, tương đồng giữa lớp.
- **Feature engineering:** bbox/trọng tâm, phân bố 10 feature theo lớp và tương quan feature trên train.
- **Mỗi model:** learning curves, checkpoint best, prediction probabilities và classification report.
- **So sánh ba model:** validation curves, Accuracy/Macro F1, chi phí/số tham số, confusion matrices ghi số ảnh từng ô (kể cả 0), màu chuẩn hóa theo lớp thật trong [0,1], và prediction grids chung ảnh.
- **ROC/PR:** một ảnh hai panel OvR cho mỗi model, mỗi panel có 10 đường theo lớp; lưu ROC-AUC/AP và tọa độ CSV. Không có F1 heatmap hay biểu đồ cột TPR/FPR.
- **Optimizer:** fixed-LR validation curves, tiến trình best score theo trial, LR-score, fixed-vs-tuned và bộ đánh giá refit cuối.

Không phải mọi hình đều tự hiển thị trong notebook; hình phụ và CSV vẫn được lưu. PR báo Average Precision (AP); không đánh đồng AP, ROC-AUC và accuracy.

---

## 5. Kiểm thử và kiểm chứng sau hợp nhất

```powershell
$env:MPLBACKEND = "Agg"
python -m pytest labs/lab01/tests -q -k "not experiment_runner_signatures"
```

Kết quả đã xác minh sau đợt rà soát confusion matrix, roadmap và cấu hình:

```text
47 passed, 1 deselected
```

Test `experiment_runner_signatures` lịch sử chạy nhiều lượt training nên không thuộc bộ regression ngắn trên. Test suite gồm split/integrity, bbox và feature, input 784/794, forward/backward, one-batch overfit, checkpoint, logging khi lỗi/pruning, Optuna integration, artifact export, số đếm/tỷ lệ confusion matrix, chế độ view không train, phát hiện comparison lệch model run và kiến trúc tùy chỉnh qua cấu hình.

Hai notebook đã thực thi toàn bộ ở MODE=view. Đánh giá lại checkpoint MainMLP giữ nguyên Accuracy/Macro F1. Đối chiếu feature của 256 ảnh với artifact trước dọn đạt tolerance 2e-6; sai lệch tối đa khoảng 4.77e-7. Các fixture synthetic chỉ phục vụ kiểm thử, không phải benchmark FashionMNIST.

Log kiểm chứng: [regression](outputs/runs/20261022T145701Z_audit_final_checks_2707f5d0/console.log), [checkpoint/feature validation](outputs/runs/20261022T142210Z_cleanup_final_validation_04387db6/validation.json), [notebook view execution](outputs/runs/20261022T145250Z_audit_notebook_execution_6fcbd376/).

---

## 6. Kết quả thực nghiệm hiện hành

### 6.1. Ba cấu hình model

Bảng dưới đọc từ `metrics.json` của đúng các run được đăng ký trong workflow.json. Các giá trị là kết quả test; không trộn best validation của một checkpoint với test của checkpoint khác.

| Model | Test Accuracy | Macro F1 | Số tham số | Nguồn artifact |
| :--- | ---: | ---: | ---: | :--- |
| SmallMLP | 88.52% | 0.884948 | 101,770 | [metrics](outputs/runs/20261022T133355Z_SmallMLP_cdad89a5/metrics.json) |
| MainMLP | 89.41% | 0.893813 | 235,146 | [metrics](outputs/runs/20261022T133406Z_MainMLP_8f90f1a2/metrics.json) |
| MainMLP_FE | 89.60% | 0.895752 | 237,706 | [metrics](outputs/runs/20261022T133425Z_MainMLP_FE_d8323126/metrics.json) |

Small/Main là comparison về dung lượng; Main/Main_FE là comparison về bổ sung feature. Với một seed, chênh lệch nhỏ chưa đủ chứng minh ý nghĩa thống kê. Kết luận chi tiết và hình liên quan nằm trong walkthrough.

### 6.2. Fixed LR và hyperparameter search

| Optimizer | Best validation: fixed LR | Best validation: study | Trial COMPLETE / PRUNED |
| :--- | ---: | ---: | :---: |
| SGD | 85.0833% | 90.8000% | 5 / 5 |
| RMSprop | 90.7333% | 90.5667% | 6 / 4 |
| Adam | 90.9333% | 90.8500% | 5 / 5 |

Nguồn: [fixed-LR summary](outputs/runs/20261022T130020Z_optimizer_fixed_lr_dbe789e0/summary.csv) và [study best configs](outputs/runs/20261022T130101Z_optimizer_tuning_d8986ca8/best_configs.csv).

Tổng 30 trial: 16 COMPLETE, 14 PRUNED. SGD cải thiện rõ khi tìm LR/momentum phù hợp; RMSprop và Adam trong ngân sách này chưa vượt fixed LR. Adam fixed vẫn có validation accuracy cao hơn best của ba study, nên không gọi tuned là cấu hình tốt nhất toàn bộ thí nghiệm.

Bản refit **best trong study** dùng Adam, LR=0.0015930522616241021 và weight_decay=0.0003470266988650413; test Accuracy **89.14%**, Macro F1 **0.891219**, loss **0.303679**. Nguồn: [refit metrics](outputs/runs/20261022T130609Z_MainMLP_0f179a92/metrics.json). Đây là đánh giá sau lựa chọn bằng validation, không phải mục tiêu để tiếp tục chọn trial.

---

## 7. Tài liệu, lịch sử và lưu trữ

- [Description](description_lab_01.md): đề bài gốc, phạm vi và các quyết định hiện hành.
- [Walkthrough](walkthrough_lab_01.md): giải thích phương pháp, triển khai và kết quả có artifact; nguồn để phát triển báo cáo lab.
- [Internal changes](internal_changes.md): nhật ký chỉnh sửa/kiểm tra nội bộ, không đưa vào báo cáo PDF.
- [Archive](archive/README.md): bản tài liệu, notebook/config và CLI trước hợp nhất; không phải entry point hiện hành.

Không xóa run bất lợi hoặc run lỗi để làm đẹp kết quả. `outputs/runs/` được gitignore vì chứa dữ liệu và weights; `git pull` không tải các artifact này. Khi chuyển máy/Kaggle phải đồng bộ run directory riêng và cập nhật đường dẫn trong workflow.json.

Chưa tạo PDF. Báo cáo xuất sau này có mục tiêu >=25 trang, dựa trên phương pháp và output thật; không đưa nhật ký chỉnh sửa hoặc phân tích lỗi nội bộ vào bản nộp.

Notebook modeling còn hiển thị loss/accuracy train-validation và history của refit sau search. Phần cuối Save / Load the Model thực hiện lưu/tải thật, giữ model trong biến LOADED_MODEL và lưu bằng chứng so sánh xác suất vào run riêng; cell này tạo artifact cả ở MODE=view, không train lại.

Sau search, notebook hiển thị loss/accuracy train-validation của cả ba optimizer từ best completed trial riêng, kèm best parameters và CSV history. Model được load lại ở cuối được đánh giá toàn bộ test, xuất metric/report/probabilities và lưới 16 ảnh đúng/sai có nhãn thật, dự đoán và xác suất.
