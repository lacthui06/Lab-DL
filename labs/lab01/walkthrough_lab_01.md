# Walkthrough — Lab01: FashionMNIST

Tài liệu hiện hành hợp nhất phương pháp và kết quả có artifact; nhật ký dọn/chỉnh sửa ở internal_changes.md. Bản trước hợp nhất được truy vết qua lịch sử Git (commit e0fd041). Không sử dụng các kết quả lịch sử K-fold/75-run sweep chưa xác minh làm bằng chứng cho pipeline hiện hành.

## 1. Đặc tả và thiết kế

Mục tiêu: thực hành PyTorch, so sánh dung lượng hai MLP, tác động 10 handcrafted features và optimizer/hyperparameter trên cùng split. Tiêu chí đánh giá: Accuracy, Macro F1, Macro Precision/Recall, loss và kết quả theo lớp; chọn checkpoint/config bằng validation, không dùng test để chỉnh tham số.

### 1. Quyết định đã thống nhất và mục tiêu thực nghiệm

Mục tiêu là so sánh dung lượng MLP và tác động của handcrafted features trên cùng FashionMNIST. SmallMLP nhận 784 pixel, có hidden layer 128. MainMLP nhận 784 pixel, có hidden layers 256 và 128. MainMLP_FE nhận 794 giá trị, giữ hidden layers 256 và 128. Các output đều có 10 logits. Class FashionMLP được dùng chung, không sao chép ba implementation.

SmallMLP so với MainMLP trả lời câu hỏi tăng dung lượng có cải thiện kết quả không. MainMLP so với MainMLP_FE trả lời câu hỏi việc cung cấp mô tả hình học/cường độ có hữu ích không. Không mặc định FE phải tăng accuracy; kết luận chỉ được viết sau lượt chạy thật. Mean/std pixel và feature scaler phải fit train, không sử dụng test để điều chỉnh cấu hình.

Notebook 01 chuẩn bị dữ liệu và minh họa phương pháp; notebook 02 điều phối huấn luyện, đánh giá và so sánh. Notebook lab01.ipynb cũ là lịch sử. Mã tính toán tái sử dụng đặt ở src; notebook không chứa bản sao vòng lặp train.

### 2. Phân biệt rescaling và standardization

Rescaling chia giá trị uint8 cho 255 để đưa ảnh về [0,1]. Feature được đo trên ảnh này. Standardization pixel dùng (x - mean_train)/std_train; đây là phép torchvision Normalize thực hiện. Feature có scaler riêng với mean và std theo từng cột. Không dùng ngưỡng foreground 0.1 trực tiếp trên ảnh đã standardize.

Đầu vào pixel là vector 784 giá trị chuẩn hóa. Đầu vào FE là vector này nối thêm 10 feature chuẩn hóa. Comparison đầu tiên không augmentation để tránh feature của ảnh gốc không khớp ảnh đã biến đổi. Nếu thêm augmentation sau này, phải trích xuất feature từ chính ảnh đã augment.

### 3. Định nghĩa 10 feature và lý do lựa chọn

- Mean và std của 784 pixel: mô tả cường độ tổng thể và tương phản.
- Foreground ratio: tỷ lệ pixel lớn hơn ngưỡng cố định 0.1.
- Bounding-box width và height: kích thước hộp bao foreground, chia 28.
- Aspect ratio: width/height; ảnh không foreground nhận 0.
- Centroid x và y: trung bình tọa độ có trọng số cường độ, chia 27; tổng cường độ bằng 0 nhận 0.
- Symmetry: 1 trừ sai khác tuyệt đối trung bình giữa ảnh và ảnh lật ngang. Ảnh rỗng có symmetry bằng 1, không được diễn giải thành quần áo đối xứng.
- Edge density: tỷ lệ cặp pixel lân cận theo hai trục có chênh lệch tuyệt đối lớn hơn 0.1. Đây là gradient adjacency đơn giản, không phải Canny.

Các feature chọn từ đặc điểm ảnh grayscale nhỏ: mức sáng, diện tích, dáng và đường biên. Chưa chọn đặc trưng màu vì ảnh có một kênh; chưa dùng HOG/LBP/PCA để giữ thí nghiệm dễ giải thích. Tương quan mạnh giữa width, height và aspect ratio không tự chứng minh chúng vô ích. Ngưỡng 0.1 là cấu hình cố định của phiên bản này, không phải ngưỡng tối ưu đã được chứng minh.

### 4. Triển khai và hợp đồng artifact

- src/datasets/features.py: trích xuất hữu hạn N x 10 từ N x 28 x 28 trong [0,1], từ chối input sai miền/shape.
- src/datasets/preparation.py: chạy lại 9 bước EDA hiện có trong thư mục run riêng, giữ split, đo thống kê train, trích xuất feature mỗi ảnh và chuẩn bị NPZ.
- src/utils/runs.py: thư mục UTC + tên + UUID; config, environment snapshot, console UTF-8, status và SHA256 manifest. Exception vẫn để lại log; không ghi đè run trước.
- src/training/comparison.py: sanity overfit, reset seed/model/loader, Adam + CosineAnnealingLR, history JSON/CSV sau mỗi epoch, best checkpoint theo validation accuracy, probabilities/labels/index test, metric và classification report.
- src/utils/comparison_plots.py: đọc lại artifact, kiểm tra cùng seed/hash dữ liệu/config, tạo bảng so sánh và hình; các tọa độ PR/ROC được lưu CSV.

Run dữ liệu lưu prepared.npz với pixel/feature/labels/original indices cho từng split; train_features.csv, val_features.csv và test_features.csv là bảng theo từng ảnh. Scaler JSON chứa tên feature, mean, scale và fit_split. Bảng tổng hợp theo lớp từ EDA không được dùng thay bảng train.

### 5. Kết quả chuẩn bị dữ liệu đã xác minh

Run thực: `outputs/runs/20261022T122157Z_preparation_dcaece7b/`.

Train có 54.000 ảnh; validation 6.000; test 10.000. Pixel tương ứng là (54000,784), (6000,784), (10000,784). Feature tương ứng là (54000,10), (6000,10), (10000,10). EDA ghi nhận không có ảnh trùng nội bộ và không trùng train-pool với test theo phép kiểm tra hiện có. Mean/std train xấp xỉ 0.2862/0.3532.

Các hình bổ sung:

![Bounding box và trọng tâm](outputs/runs/20261022T122157Z_preparation_dcaece7b/feature_geometry.png)

Hình được kiểm tra trực tiếp: bounding box đỏ bao vùng foreground, điểm cam đánh dấu trọng tâm cường độ. Quần dài có hộp hẹp và cao; sandal/sneaker có hộp thấp và rộng ở các ảnh minh họa. Đây là ví dụ trực quan, chưa chứng minh khả năng phân loại trên toàn bộ dữ liệu.

![Phân bố feature theo lớp](outputs/runs/20261022T122157Z_preparation_dcaece7b/feature_distributions.png)

![Tương quan feature trên train](outputs/runs/20261022T122157Z_preparation_dcaece7b/feature_correlations.png)

Trong boxplot của preparation run lịch sử đang được dẫn, vị trí 1–10 tương ứng lần lượt class index 0–9; không có lớp mang nhãn 10. Mã sinh các run mới đã ghi đúng tick label 0–9, giữ nguyên thứ tự và giá trị feature. Hai hình trên là bằng chứng khảo sát, không được suy ra feature importance từ tương quan. CSV đi kèm cho phép tái kiểm tra giá trị thay vì chỉ đọc ảnh.

### 6. Thiết kế notebook modeling và bộ hình so sánh

Các cell lần lượt: setup; chọn artifact dữ liệu tường minh; cấu hình; sanity và train riêng từng model; tổng hợp từ best checkpoint; hiển thị hình đã lưu. Khi đã có artifact có thể dựng hình lại mà không train lại. Optimizer fixed-LR/Optuna là các cell riêng trên MainMLP pixel. MODE=view đọc artifact đang chọn; MODE=train chạy các giai đoạn khi thực thi cell tương ứng. Thiếu Optuna phải báo lỗi thay vì dựng trial giả.

Bộ output gồm learning curves từng model, validation curves chung, Accuracy/Macro F1, chi phí train/số tham số, confusion matrices và prediction grids chung 16 ảnh. Mỗi model có một ảnh gồm hai panel ROC và Precision–Recall one-vs-rest; mỗi panel chứa 10 đường theo lớp. F1 theo lớp được giữ trong CSV. Không xuất F1 heatmap, ROC/PR micro-average hay biểu đồ cột TPR/FPR trong bộ so sánh hiện hành. PR ghi Average Precision (AP), không gọi AP là diện tích tích phân hình thang.

Confusion matrix ghi số ảnh dự đoán ở mọi ô, kể cả số 0; màu và colorbar dùng tỷ lệ chuẩn hóa theo hàng với cùng miền [0,1]. Trục dọc là lớp thật, trục ngang là lớp dự đoán. Counts CSV lưu số nguyên và normalized CSV của bộ ba model lưu tỷ lệ để đối chiếu. Hình được dựng từ test_predictions.npz của từng run, không suy ra số liệu từ màu ảnh.


## 2. Thực nghiệm và bằng chứng dữ liệu

### 1.9. EDA and Feature Engineering Results (Measured — 9 Canonical Steps)

Historical EDA evidence (before the current two-notebook workflow): all numbers below were produced by executing `run_eda_and_fe_pipeline()` in `src/datasets/eda.py` from `notebooks/lab01.ipynb` (Step 2). The 7 CSV tables live in `outputs/eda/tables/*.csv`, the 7 diagnostic figures live in `outputs/eda/figures/*.png`, and the locked split/normalization artifacts live in `outputs/eda/`.

#### 1.9.1. Structural and Integrity Findings (Steps 1, 2 & 3)

| Check | Result | Evidence |
|:---|:---|:---|
| Image shape / dtype | `(28, 28)` `uint8`, 1 channel | `tables/structural_overview.csv` |
| Pixel range | `[0, 255]` in both splits, no out-of-range value | `tables/integrity_report.csv` |
| Class counts | train 5,400/class, val 600/class, test 1,000/class | `tables/class_distribution.csv`, `figures/class_distribution.png` |
| Labels in `[0, 9]` | 0 violations | `tables/integrity_report.csv` |
| NaN / Inf | 0 | `tables/integrity_report.csv` |
| Constant / All-black images | 0 | `tables/integrity_report.csv` |
| Exact duplicates within train | 43 pairs (0.072%) | `tables/integrity_report.csv` |
| Exact duplicates within test | 2 pairs (0.020%) | `tables/integrity_report.csv` |
| Train ↔ test exact overlap | 16 pairs (0.027%) | `tables/integrity_report.csv` |
| Split index disjointness and coverage | 0 violations (54,000 train / 6,000 val) | `split_metadata.json` |

#### 1.9.2. Visual Inspection & Pixel Intensity Findings (Steps 4 & 5)

| Quantity | Value | Note |
|:---|---:|:---|
| Global mean (train 54,000 samples, scaled `[0,1]`) | `0.286209` | pooled over all pixels (`tables/pixel_statistics.csv`) |
| Global std, `ddof=0` (train, scaled) | `0.353160` | **locked in `preprocessing_config.json` for `Normalize`** |
| Mean of per-image std | `0.320673` | diagnostic only (`preprocessing_config.json`) |
| Zero-pixel fraction | `0.501887` | about half of all pixels are background (`figures/pixel_histogram.png`) |

Visual samples and per-class mean silhouettes are saved to `figures/sample_images.png` and `figures/class_mean_images.png`.

#### 1.9.3. Spatial & Handcrafted Feature Findings (Steps 6 & 7)

Foreground is defined by `FOREGROUND_THRESHOLD = 0.1` on the scaled view (`src/datasets/eda.py`).
- Spatial maps across the $28 \times 28$ grid are saved to `figures/mean_pixel_heatmap.png` and `figures/pixel_variance_heatmap.png`, with per-image descriptors in `tables/spatial_features.csv`.
- Per-class handcrafted feature summaries (mean intensity, std intensity, foreground ratio, bounding-box height/width/aspect/area ratio, and edge density) are persisted in `tables/handcrafted_features.csv`. For example, `Trouser` exhibits a narrow bounding box width (`14.79` px, aspect ratio `0.5416`), whereas `Coat` (`mean_intensity = 0.3852`, `foreground_ratio = 0.6599`) and `Pullover` (`mean_intensity = 0.3769`, `foreground_ratio = 0.6656`) share nearly identical spatial footprints.

#### 1.9.4. Inter-Class Similarity (Step 8)

Top cosine similarities between class mean images (`tables/class_similarity_matrix.csv`, `figures/class_similarity_heatmap.png`):

| Class A | Class B | Cosine similarity |
|:---|:---|---:|
| Coat | Shirt | 0.987391 |
| Pullover | Coat | 0.984122 |
| Pullover | Shirt | 0.982908 |
| T-shirt/top | Shirt | 0.949657 |
| Sandal | Ankle boot | 0.925568 |

#### 1.9.5. Handoff to Feature Engineering & Pipeline Lock (Step 9)

`lock_eda_fe_artifacts()` writes `outputs/eda/split_indices.npz`, `outputs/eda/split_metadata.json` (`split_v2_stratified`), and `outputs/eda/preprocessing_config.json` (`preproc_v2_train_global_std`). Before training begins, `verify_pipeline_handoff()` asserts 100% agreement between the loaded `DataLoader` instances and the locked EDA artifacts.

---

> **Nguồn bằng chứng:** phần EDA trên mô tả artifact lịch sử trong `outputs/eda/`; không suy từ các file đó rằng toàn bộ kết quả hiện hành đến từ Kaggle. Kết quả ba model và optimizer bên dưới thuộc các run local CPU được nêu riêng, có config, environment snapshot, prediction và metric. Không dùng notebook hoặc file kết quả không tồn tại để chứng thực số liệu.


Các bảng EDA lịch sử về handcrafted feature cần đọc theo định nghĩa ở run đó. Đặc biệt edge_density lịch sử dùng tổng chênh lệch hai trục với ngưỡng 0.15; pipeline feature modeling dùng tỷ lệ cặp lân cận vượt 0.1. Sau hợp nhất, bảng handcrafted mới dùng measure_intensity chung với extract_features; artifact cũ không được sửa lại hoặc gán công thức mới.

## 3. Kết quả hiện hành

### So sánh ba cấu hình

Các run được chọn tường minh trong config/workflow.json. Kết quả dưới đây được đọc trực tiếp từ metrics.json của từng run, không phải số liệu khai báo trước huấn luyện.

- **SmallMLP:** test Accuracy 88.52%, Macro F1 0.884948, 101,770 tham số. Artifact: `outputs/runs/20261022T133355Z_SmallMLP_cdad89a5/`.

- **MainMLP:** test Accuracy 89.41%, Macro F1 0.893813, 235,146 tham số. Artifact: `outputs/runs/20261022T133406Z_MainMLP_8f90f1a2/`.

- **MainMLP_FE:** test Accuracy 89.60%, Macro F1 0.895752, 237,706 tham số. Artifact: `outputs/runs/20261022T133425Z_MainMLP_FE_d8323126/`.

![Confusion matrices](outputs/runs/20261022T145033Z_comparison_c83937e3/confusion_comparison.png)

![ROC và PR OvR SmallMLP](outputs/runs/20261022T145033Z_comparison_c83937e3/SmallMLP_ovr_curves.png)

![ROC và PR OvR MainMLP](outputs/runs/20261022T145033Z_comparison_c83937e3/MainMLP_ovr_curves.png)

![ROC và PR OvR MainMLP_FE](outputs/runs/20261022T145033Z_comparison_c83937e3/MainMLP_FE_ovr_curves.png)

## Thực nghiệm optimizer và hyperparameter tuning

### Quyết định và phương pháp

Ba cấu hình kiến trúc/FE vẫn giữ kết quả với thiết lập chung ban đầu. Phần mới cố định MainMLP nhận 784 pixel để tránh trộn thay đổi kiến trúc với optimizer. Kịch bản fixed-LR dùng SGD momentum 0.9, RMSprop và Adam cùng lr=0.001, weight_decay=0, batch=128, seed=42, 10 epoch và CosineAnnealingLR. Cùng LR là điều kiện kiểm soát, không phải bảo đảm từng optimizer được cấu hình tốt nhất.

Tuning dùng ba study độc lập, mỗi optimizer 10 trial với tối đa 10 epoch. SGD tìm LR log trong [0.001,0.1], momentum [0.8,0.95]; RMSprop/Adam tìm LR log trong [0.0001,0.01]. Tất cả tìm weight_decay log trong [1e-7,0.01]. Không tune batch, hidden layers, feature threshold hoặc scheduler ở đợt này. Mỗi study dùng TPE seed=42, MedianPruner với 3 startup trials và 2 warmup steps. Ngân sách bằng nhau theo trial/epoch tối đa, thời gian thực có thể khác do pruning và thuật toán.

Objective là best validation accuracy; mỗi candidate khởi tạo lại model/seed/loader. Không đánh giá test trong fixed-LR hoặc objective. Winner được chọn hoàn toàn bằng validation rồi refit từ đầu với cùng seed/epochs/batch, dùng best validation checkpoint để đánh giá test. Kết quả tuned không được dùng để kết luận MainMLP tốt hơn SmallMLP/FE chưa tune theo kiến trúc.

### Triển khai và artifact

`src/training/tuning.py` có run_fixed_lr, run_tuning, refit_winner và summarize_optimizer_runs. Mỗi candidate lưu config (params, architecture, data SHA256, device, seed), log, history từng epoch, validation.json và best.pt. SQLite lưu studies, intermediate values và trạng thái COMPLETE/PRUNED; CSV trials lưu sau mỗi trial. Trial pruned có status pruned riêng, không nhầm với lỗi phần mềm. Refit kiểm tra hash dữ liệu và các điều kiện cùng study trước khi train.

Notebook 02 có các cell riêng cho cấu hình, fixed LR, tuning, refit và tổng hợp. Output hình giới hạn: fixed validation curves; tuning best-score/LR-score; fixed-vs-tuned; một hình ROC+PR OvR của cấu hình cuối. Không sinh PR/ROC cho mỗi trial. Confusion matrix cuối và các tọa độ đường cong vẫn lưu artifact.


### Kết quả chạy thật local CPU ngày 22/10/2026

Các run đã hoàn thành: fixed-LR `outputs/runs/20261022T130020Z_optimizer_fixed_lr_dbe789e0/`; tuning `outputs/runs/20261022T130101Z_optimizer_tuning_d8986ca8/`; refit `outputs/runs/20261022T130609Z_MainMLP_0f179a92/`; tổng hợp `outputs/runs/20261022T145040Z_optimizer_results_da3a081a/`. Suite `outputs/runs/20261022T130020Z_optimizer_experiment_suite_809c1fd3/phase_paths.json` lưu đường dẫn từng giai đoạn. Không dùng output gán sẵn hoặc kết quả synthetic làm benchmark.

Best validation accuracy (fixed LR -> best trong study), đơn vị phần trăm:

- SGD: 85.0833% -> 90.8000%; chênh lệch +5.7167 điểm phần trăm.
- RMSprop: 90.7333% -> 90.5667%; chênh lệch -0.1667 điểm phần trăm.
- Adam: 90.9333% -> 90.8500%; chênh lệch -0.0833 điểm phần trăm.

Có đúng 10 trial mỗi optimizer: SGD 5 COMPLETE/5 PRUNED, RMSprop 6 COMPLETE/4 PRUNED, Adam 5 COMPLETE/5 PRUNED. Tổng 16 COMPLETE và 14 PRUNED, không có trial lỗi. Ngân sách trial/epoch tối đa bằng nhau; không coi thời gian từng study là chi phí một lần train.

![Fixed LR validation curves](outputs/runs/20261022T130020Z_optimizer_fixed_lr_dbe789e0/fixed_lr_curves.png)

![Tuning progress và learning rate](outputs/runs/20261022T130101Z_optimizer_tuning_d8986ca8/tuning_progress.png)

Hình tiến trình chỉ tổng hợp trial COMPLETE; trial PRUNED vẫn nằm trong CSV/SQLite, không bị xóa. Mức best của SGD tăng khi LR được tìm ở vùng cao hơn 0.001. RMSprop và Adam trong ngân sách này chưa vượt cấu hình fixed. Vì vậy không kết luận tuning luôn cải thiện hoặc optimizer nào tốt tuyệt đối.

Cấu hình best trong ba study là Adam, lr=0.0015930522616241021, weight_decay=0.0003470266988650413. Best validation accuracy=90.85%, epoch 10. Refit từ đầu tái tạo đúng mức validation này. Đây là best trong nhóm tuning, không phải best của toàn bộ fixed+tuned: Adam fixed có validation 90.9333% nên vẫn là mốc tốt hơn trong comparison này. Không chọn cấu hình cuối bằng test accuracy.

Refit best study trên 10.000 ảnh test: Accuracy=89.14%, Macro F1=0.891219, Macro Precision=0.891262, Macro Recall=0.891400, CrossEntropy=0.303679. Checkpoint, history, class report và probabilities nằm ở run refit.


![ROC và Precision–Recall của refit best study](outputs/runs/20261022T145040Z_optimizer_results_da3a081a/final_ovr_curves.png)

Đã đọc trực tiếp hình tiến trình và ROC/PR: legend có đủ 10 lớp, mỗi panel là one-vs-rest; ROC hiển thị AUC, PR hiển thị Average Precision. Không đánh đồng AUC-ROC với accuracy hoặc AP. Confusion matrix cuối và toàn bộ tọa độ đường cong được lưu trong artifact, không tự in hết PNG trong notebook.

Regression: 41 tests passed, 1 historical full-training signature test deselected; log tại `outputs/runs/20261022T130150Z_full_regression_with_tuning_532fbb07/`. Kiểm thử callback/summary bổ sung có log tại `outputs/runs/20261022T130531Z_tuning_callback_regression_8f62d37b/`.

Notebook 02 hiện dùng cùng các cell để chạy mới hoặc đọc lại kết quả: MODE=train gọi pipeline, MODE=view đọc registry. Bản đọc lại từng được lưu riêng ở `outputs/runs/20261022T130724Z_optimizer_notebook_readback_b732509d/optimizer_results.executed.ipynb` là lịch sử trước hợp nhất; notebook hiện hành không có phần đọc lại lặp ở cuối. Roadmap liên kết trực tiếp file triển khai và ghi hàm điều phối cùng hàm tính toán.



## Giới hạn và tái lập

Một seed chưa đủ kết luận ý nghĩa thống kê cho chênh lệch nhỏ. Local đã chạy thật; chưa xác minh Kaggle end-to-end. Refit best study là best trong study, không phải best toàn bộ baseline+tuned. Cấu hình/run đăng ký trong config/workflow.json; CSV, predictions và checkpoint là bằng chứng gốc. Các run/history không bị xóa khi hợp nhất tài liệu. Test regression và nhật ký kỹ thuật nằm trong internal_changes.md và outputs/runs; không đưa nhật ký chỉnh sửa hoặc phân tích lỗi chi tiết vào báo cáo xuất sau này.

Nguồn API: [Optuna MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html), [Optuna RDB](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html).

Bản hiện hành sử dụng config/workflow.json và một chế độ train/view chung cho notebook. Source/cấu hình trước hợp nhất được truy vết qua lịch sử Git; các run nguồn của kết quả bên trên vẫn bất biến. Kiểm tra sau hợp nhất: 44 tests passed, 1 test historical training deselected; log nội bộ được lưu riêng, không thuộc bản báo cáo xuất.


### Trình bày huấn luyện sau search và lưu/tải model

Sau khi chọn best parameters bằng validation, notebook 02 hiển thị `learning_curves.png` và `history.csv` của FINAL_RUN: train/validation CrossEntropy loss và accuracy (%) theo epoch của lượt refit. Hình thuộc run refit đang chọn `outputs/runs/20261022T130609Z_MainMLP_0f179a92/`, không phải đường cong của từng trial Optuna.

![Loss và accuracy của refit best study](outputs/runs/20261022T130609Z_MainMLP_0f179a92/learning_curves.png)

Bước cuối gọi `src.inference.save_load_model`: nạp checkpoint refit đã chọn, lưu lại state_dict với metadata kiến trúc, tạo FashionMLP mới và load checkpoint ở eval mode. Xác suất trước/sau trên tối đa 16 ảnh test được lưu trong roundtrip_predictions.npz; roundtrip.json ghi sai lệch lớn nhất và trạng thái kiểm chứng. Đây là phép kiểm tra bảo toàn model, không phải một lần tìm hyperparameter hoặc huấn luyện mới. Mỗi lần thực thi cell đều có run/log/manifest riêng, kể cả khi notebook dùng MODE=view.

Kiểm chứng save/load thực tế: [roundtrip.json](outputs/runs/20261022T151050Z_model_save_load_34645627/roundtrip.json) ghi 16 ảnh test, sai lệch xác suất lớn nhất bằng 0 và toàn bộ nhãn dự đoán giữ nguyên. [Checkpoint xuất](outputs/runs/20261022T151050Z_model_save_load_34645627/best.pt) đi cùng config kiến trúc và log trong cùng thư mục run.


### So sánh đường học của ba optimizer sau search và test model tải lại

Notebook hiển thị bốn panel Train Loss, Validation Loss, Train Accuracy và Validation Accuracy, mỗi panel có đủ SGD, RMSprop, Adam. Mỗi đường lấy history của best completed trial tương ứng trong study; không gán đường refit của winner cho cả ba optimizer. Best parameters, trial number và history theo epoch được lưu CSV.

![Loss và accuracy của ba optimizer với best parameters riêng](outputs/runs/20261022T151640Z_best_optimizer_learning_curves_cbe88b2e/best_optimizer_learning_curves.png)

Model được load lại từ checkpoint xuất đã được đánh giá trên toàn bộ 10.000 ảnh test: Accuracy 89,14%, Macro F1 0,891219, CrossEntropy loss 0,303679. Phép save/load trên 16 mẫu cho sai lệch xác suất bằng 0. Kết quả toàn test nằm ở [test_metrics.json](outputs/runs/20261022T151641Z_model_save_load_2f8d78d9/test_metrics.json), đi cùng probabilities và classification report.

![Mẫu test sau khi tải model](outputs/runs/20261022T151641Z_model_save_load_2f8d78d9/loaded_model_test_samples.png)

Lưới chọn 8 dự đoán đúng đầu tiên và 8 dự đoán sai đầu tiên theo index, hiển thị nhãn thật, nhãn dự đoán và xác suất softmax của lớp được chọn. Vì được chọn cân bằng đúng/sai để minh họa, tỷ lệ trong lưới không biểu diễn accuracy toàn test. Nhóm lỗi gồm nhầm Sneaker/Ankle boot sang Sandal và các lớp áo sang Pullover/Shirt, dù một số xác suất cao; xác suất softmax không bảo đảm dự đoán đúng. CSV đi kèm lưu index, nhãn và xác suất để đối chiếu.

Notebook EDA bổ sung mục 2.1 hiển thị source features.py đang được pipeline gọi, cùng 10 feature trước StandardScaler của 10 ảnh train đầu tiên và original index/label. Source snapshot và bảng preview được lưu vào run feature_code_preview với log và manifest; không sao chép công thức thành implementation thứ hai.

Notebook EDA hiện chia đúng 7 mục tương ứng roadmap: setup; Core EDA; split/preprocessing; FE; visualization FE; handoff; tracking. Backend prepare_data vẫn tạo toàn bộ preparation run trong một lời gọi tại mục 2 ở MODE=train; các mục sau trình bày và kiểm chứng từng giai đoạn. Mục FE gọi extract_features trên 10 ảnh train thật theo original_index, đối chiếu giá trị với train_features.csv, kiểm tra nhãn/indices và áp tham số scaler train để so với prepared.npz. Artifact kiểm chứng tách riêng và không ghi đè run dữ liệu gốc.
