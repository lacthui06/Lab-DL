# Lab 02 — Transfer learning trên CIFAR-10


So sánh ResNet18 và DenseNet121 pretrained, head-only và fine-tune. Kết quả hiện tại là CPU 64×64, seed 42; pipeline tinh chỉnh LR/optimizer/unfreeze đã triển khai và kiểm thử; benchmark tuning đầy đủ chưa chạy.

## Structure

- [description_lab_02.md](description_lab_02.md): nội dung đối chiếu trực tiếp ảnh đề trang 23; phân biệt yêu cầu của đề với lựa chọn của bài làm.
- [walkthrough_lab_02.md](walkthrough_lab_02.md): báo cáo theo cấu trúc Lab01: đặc tả/thiết kế → phương pháp thực nghiệm → kết quả/thảo luận.
- [01_lab02_eda.ipynb](notebooks/01_lab02_eda.ipynb): raw → integrity → pixel/spatial EDA → preprocessing → handcrafted features; chỉ dùng `src.datasets.eda`, không nạp mạng.
- [02_lab02_modeling.ipynb](notebooks/02_lab02_modeling.ipynb): roadmap như Lab01, output ngay từng bước; đã tích hợp latency một ảnh, RAM RSS và VRAM ở mục 8. Bản [Kaggle](notebooks/02_lab02_modeling-kaggle.ipynb) có cùng bố cục.
- [03_lab02_single_image_benchmark.ipynb](notebooks/03_lab02_single_image_benchmark.ipynb): bản thực thi T4 gốc được giữ làm bằng chứng. Kết quả xem trực tiếp trong notebook 02; không cần chạy lại 03. Xem [provenance](outputs/published/single_image_benchmark/20261102T184034Z_kaggle/provenance.json).
- [config/](config/): [full.json](config/full.json) cho GPU 224×224; [cpu_pilot.json](config/cpu_pilot.json) kiểm tra nhỏ; [cpu_full.json](config/cpu_full.json) mô tả run CPU lịch sử; [tuned.json](config/tuned.json) là cấu hình tuning chạy được; thiết kế và lý do nằm trong walkthrough.
- [requirements.txt](requirements.txt): phiên bản cố định từ Windows; cài trên Linux/Kaggle chưa kiểm chứng.
- [outputs/selected_run.json](outputs/selected_run.json): run được chọn; [outputs/label_update.json](outputs/label_update.json) lưu lịch sử sửa nhãn tài liệu. `outputs/runs/` là bằng chứng gốc; `outputs/review/` chứa biểu đồ trình bày mới, không ghi đè run gốc.

### Tham chiếu source và CLI

| File | Nơi sử dụng / mục đích |
|---|---|
| [src/datasets/](src/datasets/) | [data.py](src/datasets/data.py) loader/transform và [eda.py](src/datasets/eda.py) split, thống kê, integrity; không phụ thuộc training |
| [src/models/](src/models/) | Namespace model; khởi tạo và freeze nằm trong module bên dưới |
| [src/models/models.py](src/models/models.py) | Khởi tạo mạng, freeze, cached tail |
| [src/training/](src/training/) | [experiment.py](src/training/experiment.py) train/validation/checkpoint; [feature_cache.py](src/training/feature_cache.py) cache đặc trưng/prefix và loader tensor; [tuning.py](src/training/tuning.py) search và lựa chọn theo validation |
| [src/utils/](src/utils/) | [costs.py](src/utils/costs.py) chi phí từ artifact; [reporting.py](src/utils/reporting.py), [notebook_views.py](src/utils/notebook_views.py), [delivery.py](src/utils/delivery.py), [constants.py](src/utils/constants.py) và artifact IO |
| [src/__init__.py](src/__init__.py) | Package nhẹ, chỉ khai báo namespace; dependency được nạp theo module cần dùng |
| [src/inference.py](src/inference.py) | Nạp model một lần và dự đoán một ảnh, dùng chung cho infer/benchmark |
| [src/utils/inference_benchmark.py](src/utils/inference_benchmark.py) | Latency một ảnh và lượt đo RAM/VRAM riêng, không nạp checkpoint trong phép đo |
| [scripts/analyze_data.py](scripts/analyze_data.py) | Tái đo EDA raw; `--data-dir` trỏ thư mục chứa `cifar-10-batches-py` |
| [scripts/run.py](scripts/run.py) | Điểm vào huấn luyện, dùng config/*.json |
| [scripts/infer.py](scripts/infer.py) | Nạp checkpoint và dự đoán một ảnh |
| [scripts/benchmark_inference.py](scripts/benchmark_inference.py) | Xuất ảnh test, đóng gói hai checkpoint và benchmark model đã nạp sẵn |
| [scripts/download_assets.py](scripts/download_assets.py) | README: CIFAR-10/weights; --chunked là fallback tải chia phần; --dataset-only bỏ weights |
| [scripts/resume_interrupted.py](scripts/resume_interrupted.py) | Phục hồi tại ranh giới kiến trúc, không phải resume giữa epoch |
| [scripts/execute_notebook.py](scripts/execute_notebook.py) | Thực thi notebook với interpreter hiện tại |
| [scripts/finalize.py](scripts/finalize.py) | README: finalize run; --all còn chạy test/notebook/inference theo model được validation chọn |
| [tests/test_contracts.py](tests/test_contracts.py) | Shape, freeze, split và một batch overfit |
| [tests/test_cached_prefix.py](tests/test_cached_prefix.py) | Đối chiếu logits/gradient cache và forward đầy đủ |
| [tests/test_feature_cache.py](tests/test_feature_cache.py) | Thứ tự mẫu/nhãn/chỉ số, nội dung cache, shuffle có seed và gỡ hook khi lỗi |
| [tests/test_tuning.py](tests/test_tuning.py) | LR groups, scheduler, gradient hai block, reset trial, không dùng test khi search, timing và smoke pipeline |
| [tests/test_eda.py](tests/test_eda.py) | Chặn import training, kiểm tra feature/duplicate/split, đối chiếu hash và thống kê run cũ |
| [tests/test_cleanup.py](tests/test_cleanup.py) | Graph tham chiếu production, CLI --help/import an toàn, split chung, CSV schema mới, đúng run/checkpoint và tải chia phần có checksum |
| [tests/test_notebook_review.py](tests/test_notebook_review.py) | Đối chiếu metric, ranh giới EDA và liên kết tài liệu; không cần torch |
| [tests/test_import_boundaries.py](tests/test_import_boundaries.py) | Chặn import PyTorch/model/engine khi tính chi phí và gọi view; kiểm tra export lazy |
| [tests/test_inference_benchmark.py](tests/test_inference_benchmark.py) | Output một ảnh, ranh giới thời gian, thống kê, sampler RAM và CUDA peak |

Có 4 package chức năng trong `src/`, module inference dùng chung và 8 CLI. Snapshot trong `outputs/runs/` được manifest/hash tham chiếu, giữ riêng khỏi graph code hiện hành.

Luồng chính: `run.py → experiment → data/models/tuning/reporting`; `data → eda.locked_split`; `reporting → utils.costs`; `notebook 01 → eda`; `notebook 02 → notebook_views → utils.costs`; `finalize.py → delivery → reporting`; `infer.py → data/models`.

`experiment` gọi `training.feature_cache.cached_features` để lưu tensor CPU theo schema `features/labels/indices`, và prefix NPY khi được cấu hình. `tensor_loader` dùng chung cho head, fine-tune từ cache và resume. Tên file, thứ tự mẫu và seed giữ nguyên; các import cũ qua `experiment` và export cấp `src.training` vẫn dùng được.

Các namespace `src.training` và `src.utils` chỉ nạp module khi export được yêu cầu. Chi phí đọc từ artifact qua `src.utils.costs`; recommendation dùng chính sách số liệu thuần trong `training.tuning`, không nạp PyTorch, model hoặc `training.experiment`. Các export cấp package trước đây được giữ; đường dẫn module chi phí mới là `src.utils.costs`.

Test graph bắt module production bị bỏ rơi, public function không có tham chiếu tĩnh và file thiếu link README. Tham chiếu trong test hoặc snapshot không được tính làm caller production. Docstring giải thích vai trò; không thêm import giả để giữ file.

Graph kiểm tra cả package con và relative import. Roadmap Modeling gọi `lr_search` và `roundtrip`; mọi public function trong production phải có caller, không giữ ngoại lệ cho view không dùng. TensorBoard event export và monitoring hiện có vẫn được giữ. Snapshot của finalize/resume liệt kê đệ quy các file Python/JSON trong `src/`, `scripts/`, `tests/`, `config/`; bỏ qua cache và artifact trong outputs.

## Reproduce

Chạy từ `labs/lab02`. Để **xem kết quả**, không cần dataset/weights: mở hai notebook đã có output, hoặc chạy lại bằng interpreter có NumPy, pandas, matplotlib, Pillow, scikit-learn, IPython, nbformat, nbclient và ipykernel:

Cell đầu của notebook tìm lab từ thư mục hiện tại và các thư mục cha, hoặc dùng `LAB02_ROOT` là đường dẫn tuyệt đối tới Lab02. Nếu cwd đã bị xóa, cần đặt `LAB02_ROOT` trước khi chạy cell. Đường dẫn sai được báo lỗi rõ ràng thay vì tiếp tục đọc artifact từ nơi khác.

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

Resume toàn suite: `python scripts/resume_interrupted.py outputs/runs/<run_id> --device cuda`. Chỉ hỗ trợ ranh giới kiến trúc; từ chối kiến trúc đang có trial/checkpoint dở dang hoặc thiết bị khác run gốc, không ghi đè kết quả. Khôi phục fit của một trial mới dùng `--trial` như hướng dẫn bên dưới.

### Training và TensorBoard live

`SummaryWriter` ghi event files ngay trong training, dưới `<model>/<trial>/tensorboard/`; `input_preview/` lưu ảnh đầu vào và model graph. Trong terminal thứ hai, cùng môi trường và thư mục `labs/lab02`, chạy:

```bash
tensorboard --logdir outputs/runs --port 6006
```

Mở [dashboard local](http://localhost:6006) khi server đã khởi động. Có thể mở trong lúc train, không cần chờ `finalize.py`. Để tập trung một run, đổi `--logdir` thành `outputs/runs/<run_id>`. Nếu train trên máy khác, cần truy cập server trên máy đó hoặc mang event files về local. Chạy cell hiển thị CSV/PNG không tạo events và không mở server.

Run lịch sử trong checkout thiếu checkpoint và event files gốc. `tensorboard_server_verification.json` chỉ là bằng chứng lần kiểm chứng cũ, không chứng nhận server hiện đang chạy. Không tái tạo events giả từ metric để thay bằng chứng training thật.

### Khôi phục một trial mới tại cuối epoch

Run mới lưu `best.pt` cho inference và `last.pt` cho resume. `last.pt` được thay atomically sau mỗi epoch hoàn tất, chứa model, optimizer, scheduler, best checkpoint, early-stopping counters, history, RNG Python/NumPy/Torch/CUDA và generator của hai DataLoader. Epoch đang chạy dở sẽ được chạy lại; chưa hỗ trợ khôi phục giữa batch. Không áp dụng cho checkpoint lịch sử thiếu schema mới.

```bash
python scripts/resume_interrupted.py outputs/runs/<run_id> --trial resnet18/head_trial_00 --device cuda
```

Lệnh này tiếp tục **fit của một trial**, không tự tiếp tục chọn trial, đánh giá test hay hoàn tất suite/report. Run vẫn giữ trạng thái chưa hoàn tất; không gọi `finalize.py --all` như thể đã chạy hết pipeline. Lệnh resume không có `--trial` vẫn chỉ dùng tại ranh giới kiến trúc. Không khẳng định hỗ trợ tự động resume toàn bộ suite giữa trial.

Giữ nguyên source, config, data indices, cache và môi trường. CLI kiểm tra source hash, indices và metadata trước khi resume; không đổi LR/ngân sách/cache policy. Thời gian resume cộng các đoạn fit, loại thời gian chờ; peak memory chỉ thuộc đoạn process hiện tại. Không dùng timing resume để xếp hạng benchmark mới. Event writer purge các bước chưa commit để không lặp đường cong.

### Control trước tuning mở rộng — đề xuất, chưa chạy

Giữ nguyên `full.json`/`tuned.json`; chưa tự giảm batch hoặc epoch. Control đề xuất: cả hai model, cùng split seed 42, ảnh 224×224, AdamW, head rồi mở block cuối, cùng chính sách cache và stopping. Chọn batch khả thi sau GPU sanity và đo VRAM; chưa chốt 64 hay 128. Chưa chạy grid 6 head + 13 fine trials/model trước khi biết GPU và ngân sách. Augmentation là thí nghiệm riêng: cache đặc trưng cố định không tạo augmentation ngẫu nhiên mỗi epoch.

Trước full training: kiểm tra môi trường GPU, shape/gradient và one-batch overfit; đọc validation để quyết định LR/batch/depth. Test chỉ dùng sau lựa chọn validation. Snapshot mới có phiên bản và `Requires-Python` của các package chính tại `<run>/environment_snapshot.json`; `outputs/logs/environment_snapshot.json` trỏ môi trường run mới nhất. Đây không phải xác nhận toàn bộ pins lịch sử cài được trên Linux. Lưu trữ cả run (events, `best.pt`, `last.pt`, config, indices, cache nếu cần resume, environment, manifest) ở nơi quản lý artifact; không commit weights nặng vào Git.

Python của run lịch sử: CPython 3.13.3; môi trường thực tế lưu trong `environment.json` của run. Ghim phiên bản không bảo đảm cài được trên mọi OS; chưa chạy lại full training trong đợt chỉnh trình bày này.

## Results Summary

Notebook modeling trình bày kết quả theo thứ tự: metric tổng (loss, accuracy, macro precision, macro recall/TPR, macro F1) → heatmap từng lớp (Precision, Recall/TPR, F1, FPR) → confusion matrix/ảnh sai → ROC/PR fine-tune đã lưu → chi phí. Recall và TPR dùng chung nhãn; bỏ biểu đồ cột recall/FPR lặp với heatmap. ROC dùng trục FPR–TPR qua nhiều ngưỡng, khác metric argmax; PR dùng recall–precision. Legend ROC là AUC, legend PR là AP. Bảng macro curve score lấy trung bình score từng lớp từ CSV đã lưu, không tạo đường cong hoặc inference mới. LR/batch vẫn là biểu đồ đường theo yêu cầu.

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

Mỗi seed có 6 head trials và 13 fine-tune trials **mỗi model**, fine-tune min 10/max 30 epoch/trial (validation-loss patience 7, min_delta=1e-4): đây là grid đầy đủ, cần dự trù thời gian GPU. `split_seed=42` giữ tập dữ liệu cố định khi đổi seed khởi tạo/shuffle. `LAB02_DATA` có thể trỏ tới raw CIFAR-10 đã tải. Không thay sang flower_photos trong lượt này.

- AdamW: backbone LR `{3e-6,1e-5,3e-5}`; SGD momentum 0,9: `{1e-4,3e-4,1e-3}`. Head LR = 10× backbone LR.
- ResNet mở `layer4` hoặc `layer3+layer4`; DenseNet mở `denseblock4+norm5` hoặc `denseblock3+transition3+denseblock4+norm5`; luôn học classifier. BatchNorm running statistics giữ cố định.
- Có control LR cố định; các candidate grid dùng cosine. Mọi candidate bắt đầu lại từ cùng selected head. Mở hai block buộc tắt cache prefix; tuned tắt cho cả hai.
- Chọn trong dải accuracy cách mức tốt nhất ≤0,1 điểm phần trăm, rồi loss cách mức thấp nhất trong dải ≤0,005, sau đó ưu tiên fit time nhỏ hơn. Đây là ngưỡng đánh đổi cấu hình, không phải kiểm định hai model tương đương thống kê. Test chỉ chạy sau khi khóa candidate.

`finetune_search_results.csv` lưu toàn bộ grid; `selection.json` ghi winner; `finetune/` chứa checkpoint/history được chọn, không train lại thêm lần nữa. TensorBoard lưu LR từng nhóm, optimizer/depth/scheduler, train/val loss/accuracy. `cost_comparison.csv` tách tổng search với selected recipe, không cộng lặp selected trial. Full wall time của quá trình chọn model, throughput và peak CUDA memory cũng được ghi.

Khi chất lượng gần nhau, xem thêm thời gian của selected recipe và test pass. Run cũ khác cache policy nên chưa xếp hạng tốc độ kiến trúc bằng train time. Xem [phương pháp chọn model](walkthrough_lab_02.md#28-thiết-kế-search-và-chọn-model-bằng-validation) và [chi phí](walkthrough_lab_02.md#37-chi-phí-huấn-luyện-và-đánh-giá).

Các profile grid cũ dùng batch **64/128**. Recipe mới đã chốt nằm ở `config/scheduled.json`; chưa có kết quả full training cho recipe này. Run lịch sử và checkpoint được giữ riêng.

## Recipe scheduled — tự chạy từ terminal Lab02

Kích hoạt môi trường repo (terminal đang ở `labs/lab02`):

```bash
source ../../.venv/bin/activate
python scripts/run.py --config config/scheduled.json --dry-run
python scripts/run.py --config config/scheduled.json --device cpu
```

Lệnh train chạy lần lượt ResNet18 và DenseNet121; mỗi model head 12 epoch, fine-tune tối đa 20 epoch, batch 128, cosine LR và early stopping theo description. Test được đánh giá tự động sau khi khóa checkpoint bằng validation. Với máy có CUDA, thay `--device cpu` bằng `--device cuda`. Không đóng terminal khi đang chạy.

Khi thấy `Completed: ...`, tạo ROC/PR, báo cáo bổ sung rồi cập nhật output notebook:

```bash
python scripts/finalize.py
python scripts/execute_notebook.py 02_lab02_modeling.ipynb
python scripts/execute_notebook.py 01_lab02_eda.ipynb
```

Không dùng `finalize.py --all` để chạy lại toàn bộ kiểm thử lịch sử. Mỗi run có thư mục timestamp/ID riêng ở `outputs/runs/`. Registry chỉ đổi khi suite hoàn tất; figure ở `outputs/review/<run_id>/`. Lệnh train lưu snapshot notebook hiện tại vào run cũ trước khi đổi registry; execute notebook lưu thêm bản trước/sau theo hash. Snapshot có thể có liên kết tương đối theo vị trí notebook gốc; dùng để giữ nội dung và output, không mặc định chạy lại trực tiếp từ thư mục snapshot.

TensorBoard ở terminal thứ hai, cùng môi trường: `tensorboard --logdir outputs/runs --port 6006`. Lưu `Run: ...` in ở đầu log để truy vết. Dừng giữa chừng không tự hoàn tất selection/test; xem `scripts/resume_interrupted.py --help` trước khi resume, không tự đổi source/config của run đang dở.

## Benchmark một ảnh với model đã nạp sẵn

Kết quả đã tích hợp tại **mục 8 của notebook Modeling**: các bảng latency warmed/request đầu, RAM RSS, VRAM allocated/reserved và stage diagnostics. `Run All` ở notebook 02 chỉ đọc bộ evidence được chọn tường minh trong biến `BENCHMARK`, kiểm tra checksum và training run; không khởi chạy lại GPU. Khi chuyển máy, copy cả `outputs/published/single_image_benchmark/20261102T184034Z_kaggle/`. Thiếu file sẽ hiện thông báo; sai run/checksum sẽ báo lỗi để tránh ghép nhầm kết quả.

`scripts/benchmark_inference.py` đo riêng ResNet18/DenseNet121, mỗi model một process, batch 1, FP32, eval và inference_mode. Không tính nạp checkpoint, khởi tạo model, in terminal hoặc ghi báo cáo. Tổng thời gian gồm đọc/giải mã ảnh, preprocessing từ metadata checkpoint, chuyển tensor lên thiết bị, forward, softmax/lấy kết quả về CPU và tạo chuỗi JSON.

Lượt đầu sau khi nạp model được ghi riêng. Mặc định warm-up 20 lượt rồi đo tổng 100 lượt (mean/median/p95, ms/ảnh). Chạy riêng 20 lượt đo từng bước có đồng bộ CUDA; không cộng thời gian bước để suy ra tổng. RAM/VRAM được đo ở 20 lượt riêng để sampler không làm nhiễu latency chính.

RAM là RSS process: sau nạp model, sau warm-up, trước lượt đo bộ nhớ và peak quan sát được, lấy mẫu mặc định 1 ms. Có thể bỏ lỡ đỉnh ngắn; chênh lệch peak so với baseline không phải kích thước model. VRAM là allocated/reserved của PyTorch và peak trong lượt đo warmed; không phải toàn bộ VRAM process/CUDA context. CPU ghi VRAM là null. Đây là tài nguyên quan sát được, không khẳng định cấu hình phần cứng tối thiểu. Cùng ảnh được đọc lặp nên có thể hưởng OS cache; không ép flush cache, không bật AMP/TF32/compile.

Chuẩn bị từ terminal tại `labs/lab02` (không train lại):

```bash
python scripts/benchmark_inference.py prepare-image --data-dir data --index 0 --output ../../.agents/single-image-benchmark/input.png
python scripts/benchmark_inference.py package --run outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1 --image ../../.agents/single-image-benchmark/input.png --output ../../.agents/single-image-benchmark/lab02-single-image-benchmark.zip
```

Ảnh được xuất từ tập test trong process riêng; không nạp toàn bộ test_batch trong process benchmark. ZIP chỉ chứa source cần chạy, PNG và metadata nguồn ảnh, hai `finetune/best.pt` đã chọn bằng validation và manifest SHA256. Gắn ZIP dưới dạng Kaggle Dataset **private**, giải nén vào thư mục mới dưới `/kaggle/working/` và đối chiếu manifest trước khi chạy. Giữ nguyên thư mục training cũ. Không cần cài lại pins Windows; ghi môi trường Kaggle thực tế vào kết quả.

Sau khi giải nén, chạy từ thư mục `lab02` của gói, mỗi lệnh là một process độc lập:

Có thể import notebook 03 vào một Kaggle Notebook riêng, gắn Dataset private vừa upload, chọn GPU và Run All. Runner nhận cả gói ZIP lẫn Dataset đã giải nén, kiểm tra checksum, dùng thư mục working mới và xuất bảng so sánh cùng ZIP kết quả. Notebook training cũ không cần chạy lại.

```bash
python scripts/benchmark_inference.py run --checkpoint checkpoints/resnet18.pt --image input.png --device cuda --output-dir /kaggle/working/single-image-results
python scripts/benchmark_inference.py run --checkpoint checkpoints/densenet121.pt --image input.png --device cuda --output-dir /kaggle/working/single-image-results
```

Mỗi model có benchmark ID riêng, gồm status/config/environment/source hashes, `latency_samples.csv`, `stage_samples.csv`, `summary.json`, `prediction.json` và checksum manifest. CLI yêu cầu CUDA thực, không tự chuyển sang CPU. `--device cpu` chỉ dành cho kiểm thử hoặc phép đo CPU được ghi nhãn riêng; không dùng thay kết quả Kaggle GPU. Kết quả này không cập nhật registry run training hay ghi đè metric lịch sử. Thu hồi toàn bộ thư mục kết quả và notebook đã chạy trước khi kết thúc phiên Kaggle.
## Kiến trúc và benchmark training batch 1 bổ sung

Module đo: [training_benchmark.py](src/utils/training_benchmark.py). CLI: [benchmark_training.py](scripts/benchmark_training.py). View chỉ đọc artifact: [notebook_views.py](src/utils/notebook_views.py).

Các quyết định, lý do chọn ResNet18/DenseNet121 và cấu trúc từng layer/module nằm trong [description](description_lab_02.md). Mục 2 notebook Modeling đọc kiến trúc đầy đủ và bảng tensor tham số của run; cột `head_stage_trainable` chỉ mô tả stage head, còn phần mở fine-tune lấy từ metadata/comparison.

Chạy từ root repository, sau khi có dữ liệu CIFAR-10 và hai checkpoint của run được chọn trong [cấu hình benchmark](config/training_benchmark.json):

```bash
MPLBACKEND=Agg .venv/bin/python labs/lab02/scripts/benchmark_training.py
```

Đây là vài bước training CPU batch 1, không full training. CLI không download dữ liệu/weights, không sửa checkpoint và từ chối output đã tồn tại. Muốn đo lại, chọn tên output mới trong config; muốn CUDA phải chọn `device=cuda` và chạy ở môi trường CUDA thật. Không tự thay GPU bằng CPU. Chỉ chọn run/checkpoint thực có, không tự lấy run mới nhất.

Mục 8b notebook chỉ đọc [evidence](outputs/published/training_benchmark/scheduled64_cpu_batch1), kiểm manifest và nguồn checkpoint; Run All không benchmark. Timing gồm forward/loss/backward/step với batch đã ở thiết bị. Optimizer mới dùng LR đầu recipe; đây không phải resume. Không augmentation; giữ dropout/freeze/BN theo checkpoint. RSS gồm Python/runtime; CPU VRAM không khả dụng. Inference T4 ở mục 8 có phạm vi riêng.
