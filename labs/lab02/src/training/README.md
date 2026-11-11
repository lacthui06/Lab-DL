# Lab02 · Module training
## 1. Thông tin module
- **Tên:** `src/training` — điều phối transfer learning trên CIFAR-10.
- **Điểm vào:** [scripts/run.py](../../scripts/run.py) → `run_suite()`.
- **Phạm vi:** vòng lặp học, chọn trial/checkpoint, dừng sớm và ghi kết quả.

| Ngày | Phiên bản | Tác giả | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-07 | 0.1.0 | Thường FEAI | Tạo tài liệu hướng dẫn đọc module |
| 2026-11-07 | 0.1.1 | Thường FEAI| Gắn tên file, hàm và số dòng cho từng bước |
| 2026-11-07 | 0.1.2 | Thường FEAI| Bổ sung đầu vào, output, log và biểu đồ theo giai đoạn |
| 2026-11-11 | 0.1.3 | Thường FEAI| Bổ sung lý do thiết kế, cấu hình scheduled, cơ chế LR và giới hạn thí nghiệm |
| 2026-11-11 | 0.1.4 | Thường FEAI| Sắp xếp theo mục đích → input → pipeline → thực thi → lựa chọn → output → đánh đổi → thực hành |
| 2026-11-11 | 0.1.5 | Thường FEAI, cùng Codex | Chuẩn hóa bảng source 4 cột; giải thích trial, tolerance, prefix cache và cơ chế scheduler |

## 2. Mục đích

Module huấn luyện ResNet18 và DenseNet121 qua hai giai đoạn:

- **Head-only:** giữ backbone pretrained, học classifier 10 lớp để tạo baseline.
- **Fine-tune:** bắt đầu từ head-only đã chọn, học tiếp classifier và phần backbone được mở.
- Chọn kết quả bằng validation, sau đó đánh giá test cho cả hai stage.
- Lưu history, checkpoint, log và biểu đồ để kiểm tra và tiếp tục khi gián đoạn.

## 3. Luồng huấn luyện và cơ chế

### 3.1. Pipeline

```text
Config + CIFAR-10 + weights ImageNet-1K
  ↓
run_suite(): cố định seed, chuẩn bị split và thư mục run
  ↓
run_model(): tạo từng backbone, cache feature train/validation
  ↓
Head-only: train classifier → chọn trial và checkpoint theo validation
  ↓
search_finetune(): nạp cùng head-only đã chọn cho mỗi candidate
  → mở lớp theo config → train tiếp → chọn theo validation
  ↓
Nạp checkpoint head-only/fine-tune đã chốt → đánh giá test
  ↓
Ghi comparison.csv, biểu đồ và trạng thái completed
```

### 3.2. Cấu hình và cơ chế học

Bảng áp dụng cho [scheduled.json](../../config/scheduled.json), với **seed/split seed 42**.
CIFAR-10 có 50.000 ảnh train gốc, chia thành **45.000 train + 5.000 validation**, và **10.000 test** chính thức riêng.
Weights chỉ học từ 45.000 ảnh train; validation dùng chọn kết quả, test dùng đánh giá. Các limit bằng `0` nghĩa dùng đủ ba tập này, không cắt subset.
Khi đọc kết quả cũ, dùng `config.json` của chính run đó.

| Thiết lập | Head-only | Fine-tune |
| --- | --- | --- |
| Khởi tạo | Weights ImageNet-1K, classifier mới | Head-only đã chọn của cùng backbone |
| Phần được học | Classifier | `scheduled_tail` và classifier; tối đa 50% tham số |
| LR ban đầu | `0.001` | Backbone `0.0001`, classifier `0.001` |
| Batch / epoch | 128 / 12 | 128 / tối đa 20 |
| Optimizer / weight decay | AdamW / `0.0001` | AdamW / `0.0001` |
| Scheduler | `cosine_ratio`, min ratio `0.1` | Như head-only |
| Dropout / ảnh | `0.2` / RGB 64×64 | Như head-only |
| Dữ liệu học | Vector feature đã cache | Ảnh, augmentation chỉ trên train; prefix cache tắt |

**Cấu hình thử:** candidate là một cấu hình huấn luyện được đề xuất; trial là một lượt huấn luyện để đánh giá cấu hình đó. Head-only duyệt `learning_rates × batch_sizes`; fine-tune dùng `finetune_candidates` hoặc một candidate mặc định.

**Các lần chạy còn lưu:**

| Run | Cấu hình thực đã chạy | Bằng chứng |
| --- | --- | --- |
| CPU lịch sử `20261024…71863a0e` | Head: LR `{0.001, 0.01}` × batch `{32, 64}`, **4 trial/model**, 5 epoch/trial; fine-tune 2 epoch. | [Config gốc](../../outputs/runs/20261024T075133Z_cifar10_cpu_full64_71863a0e/config.json) · [Trial ResNet18](../../outputs/runs/20261024T075133Z_cifar10_cpu_full64_71863a0e/resnet18/search_results.csv) · [Trial DenseNet121](../../outputs/runs/20261024T075133Z_cifar10_cpu_full64_71863a0e/densenet121/search_results.csv) |
| Scheduled `20261030…b895bfc1` | Head LR `0.001`, batch `128`; **1 trial mỗi stage/model**, theo bảng trên. | [Config gốc](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/config.json) · [Kết quả](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/comparison.csv) |

**Trước đây:** đã thử đủ 4 tổ hợp LR `{0.001, 0.01}` và batch `{32, 64}` cho mỗi model. LR `0.001` + batch `64` được chọn ở cả hai: validation accuracy ResNet18 **77,16%**, DenseNet121 **79,24%**.
**Hiện tại:** run scheduled dùng sẵn batch `128`, head 12 epoch và fine-tune tối đa 20 epoch, thêm scheduler/dropout/augmentation. Run này không thử lại nhiều LR/batch để chọn batch 128, nên không kết luận cải thiện chỉ do đổi batch. [Đối chiếu hai lần chạy](../../walkthrough_lab_02.md#36-kết-quả-search-trong-từng-điều-kiện-thực-nghiệm).
Các config [cpu_full](../../config/cpu_full.json), [full](../../config/full.json), [tuned](../../config/tuned.json) hiện cho phép thử batch `64/128`; kết quả đã chạy phải xem bảng bằng chứng trên. Hai run đó không dùng Optuna. Trong scheduled, **batch train là 128**; số `64` ở `feature_batch_size` là batch trích feature, còn `image_size` là ảnh 64×64.

**Một batch train** trong [pass_epoch()](experiment.py#L53):

```text
zero_grad → forward → cross_entropy → kiểm tra loss hữu hạn → backward → optimizer.step
```

Validation dùng eval/inference mode, chỉ forward và đo metric, không cập nhật weights. Loss epoch là tổng `loss_batch × số_mẫu_batch` chia tổng số mẫu; accuracy là số đúng chia số mẫu. Nếu loss NaN/Inf, code báo `FloatingPointError` trước bước cập nhật của batch đó.

**Scheduler:** mục tiêu là giảm dần độ lớn bước cập nhật trong quá trình học. Cấu hình scheduled dùng `cosine_ratio` để giảm learning rate (LR) theo đường cosin, giữ tỷ lệ giữa LR backbone và classifier. `min_ratio=0.1` đặt điểm cuối lịch bằng 10% LR ban đầu của từng nhóm. [Công thức scheduler](tuning.py#L99)

LR được ghi trước khi học epoch; `scheduler.step()` chạy sau train/validation để chuẩn bị epoch tiếp theo. Mức 10% đạt sau khi step đủ lịch, không phải LR đã dùng trong epoch cuối; dừng sớm có thể chưa đi hết lịch. Cơ chế này điều chỉnh LR theo lịch, không tự chứng minh mô hình đã hội tụ.

**Ba quyết định riêng biệt** trong [fit()](experiment.py#L90) và [select_trial()](tuning.py#L63):

| Quyết định | Điều kiện |
| --- | --- |
| Chọn checkpoint `best.pt` | Val accuracy tăng; nếu bằng nhau, chọn val loss thấp hơn; nếu cả hai bằng nhau, giữ epoch trước. |
| Chọn trial | Lọc theo accuracy tolerance, rồi loss tolerance, cuối cùng chọn thời gian fit thấp nhất. Tolerance là khoảng chênh lệch cho phép so với giá trị tốt nhất ở từng bước lọc; scheduled không đặt nên mặc định bằng 0. |
| Dừng sớm fine-tune | Đã đạt ít nhất 10 epoch và 7 epoch liên tiếp không cải thiện val loss hơn `0.0001`. |

Epoch tốt nhất theo accuracy có thể khác mốc loss dùng dừng sớm.
Lab02 chưa đặt accuracy tối thiểu để đạt bài; không lấy kết quả đã thấy để đặt ngưỡng hậu nghiệm.

### 3.3. Input, output và nơi lưu

Quy ước: `R = outputs/runs/<run_id>`, `M = R/<model>`; đây là đường dẫn tương đối từ thư mục Lab02.

| Giai đoạn | Input | Output chính |
| --- | --- | --- |
| Chuẩn bị run | Config, dữ liệu và split | `R/config.json`, `selected_indices.npz`, `data_summary.json`, `console.log` |
| Head-only | Feature `[N,512]` hoặc `[N,1024]`, nhãn | `M/head_trial_*/`; kết quả chọn ở `M/head_selected/` |
| Fine-tune scheduled | Ảnh `[B,3,64,64]`, nhãn, head-only đã chọn | `M/finetune/`, `M/selection.json` |
| Test | Checkpoint đã chốt và loader test | `metrics.json`, `test_predictions.npz`, `confusion_matrix.png` trong thư mục stage |
| Báo cáo | History và metric | `M/learning_curves.png`, `R/comparison.csv`, `R/comparison.png` |

Trong thư mục trial, `history.csv` ghi loss/accuracy train–validation, LR và thời gian theo epoch; `tensorboard/` ghi theo dõi. `training_summary.json` ghi epoch tốt nhất, số epoch thực chạy, lý do dừng và chi phí.

`best.pt` phục vụ đánh giá; `last.pt` lưu model, optimizer, scheduler, RNG và tiến độ để resume cuối epoch. Khi dùng candidate tường minh, `finetune/` nhận bản sao kết quả được chọn; resume dùng `last.pt` trong trial gốc.

Kiểm tra run có `status.json: completed`, đủ hai stage mỗi model và history khớp summary.
Đọc [registry run được chọn](../../outputs/selected_run.json) để tìm artifact; checkpoint cần đồng bộ riêng khi clone.
Metric đầy đủ ở [walkthrough](../../walkthrough_lab_02.md); cách tiếp tục run ở [hướng dẫn resume](../../README.md#36-khôi-phục-khi-gián-đoạn).

### 3.4. Bản đồ source

| Tệp mã nguồn | Ký hiệu / Hàm chính (Ctrl+Click) | Dòng code | Việc đảm nhiệm |
| :--- | :--- | :--- | :--- |
| [experiment.py](experiment.py) | [`run_suite()`](experiment.py#L474), [`run_model()`](experiment.py#L337), [`search_finetune()`](experiment.py#L259) | [L474](experiment.py#L474), [L337](experiment.py#L337), [L259](experiment.py#L259) | Điều phối run, stage, candidate và đánh giá test. |
| [experiment.py](experiment.py) | [`pass_epoch()`](experiment.py#L53), [`fit()`](experiment.py#L90) | [L53](experiment.py#L53), [L90](experiment.py#L90) | Học/đo từng batch, lặp epoch, ghi history và checkpoint. |
| [feature_cache.py](feature_cache.py) | [`cached_features()`](feature_cache.py#L13), [`tensor_loader()`](feature_cache.py#L55) | [L13](feature_cache.py#L13), [L55](feature_cache.py#L55) | Trích feature, giữ nhãn/index đúng thứ tự và tạo batch từ cache. |
| [tuning.py](tuning.py) | [`fine_candidates()`](tuning.py#L23), [`select_trial()`](tuning.py#L63), [`make_optimizer()`](tuning.py#L86), [`make_scheduler()`](tuning.py#L99) | [L23](tuning.py#L23), [L63](tuning.py#L63), [L86](tuning.py#L86), [L99](tuning.py#L99) | Kiểm tra cấu hình, chọn trial và tạo cơ chế tối ưu. |
| [__init__.py](__init__.py) | [`__getattr__()`](__init__.py#L20), [`__all__`](__init__.py#L17) | [L20](__init__.py#L20), [L17](__init__.py#L17) | Công khai API và nạp module tương ứng khi tên được yêu cầu. |

## 4. Thảo luận và quyết định kỹ thuật

### 4.1. Vì sao head-only trước fine-tune?

Head-only đo baseline từ feature pretrained; fine-tune cho phép đặc trưng thích nghi thêm. Mỗi candidate fine-tune reset seed và nạp cùng trạng thái head-only, tránh học nối từ candidate trước.

Mở thêm lớp tăng chi phí và nguy cơ overfit; không mặc định fine-tune tốt hơn. Danh sách lớp, `scheduled_tail` và BatchNorm được giải thích ở [module models](../models/README.md).

### 4.2. Vì sao cache và khi nào phải chạy từ ảnh?

Backbone cố định cho phép head-only dùng lại feature, tránh forward backbone mỗi epoch; vẫn phải tính chi phí tạo cache.

Prefix cache là đặc trưng trung gian được lưu từ phần đầu backbone cố định, để các epoch sau chỉ chạy phần cuối. Fine-tune dùng cách này chỉ hợp lệ với prefix cố định, transform xác định và độ mở `last_block`; code kiểm tra logits tương đương.

Scheduled dùng augmentation và `scheduled_tail` nên tắt prefix cache, chạy từ ảnh. Cách này giữ đường xử lý phù hợp với cấu hình học, nhưng không tiết kiệm được phép tính phần đầu backbone như khi dùng prefix cache. [Điều kiện cấu hình](tuning.py#L23)

### 4.3. Vì sao chỉ dùng validation để chọn?

Validation phục vụ chọn epoch/trial; test chỉ được đánh giá sau khi chốt cả hai stage cho model.
Không dùng test để điều chỉnh candidate hoặc chính sách chọn, nhằm giữ phép đánh giá cuối tách khỏi quá trình lựa chọn.

### 4.4. Giới hạn kết luận

Scheduled có một cấu hình/model; grid lịch sử chỉ khảo sát LR/batch trong phạm vi đã ghi. Các run cùng seed 42 chưa chứng minh độ ổn định giữa các seed hoặc tối ưu ngoài grid.
Head-only và fine-tune khác cả augmentation, phần trainable và ngân sách epoch; chênh lệch không cô lập riêng tác động unfreeze.
So sánh chi phí cần cùng điều kiện thiết bị và tính cả tạo cache, fit, validation; số tham số không thay thế phép đo thời gian.
