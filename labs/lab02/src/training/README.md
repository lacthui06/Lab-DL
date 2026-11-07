# Lab02 · Module training

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-07 | 0.1.0 | Thường FEAI, cùng Codex | Tạo tài liệu hướng dẫn đọc module |
| 2026-11-07 | 0.1.1 | Thường FEAI, cùng Codex | Gắn tên file, hàm và số dòng cho từng bước |
| 2026-11-07 | 0.1.2 | Thường FEAI, cùng Codex | Bổ sung đầu vào, output, log và biểu đồ theo giai đoạn |

## 1. Pipeline huấn luyện

```text
Input: cấu hình + CIFAR-10 + weights ImageNet-1K
  ↓
scripts/run.py → run_suite() → chuẩn bị chỉ số train / validation / test
  ↓
run_model() cho từng kiến trúc ResNet18 và DenseNet121
  ↓
Head-only → đóng băng backbone → cache đặc trưng train và validation
          → train classifier → chọn trial theo validation
  ↓
Fine-tune → bắt đầu từ cùng trạng thái head-only đã chọn
          → mở các lớp theo cấu hình → train tiếp → chọn trial theo validation
  ↓
Test → nạp lần lượt checkpoint head-only và fine-tune đã chốt để đánh giá
  ↓
Output: lịch sử train/validation, checkpoint, căn cứ chọn trial và bảng kết quả
```

| Bước | Việc thực hiện | Đoạn code và vị trí |
| --- | --- | --- |
| Điểm vào CLI | Import và gọi hàm `main()` của module training. | `from src.training.experiment import main` · [scripts/run.py, dòng 5](../../scripts/run.py#L5)<br>`main()` · [scripts/run.py, dòng 8](../../scripts/run.py#L8) |
| Chuẩn bị run | Cố định seed, rồi chuẩn bị chỉ số train, validation và test. | `seed_all(cfg["seed"])` · [experiment.py, dòng 489](experiment.py#L489)<br>`selected = prepare_indices(data_dir, run, cfg, split_dir=work/'outputs/eda')` · [experiment.py, dòng 546](experiment.py#L546) |
| Head-only | Đóng băng backbone, cache đặc trưng train và validation, rồi chọn trial. | `configure_trainable(model, name, "head")` · [experiment.py, dòng 345](experiment.py#L345)<br>`train_features, train_sec = cached_features(…)` · [experiment.py, dòng 364–365](experiment.py#L364)<br>`winner = select_trial(trials, cfg.get('selection_policy'))` · [experiment.py, dòng 390](experiment.py#L390) |
| Fine-tune | Khôi phục cùng trạng thái head-only trước mỗi trial, mở các lớp theo cấu hình, rồi chọn trial. | `model.load_state_dict(initial)` · [experiment.py, dòng 273](experiment.py#L273)<br>`configure_trainable(model, name, 'finetune', candidate['unfreeze'], …)` · [experiment.py, dòng 275–276](experiment.py#L275)<br>`winner = select_trial(trials, cfg.get('selection_policy'))` · [experiment.py, dòng 321](experiment.py#L321) |
| Test | Nạp checkpoint đã chốt rồi tạo loader cho tập test. | `saved = torch.load(stage_folder / "best.pt", …)` · [experiment.py, dòng 419](experiment.py#L419)<br>`loader = image_loader(data_dir, "test", selected["test"], …)` · [experiment.py, dòng 423](experiment.py#L423) |
| Xuất kết quả | Ghi bảng so sánh và trạng thái hoàn thành của run. | `pd.DataFrame(rows).to_csv(run / "comparison.csv", index=False)` · [experiment.py, dòng 558](experiment.py#L558)<br>`write_json(run / "status.json", {"status": "completed"})` · [experiment.py, dòng 560](experiment.py#L560) |

### Bên trong một epoch

```text
Train batch → forward → tính cross-entropy → backward → cập nhật weights
  ↓
Hết train batch → validation forward → đo loss và accuracy
  ↓
Ghi history.csv → xét lưu best.pt → xét dừng sớm → lưu last.pt
  ↓
Sang epoch tiếp theo hoặc kết thúc
```

| Bước | Dữ liệu vào | Kết quả | Đoạn code và vị trí |
| --- | --- | --- | --- |
| Train | Batch có nhãn và optimizer | Đo loss, tính gradient và cập nhật weights. | `tr = pass_epoch(model, train, device, optimizer, mode_fn)` · [experiment.py, dòng 163](experiment.py#L163)<br>`loss.backward()` → `optimizer.step()` · [experiment.py, dòng 70–71](experiment.py#L70) |
| Validation | Batch validation có nhãn, với model vừa xong lượt train | Đo `val_loss` và `val_accuracy` ở chế độ đánh giá. | `va = pass_epoch(model, val, device)` · [experiment.py, dòng 167](experiment.py#L167)<br>`model.eval()` và `torch.inference_mode()` · [experiment.py, dòng 58–60](experiment.py#L58) |
| Ghi lịch sử | Các metric train và validation của epoch | Cập nhật `history.csv`. | `pd.DataFrame(history).to_csv(folder / "history.csv", index=False)` · [experiment.py, dòng 178](experiment.py#L178) |
| Lưu checkpoint tốt nhất | Metric validation | Ghi `best.pt` khi epoch tốt hơn theo quy tắc chọn checkpoint. | `_save_checkpoint(best_checkpoint, folder / "best.pt")` · [experiment.py, dòng 197](experiment.py#L197) |
| Xét dừng sớm | Số epoch đã chạy và bộ đếm `stale` | Đánh dấu lý do dừng khi đạt điều kiện. | `if epoch >= minimum and stale >= patience:` · [experiment.py, dòng 205–206](experiment.py#L205) |
| Lưu trạng thái cuối epoch | Trạng thái model, optimizer và các dữ liệu để resume | Ghi `last.pt`. | `_save_checkpoint(…)` với đích `folder / 'last.pt'` · [experiment.py, dòng 208–220](experiment.py#L208) |

## 2. Mục đích và các file

| File | Trách nhiệm |
| --- | --- |
| [`experiment.py`](experiment.py) | Điều phối suite và các trial, chạy một lượt dữ liệu bằng `pass_epoch()`, lặp epoch bằng `fit()`, lưu checkpoint và đánh giá test. |
| [`feature_cache.py`](feature_cache.py) | `cached_features()` trích đặc trưng, giữ nhãn và chỉ số ảnh gốc đúng thứ tự. `tensor_loader()` ghép và chia batch từ các tensor đã cache. |
| [`tuning.py`](tuning.py) | Kiểm tra cấu hình candidate và early stopping, tạo optimizer và scheduler, rồi chọn trial theo kết quả validation và thời gian fit. |
| [`__init__.py`](__init__.py) | Công khai các tên như `select_trial` và `fit`. Khi caller import một tên, file mới nạp module cung cấp tên đó. |

### Hai loại cache

| Đường chạy | Dữ liệu dùng để train | Đoạn code và vị trí |
| --- | --- | --- |
| Head-only | Cache vector sau pooling, shape `[N, F]` với `F = 512` cho ResNet18 hoặc `1024` cho DenseNet121. Mỗi epoch dùng lại vector để train classifier. | `xs.append(features(model, name, images.to(device)).cpu())` · [feature_cache.py, dòng 38](feature_cache.py#L38)<br>`result = fit(head, tensor_loader(train_features, …), …)` · [experiment.py, dòng 385–387](experiment.py#L385) |
| Fine-tune có `cache_frozen_prefix` | Cache tensor ngay trước block cuối để block đó vẫn học cùng classifier. Trước khi train, code đối chiếu logits của hai đường chạy trên một mẫu nhỏ. | `last_block(model, name).register_forward_pre_hook(save_prefix)` · [feature_cache.py, dòng 34](feature_cache.py#L34)<br>`fit_model = CachedTail(model, name).to(device)` · [experiment.py, dòng 295](experiment.py#L295)<br>`torch.testing.assert_close(direct, equivalent, atol=1e-4, rtol=1e-5)` · [experiment.py, dòng 302](experiment.py#L302) |
| Fine-tune chạy từ ảnh | Đọc ảnh mỗi lượt. Khi mở lớp thuộc phần tạo cache hoặc cần augmentation mới, `fine_candidates()` yêu cầu tắt cache prefix để dùng đầu vào hiện tại. | `train_loader = image_loader(…)` · [experiment.py, dòng 310–311](experiment.py#L310)<br>`if cfg.get('train_augmentation') and cfg.get('cache_frozen_prefix'):` · [tuning.py, dòng 31–32](tuning.py#L31)<br>`if cfg.get('cache_frozen_prefix') and c['unfreeze'] != 'last_block':` · [tuning.py, dòng 57–58](tuning.py#L57) |

## 3. Input và output qua một batch

Ví dụ dò tay với hai ảnh ResNet18 sau khi trích đặc trưng cho head-only. Hai nhãn là `[3, 5]`. Giả sử sau forward, chỉ số có logit lớn nhất của từng ảnh là `[3, 2]`.

| Bước | Dữ liệu hoặc trạng thái |
| --- | --- |
| Nhận batch | `inputs` có shape `[2, 512]`, `labels` có shape `[2]`. Với fine-tune chạy từ ảnh 64×64, input tương ứng có shape `[2, 3, 64, 64]`. |
| Forward | `model(inputs)` trả logits `[2, 10]`. `cross_entropy(logits, labels)` tính loss từ logits và nhãn thật. |
| Cập nhật | Lượt train xóa gradient cũ trước forward. Sau khi kiểm tra loss hữu hạn, `loss.backward()` tính gradient, rồi `optimizer.step()` đổi các tham số mà optimizer quản lý. Lượt validation chỉ chạy forward và đo. |
| Tổng hợp | Với batch minh họa, đúng 1/2 ảnh. Nếu cả lượt chỉ có batch này, `pass_epoch()` trả `accuracy = 0.5` và loss trung bình của hai ảnh. Nếu có nhiều batch, hàm cộng theo số mẫu rồi chia cho tổng số ảnh. |

Ví dụ dùng kết quả dự đoán giả định để dò logic. Muốn biết loss cụ thể cần đủ 10 logits cho từng ảnh. [Mã `pass_epoch()`](experiment.py#L53)

## 4. Chọn checkpoint, dừng và kiểm tra run

| Quyết định | Điều kiện | Đoạn code và vị trí |
| --- | --- | --- |
| Chọn epoch trong một trial | Lưu `best.pt` nếu `val_accuracy` tăng, hoặc nếu accuracy bằng mốc tốt nhất và `val_loss` giảm. Nếu cả hai bằng nhau thì giữ epoch trước. | `if va['accuracy'] > best or (va['accuracy'] == best and va['loss'] < best_loss):` · [experiment.py, dòng 192–197](experiment.py#L192) |
| Chọn giữa nhiều trial | Lọc theo khoảng sai khác cho phép so với validation accuracy cao nhất, rồi theo validation loss thấp nhất trong nhóm còn lại. Chọn trial có thời gian fit thấp nhất trong nhóm cuối. | `best_acc = max(…)` → `best_loss = min(…)` → `return min(eligible, key=lambda r: r['train_seconds'])` · [tuning.py, dòng 79–83](tuning.py#L79) |
| Dừng sớm ở fine-tune `scheduled` | Sau khi đạt tối thiểu 10 epoch, dừng nếu đã có 7 epoch liên tiếp mà validation loss chưa giảm hơn `0.0001` so với mốc theo dõi. Epoch có loss thấp nhất có thể khác epoch được lưu làm `best.pt`. | `if va['loss'] < monitor_best-delta:` và `if epoch >= minimum and stale >= patience:` · [experiment.py, dòng 201–206](experiment.py#L201)<br>`"early_stopping": {"min_epochs": 10, "patience": 7, "min_delta": 0.0001}` · [scheduled.json, dòng 29](../../config/scheduled.json#L29) |
| Lỗi loss | Nếu loss là NaN hoặc vô cực, lượt chạy báo lỗi trước khi cập nhật weights ở batch đó. | `if not torch.isfinite(loss):` → `raise FloatingPointError(…)` · [experiment.py, dòng 67–68](experiment.py#L67) |
| Kiểm tra run | Run có trạng thái `completed`, lịch sử đủ bốn chỉ số train/validation, checkpoint, căn cứ chọn trial và kết quả test cho cả hai giai đoạn. | `row = {"epoch": epoch, "train_loss": …` · [experiment.py, dòng 172–178](experiment.py#L172)<br>`pd.DataFrame(rows).to_csv(…)` và `write_json(run / "status.json", {"status": "completed"})` · [experiment.py, dòng 558–560](experiment.py#L558) |
| Ngưỡng chất lượng | Lab02 hiện chưa quy định mức accuracy tối thiểu hay loss tối đa để đạt bài. Dùng head-only làm baseline khi đọc kết quả fine-tune. | “Head-only là baseline trên backbone pretrained.” · [walkthrough_lab_02.md, dòng 57](../../walkthrough_lab_02.md#L57)<br>“Không đặt acceptance threshold hậu nghiệm.” · [walkthrough_lab_02.md, dòng 59](../../walkthrough_lab_02.md#L59) |

## 5. Artifact và nơi lưu

### Input, output và cách kiểm tra theo giai đoạn

Graph TensorBoard ghi cấu trúc phép tính từ một ảnh mẫu trước khi fit. Learning curves dùng số đo train/validation theo epoch. Các biểu đồ test dùng dự đoán sau khi đã chọn checkpoint. Các link theo model hoặc trial trong bảng dùng ResNet18 của run đã chọn làm ví dụ.

| Giai đoạn | Input | Output và thông tin cần có | Nơi lưu | Tiêu chí kiểm tra | Đoạn code và vị trí |
| --- | --- | --- | --- | --- | --- |
| Chuẩn bị run | Config JSON, đường dẫn dữ liệu, thiết bị và seed | Bản config của run, thông tin dữ liệu đã chọn, log có run path, thiết bị và tiến độ | [config.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/config.json) · [data_summary.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/data_summary.json) · [console.log](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/console.log) | Đọc được cấu hình, thiết bị thực tế, seed và số mẫu mỗi tập. | `with (run / "console.log").open(…)` và `write_json(run / "data_summary.json", …)` · [experiment.py, dòng 543–554](experiment.py#L543) |
| Chuẩn bị model | Kiến trúc, weights, dropout, ảnh train mẫu | Cây lớp, bảng tham số, metadata tiền xử lý và graph TensorBoard | [architecture.txt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/architecture.txt) · [parameters.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/parameters.csv) · [preprocessing.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/preprocessing.json) · [input_preview/](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/input_preview/) | Bảng tham số ghi rõ trạng thái head-only. Graph và cấu hình thuộc đúng model của run. | `configure_trainable(model, name, "head")`, `writer.add_graph(…)` · [experiment.py, dòng 345–363](experiment.py#L345) |
| Fit head-only hoặc fine-tune | Hai loader train/validation, model, optimizer và lịch epoch | Mỗi epoch có train/val loss, accuracy, LR từng nhóm, thời gian epoch, thời gian train và validation riêng, throughput train. TensorBoard có metadata, scalar và histogram của tensor trainable được chọn, kèm gradient nếu có. | [Head-only history.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/head_selected/history.csv) · [Fine-tune history.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/history.csv) · [Fine-tune tensorboard/](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/tensorboard/) | Mỗi dòng CSV ứng với một epoch đã đo. Khi đọc chi phí, phân biệt `train_pass_seconds`, `validation_seconds` và tổng thời gian fit. | `writer.add_text("configuration", …)` · [experiment.py, dòng 149–150](experiment.py#L149) · `writer.add_scalar(…)`, `writer.add_histogram(…)` · [experiment.py, dòng 172–189](experiment.py#L172) |
| Chốt checkpoint hoặc resume | Metric validation, trạng thái model/optimizer và kết quả các trial | `best.pt` để đánh giá, `last.pt` để resume cuối epoch, lựa chọn trial, và summary có epoch tốt nhất, lý do dừng, số epoch thực chạy, thời gian và peak CUDA memory nếu dùng CUDA | [best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/best.pt) · [last.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/last.pt) · [selection.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/selection.json) · [training_summary.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/training_summary.json) | Checkpoint tốt nhất khớp epoch được chọn; resume dùng checkpoint của trial tương ứng. | `_save_checkpoint(…)` · [experiment.py, dòng 192–220](experiment.py#L192) · `write_json(folder / "training_summary.json", result)` · [experiment.py, dòng 224–243](experiment.py#L224) |
| Vẽ learning curves | History, training summary và ngân sách epoch của hai giai đoạn | Biểu đồ loss và accuracy theo epoch cho train/validation, tách head-only và fine-tune. Có dấu mốc checkpoint khi biết `best_epoch` và ghi lý do dừng. | [learning_curves.png](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/learning_curves.png) | Số điểm khớp số epoch trong CSV. Mốc checkpoint khớp summary. | `ax.plot(history.epoch, …)`, `save_figure(fig, destination, …)` · [reporting.py, dòng 92–121](../utils/reporting.py#L92) |
| Đánh giá test | Checkpoint đã chọn, xác suất theo lớp, nhãn thật và index test | Accuracy, loss, macro F1/precision/recall, dự đoán theo ảnh, confusion matrix và biểu đồ so sánh accuracy/macro F1 giữa các model và stage | [metrics.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/metrics.json) · [test_predictions.npz](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/test_predictions.npz) · [confusion_matrix.png](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/confusion_matrix.png) · [comparison.png](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/comparison.png) | Confusion matrix có hàng là lớp thật, cột là lớp dự đoán. Tổng số đếm bằng số ảnh test. Bảng so sánh có đủ hai stage của mỗi model. | `np.savez_compressed(…)`, `confusion_matrix(labels, predictions, …)` · [reporting.py, dòng 28–56](../utils/reporting.py#L28) · `ax.bar(names, comparison[col], …)` · [reporting.py, dòng 136–145](../utils/reporting.py#L136) |

Ví dụ dưới đây là run `20261030T052121Z_cifar10_scheduled64_b895bfc1` được registry chọn khi viết tài liệu. Khi đọc run khác, kiểm tra registry và trạng thái run trước khi đối chiếu số liệu. [Run đang được chọn](../../outputs/selected_run.json) · [Cấu hình đã chạy](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/config.json) · [Trạng thái run](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/status.json) · [Bảng kết quả test](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/comparison.csv)

| Artifact | ResNet18 | DenseNet121 |
| --- | --- | --- |
| Kết quả head-only | [head_selected/history.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/head_selected/history.csv) · [head_selected/best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/head_selected/best.pt) | [head_selected/history.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/head_selected/history.csv) · [head_selected/best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/head_selected/best.pt) |
| Kết quả fine-tune | [finetune/history.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/history.csv) · [finetune/best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/best.pt) | [finetune/history.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/finetune/history.csv) · [finetune/best.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/finetune/best.pt) |
| Trạng thái fine-tune và resume | [finetune/training_summary.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/training_summary.json) · [finetune/last.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/last.pt) | [finetune/training_summary.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/finetune/training_summary.json) · [finetune/last.pt](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/finetune/last.pt) |
| Căn cứ chọn trial | [selection.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/selection.json) | [selection.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/densenet121/selection.json) |

`best.pt` lưu checkpoint được chọn. `last.pt` lưu thêm trạng thái phục vụ resume cuối epoch và nằm trong thư mục trial đã chạy. Các link `.pt` cần checkpoint cục bộ. Sau khi clone sang máy khác, cần đồng bộ checkpoint trước khi mở link. [Hướng dẫn khôi phục](../../README.md#36-khôi-phục-khi-gián-đoạn)

## 6. Cách đọc và thảo luận với AI

1. Đọc sơ đồ rồi lần theo `run_suite()`, `run_model()` và `fit()`. Với mỗi hàm, chỉ ra dữ liệu vào, kết quả trả về, dòng thay đổi trạng thái và file được ghi.
2. Dò một batch trong `pass_epoch()`. Hỏi mỗi dòng làm gì và bỏ dòng đó thì hệ quả gì. Ví dụ nếu bỏ `optimizer.step()`, gradient vẫn được tính nhưng trọng số sẽ không được optimizer cập nhật.
3. Mở một `history.csv` cùng `training_summary.json`. Chỉ ra epoch được chọn, lý do dừng và vì sao hai quyết định đó có thể dựa trên hai mốc khác nhau.
4. Đối chiếu `selection.json` với `comparison.csv` để phân biệt chỉ số dùng chọn trial và chỉ số đo sau cùng. Ghi rõ phần nào đọc từ source, dò tay hoặc đã chạy. Sau khi duyệt nội dung bổ sung, cập nhật ngày, người cập nhật và phiên bản ở đầu file.
