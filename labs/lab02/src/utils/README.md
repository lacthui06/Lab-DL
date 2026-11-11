# Lab02 · Module utils

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-07 | 0.1.0 | Thường FEAI, cùng Codex | Tạo hướng dẫn đọc module |
| 2026-11-07 | 0.1.1 | Thường FEAI, cùng Codex | Mô tả rõ nhiệm vụ từng file và rút gọn link source |
| 2026-11-11 | 0.1.2 | Thường FEAI, cùng Codex | Tổ chức theo mẫu utils Lab01; giải thích cơ chế báo cáo, kiểm chứng và phạm vi benchmark theo source và artifact Lab02 |

## 1. Pipeline và mục đích

```text
Dự đoán, nhãn và lịch sử train
  → reporting + costs → metric, biểu đồ, chi phí và đề xuất theo validation

Run đã chọn
  → delivery → kiểm chứng, báo cáo và manifest
  → notebook_views → đọc bằng chứng và xuất hình trình bày

Checkpoint và một mẫu
  ├─ training_benchmark → số đo một bước training → bảng và hình benchmark
  └─ inference_benchmark → số đo một yêu cầu dự đoán → mẫu đo và summary
```

Module gom các chức năng để kiểm tra, trình bày và đo kết quả của Lab02. Training gọi phần báo cáo sau khi đánh giá các checkpoint. Notebook đọc các artifact đã có và có thể tạo hình trình bày. Các phép đo benchmark do CLI riêng thực thi. [Training gọi xuất đánh giá](../training/experiment.py#L429) · [Điểm vào finalize](../../scripts/finalize.py#L6)

## 2. Mục Đích (Purpose)

Lab02 khảo sát hai mô hình ResNet18 và DenseNet121 trên CIFAR-10, với hai giai đoạn: chỉ học bộ phân loại cuối (head-only) và mở một phần backbone để học tiếp (fine-tune). Module utils giúp chuyển kết quả của các giai đoạn này thành bằng chứng có thể đọc, kiểm tra và đối chiếu. [Mục tiêu thực nghiệm](reporting.py#L170)

Các công cụ được tổ chức để trả lời ba câu hỏi: mô hình dự đoán thế nào, cần bao nhiêu chi phí thực nghiệm và các số liệu có truy được về đúng lần chạy hay không. Artifact là các tệp được lưu sau khi chạy, chẳng hạn bảng metric, xác suất dự đoán, biểu đồ và cấu hình.

Module còn hỗ trợ đo chi phí của một bước training và một yêu cầu inference. Hai phép đo có phạm vi khác nhau, nên phải đọc cùng cấu hình và phương pháp đo trước khi so sánh.

## 3. Khởi Tạo & Các Công Cụ Nền Tảng (Init Utils & Toolset)

### 3.1. Cấu Trúc Các Nhóm Công Cụ Hỗ Trợ (Toolset Structure)

```text
1. Hằng số & Truy cập tiện ích:
   constants + __init__ → tên lớp, checksum và nạp module khi được yêu cầu.

2. Báo cáo chất lượng & Chi phí:
   reporting → metric, confusion matrix, ảnh dự đoán và đường học.
   costs → chi phí toàn bộ tìm kiếm, chi phí recipe đã chọn và đề xuất theo validation.

3. Kiểm chứng & Bàn giao:
   delivery → xác định run, kiểm tra bằng chứng, xuất ROC/PR và manifest hash.

4. Trình bày trong notebook:
   notebook_views → đọc artifact của run và lưu hình vào outputs/review/<run_id>/.

5. Đo chi phí tính toán:
   training_benchmark → đo một bước cập nhật trọng số trên một ảnh.
   inference_benchmark → đo một yêu cầu dự đoán với model đã nạp.
```

`resolve_run()` lấy run được truyền tường minh hoặc đọc registry; notebook cho phép chọn qua `LAB02_RUN`. Cách này giúp gắn bảng và hình với một run xác định, thay vì tự lấy thư mục mới nhất. [Đường dẫn run](delivery.py#L42) · [Ngữ cảnh notebook](notebook_views.py#L16)

Registry hiện trỏ tới `20261030T052121Z_cifar10_scheduled64_b895bfc1`. Config lưu trong run ghi ảnh 64×64, batch head 128 và ngân sách 12 epoch head / tối đa 20 epoch fine-tune. Đây là cấu hình của run; số epoch thực chạy phải đọc từ history và summary, không suy ra từ ngân sách. [Registry](../../outputs/selected_run.json) · [Config đã lưu](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/config.json) · [Cách đọc epoch](reporting.py#L95)

### 3.2. Đầu Vào và Đầu Ra của Các Tiện Ích (Input / Output Utils)

Trong bảng, `<run>` là thư mục run được chọn; `<model>/<stage>` xác định mô hình và giai đoạn. Thư mục kết quả head-only mang tên `head_selected`, còn fine-tune mang tên `finetune`.

| Công cụ / Tiện ích | Dữ liệu đầu vào tiếp nhận | Sản phẩm đầu ra (Artifact Path) | Ý nghĩa sử dụng |
| :--- | :--- | :--- | :--- |
| Báo cáo đánh giá (`reporting`) | Xác suất `[N, 10]`, nhãn thật `[N]`, index gốc `[N]` và dữ liệu test | `<run>/<model>/<stage>/metrics.json`, `test_predictions.npz`, `classification_report.csv`, `confusion_counts.csv`, `confusion_matrix.png` | Tính accuracy, loss và macro precision/recall/F1; lưu dự đoán để kiểm tra lại. Macro là trung bình các chỉ số tính riêng cho từng lớp, với trọng số ngang nhau. |
| Đồ thị học tập (`reporting`) | `history.csv`, `training_summary.json` và config của run | `<run>/<model>/learning_curves.png` | Tách head-only và fine-tune thành hai hàng, mỗi hàng có loss và accuracy của train/validation; đánh dấu checkpoint được chọn khi có dữ liệu. |
| Chi phí và đề xuất (`costs`) | Bảng so sánh, summary head, kết quả validation và chính sách chọn | `<run>/cost_comparison.csv`, `validation_recommendation.json` | Phân biệt chi phí toàn bộ tìm kiếm với chi phí thực hiện recipe đã chọn; chỉ dùng thời gian để phân hạng khi kiểu thực thi trùng nhau. |
| Kiểm chứng (`delivery`) | Run hoàn thành, config, split, dự đoán, metric, roundtrip và log TensorBoard | `<run>/verification.json`, `manifest.json`, `tensorboard_scalars.csv`; `<model>/<stage>/roc_pr.png`, `roc_pr_metrics.csv` | Đối chiếu bằng chứng và tạo manifest chứa mã hash của tệp. ROC/PR One-vs-Rest đánh giá lần lượt từng lớp so với các lớp còn lại. |
| Trình bày (`notebook_views`) | Artifact của run và bộ bằng chứng benchmark được chỉ định | `outputs/review/<run_id>/*.png` và bảng hiển thị trong notebook | Dựng hình trình bày từ dữ liệu đã lưu; các hàm xem benchmark đọc số đo có sẵn. |
| Benchmark training | Checkpoint fine-tune, config và một ảnh train cùng nhãn đã đưa lên thiết bị | `outputs/published/training_benchmark/<benchmark_id>/`: `measurement.json` theo model, `training_performance.csv`, `training_resources.png`, config và provenance | Đo thời gian bước cập nhật trọng số và bộ nhớ ở lượt riêng; không đại diện cho thời gian huấn luyện cả mô hình. |
| Benchmark inference | Model đã nạp ở chế độ eval, một ảnh PNG và cấu hình đo | `outputs/published/single_image_benchmark/<benchmark_id>/<model>/`: `summary.json`, `latency_samples.csv`, `stage_samples.csv`, config và manifest | Đo yêu cầu đầu tiên, thời gian sau warm-up và bộ nhớ ở lượt riêng; không bao gồm thời gian tải model. |

### 3.3. Tệp Mã Nguồn & Bản Đồ Điều Hướng Code (Source Files)

Giữ phím **`Ctrl + Click`** để mở ký hiệu hoặc vị trí mã nguồn trong trình soạn thảo.

| Tệp mã nguồn | Ký hiệu / Hàm chính (Ctrl+Click) | Dòng code | Việc đảm nhiệm |
| :--- | :--- | :--- | :--- |
| [`reporting.py`](reporting.py) | [`export_evaluation()`](reporting.py#L28)<br>[`plot_learning_curves()`](reporting.py#L80)<br>[`write_final_report()`](reporting.py#L214) | [L28](reporting.py#L28), [L80](reporting.py#L80), [L214](reporting.py#L214) | Xuất metric, dự đoán và biểu đồ; tổng hợp báo cáo Markdown/HTML với phần chẩn đoán do delivery cung cấp. |
| [`costs.py`](costs.py) | [`cost_table()`](costs.py#L7)<br>[`validation_recommendation()`](costs.py#L29) | [L7](costs.py#L7), [L29](costs.py#L29) | Tính hai phạm vi chi phí và đề xuất mô hình dựa trên validation, có xét khả năng so sánh thời gian. |
| [`delivery.py`](delivery.py) | [`resolve_run()`](delivery.py#L42)<br>[`finalize_run()`](delivery.py#L71)<br>[`complete_delivery()`](delivery.py#L308) | [L42](delivery.py#L42), [L71](delivery.py#L71), [L308](delivery.py#L308) | Xác định run, kiểm chứng artifact và tạo bằng chứng bàn giao; luồng đầy đủ còn chạy kiểm thử, notebook và kiểm tra inference. |
| [`notebook_views.py`](notebook_views.py) | [`context()`](notebook_views.py#L16)<br>[`training_benchmark()`](notebook_views.py#L191)<br>[`single_image_benchmark()`](notebook_views.py#L375) | [L16](notebook_views.py#L16), [L191](notebook_views.py#L191), [L375](notebook_views.py#L375) | Chọn ngữ cảnh trình bày, đọc artifact và dựng bảng/hình cho notebook. |
| [`training_benchmark.py`](training_benchmark.py) | [`prepare_training()`](training_benchmark.py#L29)<br>[`benchmark_steps()`](training_benchmark.py#L52)<br>[`measure_checkpoint()`](training_benchmark.py#L105) | [L29](training_benchmark.py#L29), [L52](training_benchmark.py#L52), [L105](training_benchmark.py#L105) | Khôi phục model và chính sách train từ checkpoint, chuẩn bị một ảnh rồi đo thời gian và bộ nhớ của bước training. |
| [`inference_benchmark.py`](inference_benchmark.py) | [`timing_summary()`](inference_benchmark.py#L19)<br>[`_RSSSampler`](inference_benchmark.py#L41)<br>[`benchmark_loaded_model()`](inference_benchmark.py#L113) | [L19](inference_benchmark.py#L19), [L41](inference_benchmark.py#L41), [L113](inference_benchmark.py#L113) | Tổng hợp phân phối thời gian, lấy mẫu RAM của tiến trình và đo yêu cầu dự đoán với model đã nạp. |
| [`constants.py`](constants.py) | [`CLASSES`](constants.py#L7)<br>[`CHECKSUMS`](constants.py#L20) | [L7](constants.py#L7), [L20](constants.py#L20) | Giữ thứ tự 10 lớp CIFAR-10, checksum dữ liệu và các hằng số chung. |
| [`__init__.py`](__init__.py) | [`__getattr__()`](__init__.py#L27) | [L27](__init__.py#L27) | Nạp module khi tên được yêu cầu, rồi lưu kết quả để các lần truy cập sau dùng lại. |

## 4. Các Thảo Luận & Quyết Định Kỹ Thuật (Discussions from Description)

### 4.1. Vì sao phải tách chi phí tìm kiếm khỏi chi phí recipe đã chọn?

Mục tiêu là phân biệt công sức tìm cấu hình với chi phí thực hiện lại cấu hình đã chọn. Recipe ở đây là tổ hợp thiết lập huấn luyện, chẳng hạn optimizer, learning rate và batch size.

`cost_table()` tính chi phí toàn pipeline bằng thời gian trích đặc trưng cộng toàn bộ trial head và fine-tune. Chi phí recipe chỉ cộng trích đặc trưng, head được chọn và fine-tune được chọn nếu có; thời gian test nằm ở cột riêng. [Công thức chi phí](costs.py#L12)

Config hiện tại `scheduled.json` và config lưu trong run được chọn đều ghi một learning rate head `0.001` và một batch size `128`. Vì vậy, không mô tả run này như đã khảo sát nhiều tổ hợp LR/batch chỉ vì module có khả năng tổng hợp search. [Config hiện tại](../../config/scheduled.json) · [Config của run](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/config.json)

Đề xuất mô hình dùng validation; test phục vụ đánh giá sau lựa chọn. Nếu các mô hình có kiểu thực thi khác nhau, hàm đề xuất xếp theo validation accuracy rồi loss và ghi `time_comparable=false`. Điều này tránh dùng thời gian từ hai cách thực thi khác nhau làm căn cứ phân hạng. [Chính sách đề xuất](costs.py#L29)

### 4.2. Vì sao confusion matrix dùng màu theo tỷ lệ nhưng vẫn ghi số đếm?

Mục tiêu là thấy cả lớp nào bị nhầm nhiều và có bao nhiêu ảnh bị nhầm. Mỗi hàng tương ứng nhãn thật, mỗi cột tương ứng nhãn dự đoán.

`export_evaluation()` tính ma trận số đếm rồi chuẩn hóa từng hàng để tô màu. Chữ trên ô vẫn là số ảnh thực tế; `confusion_counts.csv` giữ số đếm để đối chiếu. Nếu nhãn thật là `cat` nhưng dự đoán là `dog`, số đếm ở hàng `cat`, cột `dog` tăng một. [Cách dựng ma trận](reporting.py#L42)

Lưới dự đoán lấy tối đa 8 ảnh đúng và 8 ảnh sai đầu tiên để minh họa. Đây là lựa chọn có chủ đích, nên tỷ lệ đúng trong lưới không đại diện cho accuracy của toàn tập test. [Cách chọn ảnh](reporting.py#L58)

### 4.3. Vì sao benchmark training và inference phải có giao thức riêng?

Benchmark training khảo sát chi phí cập nhật trọng số: `forward → loss → backward → optimizer.step()`. Ảnh và nhãn đã ở trên thiết bị trước khi đo; tải checkpoint, tiền xử lý, truyền dữ liệu và validation nằm ngoài phạm vi. Optimizer được tạo mới ở learning rate ban đầu của recipe, nên đây không phải phép tiếp tục nguyên trạng một phiên train. [Bước đo](training_benchmark.py#L52) · [Phương pháp đã lưu](../../outputs/published/training_benchmark/scheduled64_cpu_batch1/methodology.json)

Config benchmark training hiện tại và config đã lưu đều ghi CPU, batch 1, seed 42, 2 luồng, 3 bước warm-up và 10 bước đo. Warm-up là các lượt chạy trước khi thu thời gian chính thức. CLI tạo tiến trình riêng cho từng model và kiểm tra hai model nhận cùng index ảnh, cùng hash tensor đầu vào. [Config hiện tại](../../config/training_benchmark.json) · [Config đã lưu](../../outputs/published/training_benchmark/scheduled64_cpu_batch1/config.json) · [CLI đo](../../scripts/benchmark_training.py#L66)

Benchmark inference đo từ đọc/giải mã PNG đến tạo chuỗi JSON, với model đã nạp ở chế độ eval. Config ResNet18 trong bộ bằng chứng `20261102T184034Z_kaggle` ghi ảnh 64×64, batch 1, FP32, 20 lượt warm-up và 100 lượt đo tổng latency; lượt đo từng công đoạn và bộ nhớ đều là 20. Đây là cấu hình đã lưu của bộ đo đó. [Config đã lưu](../../outputs/published/single_image_benchmark/20261102T184034Z_kaggle/resnet18/config.json) · [Giao thức inference](inference_benchmark.py#L113)

Thời gian từng công đoạn được thu ở lượt riêng, nên không cộng chúng để suy ra tổng latency. RSS là RAM đang được tiến trình sử dụng, có cả môi trường chạy; lấy mẫu có thể bỏ lỡ đỉnh ngắn. CUDA allocated/reserved chỉ phản ánh bộ cấp phát của PyTorch. Các số đo trên một ảnh không chứng minh tốc độ trên mọi đầu vào hoặc lượng RAM/VRAM tối thiểu cần có. [Giới hạn phép đo](inference_benchmark.py#L113)

### 4.4. Vì sao cần kiểm chứng trước khi dùng artifact trong báo cáo?

Một bảng metric chỉ có ý nghĩa khi gắn đúng split, mô hình và checkpoint. `finalize_run()` đối chiếu index test, nhãn gốc, xác suất hữu hạn, tổng xác suất mỗi hàng và accuracy/macro F1 tính lại; đồng thời kiểm tra bằng chứng lưu/nạp và tổng số đếm confusion matrix. [Các bước đối chiếu](delivery.py#L121)

Sau đó, hàm xuất ROC/PR, bằng chứng TensorBoard, kết quả kiểm chứng và manifest hash. Manifest giúp phát hiện tệp thay đổi so với thời điểm lập manifest; nó không tự chứng minh thiết kế thí nghiệm là đúng. [Xuất ROC/PR](delivery.py#L149) · [Lập manifest](delivery.py#L283)

Đọc notebook từ artifact đã lưu và chạy lại luồng bàn giao là hai nhu cầu khác nhau. `complete_delivery()` còn chạy kiểm thử, thực thi notebook và kiểm tra inference, nên cần dữ liệu/checkpoint tương ứng. Việc đã đọc được hình không đồng nghĩa toàn bộ luồng này vừa được kiểm chứng lại. [Luồng bàn giao đầy đủ](delivery.py#L308)
