# Lab02 · Module utils

| Ngày | Phiên bản | Tác giả hoặc người cập nhật | Thay đổi |
| --- | --- | --- | --- |
| 2026-11-07 | 0.1.0 | Thường FEAI, cùng Codex | Tạo hướng dẫn đọc module |
| 2026-11-07 | 0.1.1 | Thường FEAI, cùng Codex | Mô tả rõ nhiệm vụ từng file và rút gọn link source |

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

## 2. Nhiệm vụ từng file

| File | Trách nhiệm | Source chính |
| --- | --- | --- |
| `reporting.py` | Nhận xác suất dự đoán, nhãn thật và index test để tính metric, xuất báo cáo theo lớp và vẽ confusion matrix. Đọc lịch sử train/validation để vẽ loss và accuracy theo epoch, rồi tổng hợp kết quả theo model và giai đoạn. | [`export_evaluation()` · L28](reporting.py#L28) |
| `costs.py` | Đọc kết quả và lịch sử của run để tách chi phí tìm cấu hình khỏi chi phí chạy lại recipe đã chọn. Đề xuất model dựa trên validation và chỉ dùng thời gian để phân hạng khi các cấu hình có cùng kiểu thực thi. | [`cost_table()` · L7](costs.py#L7) |
| `delivery.py` | Xác định đường dẫn dữ liệu và run, ghi JSON và tính hash file. Khi finalize, kiểm tra split, dự đoán, metric và log của run hoàn thành rồi xuất bằng chứng kiểm chứng, ROC/PR và manifest. | [`finalize_run()` · L71](delivery.py#L71) |
| `notebook_views.py` | Xác định run cần trình bày, đọc bảng kết quả, lịch sử train và bằng chứng benchmark đã lưu. Từ đó dựng bảng và hình cho notebook, đồng thời lưu hình vào thư mục review của run. | [`context()` · L16](notebook_views.py#L16) |
| `training_benchmark.py` | Khôi phục model từ checkpoint và chuẩn bị một ảnh train cùng nhãn. Đo thời gian của bước forward, tính loss, backward và optimizer update trên batch đã ở thiết bị. Đo bộ nhớ ở lượt riêng. | [`benchmark_steps()` · L52](training_benchmark.py#L52) |
| `inference_benchmark.py` | Nhận một ảnh và model đã nạp ở chế độ đánh giá. Đo yêu cầu đầu tiên, các yêu cầu sau warm-up, thời gian từng công đoạn và bộ nhớ ở lượt riêng. Trả số liệu cho CLI ghi thành artifact. | [`benchmark_loaded_model()` · L113](inference_benchmark.py#L113) |
| `constants.py` | Cung cấp thứ tự tên lớp để ánh xạ chỉ số dự đoán, checksum để kiểm tra các file CIFAR-10 gốc và các hằng số chung như số lớp, số kênh, kích thước ảnh mặc định và seed. | [`CLASSES`, `CHECKSUMS` · L7](constants.py#L7) |
| `__init__.py` | Khai báo các tên có thể import từ `src.utils`. Khi caller yêu cầu một tên, file nạp module tương ứng và lưu kết quả để những lần truy cập tiếp theo dùng lại. | [`__getattr__()` · L27](__init__.py#L27) |

## 3. Input, output và nơi lưu

Các link theo run dùng run CIFAR-10 `20261030T052121Z_cifar10_scheduled64_b895bfc1` làm ví dụ. Hai benchmark có bộ bằng chứng riêng. [Registry](../../outputs/selected_run.json)

### Báo cáo và chi phí

| Giai đoạn | Input | Output và thông tin cần có | Nơi lưu | Tiêu chí kiểm tra | Đoạn code và vị trí |
| --- | --- | --- | --- | --- | --- |
| Xuất đánh giá test | Xác suất `[N, 10]`, nhãn `[N]`, index gốc `[N]` và dữ liệu test | Số mẫu, accuracy, loss, macro precision/recall/F1, dự đoán theo ảnh, báo cáo từng lớp và confusion matrix. Hình có hàng là nhãn thật, cột là dự đoán, màu theo tỷ lệ trong hàng và số đếm ghi trên ô. | [metrics.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/metrics.json) · [test_predictions.npz](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/test_predictions.npz) · [classification_report.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/classification_report.csv) · [confusion_matrix.png](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/confusion_matrix.png) | Nhãn và index khớp tập test đã chọn. Tổng số đếm bằng `N`. Metric tính lại từ dự đoán khớp file JSON. | `predictions = probabilities.argmax(1)` và `counts = confusion_matrix(labels, predictions, …)` · [reporting.py, dòng 28–56](reporting.py#L28) |
| Vẽ biểu đồ | `history.csv`, summary và cấu hình epoch, hoặc `comparison.csv` | Loss và accuracy theo epoch cho train/validation của từng stage. Biểu đồ tổng hợp so test accuracy và macro F1 giữa model và stage. | [learning_curves.png](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/learning_curves.png) · [comparison.png](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/comparison.png) | Số điểm khớp số epoch đã chạy. Khi có `best_epoch`, mốc checkpoint khớp summary. Mỗi cột so sánh khớp hàng model/stage tương ứng. | `ax.plot(history.epoch, …)` · [reporting.py, dòng 95–110](reporting.py#L95) · `ax.bar(names, comparison[col], …)` · [reporting.py, dòng 136–145](reporting.py#L136) |
| Tổng hợp chi phí và đề xuất | Bảng so sánh, summary head-only, metric validation, chính sách chọn và kiểu thực thi | Bảng thời gian trích đặc trưng, toàn bộ trial, recipe đã chọn và test riêng. Đề xuất model ghi validation accuracy/loss, chi phí recipe và cờ `time_comparable`. | [cost_comparison.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/cost_comparison.csv) · [validation_recommendation.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/validation_recommendation.json) | Chi phí pipeline cộng trích đặc trưng và các trial. Chi phí recipe cộng trích đặc trưng và các trial được chọn. Chỉ dùng thời gian để xếp hạng khi cùng kiểu thực thi. | `selected_recipe_training_seconds=extraction+selected_head+…` · [costs.py, dòng 18–25](costs.py#L18) · `comparable = len({r['execution'] for r in records}) == 1` · [costs.py, dòng 44–53](costs.py#L44) · `cost_table(run, comparison).to_csv(…)` · [reporting.py, dòng 130–135](reporting.py#L130) |

### Kiểm chứng và trình bày artifact

| Giai đoạn | Input | Output và thông tin cần có | Nơi lưu | Tiêu chí kiểm tra | Đoạn code và vị trí |
| --- | --- | --- | --- | --- | --- |
| Xác định run và dữ liệu | Thư mục lab, đường dẫn tường minh, biến cấu hình và registry | Đường dẫn tuyệt đối cho caller. Dữ liệu ưu tiên đối số, rồi `LAB02_DATA`, rồi `root/data`. CLI finalize lấy run tường minh hoặc registry. Notebook hỗ trợ `LAB02_RUN`. | Giá trị `Path` trong bộ nhớ · [selected_run.json](../../outputs/selected_run.json) | Đường dẫn trỏ tới run chủ định, độc lập với thư mục gọi CLI. | `Path(explicit or os.getenv('LAB02_DATA') or root/'data')` · [delivery.py, dòng 37–47](delivery.py#L37) · `run = Path(os.environ.get('LAB02_RUN', selected['run']))` · [notebook_views.py, dòng 28–33](notebook_views.py#L28) |
| Finalize | Run hoàn thành, cấu hình CIFAR-10, split, dự đoán và log TensorBoard | Kết quả kiểm chứng, manifest hash, ROC/PR theo lớp và bảng scalar đã trích. Các hình ROC/PR dùng xác suất theo lớp và nhãn test. | [verification.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/verification.json) · [manifest.json](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/manifest.json) · [roc_pr.png](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/roc_pr.png) · [roc_pr_metrics.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/resnet18/finetune/roc_pr_metrics.csv) · [tensorboard_scalars.csv](../../outputs/runs/20261030T052121Z_cifar10_scheduled64_b895bfc1/tensorboard_scalars.csv) | Split và dự đoán khớp chỉ số đã chọn, xác suất hữu hạn và tổng mỗi hàng gần 1, metric tính lại khớp, manifest kiểm tra được. | `np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)` · [delivery.py, dòng 126–135](delivery.py#L126) · `fig.savefig(folder / 'roc_pr.png', …)` · [delivery.py, dòng 166–168](delivery.py#L166) · `write_json(run / "manifest.json", manifest)` · [delivery.py, dòng 285–286](delivery.py#L285) |
| Hiển thị notebook | Artifact của run và bằng chứng benchmark được chỉ định | Bảng hiển thị và các hình trình bày, ví dụ chất lượng/chi phí, đường học hoặc benchmark. Notebook lấy số đo từ artifact có sẵn. | [quality_cost.png](../../outputs/review/20261030T052121Z_cifar10_scheduled64_b895bfc1/quality_cost.png) · [learning_curves_resnet18.png](../../outputs/review/20261030T052121Z_cifar10_scheduled64_b895bfc1/learning_curves_resnet18.png) · [single_image_benchmark.png](../../outputs/review/20261030T052121Z_cifar10_scheduled64_b895bfc1/single_image_benchmark.png) | Hình và bảng truy được về run hoặc bộ benchmark tương ứng. Khi đổi run, dùng đúng thư mục review của run mới. | `save_figure(fig, output / f'{name}.png', …)` · [notebook_views.py, dòng 36–42](notebook_views.py#L36) · `def single_image_benchmark(…)` · [notebook_views.py, dòng 375](notebook_views.py#L375) |

`complete_delivery()` bổ sung kiểm thử, finalize, thực thi hai notebook và kiểm tra inference. Nó ghi log và bổ sung bằng chứng vào run. [delivery.py, dòng 308–370](delivery.py#L308)

### Benchmark

| Giai đoạn | Input | Output và thông tin cần có | Nơi lưu | Tiêu chí kiểm tra | Đoạn code và vị trí |
| --- | --- | --- | --- | --- | --- |
| Đo một bước training | Checkpoint fine-tune, config, một ảnh train đã chuẩn hóa, nhãn và thiết bị | Mẫu thời gian bước `forward → loss → backward → optimizer.step()`, median/mean/P95, RSS và CUDA peak khi có. Config và provenance ghi seed, batch size, warm-up, thiết bị và hash checkpoint. | [methodology.json](../../outputs/published/training_benchmark/scheduled64_cpu_batch1/methodology.json) · [config.json](../../outputs/published/training_benchmark/scheduled64_cpu_batch1/config.json) · [measurement.json](../../outputs/published/training_benchmark/scheduled64_cpu_batch1/resnet18/measurement.json) · [training_performance.csv](../../outputs/published/training_benchmark/scheduled64_cpu_batch1/training_performance.csv) · [training_resources.png](../../outputs/published/training_benchmark/scheduled64_cpu_batch1/training_resources.png) · [provenance.json](../../outputs/published/training_benchmark/scheduled64_cpu_batch1/provenance.json) | Hai model nhận cùng chỉ số ảnh và cùng hash tensor. Batch đã ở trên thiết bị. Số đo theo bước và loại trừ tải model, tiền xử lý, truyền dữ liệu và validation. | `loss = torch.nn.functional.cross_entropy(model(images), targets)` · [training_benchmark.py, dòng 68–73](training_benchmark.py#L68) · `writer.writerows(rows)` · [scripts/benchmark_training.py, dòng 76–87](../../scripts/benchmark_training.py#L76) |
| Đo inference một ảnh | Model đã nạp ở chế độ eval, ảnh PNG, số lượt warm-up/lặp và cấu hình lấy mẫu bộ nhớ | Latency yêu cầu đầu, phân phối sau warm-up, số đo từng công đoạn, dự đoán, RSS và CUDA allocated/reserved khi có. Tổng latency gồm đọc ảnh đến tạo chuỗi JSON, loại trừ tải model và ghi kết quả ra đĩa. | [config.json](../../outputs/published/single_image_benchmark/20261102T184034Z_kaggle/resnet18/config.json) · [environment_snapshot.json](../../outputs/published/single_image_benchmark/20261102T184034Z_kaggle/resnet18/environment_snapshot.json) · [latency_samples.csv](../../outputs/published/single_image_benchmark/20261102T184034Z_kaggle/resnet18/latency_samples.csv) · [stage_samples.csv](../../outputs/published/single_image_benchmark/20261102T184034Z_kaggle/resnet18/stage_samples.csv) · [summary.json](../../outputs/published/single_image_benchmark/20261102T184034Z_kaggle/resnet18/summary.json) · [manifest.json](../../outputs/published/single_image_benchmark/20261102T184034Z_kaggle/resnet18/manifest.json) | Kiểm tra ảnh và checkpoint qua hash, đúng batch 1, FP32, thiết bị và số lượt đo. Đọc thời gian các công đoạn cùng phương pháp đo vì chúng được thu riêng với tổng latency. | `elapsed, prediction, _ = _timed_request(session, image)` · [inference_benchmark.py, dòng 130–141](inference_benchmark.py#L130) · `write_json(run/'summary.json', summary)` · [scripts/benchmark_inference.py, dòng 107–128](../../scripts/benchmark_inference.py#L107) |

## 4. Tiêu chí hoàn thành và cách đọc

1. **Truy được output về input.** Từ metric hoặc hình, xác định đúng run, model và stage. Từ benchmark, xác định thêm thiết bị, cấu hình đo và checkpoint. Kiểm tra trạng thái và manifest trước khi dùng số liệu.
2. **Dò một dự đoán.** Nếu nhãn thật của một ảnh là `3` và điểm cao nhất thuộc lớp `5`, ảnh đó tăng một đếm ở hàng `3`, cột `5` của confusion matrix. Đây là ví dụ dò tay. [Tính lớp dự đoán và ma trận](reporting.py#L30)
3. **Dò một hàng chi phí.** Ví dụ giả định trích đặc trưng 10 giây, tìm head 8 giây, head được chọn 3 giây, tìm fine-tune 20 giây và fine-tune được chọn 12 giây. Chi phí pipeline là 38 giây, còn recipe được chọn là 25 giây. Đối chiếu hai công thức trước khi diễn giải biểu đồ. [Hai cột chi phí](costs.py#L18)
4. **Giữ đúng phạm vi đo.** Thời gian một bước training, thời gian fit cả run và latency một yêu cầu inference có đầu vào và phạm vi đo khác nhau. Khi trao đổi với AI, yêu cầu chỉ ra file, dòng thực thi và artifact chứng minh từng số liệu. Ghi ngày, người cập nhật và phiên bản sau khi nội dung bổ sung được duyệt.
