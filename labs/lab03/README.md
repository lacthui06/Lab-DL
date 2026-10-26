# Lab 03 — Getting started with Hugging Face

Bài thực hành Deep Learning — UTH: sentiment analysis bằng mô hình pretrained và fine-tuning mô hình phân loại văn bản nhị phân bằng Hugging Face Trainer.

**Kết quả lịch sử 3 epoch đã kiểm chứng:** dùng toàn bộ Rotten Tomatoes, 8.530 train / 1.066 validation / 1.066 test. Test Accuracy **83,96%**, Binary F1 **84,06%**. Mã nguồn, notebook và báo cáo dùng chung run registry, có thể xem kết quả mà không train lại.

**Study 10 epoch đã chạy và kiểm chứng:** [Notebook kết quả, hình và thảo luận](notebooks/05_study_results_and_discussion.ipynb). Ba LR đều hoàn thành 10 epoch; winner LR **3e-5**, best checkpoint **step 1700 / epoch 3,184**, threshold validation **0,5**. Test Accuracy **83,40%**, Macro-F1 **83,40%**, đúng **889/1.066**. Hơn baseline TF-IDF nhưng thấp hơn argmax lịch sử 6 mẫu; không tuyên bố chạy nhiều epoch luôn tốt hơn. [Notebook 04](notebooks/04_kaggle_10epoch_study.ipynb) là template để chạy thí nghiệm GPU mới.

## 1. Xem bài và kết quả

- [Notebook kết quả study 10 epoch](notebooks/05_study_results_and_discussion.ipynb): audit, đủ learning curves, threshold, baseline, test và thảo luận.
- [Notebook Exercise 1](notebooks/01_sentiment_analysis_pipeline.ipynb): tokenization, pipeline và manual forward.
- [Notebook Exercise 2](notebooks/02_finetuning_binary_classification.ipynb): dữ liệu, Trainer, metrics, threshold, biểu đồ và phân tích lỗi.
- [Notebook tích hợp](notebooks/lab03_huggingface_complete.ipynb): cả hai Exercise, phù hợp xem toàn bộ bài.
- [Báo cáo thực hành](walkthrough_lab_03.md): lý thuyết, phương pháp, kết quả và giới hạn có bằng chứng.
- [Đề bài](description_lab_03.md).

Các notebook báo cáo đã lưu output của một lượt thực thi **`MODE="view"`** trên artifacts Kaggle. Notebook 04 là template GPU, được renderer offline bỏ qua để tránh vô tình train. Có thể đọc trực tiếp trên GitHub. Khi chạy lại notebook, mặc định view đọc dữ liệu đã commit, không tải model/dataset, không huấn luyện và không cần GPU. Đổi `MODE="train"` ở cell setup để tạo thí nghiệm mới; phần fine-tuning yêu cầu GPU theo mặc định.

## 2. Cấu trúc

```text
labs/lab03/
  config/
    config.json             tham số cho lần chạy mới
    workflow.json           active runs và lịch sử
    study.json              thiết kế 10 epoch, LR grid và baseline
  notebooks/                ba notebook lịch sử có output và notebook Kaggle 10 epoch
  src/
    datasets/               tải official splits, kiểm tra kích thước và tokenization
    models/                 model, pipeline/manual, kiểm tra checkpoint
    training/               Trainer, metrics, threshold và sanity overfit dùng riêng trong tests
    utils/                  log, registry, manifest và plots
    workflow.py             luồng CLI/notebook dùng chung
    evidence.py             audit đọc-only, tính lại metrics từ dự đoán
  scripts/                  train, eval, infer, audit và recovery
  tests/                    tests chức năng và kiểm tra artifacts thật
  outputs/runs/             artifacts Kaggle nhẹ, giữ nguyên manifest gốc
  validation/               kết quả audit và thực thi notebook tại môi trường kiểm chứng
  description_lab_03.md     đề bài, thảo luận và quyết định
  walkthrough_lab_03.md     báo cáo thực nghiệm để nộp
```

Cấu hình và kết quả nằm trong `labs/lab03/config/` và `labs/lab03/outputs/`, dùng chung một registry. Các byte artifacts gốc không bị chỉnh sửa; 86 file nguồn mới hiện có khớp manifest; raw logs và checkpoint được kê khai riêng.

## 3. Cài đặt

Yêu cầu `Requires-Python` giao nhau của các pin trực tiếp là **>=3.10**; tính sẵn có của wheel/phụ thuộc bắc cầu còn tùy nền tảng. Quy trình hỗ trợ đã kiểm chứng dùng Python **3.12**: đã kiểm chứng CPU Linux 3.12.14 và run Kaggle ghi nhận 3.12.15. Dùng môi trường riêng, tránh hạ cấp thư viện nền Kaggle.

Từ gốc repo, trên Linux/Kaggle:

```bash
python -m pip install uv
uv python install 3.12
uv venv --python 3.12 .venv-lab03
uv pip install --python .venv-lab03/bin/python -r labs/lab03/requirements.txt
uv pip check --python .venv-lab03/bin/python
```

Trên Windows dùng `.venv-lab03\Scripts\python.exe` thay cho `.venv-lab03/bin/python`. Không cần kích hoạt nếu gọi trực tiếp interpreter.

Trên Kaggle, bật **Settings -> Accelerator -> GPU** và **Internet** trước training; session restart sẽ mất biến Python và cần chạy lại các cell setup. Có thể đặt venv ngoài repo tại `/kaggle/working/lab03-py312` như phiên thực nghiệm đã chạy.

`torch==2.6.0` bản Linux PyPI cung cấp CUDA build phù hợp của phiên đã kiểm chứng. Nếu cần cài wheel cụ thể trước requirements:

```bash
# CUDA 12.4: cần driver hỗ trợ; kiểm tra bằng nvidia-smi
uv pip install --python .venv-lab03/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# Hoặc máy chỉ CPU dùng https://download.pytorch.org/whl/cpu
```

Requirements pin dependencies trực tiếp, chưa là lockfile toàn bộ dependency bắc cầu. Không cần torchvision. Dataset và model công khai không cần token; mạng phải truy cập được Hugging Face và các CDN/Xet chuyển hướng. Không tắt kiểm tra TLS khi gặp lỗi tải.

## 4. Kiểm thử và audit (không cần GPU)

```bash
cd labs/lab03
../../.venv-lab03/bin/python -m pytest -q
cd ../..
.venv-lab03/bin/python labs/lab03/scripts/audit_outputs.py
```

- **75 tests đạt trên CPU** (bản tích hợp kết quả, xem `validation/study_completion.json`; preflight trước training ở `validation/ten_epoch_readiness.json`): dataset contract, dynamic padding, pipeline/manual, checkpoint, sanity overfit, Trainer, run tracking, năm plots, recovery, metrics/artifact audit và phát hiện dữ liệu bị sửa.
- Audit kiểm tra 51 file khoa học hiện hành, tính lại validation sweep, test metrics và confusion matrices. Selection dùng validation phase trong training run. Báo cáo audit đầy đủ được lưu trong một run `artifact_audit` dưới `outputs/`, kèm bản tiện tra cứu tại `validation/`.
- 30 file checkpoint không có trong Git được liệt kê rõ là thiếu có chủ đích, không coi là đã kiểm tra tại môi trường kiểm chứng.
- Ba notebook chạy hết ở view, Hugging Face offline, không có cell lỗi. Xem `validation/notebook_execution.json`.

Tests có fixture nhỏ để kiểm tra pipeline; chúng không phải benchmark Rotten Tomatoes. Audit thực dùng `all_predictions.json` từ Kaggle và không suy luận lại từ weights.

## 5. Chạy thí nghiệm mới trên GPU

Từ gốc repo, dùng interpreter của môi trường vừa cài:

```bash
.venv-lab03/bin/python labs/lab03/scripts/run_exercise1.py
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false .venv-lab03/bin/python labs/lab03/scripts/train_finetune.py
```

Trên Kaggle có thể gọi các lệnh bằng `subprocess.run([...], cwd=REPO, env=RUN_ENV, check=True)`, trong đó `RUN_ENV['CUDA_VISIBLE_DEVICES']='0'`; không dùng Python mặc định của notebook nếu requirements nằm trong venv khác.

Lệnh `train_finetune.py` chạy một cấu hình đủ **10 epoch** với LR 2e-5. Quy trình mở rộng khuyến nghị dùng notebook `04_kaggle_10epoch_study.ipynb` hoặc:

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-lab03/bin/python labs/lab03/scripts/run_study.py
.venv-lab03/bin/python labs/lab03/scripts/package_study.py
```

Study thử LR 1e-5, 2e-5, 3e-5; **mỗi LR đủ 10 epoch**, tổng 30 epoch giữa ba candidate. Log mỗi 25 step, validation/checkpoint mỗi 100 step và cuối epoch. Chọn checkpoint/LR bằng validation Macro-F1, sau đó dùng validation chọn threshold, cuối cùng đánh giá test. Mỗi candidate chỉ có validation predictions, không có test predictions. Không chạy full study trên CPU.

Batch train/eval 16/32, max length 128, AdamW, weight decay 0.01, warmup 10%, linear scheduler, clip gradient 1.0, FP16 trên CUDA, dynamic padding bội 8. Lưu model-only checkpoint để giảm dung lượng; không resume optimizer. Mỗi lần chạy tạo run mới. Sanity test dùng mô hình riêng, không làm thay đổi weights pretrained của thí nghiệm chính.

Có thể thực thi ba notebook với MODE=train thay CLI. Không chạy notebook tích hợp và notebook từng bài đồng thời nếu chỉ muốn một lần thí nghiệm.

### Chạy từ notebook Kaggle mới

Bật GPU và Internet trước. Cell sau clone đúng nhánh `lab03`, tạo Python 3.12 riêng và dùng cùng interpreter cho kiểm tra/training.

```python
import os
import sys
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/lacthui06/Lab-DL.git"
REPO = Path("/kaggle/working/Lab-DL")
ENV = Path("/kaggle/working/lab03-py312")
PYTHON = str(ENV / "bin/python")

if not REPO.exists():
    subprocess.run(
        ["git", "clone", "--branch", "lab03", "--single-branch", REPO_URL, str(REPO)],
        check=True,
    )
# Nếu đã clone từ trước, kiểm tra thay đổi rồi git pull --ff-only origin lab03.
assert (REPO / "labs/lab03/requirements.txt").is_file()

RUN_ENV = os.environ.copy()
RUN_ENV["CUDA_VISIBLE_DEVICES"] = "0"
RUN_ENV["TOKENIZERS_PARALLELISM"] = "false"
RUN_ENV["PYTHONUNBUFFERED"] = "1"

def run(*args):
    return subprocess.run(list(args), cwd=REPO, env=RUN_ENV, check=True)

run(sys.executable, "-m", "pip", "install", "uv")
UV = [sys.executable, "-m", "uv"]
run(*UV, "python", "install", "3.12")
if not Path(PYTHON).exists():
    run(*UV, "venv", "--python", "3.12", str(ENV))
run(*UV, "pip", "install", "--python", PYTHON,
    "-r", str(REPO / "labs/lab03/requirements.txt"))
run(*UV, "pip", "check", "--python", PYTHON)
run(PYTHON, "-c", "import torch; print(torch.__version__); "
    "assert torch.cuda.is_available(), 'Enable GPU in Kaggle Settings'; "
    "print(torch.cuda.get_device_name(0))")
run(PYTHON, "-m", "pytest", "labs/lab03/tests", "-q", "-c", "labs/lab03/pytest.ini")
run(PYTHON, "-u", "labs/lab03/scripts/run_exercise1.py")
run(PYTHON, "-u", "labs/lab03/scripts/train_finetune.py")
```

Đây là một lần train mới; chỉ chạy khi cần làm thí nghiệm lại. Đọc kết quả đã công bố hoặc bổ sung threshold từ checkpoint cũ không cần train lại. Sau train, `config/workflow.json` trỏ đến run mới; không ghi đè run Kaggle tham chiếu. Muốn lưu lại toàn bộ cell output, đăng ký kernel của môi trường này rồi render notebook view:

```python
run(PYTHON, "-m", "ipykernel", "install", "--user", "--name", "lab03")
run(PYTHON, "labs/lab03/scripts/render_notebooks.py", "--kernel", "lab03")
```

## 6. Kết quả lịch sử 3 epoch và truy vết

| Giai đoạn | Run được registry chọn |
|---|---|
| Exercise 1 | `20261026T155744Z_exercise1_65dfe784` |
| Fine-tuning | `20261026T155822Z_finetuning_2c2f0401` |

| Test metric | Giá trị |
|---|---:|
| Accuracy | 83,9587% |
| Binary Precision / Recall / F1 | 83,5185% / 84,6154% / 84,0634% |
| Macro Precision / Recall / F1 | 83,9646% / 83,9587% / 83,9580% |
| Đúng / tổng | 895 / 1.066 |

Confusion matrix `[TN FP; FN TP]` = `[[444,89],[82,451]]`. Checkpoint epoch 3 được chọn theo validation accuracy 84,33%; epoch 2 có loss validation thấp hơn. Có dấu hiệu overfitting; đề không đặt ngưỡng accuracy 85%.

Nguồn: [metrics](outputs/published/finetuning/test_metrics.json), [toàn bộ dự đoán](outputs/published/finetuning/all_predictions.json), [lịch sử](outputs/published/finetuning/trainer_history.json), [audit độc lập](validation/artifact_audit.json).

## 7. Checkpoint, inference và lưu trữ

**GitHub không chứa trọng số.** Run Kaggle lưu bằng chứng reload thành công trong `reload_check.json`; môi trường kiểm chứng không chạy lại checkpoint. Không có URL weights công khai đã xác minh, nên cần giữ ZIP Kaggle của bạn.

Để đánh giá/suy luận lại, giải nén `best_model/` từ ZIP vào đúng đường dẫn:

```text
labs/lab03/outputs/runs/20261026T155822Z_finetuning_2c2f0401/best_model/
```

Rồi chạy:

```bash
.venv-lab03/bin/python labs/lab03/scripts/eval.py --model-path labs/lab03/outputs/runs/20261026T155822Z_finetuning_2c2f0401/best_model
.venv-lab03/bin/python labs/lab03/scripts/infer.py --model-path labs/lab03/outputs/runs/20261026T155822Z_finetuning_2c2f0401/best_model --text "A wonderful movie."
```

Eval/infer với `best_model/` ưu tiên config snapshot của run (gồm max length và dataset), nên thay đổi config hiện tại không làm lệch đánh giá run cũ. Eval chỉ tải tokenizer trong checkpoint, không tải lại backbone/tokenizer không liên quan.

Không truyền `--model-path` thì infer dùng sentiment pretrained SST-2 của Exercise 1, **không phải** mô hình Rotten Tomatoes vừa train. Eval không âm thầm fallback khi thiếu checkpoint.

SHA256 weights được giữ trong manifest: `5a63c9c2a6391132c86b0e6c08277cc23ccc692dd080b78ba2db074225bb5105`. Khi đưa weights vào lại, audit sẽ kiểm tra hash thật. Không upload weights/optimizer hoặc ZIP lớn vào Git.

## 8. Artifacts và chia sẻ run mới

Mỗi run có config, snapshot, console log, provenance, status, SHA256 manifest, metrics, lịch sử, dự đoán và report. Năm hình: learning curves, confusion matrix chuẩn hóa kèm counts, binary metrics bar, train token-length KDE/cutoff và bảng mẫu đúng/sai.

`outputs/` mặc định bị ignore để tránh đẩy nhầm model; các artifacts nhẹ của các run hiện hành đã được commit. Với run mới, chỉ stage phần nhẹ:

```bash
git add -f -- labs/lab03/outputs/runs/<run_id> ':(glob,exclude)**/best_model/**' ':(glob,exclude)**/checkpoints/**'
git add labs/lab03/config/workflow.json
git diff --cached --stat
```

Giữ manifest gốc kể cả khi weights không đi cùng. Không chỉnh sửa artifacts lịch sử để làm số liệu đẹp hơn; kết quả rà soát bổ sung ghi ở `validation/`. Run registry cần đi cùng run directory khi chuyển máy.

## 9. Notebook mở rộng, roadmap và artifact trực quan

Bản mở rộng đặt roadmap ngay đầu notebook, gồm phase, mục tiêu, module/file tham chiếu và đường dẫn artifact. Notebook không chỉ in JSON: bảng token, so sánh pipeline/manual, ROC–PR, phân bố xác suất, reliability/calibration, metric theo lớp và phân tích threshold đều có PNG đã lưu.

Chạy lại phần phân tích của run cũ **không cần checkpoint/GPU/Internet**:

```bash
.venv-lab03/bin/python labs/lab03/scripts/analyze_outputs.py
```

`src/analysis.py` tạo các run `exercise1_analysis` và `analysis` độc lập, lưu hình cùng JSON nguồn. Cache chỉ dùng khi manifest nguồn và code SHA256 còn khớp. Không sửa file hoặc manifest của run Kaggle cũ.

Để thực thi toàn bộ notebook view và lưu **mọi cell output** dưới `outputs/`:

```bash
.venv-lab03/bin/python -m ipykernel install --user --name lab03
.venv-lab03/bin/python labs/lab03/scripts/render_notebooks.py --kernel lab03
```

Run `notebook_render` chứa bản notebook đã thực thi và `execution_summary.json`; các notebook ở thư mục `notebooks/` cũng được cập nhật output. Quy trình chỉ chạy `MODE="view"`. Thực thi interactively vẫn đọc các PNG/bảng từ run phân tích đã lưu; mọi biểu đồ trình bày có file tương ứng trong `outputs/runs/`.

## 10. Quét threshold binary đúng quy trình

Quy tắc dự đoán mở rộng: Positive khi `P(Positive) >= threshold`. Với xác suất đúng bằng 0.5, quy tắc `>=` có thể khác argmax cũ ở trường hợp hòa điểm; không giả định hai cách luôn giống nhau.

Train mới thực hiện:

1. Chọn checkpoint tốt nhất theo **validation accuracy**.
2. Predict lại **validation** bằng checkpoint đó và lưu `validation_predictions.json`.
3. Quét 19 ngưỡng từ 0.05 tới 0.95, bước 0.05; chọn **validation Macro F1** lớn nhất. Khi hòa: gần 0.5 nhất, sau đó ngưỡng nhỏ hơn.
4. Lưu và cố định `decision_threshold.json`; lưu scan, hình và selected policy trong `validation_threshold/`.
5. Sau đó mới predict test. Giữ `test_metrics.json` theo argmax để so sánh; lưu riêng `test_threshold_metrics.json` cho ngưỡng validation đã cố định.

Code `src/training/thresholds.py` chặn chọn ngưỡng khi `split != 'validation'`. `scripts/select_threshold.py` chỉ đọc validation để chọn, không đọc test predictions. Test suite kiểm tra cả quy tắc này và lưu artifacts. Việc gắn nhãn split trong API không tự chứng minh nguồn dữ liệu; notebook/CLI chịu trách nhiệm truyền đúng split.

Run hiện hành đã lưu đủ validation probabilities, loss và policy **0,8** chọn theo validation Macro F1; weights tiếp tục nằm ngoài Git. Đồ thị `threshold_test_diagnostic/threshold_curves.png` chỉ mô tả độ nhạy trên test đã quan sát; **không chọn ngưỡng tối ưu từ test** và không thay đổi kết quả 83,96% của run gốc.

Nếu còn ZIP checkpoint, khôi phục `best_model/` rồi chọn ngưỡng mà không train lại:

```bash
.venv-lab03/bin/python labs/lab03/scripts/select_threshold.py \
  --model-path labs/lab03/outputs/runs/20261026T155822Z_finetuning_2c2f0401/best_model
```

Lệnh in đường dẫn policy trong run `threshold_selection` mới. Dùng policy đó khi đánh giá/suy luận:

```bash
.venv-lab03/bin/python labs/lab03/scripts/eval.py \
  --model-path <best_model_directory> \
  --threshold-file <threshold_selection_run>/decision_threshold.json

.venv-lab03/bin/python labs/lab03/scripts/infer.py \
  --model-path <best_model_directory> \
  --threshold-file <threshold_selection_run>/decision_threshold.json \
  --text "The film was enjoyable."
```

Checkpoint train mới mang policy cùng `best_model/`; eval/infer tự đọc nếu có. Checkpoint cũ không policy giữ argmax. Mọi inference CLI giờ được lưu trong run riêng gồm predictions JSON, hình và log, không chỉ in ra terminal.

Test set cũ đã được quan sát trong phát triển; một vòng thử ngưỡng mới trên benchmark đó vẫn mang tính thăm dò, không được trình bày như kiểm định mù độc lập. Không dùng hình test diagnostic để điều chỉnh lựa chọn ngưỡng.

Policy threshold được gắn SHA256 của checkpoint; không áp policy của model A vào model B. Inference lưu cả policy, hình và predictions. Dependency `ipywidgets` được pin để notebook có progress widget đúng môi trường.


## 11. Threshold của lịch sử 3 epoch và hình so sánh

| Vai trò | Run Kaggle nguồn |
|---|---|
| Chọn threshold trên validation | `20261026T155822Z_finetuning_2c2f0401` |
| Tải lại checkpoint, đánh giá full test | `20261026T160058Z_evaluation_07c24854` |
| Inference với policy đóng băng | `20261026T160111Z_inference_7f2b16f0` |

| Test metric | Argmax | Policy validation 0,8 |
|---|---:|---:|
| Accuracy | 83,9587% | 84,0525% |
| Positive Precision | 83,5185% | 86,5191% |
| Positive Recall | 84,6154% | 80,6754% |
| Binary F1 | 84,0634% | 83,4951% |
| Macro F1 | 83,9580% | 84,0343% |
| TN / FP / FN / TP | 444 / 89 / 82 / 451 | 466 / 67 / 103 / 430 |

Validation Macro F1: **0,843279 → 0,852663**. Test chỉ thêm một mẫu đúng, giảm 22 FP nhưng tăng 21 FN; không khẳng định 0,8 tốt hơn ở mọi mục tiêu. Measured test loss sau reload là **0,5442677736**, không đổi khi thay decision threshold. Giữ riêng loss này với loss 0,5443162918 lúc fine-tuning và NLL 0,5442964394 tái tính từ probabilities.

[Walkthrough](walkthrough_lab_03.md) giải thích đầy đủ quyết định và đánh đổi. Module `src/threshold_evidence.py` đối chiếu policy giữa ba run, SHA256 checkpoint, validation fingerprint, 19 ngưỡng và toàn bộ predictions. Module `src/threshold_analysis.py` xuất chín hình so sánh cùng bảng có thể tính lại; notebook hiển thị trực tiếp các hình đó.

```bash
.venv-lab03/bin/python labs/lab03/scripts/analyze_outputs.py
.venv-lab03/bin/python labs/lab03/scripts/audit_outputs.py
```

Hai lệnh này dùng evidence đã lưu, không tải weights và không train lại. Notebook `view` tự dùng kết quả bổ sung khi nó thuộc đúng training run; không gắn policy cũ vào một model mới. Bản notebook đã chạy được sao lưu đầy đủ trong run `notebook_render`, bao gồm cả bảng và hình hiển thị.

Run train mới có policy riêng được phân tích trực tiếp từ validation/test artifacts của chính run đó; xuất thêm `training_threshold_metrics.png`, `training_threshold_per_class.png` và `training_threshold_confusion.png`. Không cần tạo ba run selection/evaluation/inference riêng để notebook hiển thị so sánh threshold. `measured_test_loss` lấy từ `test_runtime_metrics.json` khi có; run gốc không log loss vẫn giữ trạng thái thiếu. Registry chọn bộ evidence cùng checkpoint/policy, không ghép tùy tiện các active run khác model.


## 12. Notebook trực quan và artifact để làm report

Notebook hiển thị hình/bảng ngay tại từng phase và lưu chính các kết quả đó dưới `outputs/`. Với notebook Kaggle điều phối CLI bằng subprocess, chạy cell hiển thị trong kernel notebook:

```python
import runpy
from pathlib import Path
LAB = Path("/kaggle/working/Lab-DL/labs/lab03")
_ = runpy.run_path(str(LAB / "scripts/display_results.py"), run_name="__main__")
```

Bộ report chọn run tường minh tại `config/report.json`. Chạy lệnh sau để sao chép hình/bảng đã có, không tải model và không vẽ lại đồ thị:

```bash
.venv-lab03/bin/python labs/lab03/scripts/publish_artifacts.py
```

Mở `outputs/published/index.md` để xem bảng kết quả và toàn bộ hình. `metrics_summary.json`, `artifact_catalog.json` và `source_runs.json` giúp dùng lại artifact cho report. Đổi các run được chọn rồi xuất lại khi cần; không cần trích raw outputs hoặc tạo lại toàn bộ phần phân tích cho mỗi bản trình bày. Checkpoint và raw logs không được đưa vào bộ published.

Phiên Kaggle hiện hành đã chốt dataset revision `aa13bc287fa6fcab6daf52f0dfb9994269ffea28` và model revisions trong config. 86 file của bảy run mới hiện có khớp manifest; bảy console logs không có trong bộ công bố được kê khai bằng `artifact_availability.json`. Manifest gốc được giữ nguyên, dữ liệu khoa học không được phép khai báo thiếu dưới dạng log.


## 13. Notebook Kaggle 10 epoch: chạy và thu hồi

1. Cập nhật nhánh `lab03`; trên Kaggle chọn **File -> Import Notebook -> GitHub** và nhập URL của `notebooks/04_kaggle_10epoch_study.ipynb` trên nhánh này.
2. Bật GPU T4 và Internet, chạy các cell từ trên xuống. Python 3.12 nằm trong venv riêng, không cài đè PyTorch của kernel Kaggle.
3. Preflight chạy tests CPU và lưu XML/log trong `outputs/preflight/`. Training chỉ bắt đầu sau khi tests pass.
4. Study fit TF-IDF trên train, chọn C bằng validation; ba DistilBERT candidate chạy đủ 10 epoch. Best checkpoint được chọn từ mọi mốc validation, không mặc định lấy epoch 10.
5. Cell hiển thị chạy ngay trong kernel notebook: learning curves từng LR, threshold sweep, bảng kết quả, confusion matrices, ROC/PR, confidence và inference. Những hình này có file gốc dưới `outputs/runs/`.
6. Tải **review ZIP** để xem xét/report; tải **checkpoint ZIP** để giữ model/tokenizer/policy. Không commit checkpoint ZIP lên Git thông thường.
7. **Save Version** và **File -> Download Notebook** để giữ cả output cell. Review ZIP không tự thu hồi notebook tương tác trong trình duyệt.

Có thể copy từng block `# %%` từ [kaggle_10epoch_cells.py](scripts/kaggle_10epoch_cells.py) vào các cell Code thay vì Import Notebook. Đây là cùng các cell của notebook, không phải một pipeline training khác.

Study tự xuất `study_report.md` với bảng và ảnh từ dữ liệu đã chạy. `candidate_comparison.json`, `frozen_selection.json`, `probability_diagnostics.json`, `threshold_errors.json` và `test_comparison.json` là nguồn số liệu đi kèm, không thay thế ảnh inline.

CI bootstrap 95% dùng 2.000 lần resample phân tầng test cho model/policy cố định. Nó không đo biến thiên giữa seed train. Test lịch sử đã từng được xem; kết quả mới không phải bằng chứng trên một blind test hoàn toàn mới. Ba learning rate là tìm kiếm có giới hạn, không có tuyên bố tối ưu toàn cục hoặc đảm bảo tăng accuracy.

Khi thu hồi kết quả mới, cập nhật Section 2/3 của walkthrough từ đúng run study; các bảng/hình lịch sử vẫn giữ đúng nguồn và không bị đổi nhãn thành 10 epoch.


## 14. Kết quả thật của study 10 epoch

Study: `20261026T184219Z_study_94f2ba7b`. Review ZIP có 128 file khớp SHA256; source hashes khớp code revision `ca58b27`. Mỗi candidate có 5.340 Trainer steps, 63 validation milestones và 10 train-eval milestones. AMP skipped 5/4/5 updates được giữ rõ trong metadata; không đánh đồng Trainer steps với successful optimizer updates.

| LR | Epoch hoàn tất | Best step | Best epoch | Validation Macro-F1 | Validation loss tại best |
|---|---:|---:|---:|---:|---:|
| 1e-5 | 10 | 3200 | 5,993 | 84,9906% | 0,755625 |
| 2e-5 | 10 | 2100 | 3,933 | 84,8920% | 0,747378 |
| 3e-5 | 10 | 1700 | 3,184 | 85,6473% | 0,702748 |

Winner được chọn bằng validation Macro-F1 trước test. Threshold sweep đủ 19 điểm chọn **0,5** cho winner; không có yêu cầu ngưỡng đã quét phải khác 0,5. Không dùng test để đổi winner hoặc policy.

| Model / policy | Test Accuracy | Macro-F1 | Binary-F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression, 0,5 | 78,2364% | 78,2352% | 78,0718% |
| TF-IDF, validation threshold 0,4 | 78,7054% | 78,6158% | 80,0000% |
| DistilBERT winner, threshold 0,5 | 83,3959% | 83,3952% | 83,2861% |

Winner test: **TN=448, FP=85, FN=92, TP=441**; Precision Positive 83,8403%, Recall 82,7392%. Measured test loss **0,7980755568**; reconstructed NLL **0,7980772048** được giữ riêng. ROC-AUC 0,918416; AP 0,925989; Brier 0,150674; confidence ECE 0,146071.

So với argmax lịch sử trên cùng 1.066 index/text/label, study sửa 42 lỗi nhưng tạo 48 lỗi mới: giảm 6 mẫu đúng. Paired bootstrap CI cho chênh accuracy chứa 0, không có kết luận vượt trội thống kê. Protocol còn khác LR/scheduler/criterion nên không phải ablation chỉ thay epoch.

**Không cần train lại để hoàn thiện báo cáo.** Hình đã hiển thị ở notebook 05 và phần mở rộng của notebook tích hợp; `outputs/published/index.md` có bảng/ảnh cho report với `primary_result="study"`. Dữ liệu lịch sử giữ riêng theo role, không bị thay bằng số study.

Raw logs của phiên này có đủ trong Git; checkpoint cố ý nằm ngoài Git. Review tại đây kiểm tra predictions/policy/manifest; không tải weights và không tuyên bố đã lặp lại GPU training. Giữ checkpoint ZIP để inference hoặc reload sau này.
