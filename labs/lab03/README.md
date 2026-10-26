# Lab 03 — Getting started with Hugging Face

Bài thực hành Deep Learning — UTH: sentiment analysis bằng mô hình pretrained và fine-tuning mô hình phân loại văn bản nhị phân bằng Hugging Face Trainer.

**Đã có kết quả Kaggle thật:** dùng toàn bộ Rotten Tomatoes, 8.530 train / 1.066 validation / 1.066 test. Test Accuracy **83,96%**, Binary F1 **84,06%**. Mã nguồn, notebook và báo cáo dùng chung run registry, có thể xem kết quả mà không train lại.

## 1. Xem bài và kết quả

- [Notebook Exercise 1](notebooks/01_sentiment_analysis_pipeline.ipynb): tokenization, pipeline và manual forward.
- [Notebook Exercise 2](notebooks/02_finetuning_binary_classification.ipynb): dữ liệu, Trainer, metrics, năm biểu đồ và phân tích lỗi.
- [Notebook tích hợp](notebooks/lab03_huggingface_complete.ipynb): cả hai Exercise, phù hợp xem toàn bộ bài.
- [Báo cáo thực hành](walkthrough_lab_03.md): lý thuyết, phương pháp, kết quả và giới hạn có bằng chứng.
- [Đề bài](description_lab_03.md).

Các notebook đã lưu output của một lượt thực thi **`MODE="view"`** trên artifacts Kaggle. Có thể đọc trực tiếp trên GitHub. Khi chạy lại notebook, mặc định view đọc dữ liệu đã commit, không tải model/dataset, không huấn luyện và không cần GPU. Đổi `MODE="train"` ở cell setup để tạo thí nghiệm mới; phần fine-tuning yêu cầu GPU theo mặc định.

## 2. Cấu trúc

```text
labs/lab03/
  config/
    config.json             tham số cho lần chạy mới
    workflow.json           active runs và lịch sử
  notebooks/                ba notebook có output
  src/
    datasets/               tải official splits, kiểm tra kích thước và tokenization
    models/                 model, pipeline/manual, kiểm tra checkpoint
    training/               Trainer, metrics, sanity overfit dùng riêng trong tests
    utils/                  log, registry, manifest và plots
    workflow.py             luồng CLI/notebook dùng chung
    evidence.py             audit đọc-only, tính lại metrics từ dự đoán
  scripts/                  train, eval, infer, audit và recovery
  tests/                    tests chức năng và kiểm tra artifacts thật
  outputs/runs/             artifacts Kaggle nhẹ, giữ nguyên manifest gốc
  validation/               kết quả audit và thực thi notebook tại máy rà soát
  description_lab_03.md     đề bài
  walkthrough_lab_03.md     báo cáo thực nghiệm để nộp
  internal_changes.md       lịch sử kỹ thuật; không thuộc báo cáo nộp
```

Đã hợp nhất các thư mục `config/` và `outputs/` được upload ở gốc repo vào `labs/lab03/`. Không có hai registry độc lập. Các byte artifacts gốc không bị chỉnh sửa; 24 file hiện có khớp manifest.

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

- **43 tests đạt trên CPU**: dataset contract, dynamic padding, pipeline/manual, checkpoint, sanity overfit, Trainer, run tracking, năm plots, recovery, metrics/artifact audit và phát hiện dữ liệu bị sửa.
- Audit kiểm tra 24 file SHA256, tính lại toàn bộ bảy metrics từ 1.066 predictions và xác nhận confusion matrix.
- 29 file checkpoint không có trong Git được liệt kê rõ là thiếu có chủ đích, không coi là đã kiểm tra tại máy rà soát.
- Ba notebook chạy hết ở view, Hugging Face offline, không có cell lỗi. Xem `validation/notebook_execution.json`.

Tests có fixture nhỏ để kiểm tra pipeline; chúng không phải benchmark Rotten Tomatoes. Audit thực dùng `all_predictions.json` từ Kaggle và không suy luận lại từ weights.

## 5. Chạy thí nghiệm mới trên GPU

Từ gốc repo, dùng interpreter của môi trường vừa cài:

```bash
.venv-lab03/bin/python labs/lab03/scripts/run_exercise1.py
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false .venv-lab03/bin/python labs/lab03/scripts/train_finetune.py
```

Trên Kaggle có thể gọi các lệnh bằng `subprocess.run([...], cwd=REPO, env=RUN_ENV, check=True)`, trong đó `RUN_ENV['CUDA_VISIBLE_DEVICES']='0'`; không dùng Python mặc định của notebook nếu requirements nằm trong venv khác.

Quy trình dùng toàn bộ official splits, max length 128, 3 epochs, batch train/eval 16/32, LR 2e-5, weight decay 0.01, warmup ratio 0.1, FP16 chỉ khi CUDA có sẵn. Lưu/evaluate mỗi epoch; chọn best theo validation accuracy. Sanity test không tác động mô hình train chính. Chạy lại tạo run mới, không resume hoặc ghi đè run cũ. Chỉ dùng `--allow-cpu` nếu chủ động chấp nhận thời gian full training CPU.

Có thể thực thi ba notebook với MODE=train thay CLI. Không chạy notebook tích hợp và notebook từng bài đồng thời nếu chỉ muốn một lần thí nghiệm.

## 6. Kết quả hiện hành và truy vết

| Giai đoạn | Run được registry chọn |
|---|---|
| Exercise 1 | `20261025T181532Z_exercise1_71da6f40` |
| Fine-tuning | `20261025T181550Z_finetuning_e02bced7` |

| Test metric | Giá trị |
|---|---:|
| Accuracy | 83,9587% |
| Binary Precision / Recall / F1 | 83,5185% / 84,6154% / 84,0634% |
| Macro Precision / Recall / F1 | 83,9646% / 83,9587% / 83,9580% |
| Đúng / tổng | 895 / 1.066 |

Confusion matrix `[TN FP; FN TP]` = `[[444,89],[82,451]]`. Checkpoint epoch 3 được chọn theo validation accuracy 84,33%; epoch 2 có loss validation thấp hơn. Có dấu hiệu overfitting; đề không đặt ngưỡng accuracy 85%.

Nguồn: [metrics](outputs/runs/20261025T181550Z_finetuning_e02bced7/test_metrics.json), [toàn bộ dự đoán](outputs/runs/20261025T181550Z_finetuning_e02bced7/all_predictions.json), [lịch sử](outputs/runs/20261025T181550Z_finetuning_e02bced7/trainer_history.json), [audit độc lập](validation/artifact_audit.json).

## 7. Checkpoint, inference và lưu trữ

**GitHub không chứa trọng số.** Run Kaggle lưu bằng chứng reload thành công trong `reload_check.json`; máy rà soát không chạy lại checkpoint. Không có URL weights công khai đã xác minh, nên cần giữ ZIP Kaggle của bạn.

Để đánh giá/suy luận lại, giải nén `best_model/` từ ZIP vào đúng đường dẫn:

```text
labs/lab03/outputs/runs/20261025T181550Z_finetuning_e02bced7/best_model/
```

Rồi chạy:

```bash
.venv-lab03/bin/python labs/lab03/scripts/eval.py --model-path labs/lab03/outputs/runs/20261025T181550Z_finetuning_e02bced7/best_model
.venv-lab03/bin/python labs/lab03/scripts/infer.py --model-path labs/lab03/outputs/runs/20261025T181550Z_finetuning_e02bced7/best_model --text "A wonderful movie."
```

Eval/infer với `best_model/` ưu tiên config snapshot của run (gồm max length và dataset), nên thay đổi config hiện tại không làm lệch đánh giá run cũ. Eval chỉ tải tokenizer trong checkpoint, không tải lại backbone/tokenizer không liên quan.

Không truyền `--model-path` thì infer dùng sentiment pretrained SST-2 của Exercise 1, **không phải** mô hình Rotten Tomatoes vừa train. Eval không âm thầm fallback khi thiếu checkpoint.

SHA256 weights được giữ trong manifest: `5a63c9c2a6391132c86b0e6c08277cc23ccc692dd080b78ba2db074225bb5105`. Khi đưa weights vào lại, audit sẽ kiểm tra hash thật. Không upload weights/optimizer hoặc ZIP lớn vào Git.

## 8. Artifacts và chia sẻ run mới

Mỗi run có config, snapshot, console log, provenance, status, SHA256 manifest, metrics, lịch sử, dự đoán và report. Năm hình: learning curves, confusion matrix chuẩn hóa kèm counts, binary metrics bar, train token-length KDE/cutoff và bảng mẫu đúng/sai.

`outputs/` mặc định bị ignore để tránh đẩy nhầm model; các artifacts nhẹ của hai run hiện hành đã được commit. Với run mới, chỉ stage phần nhẹ:

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

Run Kaggle lịch sử **không có validation probabilities và không có weights trong Git**, vì vậy chưa thể khẳng định đã chọn ngưỡng validation cho run đó. Đồ thị `threshold_test_diagnostic/threshold_curves.png` chỉ mô tả độ nhạy trên test đã quan sát; **không chọn ngưỡng tối ưu từ test** và không thay đổi kết quả 83,96% của run gốc.

Nếu còn ZIP checkpoint, khôi phục `best_model/` rồi chọn ngưỡng mà không train lại:

```bash
.venv-lab03/bin/python labs/lab03/scripts/select_threshold.py \
  --model-path labs/lab03/outputs/runs/20261025T181550Z_finetuning_e02bced7/best_model
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
