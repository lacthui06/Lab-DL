# New Lab03 — Binary sentiment classification, 3 epoch

Thí nghiệm phân loại cảm xúc tiếng Anh trên Rotten Tomatoes bằng Hugging Face. Chạy từ pretrained base với ngân sách tối đa **3 epoch**, seed 42, regularization và early stopping theo validation loss.

Chưa có kết quả full GPU cho thí nghiệm này. Hai notebook là bản sẵn sàng thực thi, không chứa cached outputs của lượt chạy khác. Accuracy, loss, threshold, thời gian và bộ nhớ chỉ được báo cáo sau Run All.

## Hai notebook

| Notebook | Nội dung | Mặc định |
|---|---|---|
| [01 — EDA, preprocessing, feature engineering](notebooks/01_eda_preprocessing_feature_engineering.ipynb) | Quality/leakage audit, nhãn/độ dài, cutoff, tokens/IDs/mask, train-only TF-IDF và augmentation | `MODE="prepare"`, `DEVICE="cpu"` |
| [02 — Modeling](notebooks/02_modeling.ipynb) | Text architectures, pretrained sentiment, baseline, train/evaluate, performance, threshold, errors và ZIP | `MODE="train"`, `DEVICE="cuda"` |

Mỗi notebook có roadmap/module ở đầu. Mỗi biểu đồ hiển thị inline từ cùng PNG lưu trong outputs/. JSON giữ số liệu chính xác để kiểm tra, không thay thế visualization.

## Kaggle

Import lần lượt từ GitHub:

- [Notebook 01](https://github.com/lacthui06/Lab-DL/blob/new-lab03/labs/new-lab03/notebooks/01_eda_preprocessing_feature_engineering.ipynb)
- [Notebook 02](https://github.com/lacthui06/Lab-DL/blob/new-lab03/labs/new-lab03/notebooks/02_modeling.ipynb)

Bật Internet; notebook02 cần bật GPU. Source đã đặt MODE phù hợp, chỉ cần Run All từ trên xuống. Không cần upload dataset hoặc checkpoint đầu vào. Python3.12 riêng được tạo để tránh thay đổi môi trường ML của kernel.

Notebook01 xuất ZIP EDA riêng. Notebook02 tự chạy cùng preprocessing/feature pipeline nên không phụ thuộc biến Python của notebook01. Sau khi đã có artifact, chọn `MODE="view"` để đọc lại mà không train hoặc tải dataset.

## Thiết kế thực nghiệm

- Full official splits: **8.530 train / 1.066 validation / 1.066 test**; model và dataset revisions được pin.
- Không xóa phủ định/punctuation. WordPiece, max length128, dynamic padding; pad multiple8 khi dùng CUDA FP16.
- Augmentation chỉ trên train, thay một từ/cụm cùng cực tính; chặn phủ định/tương phản, trùng và collision; tối đa25%, không ép đủ bằng sao chép.
- Baseline TF-IDF unigram/bigram + Logistic Regression; fit train-only, chọn C bằng validation Macro-F1.
- DistilBERT base có head mới; freeze embedding và4/6encoder blocks.
- Encoder/head LR **5e-6/2e-5**, dropout **0.2/0.2/0.4**, AdamW decay **0.1**, warmup **10%**, gradient clipping **1**.
- Budget **3 epoch**, evaluate/save mỗi epoch; early stopping **patience2/min_delta0.001**; reload checkpoint có validation loss nhỏ nhất.
- Với budget3/eval mỗi epoch, patience2 không thể gây dừng trước epoch3. Không đổi policy để ép dừng sớm.
- Quét19thresholds trên validation Macro-F1 rồi khóa policy trước predict test.
- Lưu full train-eval/validation learning curves, time/sample, RAM RSS, CUDA peak và phạm vi đo.
- Save/reload consistency, source/config/environment, raw log và SHA256 manifest.

Các kỹ thuật là lựa chọn định trước để kiểm chứng; không bảo đảm accuracy tối đa hoặc hoàn toàn không overfit. Kết quả một seed cần được diễn giải cùng baseline và giới hạn thực nghiệm.

## Folder

| Folder/file | Vai trò |
|---|---|
| `config/frozen_trial.json` | Cấu hình modeling 3epoch |
| `config/workflow.json` | Registry các run của phiên thực thi |
| `notebooks/` | Hai notebook thực thi và trình bày |
| `src/data_preparation.py`, `src/datasets/` | EDA, features, official loader, tokenizer, augmentation |
| `src/models/`, `src/training/` | Classifier, freeze/optimizer, Trainer, early stopping, metrics, threshold |
| `src/performance.py` | Time/RSS/CUDA telemetry và đơn vị đo |
| `scripts/` | CLI dùng chung module với notebook |
| `tests/` | Tiny model và synthetic fixtures sinh trong pytest tmp; không phụ thuộc kết quả training |
| `outputs/runs/` | Raw artifacts của từng run thực tế |
| `outputs/published/` | Hình/bảng/log/catalog/hash để làm report sau training |
| `outputs/latest_preparation.json` | Selection của phiên EDA hoàn thành |
| `outputs/latest_report.json` | Selection của phiên modeling hoàn thành |
| `validation/` | Bằng chứng kiểm tra phần mềm, không phải dataset validation |

## Local

Dùng Python3.12; cài requirements riêng cho bài này. Từ thư mục gốc repo:

```bash
python -m pip install -r labs/new-lab03/requirements.txt
python -m pytest labs/new-lab03/tests -q
python labs/new-lab03/scripts/preview_dataset.py
python labs/new-lab03/scripts/preview_augmentation.py
python labs/new-lab03/scripts/run_exercise1.py
python labs/new-lab03/scripts/run_frozen_trial.py --config labs/new-lab03/config/frozen_trial.json --device cuda
```

Local CPU được hỗ trợ qua `--device cpu`, nhưng full training có thể chậm. Notebook01 không cần GPU. Kiểm thử chỉ dùng dữ liệu/mô hình nhỏ và synthetic contract fixtures, không full train corpus.

## Artifact và báo cáo

Sau training thành công, `outputs/published/index.md` có hình và bảng đọc trực tiếp. Raw log trong published trỏ đúng source run; catalog/manifest giữ provenance và integrity. Không cần train lại để lấy hình làm report.

Kaggle xuất hai ZIP tại `/kaggle/working`; local xuất trong `labs/new-lab03/outputs`:

- `lab03_review_<run_id>.zip`: config, history, predictions, resource metrics, PNG, log và published; không chứa weights.
- `lab03_checkpoint_<run_id>.zip`: best model, tokenizer, training config và policy; giữ riêng để reload/infer.

Notebook01 có `lab03_preparation_<run_id>.zip`. Tải executed notebook của Kaggle để giữ cell outputs sau Run All. Không commit dataset/checkpoint/ZIP lớn.

- [Description](description_lab_03.md): đề bài, architecture, discussion và quyết định.
- [Walkthrough](walkthrough_lab_03.md): phương pháp, lý do, cách đọc kết quả và gate trước kết luận.
- [Kaggle runbook](KAGGLE_RUNBOOK.md): import, chạy, tải ZIP và xử lý lỗi.
- [Trạng thái outputs](outputs/README.md): chưa có kết quả mô hình trước khi train.
