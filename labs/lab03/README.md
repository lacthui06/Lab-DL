# Lab03 — Binary Sentiment Classification with Hugging Face

- **Tác giả:** Thái Anh Lạc
- **Phạm vi:** Phân loại cảm xúc tiếng Anh trên Rotten Tomatoes: pretrained sentiment/tokenization và fine-tuning DistilBERT. Pipeline gồm **ba notebook chuyên biệt**, có roadmap, module tham chiếu, ảnh inline và artifact được lưu để dùng cho report.

## Kết quả đã hoàn thành và xem ngay

Run Kaggle `20261103T110437Z_frozen_trial_b021d6eb` hoàn thành **5/5 epoch**, checkpoint epoch5, threshold validation **0,45**. Test: **Accuracy83,11%, Binary F1 83,30%**, 886/1.066 mẫu đúng. TF-IDF baseline:78,99%.

- [Report hình/bảng/log](outputs/published/index.md)
- [Kết quả và diễn giải chi tiết](reports/walkthrough_lab_03.md#30-kết-quả-thực-nghiệm-đã-xác-minh-từ-zip-kaggle)
- [Kiến trúc hai model dạng text](outputs/published/architecture/model_architectures.txt), [bảng tóm tắt](outputs/published/architecture/summary.json).
- [Metrics theo lớp](outputs/published/result_analysis/per_class_metrics.png), [patience replay](outputs/published/result_analysis/early_stopping_replay.json).
- [Selection](outputs/latest_report.json), [audit](validation/final_review_audit.json)

Hai notebook đã có output inline; chọn `MODE="view"` để xem offline. Source mặc định view; đổi prepare/train cho phiên Kaggle mới; không cần train/tải dataset để đọc báo cáo.

## Chạy trên Kaggle

Import notebook modeling từ GitHub:

```text
https://github.com/lacthui06/Lab-DL/blob/lab03/labs/lab03/notebooks/03_modeling.ipynb
```

Bật **GPU + Internet**, rồi **Run All**. Để train mới, đổi cell setup thành `MODE="train"`, `DEVICE="cuda"`; bắt đầu từ pretrained base, tải official dataset và thực hiện trọn pipeline. Không cần cung cấp checkpoint đầu vào.

Cấu hình: **tối đa 5 epoch**, early stopping theo validation loss, patience 2, min_delta 0.001. Số epoch thực chạy có thể ít hơn 5. Kết quả, selected checkpoint và threshold được lấy từ chính phiên thực thi.

## Ba notebook chuyên biệt

| Notebook | Nội dung chính | Chế độ mặc định |
|---|---|---|
| [01 — EDA](notebooks/01_eda.ipynb) | Phân tích 6 bước theo docs/eda.md: Tổng quan schema, 5-number summary, phân bố đơn/song biến, Boxplot độ dài theo nhãn, dò ngoại lệ IQR, kiểm toán missing/leakage | `view` (đổi `prepare` để tạo mới) |
| [02 — Feature Engineering](notebooks/02_feature_engineering.ipynb) | Kỹ nghệ đặc trưng 6 bước theo docs/feature.md: Truncation cutoff, WordPiece tokenization, Handcrafted linguistic features, TF-IDF vectorization, Chi-Square feature selection, text augmentation | `view` (đổi `prepare` để tạo mới) |
| [03 — Modeling](notebooks/03_modeling.ipynb) | Huấn luyện và đánh giá: Kiến trúc DistilBERT, Pretrained SST-2, baseline so sánh, fine-tune với frozen backbone, quét threshold, ma trận nhầm lẫn, phân tích lỗi, xuất bản artifact | `view` (đổi `train` để tạo mới) |

Ảnh được hiển thị từ cùng PNG đã lưu trong `outputs/runs/<run_id>/`. Sau khi có kết quả, `MODE="view"` đọc selection trong `outputs/latest_preparation.json` hoặc `outputs/latest_report.json`; không train để lấy lại hình. Các notebook dữ liệu (01, 02) không phải điều kiện bắt buộc trước notebook03: script modeling tự thực hiện cùng data/feature pipeline.

## Phương pháp

- Official splits: **8.530 train / 1.066 validation / 1.066 test**; seed 42; không subsample hoặc chia lại test.
- Giữ phủ định/punctuation; WordPiece tokenizer đã pin; max length 128 và dynamic padding.
- Notebook01 & 02 báo cáo đầy đủ 5-number summary (Min, Q1, Median, Q3, Max, Mean, Std), phát hiện outlier IQR, so sánh boxplot độ dài câu theo nhãn, trích xuất đặc trưng ngôn ngữ học (uppercase, punctuation, word length) và lựa chọn top-25 đặc trưng bằng kiểm định Chi-Square ($\chi^2$).
- Train-only augmentation bằng thay từ/cụm cùng cực tính, loại câu có phủ định/tương phản, dedup, cân bằng variants; tối đa 25%, không ép đủ bằng sao chép.
- Baseline: TF-IDF unigram/bigram + Logistic Regression; fit train-only, chọn C bằng validation Macro-F1.
- DistilBERT base: embedding và 4/6 encoder blocks frozen; encoder/head LR 5e-6/2e-5; dropout 0.2/0.2/0.4; AdamW decay 0.1; warmup 10%; gradient clipping 1.
- Max 5 epoch + early stopping; chọn checkpoint bằng minimum validation loss.
- Quét 19 thresholds trên validation Macro-F1, khóa policy rồi test.
- Đo accuracy/loss theo epoch, time/sample, RAM RSS và CUDA peak/batch slot; hiển thị learning curves, architecture và phân bổ frozen/trainable parameters, metrics, confusion, ROC/PR, calibration và sample predictions.
- Save/reload consistency, raw log, environment/config/source hashes và SHA256 manifest.

Các kỹ thuật là lựa chọn được định trước để kiểm chứng bằng thực nghiệm; không bảo đảm không overfit hoặc luôn hơn baseline.

## Folder và tài liệu

| Folder/file | Vai trò |
|---|---|
| `config/frozen_trial.json` | Profile thí nghiệm, hyperparameters và early stopping |
| `config/workflow.json` | Registry các run runtime |
| `notebooks/` | Hai notebook trình bày và thực thi |
| `src/datasets/` | Official loader, tokenizer, augmentation |
| `src/data_preparation.py` | EDA, quality/cutoff audit, TF-IDF và token examples |
| `src/models/` | Load classifier, inference và save/reload |
| `src/training/` | Trainer, early stopping, optimizer/freeze, metrics, threshold |
| `src/performance.py` | Timer, RSS/CUDA peak, phạm vi và đơn vị đo |
| `src/utils/` | Run isolation, snapshot, manifest và figures |
| `scripts/` | CLI dùng chung module với notebook |
| `tests/` | Tiny fixtures và kiểm thử data/model/training/threshold/artifact |
| `outputs/runs/<id>/` | Artifact nguồn của mỗi run |
| `outputs/published/` | Hình/bảng/log được chọn cho report, catalog và manifest |
| `outputs/latest_report.json` | Selection của phiên modeling hoàn thành |
| `outputs/latest_preparation.json` | Selection của phiên EDA hoàn thành |
| `outputs/preflight/` | Pytest log/XML trước training |
| `validation/` | Kiểm chứng phần mềm, khác dataset validation |
| [Description](description_lab_03.md) | Đề bài, architecture, discussion và quyết định |
| [Reports & Walkthrough](reports/walkthrough_lab_03.md) | Báo cáo phương pháp, vận hành, thu thập bằng chứng (kèm bản HTML trong `reports/`) |
| [Kaggle runbook](KAGGLE_RUNBOOK.md) | Import/push kernel, chạy và lấy kết quả |

## Local CLI

Dùng Python3.12, từ thư mục gốc repo:

```bash
python -m venv .venv
source .venv/bin/activate  # Git Bash trên Windows: source .venv/Scripts/activate
python -m pip install -r labs/lab03/requirements.txt
python -m pytest labs/lab03/tests -q
python labs/lab03/scripts/preview_dataset.py
python labs/lab03/scripts/preview_augmentation.py
python labs/lab03/scripts/run_exercise1.py
python labs/lab03/scripts/run_frozen_trial.py --config labs/lab03/config/frozen_trial.json --device cuda
```

Local verification chỉ dùng tiny fixtures/sanity minibatch. Full training khuyến nghị GPU; `--device cpu` chỉ dùng khi chủ động chấp nhận thời gian chạy.

## Artifact và report

Sau training thành công, hình/bảng/raw log được tự publish. Mỗi session có ID riêng; artifact đi kèm trong ZIP phải thuộc phiên tương ứng. Có hai ZIP tại `/kaggle/working` trên Kaggle hoặc `labs/lab03/outputs` trên local:

- `lab03_review_<run_id>.zip`: config, log, predictions, history, performance, figures, published và manifest; không chứa weights.
- `lab03_checkpoint_<run_id>.zip`: best model, tokenizer, training config và decision policy; giữ riêng để evaluate/infer.

Notebook hiển thị link tải. Nếu không tải được, mở Output → `/kaggle/working` → Refresh → Download. Tải notebook có cell outputs bằng File → Download Notebook. Không commit weights, dataset thô hoặc ZIP vào Git; commit hình/bảng/log nhẹ và notebook đã chạy.

RAM/VRAM per batch slot gồm shared model memory, không phải bộ nhớ riêng của một câu. Test ms/sample là batch throughput, không phải latency request đơn lẻ. Metric chỉ được kết luận sau khi đọc artifact thật của lượt chạy.
