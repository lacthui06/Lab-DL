# Lab03 — Getting started with Hugging Face

Hai Exercise: pretrained sentiment/tokenization và fine-tuning binary sentiment trên Rotten Tomatoes. Có đúng hai notebook: EDA → preprocessing → feature engineering, rồi modeling. Mỗi notebook có roadmap/module tham chiếu, bảng và ảnh inline; cùng hình/bảng/log được lưu trong `outputs/` để dùng lại cho report.

**Profile train hiện hành:** tối đa **5 epoch**, early stopping theo validation loss (patience2, min_delta0.001). Đây là cấu hình mới chờ lượt GPU. Bằng chứng đã publish thuộc lượt frozen **10epoch lịch sử**, không gọi đó là kết quả5epoch.

## Notebook và cách chạy

- [01 — EDA, preprocessing, feature engineering](notebooks/01_eda_preprocessing_feature_engineering.ipynb): `MODE="view"` xem hình có sẵn; `MODE="prepare"` tạo lại phân tích dữ liệu, tokenization, TF-IDF và augmentation. Không train mạng neural.
- [02 — Modeling](notebooks/02_modeling.ipynb): `MODE="view"` xem kết quả đã lưu; đổi `MODE="train"`, `DEVICE="cuda"` để chạy mới trên Kaggle GPU. Chạy từ đầu theo thứ tự, không resume checkpoint cũ. Local dùng cùng notebook/script và chọn device phù hợp.
- [Kaggle runbook](KAGGLE_RUNBOOK.md): import GitHub hoặc tạo kernel bằng CLI, bật GPU/Internet, lấy ZIP.

Import notebook02 vào Kaggle từ URL:

```text
https://github.com/lacthui06/Lab-DL/blob/lab03/labs/lab03/notebooks/02_modeling.ipynb
```

Notebook gốc mặc định view để mở báo cáo an toàn; phải đổi MODE trước Run All nếu muốn train. Kaggle tạo Python3.12 riêng, tránh xung đột thư viện kernel. Sau train có review ZIP nhẹ và checkpoint ZIP riêng tại `/kaggle/working`.

## Luồng và cấu hình

1. Kiểm thử fixture/sanity overfit trước GPU training.
2. Official train8.530 / validation1.066 / test1.066; seed42; không subsample hay chia lại test.
3. Kiểm tra chất lượng, phân bố nhãn/độ dài, WordPiece và attention mask; max_length128, dynamic padding.
4. Augmentation chỉ từ train: cùng cực tính, tránh phủ định/tương phản, dedup và cân bằng. Trần25%, số hợp lệ thực tế hiện có274, effective train8.804; synthetic label có thể nhiễu.
5. Baseline TF-IDF unigram/bigram + Logistic Regression; fit train-only, chọn C bằng validation.
6. DistilBERT base: embedding +4/6 blocks frozen, hai blocks cuối/head trainable. EncoderLR5e-6, headLR2e-5, dropout0.2/0.2/0.4, AdamW decay0.1, warmup10%, clipping1. Profile là giả thuyết đã định trước, chưa chứng minh tối ưu.
7. Max5epoch, full train/validation evaluation mỗi epoch; early stopping validation-only, reload checkpoint min validation loss. Biểu đồ có đúng số epoch thực chạy, có thể ít hơn5.
8. Quét19thresholds0.05–0.95 trên validation Macro-F1, khóa policy rồi mới test. Không chọn threshold theo test.
9. Hiện learning curves, kiến trúc, accuracy/loss, time/sample, RAM/VRAM, confusion, ROC/PR, calibration và sample errors ngay trên notebook; đồng thời lưu artifact.
10. Save/reload check, SHA256, tự publish hình/bảng/log và tạo hai ZIP.

## Vai trò folder/file

| Folder/file | Vai trò |
|---|---|
| `config/frozen_trial.json` | Profile mới5epoch + early stopping/regularization |
| `config/report.json` | Chọn tường minh run lịch sử cho notebook view/report; không ghi đè khi train mới |
| `config/workflow.json` | Registry run runtime và active runs |
| `notebooks/` | Đúng hai notebook, không nhân bản theo từng experiment |
| `src/datasets/` | Loader official splits, tokenization và augmentation |
| `src/data_preparation.py` | EDA, quality/cutoff audit, token/mask và TF-IDF train-only |
| `src/models/` | Pretrained classifier và save/reload consistency |
| `src/training/` | Trainer/early stopping, optimizer/freeze, metrics và threshold |
| `src/performance.py` | Time/RSS/CUDA peak, phạm vi và đơn vị đo |
| `src/utils/` | Isolated runs, snapshots, manifest, visualization |
| `scripts/` | Entry points gọi cùng module với notebook; không chứa pipeline copy |
| `tests/` | Fixture nhỏ: leakage, optimizer, early stop, reload, threshold, telemetry/artifact |
| `outputs/runs/<id>/` | Nguồn bất biến của từng run, ảnh/bảng/log/config/environment/hash |
| `outputs/published/` | Hình/bảng/log đã chọn và catalog dùng ngay cho report |
| `outputs/latest_report.json` | Nguồn published của lượt train mới; không thay report lịch sử |
| `outputs/preflight/` | Pytest log/XML trước training |
| `outputs/imports/` | Manifest/provenance của kết quả Kaggle được nhập |
| `validation/` | Bằng chứng kiểm thử/audit phần mềm, khác dataset validation |
| `description_lab_03.md` | Đề, discussion/decision và architecture/rationale |
| `walkthrough_lab_03.md` | Spec, quy trình thí nghiệm, kết quả, hình, lý do và giới hạn |

Các config/module cho study cũ được giữ vì dùng kiểm toán/so sánh run đã công bố; không cần chạy lại khi train profile mới. Không commit dataset thô, ZIP hoặc weights.

## Local CLI

Từ thư mục gốc repo, dùng Python3.12 và môi trường riêng:

```bash
python -m venv .venv
source .venv/bin/activate  # Git Bash/Windows: source .venv/Scripts/activate
python -m pip install -r labs/lab03/requirements.txt
python -m pytest labs/lab03/tests -q
python labs/lab03/scripts/preview_dataset.py
python labs/lab03/scripts/preview_augmentation.py
python labs/lab03/scripts/run_exercise1.py
python labs/lab03/scripts/run_frozen_trial.py --config labs/lab03/config/frozen_trial.json --device cuda
```

CLI GPU training tương đương MODE=train. CPU đầy đủ có hỗ trợ qua `--device cpu` nhưng chậm; local verification chỉ dùng fixture nhỏ. Evaluate/infer lại cần checkpoint đã tải riêng:

```bash
python labs/lab03/scripts/eval.py --model-path /path/to/best_model
python labs/lab03/scripts/infer.py --model-path /path/to/best_model --text "The acting was good, but the story was disappointing."
```

Không đổi threshold sau khi xem test. Decision policy gắn với hash weights của checkpoint.

## Bằng chứng đọc ngay sau khi pull

- [Catalog hình/bảng/log](outputs/published/index.md), [manifest SHA256](outputs/published/manifest.json).
- [EDA chất lượng](outputs/published/data_preparation/data_quality.png), [TF-IDF](outputs/published/data_preparation/tfidf_features.png), [nguồn EDA](outputs/published/data_preparation/data_source_provenance.json).
- [Kiến trúc](outputs/published/performance_report/model_architecture.png), [learning curves lịch sử](outputs/published/frozen_trial/learning_curves.png).
- [Performance](outputs/published/performance_report/performance.png), [định nghĩa/đơn vị đo](outputs/published/performance_report/performance_summary.json), [raw log](outputs/published/logs/frozen_trial/console.log).

Frozen10epoch lịch sử: selected epoch9, validation threshold0.45, test883/1.066 đúng (accuracy82.83%, Macro-F182.82%). Test evaluation khoảng0.466792ms/mẫu là throughput batch, không phải latency request đơn lẻ. Run cũ không ghi training RAM/VRAM nên hiển thị **NOT LOGGED**, không tạo số giả; lượt train mới có resource_history/test_resource_metrics và hình tương ứng.

EDA bổ sung dùng corpus công khai phục dựng theo legacy official builder, đối chiếu2.132heldout và274augmentation records với run Kaggle; chưa đối chiếu toàn bộ train/HF fingerprints. Provenance ghi rõ giới hạn. Lần train/prepare mới vẫn dùng pinned HF official dataset. Sau khi có reviewZIP5epoch, cập nhật kết quả trong walkthrough và report selection; chưa coi phép đo GPU mới hoàn tất trước lúc đó.
