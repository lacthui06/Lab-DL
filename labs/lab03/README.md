# Lab 03 — Getting started with Hugging Face

Bài thực hành gồm pretrained sentiment analysis và fine-tuning DistilBERT cho phân loại văn bản nhị phân trên toàn bộ Rotten Tomatoes.

**Kết quả chính đã chạy trên Kaggle:** frozen DistilBERT, đủ **10 epoch**, chọn checkpoint **epoch 9 theo validation loss**, threshold **0.45 chọn bằng validation Macro-F1**. Test đúng **883/1.066**, Accuracy **82.83%**, Binary F1 **83.23%**, Macro-F1 **82.82%**, test loss **0.412560**.

**Mục tiêu của bản chính là giảm dấu hiệu overfitting**, không chọn accuracy cao nhất từ các lần test. Epoch 10: train/validation accuracy 84.25%/83.77%, chênh 0.47 điểm phần trăm; train/validation loss 0.3581/0.3696. Bản full fine-tuning có accuracy cao hơn nhưng gap lớn hơn. Một seed và test đã được xem trước không đủ để khẳng định tối ưu tuyệt đối hoặc không bao giờ overfit.

## 1. Chỉ hai notebook

| Notebook | Nội dung | Chế độ |
|---|---|---|
| [01 — EDA, preprocessing, feature engineering](notebooks/01_eda_preprocessing_feature_engineering.ipynb) | Full splits, phân bố nhãn/độ dài, WordPiece/IDs/mask, augmentation trước/sau | `MODE="view"` xem artifact; `MODE="prepare"` tạo lại phần chuẩn bị |
| [02 — Modeling](notebooks/02_modeling.ipynb) | Exercise 1 pretrained sentiment, baseline, frozen DistilBERT, train 10 epoch, threshold, test và lỗi | `MODE="view"` xem kết quả; `MODE="train"` train mới |

Cả hai có roadmap/module ở đầu, giải thích theo phase, bảng và ảnh inline. Ảnh được hiển thị từ chính PNG đã lưu outputs/. Module .py giữ thuật toán dùng chung; không duy trì thêm bản copy-cell hoặc notebook riêng cho từng thí nghiệm. Các notebook cũ có thể phục hồi từ lịch sử Git.

Đọc [description](description_lab_03.md) để xem đề/thảo luận/quyết định; [walkthrough](walkthrough_lab_03.md) để xem báo cáo và câu hỏi bảo vệ. [Artifact report](outputs/published/index.md) chứa hình/bảng dùng trực tiếp, không cần extract ZIP hoặc viết lại code vẽ khi làm report.

## 2. Pipeline và quyết định chính

1. Exercise 1 dùng `distilbert-base-uncased-finetuned-sst-2-english`, đối chiếu pipeline và manual softmax.
2. Exercise 2 dùng DistilBERT **base**, classification head mới; không lấy sentiment model của Exercise 1 để giả làm fine-tuning mới.
3. Giữ official splits **8.530 train / 1.066 validation / 1.066 test**, không cắt nhỏ hoặc tự chia lại.
4. WordPiece tokenizer, max_length=128, dynamic padding; giữ từ phủ định/tương phản.
5. Augmentation train-only thêm **274 mẫu** (137 mỗi lớp), effective train **8.804**. Nhãn kế thừa có thể nhiễu; không phải 274 ground-truth mới do người gán.
6. Đóng băng embedding và **4/6 encoder layers đầu**; cập nhật 14.767.874/66.955.010 tham số.
7. Encoder LR **2e-6**, head LR **1e-5**; warmup 10%, linear decay, AdamW decay **0.1**, bias/LayerNorm không decay.
8. Dropout encoder/attention/classifier **0.2/0.2/0.4**, chỉ hoạt động khi train. Metrics full train và validation đều đo trong eval mode.
9. Chạy đủ **10 epoch**, không early stopping; checkpoint có validation loss thấp nhất được nạp lại. Quét 19 threshold validation sau đó khóa policy trước test.
10. Lưu run riêng với config, environment, raw console, predictions, PNG và SHA256 manifest. Notebook hiển thị từ cùng PNG đã lưu.

## 3. Vai trò các thư mục

```text
labs/lab03/
  config/                 cấu hình và lựa chọn run tường minh
    frozen_trial.json     profile train bản chính
    report.json           run nguồn và primary_result của report
    workflow.json         registry runtime; không thay report.json tự động
  notebooks/              đúng hai notebook: chuẩn bị dữ liệu và modeling
  src/
    datasets/             full splits, tokenizer, train-only augmentation
    models/               pretrained/model loading, manual inference, reload check
    training/             Trainer, optimizer/freeze, metrics, threshold
    utils/                run context, registry, manifest và plots
    final_review.py       audit frozen và so sánh từ predictions thật
  scripts/                CLI train, preview, audit, xuất report và render notebook
  tests/                  fixture sanity, freeze/optimizer, leakage guards, artifact audits
  outputs/
    runs/                 evidence thí nghiệm gốc và phân tích dẫn xuất riêng
    imports/              bundle metadata và đối chiếu source commit của dữ liệu nhập
    preflight/            log/XML tests chạy trước training
    published/            ảnh/bảng nhẹ được chọn để viết report
    experiment_results.json  tóm tắt máy đọc của kết quả chính
  validation/             kết quả kiểm chứng phần mềm/notebook; không phải dataset validation
  description_lab_03.md   đề, thảo luận và quyết định
  walkthrough_lab_03.md   tài liệu sống: spec, experiments, results
```

`report.json` cố định nguồn để notebook không tự lấy nhầm một fixture hoặc run vừa chạy. `workflow.json` ghi active run phục vụ runtime. Dataset validation được tải qua Hugging Face; không nằm trong thư mục `validation/` của source.

## 4. Chạy notebook trên Kaggle hoặc local

**Kaggle:** import notebook cần dùng từ nhánh code hiện tại, dùng session mới và bật Internet. Với modeling chọn GPU nếu có. Cell đầu tự clone repo nếu chưa có và tạo Python 3.12 riêng bằng uv. Không cần upload checkpoint cũ.

**Local:** clone repo, cài requirements và dùng kernel Python 3.12 theo Mục 5. Mở notebook từ thư mục repo hoặc notebooks/.

Chỉ sửa hai biến ở cell đầu:

```python
MODE = "view"       # NB01: view/prepare; NB02: view/train
DEVICE = "auto"     # auto/cuda/cpu
```

- Notebook 01: `prepare` tải dataset/tokenizer và tạo artifact EDA/augmentation, không train mô hình.
- Notebook 02: `train` chạy preflight, Exercise 1, baseline và frozen training đủ 10 epoch; `view` mặc định không train lại.
- `auto` chọn CUDA nếu có, nếu không CPU. `cuda` báo lỗi rõ nếu không có GPU. `cpu` chạy local được nhưng full training chậm; khuyến nghị CUDA. Kiểm thử phần mềm dùng fixture nhỏ; không cần full training để xác nhận bố cục notebook.

Notebook modeling gọi training một lần. Pipeline khóa checkpoint bằng validation loss, chọn threshold validation rồi mới đo test; các cell sau chỉ hiển thị artifact của đúng run vừa tạo. Không dùng ảnh review lịch sử thay cho kết quả train mới. Run All theo thứ tự; session restart làm mất biến dù output cũ còn hiển thị.

## 5. Cài đặt và CLI

Requirements trực tiếp có giao `Requires-Python >=3.10`, nhưng môi trường hỗ trợ đã kiểm chứng là **Python 3.12**. Run Kaggle ghi 3.12.15; môi trường CPU kiểm chứng ghi trong artifact riêng. Không ghi `python==...` vào requirements pip.

Từ gốc repo trên Linux:

```bash
python -m pip install uv
uv python install 3.12
uv venv --python 3.12 .venv-lab03
uv pip install --python .venv-lab03/bin/python -r labs/lab03/requirements.txt
uv pip check --python .venv-lab03/bin/python
.venv-lab03/bin/python -m pytest labs/lab03/tests -q
```

Windows dùng `.venv-lab03/Scripts/python.exe`. Kiểm thử cục bộ chỉ dùng sanity/fixture nhỏ. Full training do người chạy notebook/CLI thực hiện trên Kaggle hoặc local; CUDA được khuyến nghị.

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false .venv-lab03/bin/python labs/lab03/scripts/run_frozen_trial.py --config labs/lab03/config/frozen_trial.json --device auto
```

Tái kiểm toán và xuất artifact report từ kết quả đã lưu, không train:

```bash
.venv-lab03/bin/python labs/lab03/scripts/finalize_results.py
```

## 6. Kết quả và đánh đổi

| Thí nghiệm | Threshold validation | Test accuracy | Macro-F1 | Test loss | Confidence ECE |
|---|---:|---:|---:|---:|---:|
| Full fine-tuning 10 epoch cũ | 0.50 | 83.40% | 83.40% | 0.798076 | 0.146071 |
| Full fine-tuning + augmentation | 0.65 | 84.15% | 84.15% | 0.625951 | 0.130603 |
| **Frozen 4 layers — bản chính** | **0.45** | **82.83%** | **82.82%** | **0.412560** | **0.037950** |

Frozen giảm test loss và sai lệch confidence, accuracy thấp hơn bản augmented 14 mẫu, tương đương -1.31 điểm phần trăm. Các cấu hình khác cả LR, dropout, freeze và criterion checkpoint: đây là so sánh mô tả, không phải ablation xác định tác động riêng từng yếu tố. Một seed không đo biến thiên huấn luyện; test đã được xem trước nên không gọi là benchmark mới chưa đụng tới.

Ma trận bản chính tại 0.45: **TN=429, FP=104, FN=79, TP=454**. Threshold 0.5 có cùng 883 mẫu đúng nhưng khác precision/recall; không đổi policy đã chọn vì một chỉ số test nhỉnh hơn.

## 7. Thu hồi artifact và checkpoint

Sau training, hai ZIP nằm ở `/kaggle/working`:

- `lab03_review_<run>.zip`: config, log, predictions, epoch history, ảnh, source hashes và manifest; dùng kiểm tra và hoàn thiện report.
- `lab03_checkpoint_<run>.zip`: best_model, tokenizer và decision_threshold; giữ riêng để inference hoặc evaluate lại.

Nếu link cell không tải, mở **Output > /kaggle/working > Refresh > dấu ba chấm cạnh ZIP > Download**. Tải notebook có output bằng **File > Download Notebook**; script container không đọc được notebook tương tác đang mở trên web.

Git giữ source, notebook có output, PNG và JSON/log bằng chứng cần thiết. Không commit checkpoint, dataset thô hoặc ZIP nặng. Review ZIP của bản chính không có weights: audit trong repo kiểm tra records và SHA256, không giả vờ nạp lại checkpoint. `reload_check.json` gốc ghi xác minh weights và inference trên bốn câu validation tại Kaggle, không phải toàn bộ test.

## 8. Giới hạn và điều kiện kết luận

Dấu hiệu overfitting mạnh đã giảm trong 10 epoch quan sát được; không bảo đảm bằng 0 ở mọi dữ liệu hoặc ngân sách dài hơn. Hai đường gần nhau chưa chứng minh optimal: regularization quá mạnh có thể hạn chế khả năng thích nghi. Synthetic labels có thể nhiễu. Chưa có multi-seed study, semantic leakage audit hoặc kiểm toán dữ liệu pretraining. Bootstrap CI chỉ phản ánh lấy mẫu test có điều kiện trên model/threshold cố định.

Không cần train lại để đọc bài, tạo report hoặc xem ảnh. Muốn thay cấu hình phải tạo run mới, công bố tiêu chí trước khi nhìn test và giữ nguyên bằng chứng cũ.

## 9. Inference từ checkpoint frozen đã tải

Giải nén checkpoint ZIP vào thư mục lưu riêng, ngoài run evidence đã commit. Với ZIP cũ, chuẩn bị policy/config bằng cách đối chiếu SHA256 weights với frozen trial đã chọn; không train và không đổi threshold:

```bash
CHECKPOINT_DIR="/path/to/extracted/best_model"
.venv-lab03/bin/python labs/lab03/scripts/prepare_frozen_checkpoint.py --model-path "$CHECKPOINT_DIR"
.venv-lab03/bin/python labs/lab03/scripts/infer.py --model-path "$CHECKPOINT_DIR" --text "The acting was excellent, but the story was disappointing."
```

Thay đường dẫn bằng thư mục best_model thực tế. Lệnh đầu từ chối checkpoint khác SHA256 hoặc đường dẫn nằm trong raw run nguồn. Lệnh inference tự đọc threshold đã gắn và lưu predictions/PNG vào run inference riêng. Các lượt train mới xuất sẵn policy có checkpoint identity và training_config.json trong checkpoint ZIP. Việc thêm metadata không làm thay đổi weights hoặc cần train lại.


## 10. Phạm vi tinh gọn

Giữ hai notebook và các module/CLI còn dùng. Bỏ tám notebook cũ sau khi gom nội dung, bốn helper copy-cell, wrapper train.py và sáu CLI training/audit/recovery cũ không còn dùng. Bỏ các run render/báo cáo kiểm thử phần mềm cũ bị lặp; giữ bằng chứng kiểm tra mới nhất. Không xóa predictions, logs hoặc manifests của thí nghiệm dùng trong báo cáo. Cấu hình và artifact lịch sử được giữ để kiểm toán so sánh, không phải notebook người dùng cần chạy thêm.

Notebook đã có output là bản trình bày chính. Renderer lưu execution summary và SHA256 của notebook thay vì chép thêm toàn bộ notebook vào mỗi run, tránh lưu lặp ảnh base64. Không cần train lại vì thay bố cục.
