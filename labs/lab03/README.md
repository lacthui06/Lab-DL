# Lab 03 — Getting started with Hugging Face

Bài thực hành gồm pretrained sentiment analysis và fine-tuning DistilBERT cho phân loại văn bản nhị phân trên toàn bộ Rotten Tomatoes.

**Kết quả chính đã chạy trên Kaggle:** frozen DistilBERT, đủ **10 epoch**, chọn checkpoint **epoch 9 theo validation loss**, threshold **0.45 chọn bằng validation Macro-F1**. Test đúng **883/1.066**, Accuracy **82.83%**, Binary F1 **83.23%**, Macro-F1 **82.82%**, test loss **0.412560**.

**Mục tiêu của bản chính là giảm dấu hiệu overfitting**, không chọn accuracy cao nhất từ các lần test. Epoch 10: train/validation accuracy 84.25%/83.77%, chênh 0.47 điểm phần trăm; train/validation loss 0.3581/0.3696. Bản full fine-tuning có accuracy cao hơn nhưng gap lớn hơn. Một seed và test đã được xem trước không đủ để khẳng định tối ưu tuyệt đối hoặc không bao giờ overfit.

## 1. Đọc và trình bày bài

| File | Vai trò |
|---|---|
| [Notebook tích hợp](notebooks/lab03_huggingface_complete.ipynb) | Bản trình bày chính, cả hai Exercise, có roadmap/module, bảng và ảnh inline |
| [Notebook Exercise 1](notebooks/01_sentiment_analysis_pipeline.ipynb) | Tokenization, pipeline, manual forward; bằng chứng pretrained inference lịch sử |
| [Notebook Exercise 2](notebooks/02_finetuning_binary_classification.ipynb) | Frozen trial, đủ 10 epoch, threshold, test và phân tích lỗi |
| [Notebook 08 — train Kaggle](notebooks/08_kaggle_frozen_encoder_trial.ipynb) | Chạy từ pretrained base, đúng 10 cell, tự xuất ZIP |
| [Notebook 05 — study lịch sử](notebooks/05_study_results_and_discussion.ipynb) | Giữ bằng chứng full fine-tuning để so sánh |
| [Description](description_lab_03.md) | Đề gốc, các điểm thảo luận, quyết định và tiêu chí nghiệm thu |
| [Walkthrough](walkthrough_lab_03.md) | Lý thuyết, phương pháp, lịch sử thí nghiệm, báo cáo cuối và câu hỏi bảo vệ |
| [Artifact report](outputs/published/index.md) | Ảnh/bảng đã chọn để sử dụng trực tiếp khi làm report |

Notebook báo cáo đã được thực thi ở `MODE="view"`, giữ output trong file. Chế độ này đọc artifacts, không tải model/dataset và không train. Notebook 04/06/07/08 là template GPU; renderer offline bỏ qua. Chỉ notebook 08 là cấu hình train được khuyến nghị cho bản chính hiện tại; các profile cũ giữ để tái lập lịch sử.

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
  notebooks/              bản báo cáo có output và template train Kaggle
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

## 4. Chạy từ đầu trên Kaggle

Import [notebook 08](notebooks/08_kaggle_frozen_encoder_trial.ipynb) từ nhánh chứa code hiện tại, dùng session mới, bật **GPU + Internet**, chạy lần lượt 10 cell hoặc Run All. Cell 1 clone repo và tạo Python 3.12 riêng bằng uv; không hạ cấp PyTorch của kernel Kaggle. Không cần thêm checkpoint cũ vào Input vì lượt này train từ pretrained base.

| Cell | Công việc |
|---|---|
| 1 | Repo, Python 3.12 riêng, GPU, dependencies |
| 2 | Chốt frozen_trial config |
| 3 | Preflight tests và sanity fixture CPU |
| 4 | Exercise 1 và ba ảnh inline |
| 5 | EDA official dataset và ảnh inline |
| 6 | Preview augmentation trước/sau |
| 7 | Baseline, train 10 epoch, khóa policy, test; hiện epoch table/curves |
| 8 | Trình bày quyết định checkpoint/threshold đã ghi |
| 9 | Test comparison, confusion, ROC/PR, calibration và lỗi |
| 10 | Kiểm tra SHA256 và link tải review/checkpoint ZIP |

Cell 7 thực hiện train -> validation selection -> test đúng thứ tự. Cell 8–9 chỉ hiển thị bằng chứng; không chọn lại model sau khi xem test. Không chạy lại cell 7 chỉ để xem hình. Session restart làm mất biến, cần chạy lại các cell setup; output cũ hiển thị không có nghĩa biến còn tồn tại.

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

Windows dùng `.venv-lab03/Scripts/python.exe`. Chỉ sanity/fixture cực nhỏ chạy trên CPU; full training cần Kaggle GPU.

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false .venv-lab03/bin/python labs/lab03/scripts/run_frozen_trial.py --config labs/lab03/config/frozen_trial.json
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
