# Description — New Lab03: Getting started with Hugging Face

## 1. Đề bài

### Exercise1: Sentiment Analysis with Hugging Face

1. Cài Hugging Face Transformers.
2. Dùng pretrained sentiment model từ Hugging Face Hub.
3. Tokenize một câu mẫu.
4. Dự đoán sentiment của câu.

### Exercise2: Fine-tuning for Binary Text Classification

1. Cài transformers, datasets và evaluate.
2. Tải dataset phân loại văn bản nhị phân.
3. Tải pretrained model và tokenizer.
4. Preprocess dữ liệu cho mô hình.
5. Định nghĩa training arguments.
6. Tạo Trainer và fine-tune.
7. Đánh giá mô hình đã fine-tune.

Đề không bắt buộc một mức accuracy cụ thể. Các quyết định dataset, baseline, architecture, regularization, threshold, resource monitoring và artifact dưới đây làm rõ thiết kế thực nghiệm.

Thí nghiệm chạy **tối đa 3 epoch**, early stopping theo validation loss. Kết quả chỉ được ghi sau khi thực thi, không đặt trước accuracy hoặc threshold.

## 2. Phạm vi và sản phẩm

Hai notebook có roadmap/module tham chiếu ở đầu:

| Notebook | Phạm vi | Đầu ra |
|---|---|---|
| 01 | EDA → preprocessing → feature engineering | Quality/length/token/mask/TF-IDF/augmentation figures và bảng |
| 02 | Pretrained inference → baseline → fine-tune → evaluate | Learning/performance/threshold/confusion/calibration/error figures, model và report artifacts |

Ảnh hiển thị inline và cùng PNG lưu trong outputs/. JSON lưu số chính xác để kiểm toán; không thay thế visualization.
Walkthrough là tài liệu sống duy nhất cho phương pháp, diễn giải kết quả và bằng chứng.

## 3. Bài toán và data contract

- Input: một câu review tiếng Anh.
- Target: Negative=0, Positive=1.
- Dataset: official Rotten Tomatoes, pin revision trong config.
- Splits: 8.530 train, 1.066 validation, 1.066 test.
- Giữ split và thứ tự, không subsample để làm thí nghiệm chính.
- Fit features/model bằng train; validation chọn model/threshold; test đo sau khi khóa policy.

## 4. Model architecture

### 4.1. Exercise1

Dùng `distilbert/distilbert-base-uncased-finetuned-sst-2-english` cho sentiment inference.
Đối chiếu pipeline với manual forward/softmax trên cùng model/input.
Hiển thị tokens, IDs, mask và predictions.

### 4.2. Exercise2

Backbone `distilbert/distilbert-base-uncased`, classification head hai logits được khởi tạo cho nhiệm vụ.

| Khối | Tensor / cấu trúc | Vai trò |
|---|---|---|
| Input IDs và attention mask | [B,L], L<=128 | Chỉ định token và padding |
| WordPiece + position embedding | [B,L,768] | Biểu diễn token/vị trí, frozen |
| Encoder1–4 | Attention + FFN + residual/LayerNorm | Frozen pretrained representation |
| Encoder5–6 | Hidden768, 12heads, FFN3072 | Thích nghi nhiệm vụ |
| CLS pooling | [B,768] | Đại diện câu ở token đầu |
| Pre-classifier | Linear768→768 + ReLU | Chuyển biểu diễn cho head |
| Dropout | p=0.4 | Regularize head khi train |
| Classifier | Linear768→2 | Logits hai lớp |
| Objective | Cross-entropy trên logits | Học supervised |
| Decision | Softmax + threshold Positive | Policy chọn bằng validation |

Dropout encoder/attention đều0.2. Freeze ngăn gradient update, không tự tắt dropout.
Optimizer chỉ chứa weights trainable; hai nhóm LR encoder/head và decay/no-decay được ghi vào artifact.
Model.config thực tế được lưu thành model_architecture.json; hình architecture được hiển thị trong modeling.

### 4.3. Vì sao chọn kiến trúc này?

- Transfer learning tận dụng biểu diễn ngôn ngữ thay vì học Transformer từ corpus nhỏ.
- Sáu encoder blocks phù hợp ngân sách thực hành Kaggle T4 hơn một backbone lớn.
- English uncased WordPiece phù hợp loại văn bản, không yêu cầu tự học vocabulary Transformer.
- Head đơn giản tạo tensor/loss/inference flow dễ kiểm chứng theo mục tiêu Hugging Face.
- Freeze phần lớn backbone giảm số weights cần update, nhưng có thể làm adaptation yếu hơn.
- Baseline TF-IDF độc lập giúp kiểm tra lợi ích thực tế; không mặc định kiến trúc phức tạp luôn tốt hơn.

Không có bằng chứng ablation để kết luận số blocks frozen hoặc mức dropout là tối ưu trên mọi dữ liệu.

## 5. Discussion và quyết định dữ liệu

### D01 — Official splits hay tự chia lại?

**Thảo luận:** chia lại có thể làm benchmark khó so sánh và tạo rò rỉ khi vô tình dùng test để tuning.
**Quyết định:** giữ official splits/full sizes, kiểm labels và text trước pipeline.
**Kiểm chứng:** data_preview/data_statistics, fingerprints, quality và overlap audit.

### D02 — EDA cần kiểm gì?

**Thảo luận:** chỉ xem nhãn và vài câu chưa đủ để chọn preprocessing.
**Quyết định:** kiểm missing/blank/invalid labels, exact/normalized duplicates, cross-split overlap; word/token lengths theo lớp; quantiles và tỷ lệ truncation; phủ định/tương phản; train examples.
**Kiểm chứng:** quality, length/cutoff và text-signal tables/figures từ lượt prepare.

### D03 — Có xóa stopwords/punctuation không?

**Thảo luận:** not/but và dấu câu có thể mang sentiment hoặc quan hệ tương phản.
**Quyết định:** giữ text cues; tokenizer pretrained thực hiện lowercase/WordPiece.
**Kiểm chứng:** tokens/IDs/mask trên câu train thật; review examples.

### D04 — Max length và padding?

**Thảo luận:** cutoff ngắn có thể mất nghĩa; pad cố định dài gây lãng phí.
**Quyết định:** max_length128, train-only sensitivity audit ở64/96/128/256; dynamic padding, GPU FP16 pad bội8.
**Kiểm chứng:** token quantiles, truncated fraction và attention-mask heatmap.

### D05 — Feature engineering cho hai nhánh?

**Thảo luận:** baseline sparse và pretrained Transformer cần representation khác nhau.
**Quyết định:** TF-IDF unigram/bigram + Logistic Regression; DistilBERT dùng token embeddings/contextual attention. Không nối sparse TF-IDF vào Transformer head.
**Kiểm chứng:** feature_summary/top mean TF-IDF, baseline validation candidates và model config.

### D06 — Fit TF-IDF ở đâu?

**Thảo luận:** IDF/vocabulary học từ heldout sẽ rò rỉ phân bố.
**Quyết định:** EDA vectorizer fit original train; baseline modeling fit lại trên effective augmented train, cùng vectorizer parameters. Validation chọn C, không fit vocabulary.
**Kiểm chứng:** fitted_split/train_shape; tests sentinel heldout token không vào vocabulary.

### D07 — Có tăng dữ liệu không?

**Thảo luận:** thêm diversity có thể giúp generalization, nhưng lexical substitution có nguy cơ đổi nghĩa.
**Quyết định:** thay một từ/cụm cùng cực tính, tránh phủ định/tương phản, dedup, cân bằng variants và trần25%. Giữ mọi original train, không ép đủ trần bằng duplicate.
**Kiểm chứng:** parent/variant records, digest, counts và examples inline.

Synthetic labels kế thừa parent là weak labels, không coi là ground truth được người gán lại.

## 6. Discussion và quyết định training

### D08 — Vì sao có baseline?

**Thảo luận:** không có baseline thì khó biết transfer learning mang lại giá trị thực tế.
**Quyết định:** TF-IDF + Logistic Regression, C∈{0.1,1,10}, chọn bằng validation Macro-F1 và loss tie-break.
**Kiểm chứng:** candidate table và test comparison với policy đã khóa.

### D09 — Freeze và discriminative LR?

**Thảo luận:** corpus nhỏ có thể gây thay đổi quá mức pretrained weights; head mới cần học nhanh hơn.
**Quyết định:** freeze embeddings và4 blocks đầu; encoderLR5e-6, headLR2e-5.
**Kiểm chứng:** requires_grad/optimizer groups, trainable parameters và LR history.

### D10 — Regularization nào?

**Quyết định:** encoder/attention/head dropout0.2/0.2/0.4; AdamW weight_decay0.1 cho tensors>=2D, không decay bias/LayerNorm; warmup10% rồi linear decay; clipping norm1.
**Lý do:** giảm đồng thích nghi, hạn chế weights lớn và update bất ổn; vẫn cần quan sát validation để phát hiện underfit/overfit.
**Giới hạn:** không tăng regularization tùy tiện hoặc tuyên bố mỗi kỹ thuật có causal benefit khi chưa ablation.

### D11 — Ngân sách epoch và early stopping?

**Quyết định:** tối đa 3 epoch, evaluate/save mỗi epoch; early stopping theo validation loss, patience2/min_delta0.001.
**Lý do:** dừng khi validation không đủ cải thiện để tiết kiệm compute và hạn chế train tiếp không hữu ích.
**Kiểm chứng:** requested/completed/stopped_early, consecutive epoch history, checkpoint và log.

Min_delta điều khiển patience; checkpoint vẫn chọn loss thấp nhất thực tế. Evaluate(train) không được tính thành validation failure. Biểu đồ phản ánh đúng số epoch thực chạy; với budget3/patience2, lượt hoàn thành bình thường đi tới epoch3.

### D12 — Chọn checkpoint bằng gì?

**Thảo luận:** accuracy ít nhạy với việc sai ngày càng tự tin; loss cung cấp tín hiệu xác suất.
**Quyết định:** min validation loss; reload checkpoint đó trước threshold/test, không dùng epoch cuối mặc định.
**Kiểm chứng:** selection_summary, trainer history và save/reload logits consistency.

## 7. Discussion và quyết định binary threshold

### D13 — Có dùng threshold0.5 cố định không?

**Quyết định:** giữ0.5 làm mốc, quét19ngưỡng0.05–0.95 bước0.05 trên validation Macro-F1; hòa chọn gần0.5 rồi nhỏ hơn.
**Lý do:** threshold quyết định precision/recall trade-off, không nhất thiết0.5 là operating point tốt cho objective.
**Kiểm chứng:** scan đủ19 điểm, selected policy/hash và figure inline.

Không ép chọn khác0.5. Test không tham gia sweep/selection.

### D14 — Báo những metric nào?

Accuracy, Binary/Macro-F1, precision/recall, confusion counts và row-normalized rates, test loss, ROC-AUC/PR và calibration diagnostics. Bootstrap là uncertainty conditional trên fixed model/test sampling, không thay nhiều seed training.

### D15 — Phân tích lỗi thế nào?

Đọc câu đúng/sai theo lớp, confidence, policy error examples; xem phủ định, tương phản và implicit sentiment. Sample grid ưu tiên confidence cao để kiểm lỗi, không phải random sample đại diện. Giải thích ngôn ngữ là giả thuyết, không phải causal attribution.

## 8. Performance, artifact và tiêu chí hoàn thành

| Yêu cầu | Phạm vi đo / bằng chứng |
|---|---|
| Train/validation accuracy và loss | Full datasets eval-mode mỗi epoch, learning_curves/history |
| Train time/sample | Train-phase wall time / effective train size |
| Test time/sample | Full predict runtime / test size, batch throughput |
| RAM | Parent-process RSS sampled20ms |
| VRAM | CUDA allocated/reserved peak trên selected GPU |
| Per sample memory | Peak/nominal batch slots, có shared model memory |
| Inline visualization | Cùng PNG đã lưu được display trong notebook |
| Publish | Hình/bảng/raw log, selection, catalog và SHA256 |
| Reload | Model/tokenizer/policy khớp weights và logits tolerance |

Mỗi session có ID riêng để chỉ gói thêm Exercise1/preview artifacts cùng phiên. Training không nạp report selection để lấy metric trước khi chạy. MODE=view chỉ đọc selection sau khi có kết quả.

Review ZIP không có weights; checkpoint ZIP giữ riêng cho evaluate/infer. Không commit ZIP, raw dataset hoặc model weights vào Git. Notebook có output cần tải từ Kaggle sau Run All; không tạo trước số đo hoặc hình performance khi chưa train.

Tiêu chí hoàn thành thực nghiệm: pipeline pass gates, có kết quả thật với policy validation-only, dữ liệu/weights/log/hash nhất quán, ảnh inline và published dùng được sau pull. Không đặt điều kiện accuracy tùy ý hoặc bảo đảm hoàn toàn không overfit.

## 9. Trạng thái thực nghiệm và điều kiện hoàn thành

Thí nghiệm `20261104T110654Z_frozen_trial_00eccdd1` đã hoàn thành trên Kaggle TeslaT4. Hoàn thành3/3epoch, chọn epoch3 theo validation loss0,377385; quét19thresholds validation chọn0,5. Test accuracy82,18%, Binary F182,11%, Macro-F182,18%, loss0,414510. Số liệu được tái tính từ saved predictions; source33file khớp commit a759119.

Dataset giữ8530/1066/1066; augmentation thêm274mẫu cân bằng137/lớp, effective train8804. Không có missing/blank/invalid labels hoặc normalized overlap giữa các split trong audit exact-text. TF-IDF train-only có24083features,207847nonzeros. Max token length78 nên cutoff128 không cắt mẫu train nào.

Cuối epoch3, train/validation accuracy83,60%/83,49%, gap0,109điểm phần trăm; loss0,369313/0,377385. Validation loss giảm cả3epoch. Chưa thấy dấu hiệu overfitting rõ trong cửa sổ này; không bảo đảm hết overfit hoặc cấu hình tối ưu. Dropout, decay, freeze và augmentation được đánh giá như một cấu hình kết hợp, không suy ra hiệu quả nhân quả của từng kỹ thuật.

Hình, bảng và log đã có trong [published report](outputs/published/index.md), [preparation report](outputs/preparation/index.md); hai notebook hiển thị cùng artifact ở MODE=view. Source prepare/train vẫn sẵn sàng tạo run mới, không resume weights của run này.

Review ZIP đã qua kiểm tra integrity, official splits, augmentation, optimizer/dropout, epoch/checkpoint, validation-only threshold, metrics, learning curves và resource telemetry. Không có checkpoint weights trong ZIP; local reviewer chỉ kiểm được manifest/probabilities và bản ghi reload trên Kaggle.

Patience=2 được giữ cố định. Trong budget 3 epoch, cơ chế này có thể kích hoạt từ lần validation thứ ba; không thể kết thúc trước epoch3 khi đánh giá một lần mỗi epoch. Không đổi patience chỉ để tạo ra biểu hiện dừng sớm. Min_delta điều khiển patience, không thay quy tắc chọn checkpoint minimum loss.
