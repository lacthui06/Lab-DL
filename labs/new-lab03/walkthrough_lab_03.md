# Walkthrough — New Lab03: Binary Sentiment Classification với Hugging Face

**Mục tiêu:** xây dựng pipeline đầy đủ từ khám phá dữ liệu đến fine-tuning, chọn policy và xuất artifact có thể dùng trực tiếp cho report.

**Profile:** official Rotten Tomatoes, DistilBERT base, tối đa 3 epoch, early stopping và regularization.

**Quy ước bằng chứng:** kết luận định lượng chỉ dựa trên artifact của phiên thực thi. Tài liệu này mô tả phương pháp, cách vận hành và cách đọc kết quả; không ghi trước accuracy, threshold hoặc selected epoch.

**Playbook:** `docs/eda.md`, `docs/feature.md`, `docs/dl_flow.md`, `docs/dl_data.md`, `docs/dl_training.md`.

## 1. Đặc tả và thiết kế

### 1.0. Diễn giải yêu cầu

Exercise1 yêu cầu dùng pretrained sentiment model, tokenize và dự đoán sentiment.
Exercise2 yêu cầu tải dataset/model/tokenizer, preprocess, định nghĩa Trainer, fine-tune và evaluate.

Bài triển khai bổ sung baseline để có mốc so sánh, EDA để lựa chọn preprocessing và threshold sweep vì target là binary.
Notebook phải vừa giải thích vừa trực quan hóa, không chỉ gọi một script rồi in JSON.

Mọi ảnh được hiển thị từ chính PNG lưu trong outputs/.
Bảng số liệu, predictions, config và log cũng được lưu để tái sử dụng khi viết report.

### 1.1. Roadmap

| Phase | Câu hỏi | Module/file | Bằng chứng |
|---|---|---|---|
| Setup | Môi trường có phù hợp không? | requirements, notebook_runtime | snapshot/config |
| Preflight | Model/loss/training có chạy đúng không? | tests, trainer | pytest log/XML, sanity |
| Pretrained sentiment | Pipeline/manual có nhất quán không? | classifier, run_exercise1 | tokens/probabilities/PNG |
| Data contract | Đúng full official splits chưa? | loader | sizes/labels/fingerprints |
| EDA | Có vấn đề chất lượng hoặc leakage không? | data_preparation | quality/count/length PNG |
| Preprocess | Text thành tensor như thế nào? | loader/tokenizer | IDs/mask/cutoff |
| Feature engineering | Hai nhánh dùng representation nào? | data_preparation/augmentation | TF-IDF/variants |
| Baseline | Mô hình đơn giản đạt gì? | study.fit_text_baseline | C candidates/policy |
| Fine-tuning | Model học và generalize ra sao? | frozen/trainer | epoch_history/curves |
| Performance | Compute/memory tính thế nào? | performance | timer/RSS/CUDA |
| Selection | Chọn checkpoint/threshold bằng gì? | trainer/thresholds | policy/scan/checksum |
| Test | Policy khóa có hiệu quả ra sao? | metrics/plots | confusion/metrics/calibration |
| Reuse | Pull về có lấy hình/log được không? | runs/publish | index/catalog/manifest/ZIP |

### 1.2. Problem formulation

Input là một câu review tiếng Anh.
Target có hai giá trị: Negative=0 và Positive=1.
Model trả hai logits, sau đó softmax tạo xác suất hai lớp.

Policy dự đoán Positive khi P(Positive)>=threshold.
Confidence của nhãn dự đoán không đồng nhất với xác suất Positive nếu nhãn dự đoán là Negative.
Threshold phải nhận P(Positive), không nhận confidence của lớp thắng.

Không đặt accuracy target tùy ý ngoài đề bài.
Đánh giá chất lượng gồm khả năng phân loại, confidence, generalization, tài nguyên và tính toàn vẹn bằng chứng.

### 1.3. Official dataset

Rotten Tomatoes là sentence-level movie-review polarity, không phải full-length review corpus.
Dataset name và revision được cố định trong config.

| Split | Số mẫu | Vai trò |
|---|---:|---|
| Train | 8.530 | Fit features và model, sinh variants |
| Validation | 1.066 | Chọn baseline C, checkpoint, early stopping, threshold |
| Test | 1.066 | Đánh giá policy sau khi khóa |

Không subsample cho kết quả chính.
Không chia lại test hoặc trộn test vào train để tăng accuracy.
Giữ thứ tự/index để đối chiếu predictions với raw text.

### 1.4. Data và tensor contract

Loader phải trả đủ train/validation/test với labels0/1.
Text phải là string không rỗng sau strip.
Tokenization giữ label riêng, không mã hóa label vào input.

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| input_ids | [B,L] | Token IDs từ vocabulary pretrained |
| attention_mask | [B,L] | 1=valid, 0=padding |
| labels | [B] | Class index0/1 |
| logits | [B,2] | Score trước softmax |
| probabilities | [B,2] | Tổng hai xác suất xấp xỉ1 |

Batch size và sequence length là hai chiều khác nhau.
Dynamic padding có thể làm L thay đổi giữa các batch.
Tensor shape đúng chưa đủ; còn phải kiểm labels, finite values và class order.

### 1.5. Baseline đi trước

Dùng TF-IDF unigram/bigram và Logistic Regression.
Baseline có ít phần tùy chỉnh, dễ kiểm dữ liệu và tạo mốc so sánh hữu ích.

Vocabulary/IDF/classifier chỉ fit trên effective train.
Validation dùng chọn C trong0.1/1/10 bằng Macro-F1, rồi loss và C làm tie-break.
Sau đó baseline cũng chọn threshold bằng validation.

Transformer không mặc định tốt hơn chỉ vì có nhiều tham số.
Report cần so các nhánh trong cùng split và cùng effective train pool.
Không so accuracy của các policy khác nhau mà bỏ qua precision/recall trade-off.

### 1.6. Architecture Exercise1

Dùng sentiment model SST-2 pretrained từ Hugging Face Hub.
Revision model được pin cùng config để giảm thay đổi do upstream.

Pipeline là giao diện cấp cao cho tokenizer/model/postprocessing.
Manual inference tokenizes, forward logits và softmax trực tiếp.
So hai nhánh trên cùng model/input ở eval mode để phát hiện class-order hoặc postprocessing lỗi.

Đầu ra gồm tokens, IDs, attention mask, nhãn và probabilities.
Exercise1 không cập nhật weights và không dùng làm fine-tuning checkpoint của Exercise2.

### 1.7. Architecture Exercise2

Dùng DistilBERT base với head phân loại hai lớp.
Pretrained representation được tận dụng; head học nhiệm vụ sentiment từ labels train.

Luồng:

```text
Text
  -> WordPiece + attention mask
  -> Token/position embedding
  -> Transformer encoder blocks1..6
  -> First-token [CLS] hidden state
  -> Linear768 + ReLU
  -> Dropout0.4
  -> Linear2 logits
  -> Cross-entropy khi train
  -> Softmax + threshold khi inference
```

Hidden dimension768, attention12heads và FFN intermediate3072.
Embedding và4 blocks đầu frozen;2 blocks cuối và head trainable.
Model.config thực tế được ghi vào model_architecture.json trước training.
Sơ đồ architecture trong notebook được dựng từ cấu hình đó.

### 1.8. Vì sao chọn DistilBERT và frozen representation?

Transfer learning phù hợp khi dữ liệu task nhỏ hơn nhiều so với dữ liệu cần để học ngôn ngữ từ đầu.
Backbone sáu blocks giúp giữ pipeline thực hành gọn và phù hợp Kaggle T4.

Freeze hạn chế số weights có thể thay đổi, giảm nguy cơ học thuộc corpus nhỏ.
Đổi lại, freeze nhiều có thể hạn chế adaptation và gây underfitting.

Không có ablation kiến trúc để khẳng định sáu blocks hoặc freeze4 là tối ưu toàn cục.
Lựa chọn được đánh giá qua validation, baseline và error analysis của run thật.
Số tham số không dùng thay thế phép đo RAM/VRAM.

### 1.9. Profile training

| Thành phần | Giá trị |
|---|---|
| Seed | 42 |
| Max epochs | 5 |
| Train/eval batch | 16/32 |
| Gradient accumulation | 1 |
| Encoder/head LR | 5e-6 / 2e-5 |
| Optimizer | AdamW |
| Weight decay | 0.1 cho weights>=2D |
| Warmup | 10% tổng updates |
| Scheduler | Linear decay |
| Max gradient norm | 1 |
| Encoder/attention/head dropout | 0.2 / 0.2 / 0.4 |
| Max sequence length | 128 |
| Eval/save schedule | Mỗi epoch |
| Checkpoint criterion | Min validation loss |
| Early stopping | Patience2, min_delta0.001 |
| Workers | 0 |
| AMP | FP16 khi CUDA khả dụng |

Config nguồn là `config/frozen_trial.json`.
Mỗi run lưu bản copy config và environment thực tế.
Không suy luận ngược hyperparameters chỉ từ tên notebook.

### 1.10. Regularization có vai trò gì?

Dropout giảm phụ thuộc vào một tập feature cố định trong lúc train.
Encoder và attention dropout0.2 regularize representation; head dropout0.4 regularize classifier mới.

Weight decay tác động vào weights để hạn chế nghiệm quá lớn.
Bias/LayerNorm không bị decay trong nhóm optimizer được tạo.

Warmup giảm update đột ngột đầu training; linear decay hạ LR dần.
Gradient clipping hạn chế norm lớn nhưng không hợp thức hóa NaN hoặc loss không hữu hạn.

Freeze, augmentation và early stopping là các biện pháp bổ sung.
Kết hợp chúng không bảo đảm hết overfit; regularization quá mạnh cũng có thể underfit.

### 1.11. Quy tắc chọn checkpoint và threshold

Hai quyết định được tách rõ:

1. Trong training: checkpoint có validation loss thấp nhất.
2. Sau reload checkpoint: threshold tối đa validation Macro-F1 trên grid định trước.

Loss xét cả độ tự tin của dự đoán.
Macro-F1 xét chất lượng nhãn theo cả hai lớp ở một operating point.
Hai objective phục vụ hai quyết định khác nhau; không dùng test để đổi objective.

Policy được lưu trước khi predict test và gắn hash weights.
Threshold không được chuyển sang một checkpoint khác mà không kiểm identity.

## 2. Vận hành và phương pháp thực nghiệm

### 2.1. Khởi tạo môi trường

Dùng Python3.12 cùng requirements đã pin.
Kaggle bootstrap tạo interpreter con trong `/kaggle/working`, giữ kernel hiển thị độc lập với ML dependencies.

Setup kiểm Python, PyTorch và CUDA availability.
DEVICE=cuda yêu cầu CUDA thật, không âm thầm chuyển full training sang CPU.

Local dùng kernel có requirements tương ứng.
Dùng CPU cho fixture/sanity, full training khuyến nghị GPU.
Đường dẫn ghi luôn nằm trong workspace có quyền ghi.

### 2.2. Hai notebook và chế độ chạy

Notebook01 mặc định MODE=prepare.
Notebook02 mặc định MODE=train, DEVICE=cuda.

| Chế độ | Hành vi |
|---|---|
| prepare | Tạo EDA/feature artifacts, không train model |
| train | Preflight, inference, baseline, fine-tune, evaluate, export |
| view | Đọc selection sau khi đã có artifact, không download/train |

Không cần giữ biến Python giữa notebook01 và02.
Modeling script tự dựng lại cùng loader/tokenizer/augmentation từ config.
Chạy các cell theo thứ tự để session/helper/config được định nghĩa đầy đủ.

### 2.3. Session isolation

Mỗi NotebookSession có một UUID trong LAB03_SESSION_ID.
Subprocess ML nhận cùng session ID qua environment.
Run context ghi session_id vào config artifact mà không sửa config object của caller.

MODE=train bắt đầu với paths rỗng.
Nó không dùng report selection để lấy metrics trước training.
Sau từng bước, session trỏ đến run vừa hoàn thành.

Khi publish/package, các Exercise1/preview runs chỉ được kèm nếu session ID khớp.
Điều này ngăn gói kết quả vô tình lẫn artifact của phiên khác.
Isolated run folders vẫn giữ nguyên bytes sau khi manifest được tạo.

### 2.4. Preflight và sanity overfit

Chạy pytest trước GPU job.
Tests dùng tiny fixtures để kiểm labels, masks, shapes, optimizer và logic training.

Sanity test học một minibatch nhỏ để xác nhận model/loss/optimizer nối đúng.
Model fixture là disposable, không dùng weights sau sanity cho thí nghiệm chính.

Early-stopping fixture tạo validation plateau để xác nhận dừng và giữ checkpoint đúng.
Threshold tests kiểm NaN/Inf, class order, grid, tie-break và policy identity.
Preflight failure phải được đọc từ pytest.log trước khi tiếp tục training.

### 2.5. EDA chất lượng

Đếm missing/nonstring và blank text ở từng split.
Kiểm labels có đúng0/1 và số mẫu từng lớp.

Đếm exact duplicates và normalized duplicates nội bộ.
Normalized audit dùng lowercase/whitespace normalization để tìm text trùng theo quy tắc đơn giản.

Đếm giao nhau train-validation, train-test và validation-test.
Kiểm cùng normalized train text có nhiều label khác nhau không.

Audit không chứng minh đã phát hiện mọi semantic near-duplicate.
Không tự loại test samples dựa trên lỗi model hoặc để tăng điểm.
Nếu có vấn đề chất lượng, lưu counts và thảo luận ảnh hưởng vào report.

### 2.6. EDA nhãn và ví dụ

Class-distribution figures hiển thị counts cho từng split.
Train-examples table cho thấy văn phong, punctuation và sự khác biệt giữa các nhãn.

Không chọn toàn câu dễ rồi kết luận task đơn giản.
Ví dụ EDA chỉ minh họa dữ liệu, không thay thống kê toàn tập.

Label balance giúp chọn cách diễn giải metric nhưng không loại nhu cầu confusion/per-class rates.
Khi hai lớp cân bằng, accuracy vẫn có thể che trade-off precision/recall của một policy.
Raw label không được ghép vào text input.

### 2.7. Độ dài theo lớp

Đo whitespace words và WordPiece tokens riêng.
Token count gồm special tokens, không đồng nhất số từ đọc bằng mắt.

Histogram theo Negative/Positive giúp quan sát length bias.
Quantiles giúp nhìn phần đuôi thay vì chỉ mean.

Không coi độ dài tự nó là nhãn sentiment.
Không dùng test distribution để tối ưu cutoff theo kết quả phân loại.
Lưu train_token_lengths để kiểm chứng các kết luận về padding/truncation.

### 2.8. Cutoff sensitivity

Tính tỷ lệ train có token length vượt64/96/128/256.
Configured max_length128 là upper bound khi tokenize.

Nếu một câu bị cắt, phần cuối có thể chứa kết luận hoặc đảo nghĩa.
Nếu max_length quá lớn, padding và memory có thể tăng không cần thiết.

Dynamic padding làm chi phí phụ thuộc câu dài nhất trong batch.
Vì vậy không suy ra runtime chỉ từ cutoff mà chưa đo thực tế.
Notebook hiển thị cùng bảng/PNG cutoff đã lưu.

### 2.9. Giữ phủ định và tương phản

Các từ not, never hoặc but có thể đổi polarity hay trọng tâm câu.
Không áp dụng xóa stopwords/punctuation đại trà trước pretrained tokenizer.

Regex negation/contrast dùng để khảo sát train và bảo vệ augmentation.
Đây là heuristic, có thể bỏ sót trường hợp ngôn ngữ phức tạp.

Không đưa heuristic này thành nhãn mới cho validation/test.
Không kết luận model sai vì một từ chỉ dựa trên việc từ đó xuất hiện trong câu.
Các giải thích cần gắn với text thật và được gọi là giả thuyết.

### 2.10. WordPiece, IDs và mask

Lấy ví dụ thật từ train ở hai lớp.
Lưu full text, tokens, input_ids và attention_mask trong JSON.

Bảng PNG rút gọn token/ID preview để đọc được trên notebook.
Heatmap mask cho thấy padding ở vị trí nào.

Mask0 không có nghĩa là Negative; nó chỉ có nghĩa padding.
Labels là tensor riêng cho loss.
Same tokenizer revision và max_length được dùng trong modeling.

### 2.11. TF-IDF feature engineering

Unigram biểu diễn từ đơn; bigram giữ một phần quan hệ như not good.
min_df2 giảm features chỉ xuất hiện một lần.
max_features50.000 là trần vocabulary, không phải số features bắt buộc.

Sublinear term frequency làm giảm ảnh hưởng số lần lặp term.
EDA vectorizer fit trên original train để mô tả corpus.

Baseline modeling fit lại trên effective augmented train với cùng parameters.
Không coi hai fitted vocabularies là cùng một object.
Validation/test không được dùng fit vocabulary hoặc IDF.

### 2.12. Diễn giải sparse feature figures

Top mean TF-IDF theo lớp mô tả mức feature trung bình trong corpus.
Nó không phải classifier coefficient và không phải causal importance.

Sparse matrix shape, nonzeros và density được lưu.
Sparse array bytes gồm data/indices/indptr, không gồm mọi Python/model overhead.

Không gọi sparse array footprint là RAM peak của training process.
RAM peak có phép đo riêng bằng RSS monitor.
Báo cả representation settings và fitted split khi dùng feature figure trong report.

### 2.13. Train-only augmentation

Giữ mọi original train sample.
Tạo variant bằng thay một từ/cụm với từ/cụm cùng cực tính trong lexicon.

Bỏ câu có phủ định/tương phản để giảm rủi ro đảo nghĩa.
Chặn duplicate normalized text, kể cả collision với heldout text.

Cân bằng số variants hai lớp theo khả năng sinh hợp lệ.
Giới hạn tối đa25%; không nhân bản nguyên câu để đủ quota.
Validation/test không bị thay text hoặc label.

### 2.14. Kiểm duyệt augmentation

Mỗi record lưu parent_index, original_text, synthetic text, label và replacement.
Nhãn variant kế thừa parent là weak supervision.

Cùng polarity từ vựng không đảm bảo cùng sentiment trong mọi context.
Đọc bảng trước/sau để phát hiện câu thiếu tự nhiên hoặc đổi nghĩa.

Counts/diagnostic figures cho thấy nguồn tăng mẫu và balance.
Records digest giúp tái tạo và đối chiếu cùng seed.
Không gọi synthetic labels là ground truth đã được người xác nhận.

### 2.15. Baseline selection

Fit ba Logistic Regression candidates trên cùng train-only TF-IDF pipeline.
Mỗi candidate dự đoán validation để tính Macro-F1 và loss.

Chọn bằng thứ tự: Macro-F1 cao hơn, loss thấp hơn, C nhỏ hơn.
Giữ validation candidate table để kiểm chứng quyết định.

Sau chọn baseline, quét validation threshold bằng cùng objective/tie-break policy.
Baseline test chỉ đo sau khi khóa các quyết định của nó.
Không dùng test để chọn C.

### 2.16. Frozen optimizer groups

Set requires_grad=False cho embeddings và4encoder blocks đầu.
Chỉ collect tham số trainable vào optimizer.

Tách encoder/head với LR5e-6/2e-5.
Trong mỗi nhóm lại tách decay/no-decay theo tensor dimensions.

Bias/LayerNorm không decay; weights>=2D dùng weight_decay0.1.
Lưu parameter names và group settings trước scheduler updates.
Kiểm duplicate parameter để không update một tensor nhiều lần.

### 2.17. Freeze không đồng nghĩa eval

requires_grad điều khiển việc tính/cập nhật gradient cho weights.
model.train/model.eval điều khiển hành vi như dropout.

Một block frozen vẫn có thể có dropout hoạt động khi toàn model ở train mode.
Khi evaluate, model.eval tắt dropout để metric ổn định hơn.

Do đó optimization loss và end-epoch train_loss_eval là hai ngữ cảnh khác nhau.
Không so chúng như hai phép đo cùng điều kiện.
Notebook ghi rõ curve dùng full eval-mode train/validation.

### 2.18. Epoch và optimizer step

Một epoch là một lượt qua effective train.
Một optimizer step là một lần update sau accumulation.

Steps/epoch phụ thuộc train size, batch, GPU count và accumulation.
Pipeline notebook dùng một GPU để đơn giản hóa scope và resource accounting.

Trục learning curves là epoch, mỗi epoch đã hoàn thành một điểm.
Global step được giữ để truy vết checkpoint.
Không điền thêm điểm để đủ ngân sách nếu early stopping đã dừng.

### 2.19. Scheduler và clipping

Warmup_ratio0.1 được truyền đúng API Transformers.
warmup_steps nếu dùng là số nguyên riêng, không truyền ratio vào trường này.

Sau warmup, linear scheduler giảm LR theo số updates dự kiến.
Hai optimizer groups giữ tốc độ học khác nhau nhưng cùng lịch tỷ lệ.

Max gradient norm1 hạn chế gradient lớn trước optimizer update.
Loss không hữu hạn phải dừng/ghi lỗi thay vì lọc thành một giá trị có vẻ hợp lệ.
AMP skipped updates được ghi khi có, cần đọc cùng optimization log.

### 2.20. End-epoch evaluation

Callback đo full effective train ở eval mode.
Sau đó Trainer đo full validation theo lịch epoch.

Ghép train/validation metrics của cùng epoch/step vào epoch_history.
Lưu một dòng cho mỗi epoch hoàn thành.

Console summary in epoch/budget, train loss/accuracy, validation loss/accuracy và LR.
Không lấy loss của một batch cuối làm full train loss.
Timer train phase dừng trước các lượt evaluation để tách scope.

### 2.21. Early stopping

Objective được theo dõi là validation loss.
Patience2 đếm hai validation evaluations liên tiếp không cải thiện đủ mức min_delta0.001.

Metric prefix train_* không được đếm như validation.
Callback chỉ xử lý metrics có eval_loss.

Min_delta điều khiển patience; checkpoint vẫn chọn loss nhỏ nhất thực tế.
Với budget 3 epoch, patience 2 và đánh giá mỗi epoch, early stopping sớm nhất ở epoch3. Không đổi patience giữa các thí nghiệm ngân sách.
selection_summary lưu requested/completed/stopped_early và selected_epoch.

### 2.22. Chọn và reload checkpoint

Checkpoint criterion là min validation loss, không mặc định epoch cuối.
Sau training, Trainer reload best weights.

Script đối chiếu selected step với epoch có loss nhỏ nhất trong history.
Lưu best_model, tokenizer và training configuration.

Save/reload verification so logits trên một số validation examples trong tolerance.
Không dùng test examples để lựa chọn weights hoặc sửa policy.
Model reload thành công chưa thay thế kiểm consistency của output.

### 2.23. Threshold sweep

Dùng P(Positive) của toàn validation sau reload checkpoint.
Grid mặc định19 điểm từ0.05 đến0.95, bước0.05.

Tại mỗi threshold tính confusion counts, accuracy, precision/recall và Macro-F1.
Chọn Macro-F1 lớn nhất, hòa thì gần0.5 hơn rồi threshold nhỏ hơn.

Không ép threshold khác0.5.
Lưu scan table, figure và selected policy trước khi predict test.
Policy mang weights identity để không áp cho model khác ngoài ý muốn.

### 2.24. Confusion và metrics

Confusion matrix có hàng là true label, cột là predicted label.
Row-normalized values cho biết tỷ lệ trong từng lớp thật.

```text
TN: Negative -> Negative
FP: Negative -> Positive
FN: Positive -> Negative
TP: Positive -> Positive
```

Accuracy=(TP+TN)/N.
Positive precision=TP/(TP+FP).
Positive recall=TP/(TP+FN).
Binary F1 là harmonic mean của positive precision/recall.
Macro-F1 lấy trung bình F1 của cả hai lớp.
Zero denominator dùng quy ước0 và được kiểm thử.

### 2.25. Threshold trade-off

Tăng threshold thường giảm số Positive predictions.
Điều đó có thể giảm FP nhưng tăng FN.

Không kết luận một policy tốt hơn mọi mặt chỉ vì accuracy tăng.
Đọc precision, recall, Macro-F1 và confusion counts cùng nhau.

Test comparison có baseline, Transformer threshold0.5 và validation-selected threshold.
Các policy này giúp diễn giải cùng một run, không dùng để tiếp tục tuning trên test.
Policy selection split phải được ghi rõ trong report.

### 2.26. ROC, PR và calibration

ROC/PR khảo sát ranking score trên nhiều operating points.
Reliability plot so confidence trung bình với accuracy thực tế theo bins.

Brier/ECE/NLL bổ sung thông tin xác suất mà accuracy không thể hiện.
Sai nhưng tự tin hơn có thể làm loss tăng khi accuracy gần như không đổi.

Threshold tối ưu một metric nhãn không tự sửa calibration.
Các diagnostics được tính sau policy lock và không dùng test chọn threshold.
Không gọi ROC-AUC là accuracy.

### 2.27. Uncertainty và seed

Bootstrap phân tầng giữ hai lớp khi resample test predictions.
Interval phản ánh biến thiên sampling với model/policy cố định.

Nó không bao gồm uncertainty do training seed, data augmentation hoặc kiến trúc.
Một seed42 không đủ chứng minh profile ổn định qua mọi khởi tạo.

Nếu cần khẳng định độ ổn định, phải có thiết kế nhiều seed và budget phù hợp.
Không thực hiện thêm search chỉ vì một test score chưa đạt mong muốn.
Báo giới hạn thay vì dùng interval như bằng chứng tối ưu toàn cục.

### 2.28. Performance: timing

Train timer đo train phase của epoch, gồm data loading/forward/backward/optimizer/logging.
Nó dừng trước end-epoch full train-eval/validation/save.

Epoch pipeline timer có train và end-epoch evaluation, được giữ thành trường riêng.
Không gọi pipeline timing là pure training latency.

Test ms/sample lấy predict runtime nhân1000 chia test size.
Đây là batch throughput có transfer/evaluation overhead, không phải latency một request độc lập.
Cùng đơn vị ms/sample không có nghĩa hai scope có thể so trực tiếp.

### 2.29. Performance: RAM và VRAM

RAM là RSS của parent training process sampled20ms.
Peak quan sát có thể bỏ lỡ một đỉnh ngắn hơn chu kỳ sampling.

VRAM dùng CUDA max_memory_allocated và max_memory_reserved trên selected device.
Allocated khác reserved cache; cả hai khác tổng memory mọi ứng dụng trên GPU.

Peak/batch slots gồm shared model memory, không phải bộ nhớ riêng từng câu.
Ghi batch size và device cùng normalized values.
CPU execution có VRAM N/A, không thay bằng0.

### 2.30. Visualization trong notebook

session.show hiển thị PNG qua IPython Image.
Đây là cùng file đã lưu, không dựng lại một hình khác chỉ để trình bày.

Notebook01 có quality, length, cutoff, token/mask, TF-IDF và augmentation figures.
Notebook02 có pretrained comparison, learning, architecture/performance, threshold, confusion/metrics và calibration/errors.

JSON giữ precision của số liệu, DataFrame giúp đọc bảng ngay trong notebook.
Không thay toàn bộ report bằng JSON dump.
Notebook chưa train không được gắn số đo hoặc ảnh performance tự tạo.

### 2.31. Error analysis

Sample grid có đúng/sai theo hai lớp và confidence.
Nhóm high-confidence errors hữu ích để tìm failure cases.

Đây không phải random sample đại diện toàn test.
threshold_errors lưu các câu sai dưới policy đã chọn, có index để đối chiếu.

Đọc phủ định, tương phản, implicit sentiment và văn cảnh.
Giải thích ngôn ngữ là hypothesis, không phải causal attribution.
Không sửa threshold theo một vài ví dụ test gây ấn tượng.

### 2.32. Artifact lifecycle

Run context tạo ID riêng và folder không ghi đè.
Lưu config/environment trước khi thực thi phần khoa học.

Console output được tee vào console.log.
Khi kết thúc, status và SHA256 manifest được ghi.

Failed runs không được chọn làm active result thành công.
Published là bản sao được kiểm tra từ nguồn, không sửa source artifacts.
Có thể đọc ảnh/bảng/log sau pull mà không cần train lại.

### 2.33. Publish và selection

Training thành công tạo outputs/latest_report.json trỏ đúng run.
Preprocessing tạo outputs/latest_preparation.json cho EDA view.

Publisher kiểm source manifests và kiểm audit cohort/selection trước khi xuất.
Catalog ghi role, run, file, kind và SHA256.

index.md nhúng hình và liên kết bảng/log để dùng làm report.
Raw console logs được copy nguyên bytes, không đổi newline làm sai hash.
Published manifest xác nhận toàn bộ file đã xuất.

### 2.34. Review ZIP và checkpoint ZIP

Review ZIP chứa artifact nhẹ, source selection, log, figures và manifest.
Không mang weights vào review bundle.

Checkpoint ZIP chứa best_model, tokenizer, training config và decision policy.
Giữ nó riêng để evaluate/infer và kiểm identity.

Bundle manifest cần được kiểm trước khi nhập artifact.
Không commit dataset thô, ZIP hoặc weights vào Git.
Notebook có cell output được tải riêng từ giao diện Kaggle.

## 3. Kết quả, bằng chứng và hoàn thành báo cáo

### 3.0. Trạng thái

Thí nghiệm 3 epoch chưa được full train trên GPU. Không có số đo accuracy, loss, threshold, runtime hoặc VRAM để kết luận. Các mục dưới đây là hướng dẫn đọc bằng chứng do chính phiên chạy sinh ra; không phải kết quả đã thực hiện.

Không dùng kết quả mô hình từ run khác để điền bảng. Không tạo learning curves trước training. Hai notebook hiện không có cached outputs; Run All sẽ tạo bảng và hình inline đồng thời lưu artifact.

### 3.1. Kiểm tra data contract

Đọc data_statistics.json trong model run. Official splits phải là 8.530 train, 1.066 validation, 1.066 test. Effective train bằng train gốc cộng added_count trong augmentation.json. Validation/test không tăng mẫu.

Kiểm tra parent index, text gốc, text mới và nhãn kế thừa trong augmentation records. Synthetic variants không phải dữ liệu đã được con người xác nhận lại nhãn. Nếu không có candidate phù hợp, sửa phương pháp dựa trên train thay vì ép nhân bản đủ quota.

### 3.2. Đọc learning curves

Mỗi dòng epoch_history tương ứng một epoch thực chạy. Train loss trên biểu đồ được evaluate lại trên toàn effective train với dropout tắt, có thể khác running optimization loss. Validation cũng được evaluate ở chế độ eval. So hai đại lượng cùng cách đo để đọc generalization gap.

Nếu train loss tiếp tục giảm mà validation loss tăng, đó là dấu hiệu overfit cần điều tra. Nếu hai loss đều cao và accuracy gần ngẫu nhiên, có thể underfit hoặc lỗi pipeline. Nếu gap nhỏ nhưng performance thấp, không kết luận mô hình tốt chỉ dựa vào gap.

Biểu đồ dùng epoch làm trục X; step vẫn được lưu để định danh checkpoint và truy vết optimizer. Không thêm điểm giả khi run kết thúc. Không lấy accuracy tốt nhất trên test để chọn epoch.

### 3.3. Early stopping và checkpoint

Budget tối đa 3 epoch, patience 2, min_delta 0.001, monitor validation loss. Evaluate/save mỗi epoch. Với lịch này patience không thể gây dừng trước epoch3. Đây là thuộc tính của cơ chế, không phải lỗi.

Minimum validation loss vẫn cập nhật khi loss giảm ít hơn min_delta; patience chỉ reset khi cải thiện vượt min_delta. Do đó checkpoint được chọn có thể thuộc một epoch không reset patience.

Đọc selection_summary để xác nhận requested_epochs, completed_epochs, selected_epoch, best_step, early_stopping và stopped_early. Không đánh đồng enabled với stopped_early.

### 3.4. Threshold và test

Sau khi chọn checkpoint, predict validation và quét 19 thresholds từ 0.05 đến 0.95. Chọn Macro-F1 lớn nhất, tie-break gần 0.5 rồi ngưỡng nhỏ hơn. Khóa decision_threshold trước predict test.

Báo cả argmax/threshold0.5 và threshold được chọn trên validation. Không dùng test để chọn operating point. Threshold tối ưu không được điền trước.

Đọc TN/FP/FN/TP, precision, recall, Binary F1 và Macro-F1. Confusion matrix hiển thị counts và chuẩn hóa theo hàng; hai phần bổ sung nhau.

Accuracy cho biết tỷ lệ đúng, F1 mô tả trade-off precision/recall. Một threshold có thể cải thiện Macro-F1 nhưng làm accuracy thấp hơn. Giải thích theo mục tiêu đã định trước.

### 3.5. Performance và bộ nhớ

resource_history ghi train time không gồm full train-eval/validation/save. Epoch pipeline time có phạm vi rộng hơn. Hai loại time/sample không được gọi cùng một đại lượng.

Test evaluation runtime là thời gian predict cả tập, chia cho số mẫu; đó là throughput trung bình, không phải latency một request.

RSS được lấy mẫu trên process cha. CUDA allocated/reserved peak là số đo allocator của device được chọn. CPU ghi VRAM N/A; không suy ra CUDA peak bằng số tham số.

RAM/VRAM per batch slot là bộ nhớ peak chia số slot nominal của batch, có phần model/optimizer dùng chung. Không coi đây là bộ nhớ riêng biệt mà mỗi câu cần độc lập.

### 3.6. Phân tích theo lớp

Báo precision, recall, F1, TPR, FPR và support của Negative/Positive. Với bài hai lớp, FP của lớp này liên quan FN của lớp kia; luôn ghi rõ positive label của phép one-vs-rest.

Đọc per_class_metrics.json cùng hình PNG. Kiểm tra totals bằng confusion counts để phát hiện đảo trục hoặc đổi thứ tự nhãn.

### 3.7. Calibration và uncertainty

ROC-AUC đánh giá thứ tự scores, không cố định threshold. PR phản ánh precision/recall trên nhiều operating points. ECE và reliability diagram đánh giá confidence.

Mô hình tự tin chưa chắc đúng. Không gọi score 0.99 là xác suất đúng đã được kiểm chứng nếu chưa đánh giá calibration.

Bootstrap test interval chỉ mô tả uncertainty của fixed model trên mẫu test; không bao gồm variance do seed, dataset shift hoặc lựa chọn hyperparameters.

### 3.8. Đọc các câu sai

Đọc cả false positive và false negative. Chú ý negation, contrast, sarcasm, implicit sentiment và kiến thức ngữ cảnh.

Các sample_predictions được chọn theo nhóm đúng/sai, nhãn và confidence để dễ phân tích; không phải mẫu ngẫu nhiên đại diện toàn test.

Giải thích ngôn ngữ là giả thuyết, không phải chứng minh nguyên nhân bên trong Transformer. Muốn kết luận nhân quả cần thiết kế ablation hoặc perturbation có kiểm soát.

### 3.9. Baseline

TF-IDF + Logistic Regression dùng cùng effective train. Vocabulary/IDF fit train-only. Chọn C trên validation trước test. Không fit vocabulary trên tất cả splits để tăng feature coverage.

So baseline và Transformer trên cùng test cohorts, không chỉ cùng số lượng mẫu. Index, text và true label phải khớp.

Baseline log loss được tính từ saved probabilities; Transformer loss là cross-entropy do evaluation ghi. Không gán thời gian GPU của Transformer cho baseline.

### 3.10. Publish và dùng artifact làm report

Sau training, mở outputs/published/index.md. Catalog ghi role, source run, đường dẫn và SHA256. Ảnh đã được sinh bởi pipeline; không cần train lại hoặc viết lại code để vẽ report.

Raw run giữ config, environment, predictions, epoch history, resource history, console log và manifest. Published là bản chọn từ run tường minh. Không tự lấy run mới nhất để báo cáo.

Notebook hiển thị cùng PNG được lưu. Tải executed notebook của Kaggle để giữ output cell; interactive notebook không tự nằm trong training subprocess.

### 3.11. ZIP và checkpoint

Review ZIP không chứa weights nhưng có dữ liệu để kiểm tra metrics và biểu đồ. Checkpoint ZIP giữ model/tokenizer/training config/policy để reload hoặc inference. Không đưa trọng số hoặc ZIP lớn vào Git.

Giữ checkpoint để không phải train lại khi cần dự đoán, kiểm tra save/load hoặc mở rộng đánh giá. Kiểm tra SHA256 manifest trước xử lý bundle.

### 3.12. Các gate trước kết luận

1. Preflight tests pass, sanity overfit minibatch pass.
2. Full splits và train-only fitting đúng.
3. Model architecture, dropout, frozen layers và optimizer khớp config.
4. Có đúng số epoch thực chạy trong logs/curves/selection.
5. Validation chọn checkpoint và threshold; test không tham gia lựa chọn.
6. Raw predictions tái tính được metrics và confusion counts.
7. Resource units và phạm vi đo được khai báo.
8. Reload consistency được kiểm tra trên cùng execution context.
9. Published images/logs/catalog/hash đầy đủ.
10. Bảng kết quả lấy từ run thật; chưa chạy thì ghi pending.

## 4. Câu hỏi bảo vệ

### Vì sao dùng pretrained base?

Biểu diễn ngôn ngữ đã học giúp thích nghi với lượng dữ liệu có hạn. Classification head được khởi tạo để học hai nhãn của bài toán.

### Pretrained sentiment ở Exercise1 có phải mô hình fine-tune ở Exercise2?

Không. Exercise1 dùng checkpoint đã học SST-2 để minh họa inference; Exercise2 dùng DistilBERT base và head mới.

### Vì sao cần baseline?

Baseline đo lợi ích thực tế của kiến trúc phức tạp và giúp phát hiện lỗi dữ liệu hoặc training.

### Vì sao không xóa stopwords và punctuation?

Phủ định, dấu câu và từ tương phản có thể quyết định polarity. Tokenizer pretrained đã có quy tắc chuẩn hóa phù hợp.

### Vì sao không fit TF-IDF trên test?

Vocabulary/IDF là phần của mô hình cần fit. Dùng test để fit sẽ làm test tham gia xây dựng representation.

### Feature engineering của Transformer là gì?

Tokenization, cutoff, attention mask, dynamic padding và lựa chọn dữ liệu/augmentation. Không cần nối thủ công TF-IDF vào DistilBERT.

### Freeze có tắt dropout không?

Không. requires_grad=False ngăn cập nhật gradient; eval mode mới tắt dropout.

### Vì sao có encoder/head LR khác nhau?

Encoder có tri thức pretrained nên cập nhật nhẹ; head mới cần học nhanh hơn. Đây là lựa chọn cần kiểm chứng, không bảo đảm tối ưu.

### Vì sao bias và LayerNorm không decay?

Optimizer chỉ decay nhóm weights đã định trước; bias và các vector normalization được giữ ở nhóm không decay.

### Vì sao loss có thể tăng khi accuracy gần như không đổi?

Cross-entropy nhạy với confidence của lỗi. Argmax có thể giữ nguyên nhưng mô hình tự tin hơn vào các câu sai.

### Early stopping có bảo đảm không overfit?

Không. Nó giới hạn tiếp tục học theo validation monitor, còn generalization phụ thuộc dữ liệu, architecture và protocol.

### Có bắt buộc đủ ba điểm trên biểu đồ?

Chỉ vẽ số epoch thực chạy; với budget3/patience2/eval mỗi epoch, lượt hoàn thành bình thường chạy tới epoch3. Run lỗi không được gọi hoàn thành.

### Có bảo đảm accuracy cao nhất không?

Không. Kết luận phải dựa vào kết quả thật, baseline, calibration và giới hạn một seed; không dùng số epoch để suy ra chất lượng.

### Vì sao lưu cả JSON và PNG?

JSON lưu giá trị chính xác để kiểm toán; PNG phục vụ trực quan và report. Notebook cần hiển thị PNG inline, không chỉ in đường dẫn.

### Nếu checkpoint tốt nhất không phải epoch cuối?

Reload checkpoint theo minimum validation loss đã công bố trước. Không thay quy tắc vì thấy test tốt hơn ở epoch khác.

### Vì sao cần lưu checkpoint riêng?

Để reload/infer/evaluate mà không train lại. Review artifacts không thể thay thế model weights.

### So sánh hai ngân sách huấn luyện thế nào?

Giữ dataset revision, seed, preprocessing, augmentation, model, optimizer, regularization và selection protocol. Ghi rằng LR schedule phụ thuộc tổng số steps: thay budget tạo một thí nghiệm khác, không nhất thiết bằng cắt history của run dài hơn.

Không chọn hyperparameters mới dựa trên test của từng run. So validation curves, selected checkpoint, thời gian và generalization; test là đánh giá cuối và cần thừa nhận mọi test exposure.
