# Description — Lab03: Getting started with Hugging Face

**Môn học:** Deep Learning — UTH.
**Nguồn đề:** ảnh Practice 3 do giảng viên cung cấp trong phiên làm việc.
**Vai trò tài liệu:** ghi đề nhận vào, nội dung đã thảo luận, quyết định hiện hành và điều kiện nghiệm thu.

Đề gốc được giữ riêng tại Mục 1, không trộn các yêu cầu mở rộng vào lời giảng viên.
Các phần sau ghi rõ phạm vi đã thống nhất trong quá trình xây dựng bài.
Bài làm, phương pháp, hình và phân tích kết quả nằm trong `walkthrough_lab_03.md`.
Nhật ký sửa phần mềm và xử lý môi trường nằm trong `internal_changes.md`.

## 1. Đề bài gốc

### Exercise 1: Sentiment Analysis with Hugging Face
1. Install the Hugging Face transformers library.
2. Use a pre-trained sentiment analysis model from the Hugging Face Hub.
3. Tokenize a sample sentence.
4. Perform sentiment analysis on the sentence.

### Exercise 2: Finetuning a Pretrained Model for Binary Text Classification
1. Install the necessary Hugging Face libraries (transformers, datasets, evaluate).
2. Load a simple dataset for binary text classification.
3. Load a pretrained model and its tokenizer.
4. Preprocess the dataset to be suitable for the model.
5. Define training arguments.
6. Create a Trainer object and finetune the model.
7. Evaluate the finetuned model.

### Phân biệt đề gốc và phần bổ sung

Đề gốc không chỉ định dataset, backbone, số epochs hoặc cấu trúc thư mục.
Đề gốc không đặt ngưỡng Accuracy tối thiểu 85%.
Roadmap, ba notebook, isolated runs, bộ hình mở rộng và quét threshold là yêu cầu bổ sung của người dùng.
Các phần bổ sung phải phục vụ giải thích bài và truy vết kết quả, không được dùng để gán ngược một tiêu chí chưa có cho đề giảng viên.

## 2. Phạm vi đã thống nhất

1. Thực hiện đủ hai Exercise bằng Hugging Face và PyTorch.
2. Exercise 1 dùng mô hình sentiment pretrained SST-2, đối chiếu pipeline và forward thủ công trên cùng câu.
3. Exercise 2 fine-tune DistilBERT nền trên toàn bộ Rotten Tomatoes với các split chính thức.
4. Có ba notebook: Exercise 1, Exercise 2 và bản tích hợp.
5. Mỗi notebook bắt đầu bằng roadmap có phase, mục tiêu, module/file tham chiếu và artifact.
6. Các phase trình bày theo dòng xử lý thực tế; hình và bảng xuất hiện ngay sau phần giải thích tương ứng.
7. Notebook ưu tiên trực quan hóa và diễn giải thay vì in các khối JSON dài.
8. Notebook và CLI gọi chung module; không duy trì bản sao thuật toán riêng trong notebook.
9. Mọi hình, bảng, metric, kết quả suy luận và phân tích sinh ra phải được lưu trong `outputs/`.
10. Run mới không ghi đè run cũ; có source run ID, config, snapshot, status và manifest để truy vết.
11. Bài phân loại nhị phân có quét threshold và giải thích đánh đổi Precision/Recall/FPR.
12. Threshold dùng để vận hành phải được chọn bằng validation, cố định trước khi đọc kết quả test.
13. Walkthrough là báo cáo đầy đủ có phương pháp, lý do lựa chọn, hình, công thức, kết quả và giới hạn.
14. Nội dung báo cáo hướng đến chiều sâu tương đương Lab01 hiện hành trên nhánh `main`; khoảng 700 dòng là yêu cầu về mức triển khai, không phải lý do chèn dòng rỗng hoặc lặp lại.

## 3. Thảo luận và sổ quyết định

### D01 — Chọn dữ liệu nào và sử dụng bao nhiêu?

**Thảo luận:** bản tham khảo dùng IMDB subset nên chưa đáp ứng yêu cầu mở rộng về dữ liệu đầy đủ.
**Quyết định:** dùng `cornell-movie-review-data/rotten_tomatoes`, đủ 8.530 train, 1.066 validation và 1.066 test.
**Lý do:** corpus văn bản ngắn, nhị phân, có split chính thức và vừa ngân sách GPU lab.
**Kiểm chứng:** loader từ chối kích thước khác hợp đồng; run lưu thống kê và fingerprints.

### D02 — Có chia lại train/validation hoặc lấy mẫu nhỏ không?

**Thảo luận:** lấy mẫu giúp chạy nhanh nhưng thay đổi phạm vi thí nghiệm; tự chia lại làm khó đối chiếu official split.
**Quyết định:** giữ nguyên ba split, không subsample trong luồng train chính.
**Lý do:** train cập nhật trọng số, validation chọn checkpoint/threshold, test đánh giá sau lựa chọn.
**Giới hạn:** official split không tự chứng minh đã kiểm toán toàn bộ duplicate hoặc rò rỉ ngữ nghĩa giữa các tập.

### D03 — Dùng mô hình nào cho mỗi Exercise?

**Thảo luận:** mô hình biết sentiment và backbone chưa fine-tune cho bài toán đích có vai trò khác nhau.
**Quyết định:** Exercise 1 dùng `distilbert/distilbert-base-uncased-finetuned-sst-2-english`; Exercise 2 dùng `distilbert/distilbert-base-uncased` với đầu phân loại hai nhãn.
**Lý do:** phần đầu minh họa ứng dụng pretrained; phần sau minh họa transfer learning và cập nhật trọng số thật.
**Kiểm chứng:** không dùng confidence của đầu phân loại khởi tạo mới làm kết quả sentiment có ý nghĩa.

### D04 — Chuẩn hóa cách mã hóa và padding thế nào?

**Thảo luận:** cần cân bằng khả năng giữ ngữ cảnh và chi phí batch.
**Quyết định:** WordPiece, truncation `max_length=128`, dynamic padding bằng `DataCollatorWithPadding`.
**Lý do:** không ép mọi câu ngắn thành chuỗi 128 token trong từng phép tính.
**Bằng chứng hiện có:** train của run Kaggle có độ dài token tối đa 78; tỉ lệ truncation train bằng 0%, không suy rộng con số này sang split chưa thống kê.

### D05 — Thiết kế huấn luyện và chọn checkpoint ra sao?

**Thảo luận:** cần cấu hình ngắn gọn, dễ tái lập và vừa tài nguyên Tesla T4.
**Quyết định:** seed 42, ba epochs, train/eval batch 16/32, AdamW, LR `2e-5`, weight decay `0.01`, warmup ratio `0.1`.
**Lựa chọn:** đánh giá và lưu mỗi epoch; chọn checkpoint theo validation accuracy.
**Lý do:** chốt tiêu chí trước test, không đổi sang loss sau khi quan sát test để chọn lại kết quả thuận lợi.

### D06 — Có dùng sanity-overfit model trong train chính không?

**Thảo luận:** sanity overfit kiểm tra khả năng học nhưng cũng làm thay đổi trọng số.
**Quyết định:** bài kiểm tra dùng mô hình disposable; train chính tải backbone mới.
**Lý do:** phát hiện lỗi forward/loss/gradient mà không tạo bước huấn luyện ngoài dataset trong thí nghiệm chính.
**Tài nguyên:** local chỉ kiểm thử nhỏ; full training thực hiện trên GPU.

### D07 — Notebook cần trình bày đến mức nào?

**Thảo luận:** bản notebook ngắn và chủ yếu đọc JSON chưa giải thích được bài khó hơn Lab01.
**Quyết định:** có bảng roadmap ngay đầu, lý thuyết ngắn tại từng phase, tham chiếu module/file và hình/bảng ở đúng bước.
**Lý do:** người đọc lần theo input, phép biến đổi, output và bằng chứng mà không phải tự đoán JSON thuộc giai đoạn nào.
**Ranh giới:** notebook minh họa bằng source hiện hành, không sao chép thêm một training loop thứ hai.

### D08 — Quét threshold có mục tiêu gì?

**Thảo luận:** argmax hai logits tương ứng ngưỡng 0,5 khi không có tie, nhưng một operating point có thể cần cân bằng Precision và Recall khác.
**Quyết định:** quét một lưới threshold định trước trên xác suất Positive của validation sau khi đã chọn checkpoint.
**Tiêu chí:** tối đa validation Macro F1; khi hòa chọn ngưỡng gần 0,5 hơn, tiếp tục hòa thì chọn ngưỡng nhỏ hơn.
**Lý do:** quy tắc chọn và tie-break rõ ràng, không lấy threshold tối ưu từ test.

### D09 — Đánh giá test sau quét threshold thế nào?

**Thảo luận:** dùng test để chọn threshold làm test trở thành tập tuning.
**Quyết định:** lưu threshold đã chọn trong `decision_threshold.json`, cố định quyết định, rồi đánh giá test tại ngưỡng đó.
**So sánh:** giữ metrics argmax gốc và metrics tại threshold validation chọn thành hai nhóm riêng; mốc sweep 0,5 dùng quy tắc >= được lưu riêng khi cần đối chiếu boundary.
**Báo cáo:** giữ cả lợi ích và đánh đổi khi đã đo: threshold 0,8 thêm một mẫu test đúng, giảm FP nhưng giảm Recall/Binary F1; không đổi checkpoint hay chọn lại threshold theo test.

### D10 — Hoàn tất khoảng trống validation của run lịch sử thế nào?

**Thảo luận:** run train gốc chỉ lưu xác suất test và metrics validation tổng hợp, không đủ để phục dựng validation sweep.
**Quyết định:** khôi phục đúng checkpoint trên Kaggle, dự đoán toàn bộ validation thật rồi tạo selection run riêng; không bịa scores hoặc dùng test để chọn threshold.
**Đã thực thi:** `20261026T083558Z_threshold_selection_b64cf564` lưu 1.066 validation records, chọn 0,8 theo Macro F1 0,8526632715 và gắn SHA256 weights trùng run train gốc.
**Ranh giới:** test sensitivity vẫn là descriptive; test đã quan sát trong phát triển nên phiên mở rộng không được gọi là một blind confirmatory experiment mới.

### D11 — Những output nào phải thành artifact?

**Thảo luận:** chỉ hiển thị trong notebook hoặc giữ số liệu trong RAM không đủ để truy vết và đóng gói.
**Quyết định:** mọi hình, bảng, metric, dự đoán và diagnostics sinh từ notebook/source phải ghi dưới `outputs/` với nguồn run xác định.
**Định dạng:** hình PNG để xem, JSON hoặc bảng máy đọc cho tái kiểm tra; notebook hiển thị từ chính artifact vừa lưu.
**Bảo toàn:** không sửa run Kaggle lịch sử; diagnostics dẫn xuất nằm riêng, giữ liên kết đến source run.

### D12 — Các hình mở rộng cần giải thích điều gì?

**Thảo luận:** năm hình ban đầu chưa đủ cho vai trò threshold và phân tích nhị phân.
**Quyết định:** bổ sung ROC/PR, reliability/calibration, phân bố xác suất theo nhãn, phân bố lớp/độ dài review, metrics theo lớp, validation sweep và hình so sánh operating point test 0,5/0,8.
**Lý do:** mỗi hình trả lời một câu hỏi cụ thể về khả năng phân biệt, confidence, lỗi lớp và chi phí operating point.
**Giới hạn:** review length có thể được đo bằng số từ hoặc ký tự từ text đã lưu; không gọi đó là token length nếu chưa chạy tokenizer.

### D13 — Giữ mô hình và bằng chứng ở đâu?

**Thảo luận:** weights lớn vượt giới hạn Git thông thường và làm repository tăng kích thước.
**Quyết định:** Git giữ source, config, notebook có output và artifacts nhẹ; checkpoint được lưu riêng trong ZIP Kaggle của người thực hiện.
**Truy vết:** giữ SHA256 weights trong manifest và policy; không coi file không có mặt là đã kiểm tra tại máy rà soát.
**Đã khôi phục:** run evaluation Kaggle `20261026T083612Z_evaluation_a724c16c` tải đúng `best_model/` và chạy lại toàn test; Git giữ predictions, measured loss và runtime để rà soát, không giữ weights.

### D14 — Độ đầy đủ của báo cáo được xác nhận bằng gì?

**Thảo luận:** số dòng nhiều nhưng thiếu lý do, điều kiện so sánh hoặc bằng chứng vẫn chưa là báo cáo đầy đủ.
**Quyết định:** walkthrough có ba phần thống nhất: đặc tả/cơ sở, phương pháp/thực nghiệm, kết quả/chẩn đoán; liên kết các quyết định D01–D14.
**Độ sâu:** giải thích tensor contracts, logits/loss, optimizer/scheduler, lựa chọn checkpoint/threshold, hình, lỗi ngôn ngữ và giới hạn.
**Tính trung thực:** phân biệt điều đã chạy với tính năng mới, không gán kết quả giả định cho run Kaggle cũ.

## 4. Roadmap notebook và module tham chiếu

Mỗi notebook dùng bảng roadmap riêng phù hợp phạm vi; bản tích hợp trình bày toàn bộ các phase dưới đây.
Cột artifact phải phân biệt file sinh khi train với file đọc hoặc diagnostics dẫn xuất khi view.

| Phase | Câu hỏi / công việc | Module/file thực thi | Sản phẩm dưới `outputs/` |
|---|---|---|---|
| P01 | Cấu hình, seed và môi trường | `src/workflow.py`, `src/utils/runs.py`, `config/config.json` | Config snapshot, environment snapshot, console log |
| P02 | Pretrained sentiment pipeline | `src/models/classifier.py`, `scripts/run_exercise1.py` | Dự đoán pipeline/manual |
| P03 | Tokenization và logits | `src/models/classifier.py` | Bảng token/ID/mask và xác suất |
| P04 | Kiểm tra official dataset | `src/datasets/loader.py` | Kích thước split, thống kê lớp/dữ liệu |
| P05 | Phân tích độ dài và preprocessing | `src/datasets/loader.py`, `src/utils/plots.py` | Thống kê, phân bố độ dài, truncation |
| P06 | Xây model và sanity gate | `src/models/classifier.py`, `src/training/trainer.py`, `tests/` | Bằng chứng sanity, cấu hình model |
| P07 | Fine-tuning và chọn checkpoint | `src/training/trainer.py`, `src/workflow.py` | History, train metrics, learning curves |
| P08 | Validation threshold sweep | `src/training/thresholds.py` | Lưới ngưỡng, hình trade-off, `decision_threshold.json` |
| P09 | Test với quyết định đã cố định | `src/workflow.py`, `src/training/metrics.py` | Predictions, metrics argmax và ngưỡng đã chọn |
| P10 | Chẩn đoán nhị phân và confidence | `src/analysis.py` | ROC/PR, calibration, phân bố xác suất, per-class panel |
| P11 | Sensitivity và phân tích lỗi | `src/analysis.py`, `src/utils/plots.py` | Test sensitivity mô tả, mẫu đúng/sai và bảng lỗi |
| P12 | Save/load và tính toàn vẹn | `src/models/classifier.py`, `src/evidence.py` | Reload check, SHA256 audit |
| P13 | Tổng hợp và trình bày | Notebook, `walkthrough_lab_03.md` | Các bảng/hình dẫn xuất liên kết source run |

## 5. Hợp đồng dữ liệu và thực nghiệm

Input text là chuỗi; labels 0/1 tương ứng Negative/Positive.
Tokenizer tạo `input_ids` và `attention_mask`; batch sau collator có shape `(B,S)` và labels `(B,)`.
Model trả logits `(B,2)`; cross-entropy nhận logits, không cần softmax thủ công trước loss.
Softmax được dùng khi báo xác suất, vẽ curves hoặc áp threshold trên cột Positive.

Threshold dùng quy tắc `p_positive >= threshold` để dự đoán Positive.
Quy tắc này được ghi rõ vì trường hợp xác suất đúng bằng threshold cần kết quả xác định.
Ngưỡng 0,5 và argmax có thể khác nhau tại tie chính xác; không bỏ qua trường hợp này trong tests.
Precision/Recall/F1 nhị phân lấy Positive làm lớp dương; Macro metrics trung bình đều hai lớp.
TPR/FPR theo từng lớp được tính theo cách one-vs-rest, không đánh đồng FPR với phần bù Precision.

Baseline hằng Negative trên test cân bằng đạt accuracy 50% và Macro F1 33,33%.
Đây là baseline từ nhãn, không phải kết quả của một classifier head khởi tạo ngẫu nhiên.
Metric không được báo cáo ở run gốc phải ghi “not logged” thay vì mượn giá trị từ run khác.

## 6. Hợp đồng trực quan hóa và lưu trữ

Các hình cũ được giữ nguyên: learning curves, confusion matrix, metrics bar, token-length distribution và sample predictions.
Hình mới từ `src/analysis.py` phải chỉ rõ source run, split và loại score được dùng.
ROC/PR đánh giá ranking; calibration đánh giá quan hệ giữa score và tần suất đúng; chúng không thay thế nhau.
Threshold sensitivity mô tả việc thay đổi quyết định và không biến test thành tập chọn threshold.
Bảng số liệu cần hiển thị chọn lọc cạnh hình để người đọc kiểm tra các điểm quan trọng.

Các bảng/hình phải có file tương ứng dưới `outputs/`; không chỉ tồn tại dưới dạng cell output.
Artifacts dẫn xuất phải có metadata hoặc tên thư mục gắn source run và quy trình sinh.
Không ghi đè predictions, metrics hoặc manifest của run lịch sử khi thêm diagnostics.
Manifest gốc có thể liệt kê weights không commit; audit phải báo rõ thiếu có chủ đích.
File ZIP tải kết quả chỉ là tiện ích lưu trữ, không phải nội dung thuật toán bắt buộc của bài nộp.

## 7. Bằng chứng thực nghiệm và trạng thái hoàn tất

Run fine-tuning gốc là `20261025T181550Z_finetuning_e02bced7`.
Run dùng đủ official splits, ba epochs và 1.602 optimizer steps; không train thêm trong phiên threshold.
Test argmax gốc đạt **83,9587% Accuracy**, **84,0634% Binary F1**, **83,9580% Macro F1**, đúng 895/1.066 mẫu.
Test loss không được evaluator của run train gốc ghi lại; không sửa file lịch sử để điền một số đo của phiên khác.

| Bước đã chạy | Run / nguồn | Bằng chứng chính |
|---|---|---|
| Validation selection | `20261026T083558Z_threshold_selection_b64cf564` | 1.066 validation predictions, grid 19 điểm, policy 0,8, checkpoint identity |
| Full-test checkpoint reevaluation | `20261026T083612Z_evaluation_a724c16c` | 1.066 test predictions, metrics argmax/policy, measured loss và runtime |
| Inference có policy | `20261026T083627Z_inference_0ed240e2` | Probability, nhãn, threshold và metadata nguồn validation |
| Diagnostics dẫn xuất | `20261026T080403Z_analysis_7b9e92eb` | ROC/PR, calibration, probability distribution, test sensitivity descriptive |

Validation Macro F1 tại 0,5 là **0,8432787665**; tại ngưỡng được chọn 0,8 là **0,8526632715**.
Validation Accuracy tăng từ **0,8433395872** lên **0,8527204503**, thêm 10 mẫu đúng.
Cách chọn đã chốt trước evaluation: tối đa validation Macro F1, hòa thì gần 0,5, tiếp tục hòa thì threshold nhỏ hơn.
Policy ghi split, objective, grid, sample count và weights SHA256; eval/infer kiểm tra identity trước khi áp dụng.

| Test operating point | Accuracy | Binary Precision | Binary Recall | Binary F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Argmax gốc | 83,9587% | 83,5185% | 84,6154% | 84,0634% | 83,9580% |
| Validation-selected 0,8 | 84,0525% | 86,5191% | 80,6754% | 83,4951% | 84,0343% |

Test tại 0,8 có **TN 466, FP 67, FN 103, TP 430**: đúng 896/1.066 mẫu.
So với argmax, FP giảm 22 nhưng FN tăng 21; Accuracy tăng một mẫu, Macro F1 tăng khoảng 0,0763 điểm phần trăm, Binary F1 giảm khoảng 0,5682 điểm phần trăm.
Không kết luận đây là cải thiện lớn, có ý nghĩa thống kê hoặc phù hợp mọi chi phí ứng dụng.
Threshold 0,8 được giữ vì protocol validation, không chọn lại từ đường test sensitivity.

Checkpoint reevaluation ghi **test loss 0,5442677736**, runtime **1,6827 giây**, throughput **633,519 mẫu/giây** trên Tesla T4.
Đây là loss đo thật sau reload, tách với **NLL 0,5442964394** tái tính offline từ saved probabilities và trạng thái **not logged** của run train gốc.
Threshold chỉ đổi quyết định thành nhãn; nó không thay cross-entropy của cùng probability records.
Kết quả argmax sau reload khớp metrics gốc, còn weights SHA256 trong policy khớp manifest training run.

Inference cùng policy trả Negative cho “The acting was excellent, but the story was disappointing.”, confidence **0,9933324**, `P(Positive)=0,0066675888`.
Bằng chứng full-test reload đã có trong artifacts Kaggle; checkpoint vẫn không nằm trong Git nên không tuyên bố máy rà soát đã tự chạy lại full model tại chỗ.
Bản báo cáo giữ test diagnostics descriptive và không gọi phiên mở rộng là blind test mới sau khi benchmark đã được quan sát.

## 8. Điều kiện nghiệm thu

| Nhóm | Điều kiện cần đạt | Cách kiểm tra |
|---|---|---|
| Đề bài | Đủ hai Exercise | Roadmap, code và output theo phase |
| Data | Full official splits, nhãn đúng | Loader contract và artifacts thống kê |
| Notebook | Roadmap đầu bài, module tham chiếu, giải thích và hình | Thực thi toàn bộ notebook ở view; kiểm tra links |
| Artifact | Mọi output tính toán có file trong `outputs/` | Đối chiếu bảng/hình với thư mục dẫn xuất |
| Training | Cùng workflow giữa notebook và CLI | Tests, history và config snapshot |
| Threshold | Chọn trên validation, tie-break xác định | Đã có 1.066 validation scores, policy 0,8 và selection metadata |
| Test isolation | Test không được truyền vào hàm chọn threshold | Guard split, code review, tests; selection trước evaluation riêng |
| Checkpoint reload | Toàn test được đo sau khi tải weights đúng identity | Đã có evaluation run, predictions, test loss và runtime; weights ngoài Git |
| Trình bày | Walkthrough đủ chiều sâu, khoảng 700 dòng trở lên | Rà soát nội dung, số dòng và nguồn số liệu |
| Tính trung thực | Phân biệt measured, derived, pending và not logged | Báo cáo/metadata không mâu thuẫn |
| Git | Source, notebook có output, artifacts nhẹ; không weights/cache | Checklist pre-commit và staged diff |

Hoàn tất phần mềm và hoàn tất bằng chứng thực nghiệm là hai trạng thái cần ghi riêng.
Không đánh dấu threshold validation đã đo chỉ vì unit test trên dữ liệu giả lập đã pass.
Không đánh dấu checkpoint đã được tái đánh giá toàn test chỉ vì đã kiểm tra một câu; trạng thái hoàn tất hiện dựa trên evaluation run 1.066 mẫu riêng, không dựa vào one-example reload check gốc.

## 9. Tài liệu tham chiếu

- `labs/lab01/description_lab_01.md` và `walkthrough_lab_01.md` trên nhánh `main`: cấu trúc thảo luận, roadmap và mức chi tiết báo cáo.
- `docs/dl_flow.md`, `docs/dl_data.md`, `docs/dl_training.md`: luồng thí nghiệm, data contract và huấn luyện.
- `docs/eda.md`, `docs/feature.md`: cách đặt câu hỏi dữ liệu và kiểm soát preprocessing.
- https://huggingface.co/docs/transformers/en/training
- https://huggingface.co/blog/sentiment-analysis-python
- https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
