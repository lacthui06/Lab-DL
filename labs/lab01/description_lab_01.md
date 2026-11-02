# Description — Lab01: FashionMNIST với PyTorch

## Đề bài gốc

Learn: Study PyTorch Tensors, Datasets/Loaders, Transforms, Model Building, Autograd, Optimization, and Model Saving/Loading.

Implement: Load FashionMNIST, apply transforms, build a neural network, create a training loop, evaluate accuracy, save/load the model. Experiment with network/hyperparameters, visualize loss and predicted-vs-actual images. Deliver Python code, report, loss graphs and image displays. Use PyTorch documentation/tutorials.

## Thảo luận và quyết định trước triển khai

Mục này phục hồi các vấn đề đã được ghi trong những bản description trước khi hợp nhất, đồng thời nối tiếp các yêu cầu bổ sung đã chốt trong cuộc trao đổi hiện tại. Đây là bản đối chiếu quyết định theo tài liệu còn lưu, không phải bản chép nguyên văn toàn bộ lịch sử chat. Các câu hỏi bên dưới diễn đạt lại chủ đề thảo luận; không coi là trích dẫn lời người dùng. Phương án cũ được ghi rõ trạng thái để không áp dụng nhầm vào pipeline hiện hành.

### 1. Mục tiêu bài làm và ranh giới với đề bài gốc

**Vấn đề đã làm rõ:** cần chỉ hoàn thành các thao tác PyTorch cơ bản hay mở rộng thành quy trình có EDA, so sánh kiến trúc và optimizer?

**Đã chốt:** giữ đề bài gốc làm đầu vào; mở rộng bài làm thành EDA/preprocessing/feature engineering, ba cấu hình MLP, training/validation, thí nghiệm optimizer riêng, đánh giá test và save/load. Description ghi yêu cầu, lựa chọn và tiêu chí nghiệm thu; walkthrough trình bày phương pháp/kết quả; README hướng dẫn chạy. Không trộn nhật ký sửa phần mềm vào báo cáo.

**Ghi chú nguồn đề:** bản description đầu tiên ghi tiêu đề “Exercise: PyTorch FashionMNIST Classification” và một danh sách 1–5 thiếu nội dung trong bản sao đề. Không tự đặt nội dung cho danh sách trống hoặc biến yêu cầu mở rộng thành lời nguyên văn của giảng viên.

### 2. Dữ liệu, split và chuẩn hóa lấy từ đâu?

**Đã chốt:** FashionMNIST thông qua torchvision; tập train gốc 60.000 ảnh chia stratified thành 54.000 train và 6.000 validation; test gốc 10.000 ảnh giữ riêng. Lưu indices để tái dùng đúng mẫu giữa các model. Kiểm tra dữ liệu và đo thống kê trước modeling; không dùng thống kê validation/test để fit preprocessing.

**Điều chỉnh so với bản cũ:** bản trước từng ghi cố định mean/std vào nội dung phạm vi. Pipeline hiện hành phải đọc thống kê train từ artifact, không lấy một hằng số trong tài liệu làm nguồn chuẩn hóa cho mọi run. Pixel rescale [0,255] → [0,1] trước khi trích feature; pixel và feature được standardize riêng.

### 3. Có augmentation và feature engineering không?

**Phương án từng ghi:** pipeline transform nối cấu hình augmentation YAML theo thứ tự Augmentation → ToTensor → Normalize.

**Quyết định hiện hành:** không augmentation trong comparison ba model. Trích đúng 10 feature từ ảnh [0,1]: mean, std, foreground ratio, bbox width/height, aspect ratio, centroid x/y, symmetry và edge density; ngưỡng foreground 0,1, định nghĩa rõ ảnh rỗng. Scaler feature chỉ fit train. Không giữ cấu hình augmentation cũ như một bước đang chạy. Quyết định này tránh việc feature được tính từ ảnh gốc nhưng pixel đầu vào đã bị biến đổi khác.

### 4. Vì sao MLP và vì sao chuyển từ một model sang ba cấu hình?

**Phương án ban đầu:** một FashionMLP 784 → 256 → 128 → 10, chỉ dùng Flatten, Linear và ReLU; không BatchNorm, không Dropout.

**Đã chốt sau đó:** SmallMLP 784 → 128 → 10 làm baseline gọn; MainMLP 784 → 256 → 128 → 10 khảo sát tăng dung lượng; MainMLP_FE 794 → 256 → 128 → 10 khảo sát thêm 10 feature. Cùng class FashionMLP, cùng split/seed và điều kiện so sánh. Small/Main thay đổi cả độ sâu và độ rộng; Main/Main_FE thay đổi đầu vào và số tham số lớp đầu. Không tuyên bố đã cô lập từng yếu tố hoặc đã tìm được kiến trúc tối ưu. Kiến trúc từng layer và lý do chọn chi tiết được giữ ở mục bên dưới.

### 5. Kiểm tra gì trước khi huấn luyện chính thức?

**Đã chốt:** có kiểm tra hợp đồng dữ liệu/split, kích thước đầu vào 784/794, forward/backward và sanity overfit một mini-batch. Trong runner hiện hành, sanity học 8 mẫu trong 120 bước, đạt khi loss < 0,15 và accuracy ≥ 90%; sau đó reset seed, model và loader trước training chính thức. Sanity thất bại thì dừng và lưu log. Đây là kiểm tra khả năng học/gradient, không phải kết quả đánh giá tổng quát hóa.

**Điều chỉnh tên gọi:** các tên guard/loader từng có trong source cũ chỉ là tham chiếu lịch sử; không coi `verify_pipeline_handoff` hoặc `src.dataset.get_dataloaders` là entry point bắt buộc của source hiện hành.

### 6. So sánh optimizer như thế nào để không đổi nhiều yếu tố cùng lúc?

**Đã chốt:** thí nghiệm optimizer tách riêng trên MainMLP nhận pixel. Gồm hai kịch bản: SGD momentum 0,9, RMSprop momentum 0 và Adam cùng LR=0,001/weight_decay=0; sau đó mỗi optimizer có study tuning riêng. Giữ seed, split, batch size và ngân sách epoch để đối chiếu; không giả định cùng số trial có nghĩa cùng thời gian thực khi có pruning.

**Phương án cũ đã thay thế:** bản đầu đề xuất đối đầu “peak-to-peak”, đo giây/epoch, thời gian đạt 87% validation accuracy và vẽ validation accuracy theo wall-clock. Mục tiêu diễn đạt sẵn ưu thế của Adam trong bản cũ không phải kết luận được chấp nhận trước khi có dữ liệu. Hiện giữ thời gian thực đã ghi và yêu cầu đo tài nguyên training bổ sung; chưa có bằng chứng xác nhận biểu đồ time-to-87% cho pipeline hiện hành. Không dùng benchmark ms/sample mới để thay thế phép đo hội tụ theo thời gian đó.

### 7. Tuning bao nhiêu trial, tìm tham số gì và có scheduler không?

**Phương án từng ghi:** 10–12 hoặc 12 trial nhanh, có bản ghi 3 epoch/trial; LR SGD từng bắt đầu từ 1e-2. Giai đoạn cuối từng đề xuất bỏ scheduler để phục vụ threshold scan. Đây là các thiết kế lịch sử, không phải cấu hình hiện hành.

**Đã chốt hiện hành:** 10 trial độc lập cho mỗi SGD/RMSprop/Adam, tối đa 10 epoch/trial, batch 128, seed 42, CosineAnnealingLR, MedianPruner. LR log-uniform: SGD [1e-3,1e-1], RMSprop/Adam [1e-4,1e-2]; weight_decay log [1e-7,1e-2]; momentum SGD [0,8;0,95]. Chọn best validation accuracy; test không tham gia objective. Refit cấu hình thắng theo validation rồi đánh giá test.

**Giới hạn phải nói rõ:** 10 trial hiện tại bằng 10 startup trials nên không mô tả là đã khai thác pha TPE thích nghi. Baseline weight_decay=0 chưa nằm trong miền log search, chưa enqueue baseline; không mặc định tuning phải thắng fixed-LR. Mọi thay đổi ngân sách sau này phải được ghi thành thí nghiệm mới.

### 8. Threshold scan hay argmax, ROC và Precision–Recall?

**Phương án cũ đã ghi:** quét 100 ngưỡng xác suất 0–1, theo dõi Precision/Recall/F1/Accuracy, đánh dấu ngưỡng tối đa F1 và xuất bảng bước 0,05. Không phục hồi yêu cầu này thành một bước đã thực hiện ở pipeline hiện hành.

**Quyết định hiện hành:** dự đoán 10 lớp bằng argmax; đánh giá ROC AUC và Average Precision theo one-vs-rest. ROC và PR ghép một ảnh hai panel/model, mỗi panel có 10 đường theo lớp. Không tự thêm threshold scan, pAUC, micro/macro AUC lịch sử hoặc dùng test để chọn ngưỡng. Việc cần khôi phục một thí nghiệm chọn ngưỡng là thay đổi phạm vi riêng, chưa được chốt lại trong lượt phục hồi tài liệu này.

### 9. Notebook phải trình bày những gì và tránh lặp ra sao?

**Phương án ban đầu:** một notebook gọn, có bản đặt giới hạn 12–15 cell, mỗi cell gọi module; đường dẫn notebook cũ thay đổi giữa các phiên bản.

**Đã chốt hiện hành:** hai notebook `01_eda_preprocessing_features.ipynb` và `02_modeling_comparison.ipynb`. Roadmap đầu mỗi notebook có 5 cột: Bước, Giai đoạn, Nội dung chi tiết, Module triển khai, Output/Artifact. Cell gọi code trong src; output nằm ngay sau bước tương ứng, không có phần đọc lại trùng ở cuối. EDA trình bày các mục 1–7 theo roadmap; phân biệt backend chạy gộp với các cell giải thích/kiểm chứng. Feature extraction phải chạy trên ảnh thật, hiển thị giá trị và đối chiếu CSV/scaler/NPZ, không chỉ in source.

**Chốt visualization:** giữ loss/accuracy train–validation, so sánh validation, Accuracy/Macro F1, chi phí, confusion matrix và prediction grid. F1 heatmap và biểu đồ cột TPR/FPR được bỏ; không bỏ các metric TPR/FPR trong báo cáo theo lớp. Confusion matrix có số ở mọi ô, màu chuẩn hóa theo lớp thật và colorbar. Bỏ yêu cầu cũ tự hiển thị toàn bộ PNG và PR/ROC từng trial; artifact vẫn lưu đủ.

### 10. Sau tuning và save/load phải kiểm chứng gì?

**Đã chốt:** hiển thị history và loss/accuracy train–validation của refit; đồng thời so sánh history best completed trial của đủ SGD/RMSprop/Adam với best parameters riêng, ghi rõ trial hay refit. Không chỉ hiển thị optimizer thắng.

Lưu state_dict và cấu hình kiến trúc, khởi tạo instance mới, load ở eval mode, kiểm tra dự đoán/xác suất trước và sau tải. Đánh giá model đã tải trên toàn bộ test, có nhãn thật, nhãn dự đoán và xác suất cho mẫu đúng/sai. Các phiên bản từng yêu cầu lưới 16 ảnh; yêu cầu hiện hành làm rõ tối đa 8 đúng và 8 sai, có bổ sung khi một nhóm thiếu. Hình minh họa không thay thế metric của toàn bộ test.

### 11. Môi trường chạy, quản lý artifact và báo cáo

**Đã chốt theo yêu cầu người dùng:** hỗ trợ local CPU/CUDA và Kaggle; GPU tăng tốc nhưng không là điều kiện bắt buộc của API Lab 1. Ghi thiết bị thực của từng run để diễn giải thời gian; không coi kết quả CPU và GPU là cùng điều kiện. Mục tiêu cũ hoàn thành toàn pipeline trên Kaggle trong 3–5 phút chưa được xác minh và không dùng làm cam kết thời gian hiện hành.

Giữ cấu trúc `src/datasets`, `src/models`, `src/training`, `src/utils`, notebook ngắn gọi module; config hiện hành là `config/workflow.json`, không buộc khôi phục YAML cũ. Mỗi run có config, môi trường, log stdout/stderr, status, manifest và artifact; không ghi đè run cũ, thất bại/pruned vẫn có log. Không chọn run mới nhất tự động. Walkthrough là báo cáo có bằng chứng, không chèn metric của K-fold/sweep hoặc kiến trúc lịch sử chưa xác minh. Chưa xuất PDF; mục tiêu bản dàn trang sau này ít nhất 25 trang, số trang chỉ xác nhận khi render thật. Các đường dẫn checkpoint/ZIP cố định trong bản cũ được thay bằng run riêng và publication chọn lọc; không coi ZIP cũ là artifact phải sinh lại ở mọi lần chạy.

### 12. Yêu cầu bổ sung về kiến trúc, performance và monitoring

**Đã chốt trong trao đổi bổ sung:** description in nguyên kiến trúc dạng text từng layer của cả ba model và lý do chọn, không chỉ ghi chuỗi kích thước. Notebook modeling in cùng kiểu kiến trúc, giữ train/validation accuracy và loss theo epoch, thêm time/sample và RAM–VRAM/sample.

Người dùng đã chọn **chỉ đo training**, không đo inference. Phép đo bổ sung dùng vài batch trên checkpoint bản sao trong process riêng; nêu rõ batch size, thiết bị, warmup, số lần lặp, cách chuẩn hóa theo số mẫu và phạm vi bộ nhớ. Không giả dữ liệu VRAM từ run CPU hoặc gán số đo mới cho lịch sử training cũ. Hợp đồng đo chi tiết nằm ở mục “Hợp đồng hiển thị modeling và hiệu năng training”.

Các trao đổi về TensorBoard/timm là làm rõ công cụ hiện có, chưa phải yêu cầu chuyển sang TensorBoard hoặc thêm timm. Monitoring hiện dùng console.log, history CSV/JSON và hình; việc thêm dashboard cần được chốt riêng. Tương tự, kiểm tra duplicate hiện hành là ảnh trùng chính xác, không đồng nghĩa đã kiểm tra near-duplicate.

### Nguồn khôi phục và cách cập nhật tiếp

- `fa42fa2` và `6fbe29f`: description ghi đề bài gốc, vai trò tài liệu và ghi chú danh sách 1–5 trống trong bản sao đề.
- `fb20fa8`, file `labs/lab01/description_lab_01.md`: phạm vi ban đầu, threshold scan, so optimizer theo thời gian thực, ngân sách và phương án scheduler cũ.
- `9a0bd1a`, cùng file: phạm vi một FashionMLP, Core EDA/FE, optimizer fixed-LR/Optuna và ROC/save-load.
- `e0fd041`, file `labs/lab01/archive/pre_cleanup_20261022/description_lab_01.md`: các cập nhật đã chốt ngày 22/10/2026 về hai notebook, ba model, FE, môi trường, visualization, tuning và roadmap.
- Các phần “Bổ sung trình bày sau tìm siêu tham số”, “Viết lại báo cáo đầy đủ” và trao đổi hiện tại: yêu cầu bổ sung về trình bày, save/load, kiến trúc và benchmark training.

Có thể đọc lại bản nguồn bằng `git show <commit>:<đường_dẫn_file>`. Các bản này còn mâu thuẫn giữa các giai đoạn; danh mục hiện hành bên dưới và `config/workflow.json` quyết định cấu hình đang dùng. Khi bổ sung thảo luận mới, ghi câu hỏi, phương án đã cân nhắc, quyết định, lý do có bằng chứng và trạng thái thay thế; không xóa dấu vết cũ hoặc viết lại lịch sử như một phương án duy nhất từ đầu.

## Phạm vi hiện hành đã thống nhất

1. Hai notebook: EDA/preprocessing/feature engineering và modeling/optimizer/tuning. Roadmap dạng bảng 5 cột ở đầu mỗi notebook. Chế độ view/train dùng cùng cell hiển thị, tránh phần đọc lại lặp cuối notebook.
2. FashionMNIST 28×28 grayscale, split stratified 54.000 train/6.000 validation, test gốc 10.000; giữ indices. Rescale /255, đo thống kê pixel và fit feature scaler chỉ trên train.
3. Trích xuất 10 feature từ ảnh [0,1]: mean, std, foreground ratio, bbox width/height, aspect ratio, centroid x/y, symmetry, edge density. Pixel và feature được standardize riêng. Không augmentation trong comparison này.
4. SmallMLP 784-128-10; MainMLP 784-256-128-10; MainMLP_FE 794-256-128-10. Cùng split/seed/config để so dung lượng và tác động FE. Cùng class FashionMLP.
5. Thực nghiệm optimizer riêng trên MainMLP pixel: SGD/RMSprop/Adam cùng LR=0.001; tuning độc lập 10 trial/optimizer, tối đa 10 epoch. Tune LR, weight_decay và momentum SGD. Chọn bằng validation; test chỉ đánh giá sau lựa chọn. Với 10 startup trials, lượt hiện tại chưa vào pha TPE thích nghi. Không mặc định tuned tốt hơn baseline.
6. Hình: learning curves, validation comparison, Accuracy/Macro F1, chi phí, confusion matrices, prediction grids; ROC+PR OvR một ảnh hai panel cho mỗi model. Không F1 heatmap hoặc biểu đồ cột TPR/FPR. Hiển thị chọn lọc nhưng lưu đủ artifact.
7. Hỗ trợ local CPU/CUDA và Kaggle; cấu hình nguồn ở config/workflow.json, seed=42, epoch=10, batch=128 cho kết quả hiện hành. Mọi run lưu config, environment, log, status, manifest, bảng/hình/checkpoint/prediction; run mới không ghi đè cũ.
8. Walkthrough ghi phương pháp, quyết định và kết quả có bằng chứng. Nhật ký chỉnh sửa và phân tích lỗi nội bộ tách khỏi nội dung dùng cho báo cáo. Chưa tạo PDF; mục tiêu bản xuất sau này >=25 trang bằng nội dung có bằng chứng.

## Quy ước tài liệu và lịch sử

README hướng dẫn chạy; description chứa yêu cầu và quyết định hiện hành; walkthrough trình bày bài làm/kết quả; internal_changes là nhật ký nội bộ. Bản trước hợp nhất, đề bài đã chú giải và các lựa chọn cũ được truy vết qua lịch sử Git (commit e0fd041). Không xóa outputs/runs. Danh mục run đang dùng nằm trong config/workflow.json, không tự chọn run mới nhất.

## Rà soát tính nhất quán sau triển khai

Confusion matrix phải ghi số ảnh ở mọi ô, màu biểu diễn tỷ lệ theo lớp thật và có colorbar. Roadmap phải liên kết file/hàm đang thực thi, phân biệt artifact sinh khi train và dữ liệu đọc khi view. Chỉ loại bỏ file không còn tác dụng sau khi kiểm tra tham chiếu; giữ module tương thích, notebook, bản lưu lịch sử và toàn bộ bằng chứng thí nghiệm.

## Bổ sung trình bày sau tìm siêu tham số

Notebook modeling hiển thị loss và accuracy train/validation của lượt refit best study, cùng history theo epoch. Cuối notebook phải có bước save/load model: lưu state_dict và kiến trúc/config, tải vào instance mới ở eval mode và kiểm tra dự đoán trước/sau, lưu toàn bộ bằng chứng và log vào run riêng.

Bổ sung làm rõ: loss/accuracy sau search phải so sánh đủ SGD, RMSprop, Adam với best parameters riêng; dùng history best completed trial và ghi rõ nguồn, không chỉ winner. Model tải lại phải đánh giá toàn bộ test và hiển thị mẫu đúng/sai với nhãn thật, nhãn dự đoán, xác suất. Mọi hình/bảng/metric có artifact và log.

Notebook EDA phải trình bày trực tiếp code trích xuất 10 feature và bảng giá trị từng ảnh; dùng source hiện hành, không duy trì bản sao công thức trong notebook.

Notebook EDA phải trình bày từng mục 1–7 đúng roadmap, output nằm ngay sau bước tương ứng. Phải phân biệt backend prepare_data chạy gộp và cell trình bày/kiểm chứng; bước FE phải gọi extractor trên ảnh thật rồi đối chiếu CSV/scaler/NPZ, không chỉ in source.

## Viết lại báo cáo đầy đủ

Walkthrough phải là báo cáo học thuật chi tiết, định hướng bản PDF khoảng 25 trang A4; phục hồi chiều sâu lý thuyết, phương pháp, giải thích code, EDA/FE, ba model, optimizer/tuning, metric theo lớp, hình và kết quả save/load từ artifact thật. Không đưa log sửa phần mềm, không phục hồi metric K-fold/sweep chưa xác minh. Lượt này soạn nguồn báo cáo, chưa xuất PDF; số trang chỉ xác nhận khi dàn trang thực tế.

## Kiến trúc mô hình và lý do lựa chọn

Cả ba cấu hình dùng `FashionMLP`: Flatten → các lớp Linear/ReLU → Linear 10 logits. Loss là CrossEntropyLoss, không đặt Softmax trước loss. 784 là 28×28 pixel; 794 là 784 pixel nối 10 feature được chuẩn hóa riêng.

- **SmallMLP: 784 → 128 → 10, 101.770 tham số.** Baseline một hidden layer để kiểm tra pipeline và làm mốc chi phí/dung lượng thấp.
- **MainMLP: 784 → 256 → 128 → 10, 235.146 tham số.** Tăng độ rộng và thêm một hidden layer để khảo sát dung lượng biểu diễn. So sánh này thay đổi cả độ sâu lẫn độ rộng, không cô lập riêng tác động của từng yếu tố.
- **MainMLP_FE: 794 → 256 → 128 → 10, 237.706 tham số.** Giữ hidden layers như MainMLP, thêm 10 feature hình học/cường độ để khảo sát biểu diễn đầu vào; tăng 2.560 tham số ở lớp đầu.

MLP phù hợp mục tiêu lab về Tensor, nn.Module, Autograd và optimization, đồng thời dễ đọc từng layer và kiểm chứng số tham số. MLP sau Flatten không tận dụng kết nối không gian như CNN; các kích thước hidden là lựa chọn thực nghiệm trong ngân sách lab, chưa được chứng minh tối ưu. Cùng split/seed và ngân sách epoch để đối chiếu; không chọn kiến trúc bằng test và không suy ra ưu thế phổ quát từ một seed.

### SmallMLP — kiến trúc từng layer

```text
FashionMLP(
  (net): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=128, bias=True)
    (2): ReLU()
    (3): Linear(in_features=128, out_features=10, bias=True)
  )
)
```

### MainMLP — kiến trúc từng layer

```text
FashionMLP(
  (net): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=784, out_features=256, bias=True)
    (2): ReLU()
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): ReLU()
    (5): Linear(in_features=128, out_features=10, bias=True)
  )
)
```

### MainMLP_FE — kiến trúc từng layer

```text
FashionMLP(
  (net): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=794, out_features=256, bias=True)
    (2): ReLU()
    (3): Linear(in_features=256, out_features=128, bias=True)
    (4): ReLU()
    (5): Linear(in_features=128, out_features=10, bias=True)
  )
)
```

## Hợp đồng hiển thị modeling và hiệu năng training

1. Trước mỗi model, notebook in kiến trúc dạng `FashionMLP(Sequential(...))`, bảng kích thước layer và số tham số; lưu TXT/CSV và log trong run mới.
2. Hiển thị train/validation accuracy và CrossEntropy loss theo epoch từ history thật của từng model; giữ kết quả và thời gian train cũ.
3. Đo **training** trên một batch train cố định, cùng batch size, số thread và thiết bị cho ba model. Mỗi bước gồm zero_grad, forward, CrossEntropyLoss, backward, optimizer.step. Dùng checkpoint bản sao trong process riêng, không ghi ngược checkpoint và không chạy lại full training.
4. **Time/sample:** median thời gian một batch sau warmup chia batch size, đơn vị ms/sample. Batch đã ở thiết bị; không gồm IO, chuyển CPU/GPU, preprocessing/FE, validation hoặc ghi checkpoint. Với batch 128, đây là tốc độ batch chuẩn hóa; benchmark batch size 1 mới đo trực tiếp một mẫu. Cả hai không gồm toàn pipeline và không thay thế train_seconds lịch sử.
5. **RAM/sample:** peak RSS process được lấy mẫu mỗi 1 ms trong một bước training đầu tiên, chia batch size (MiB/sample); lưu cả baseline và phần tăng so với baseline. RSS chứa Python, thư viện và vùng nhớ dùng chung; polling có thể bỏ lỡ đỉnh ngắn, không khẳng định là peak chính xác hoặc bộ nhớ độc lập của từng ảnh.
6. **VRAM/sample:** peak bộ nhớ tensor được PyTorch allocated trong bước training đầu tiên chia batch size. CUDA đồng bộ trước/sau đo; không tính reserved cache hoặc tổng VRAM toàn card. CPU ghi không khả dụng, không ghi số 0 giả.
7. Đo bộ nhớ và đo thời gian ở hai lượt riêng để sampler không làm chậm phép đo thời gian; cấu hình, môi trường, times thô và phương pháp lưu cùng bảng training_performance.csv trong run riêng. Các số benchmark mới không được coi là đã đo trong run huấn luyện cũ.

### Diễn giải thiết bị đo hiện tại

Phép đo mới chạy trên CPU với PyTorch bản `+cpu`, CUDA không khả dụng. Time/sample là thời gian bước training trên CPU chia số mẫu trong batch; RAM/sample là RSS của process trên RAM hệ thống chia số mẫu (có baseline và phần tăng). VRAM là bộ nhớ của GPU: lượt chạy CPU này không có phép đo VRAM nên ghi không khả dụng/NaN, không có nghĩa VRAM của model bằng 0 hoặc máy chắc chắn không có card đồ họa. Muốn có VRAM/sample phải đo lại trên GPU CUDA với môi trường PyTorch hỗ trợ CUDA; không suy ra số GPU từ kết quả CPU.

### Làm rõ phép đo đúng một mẫu

Yêu cầu mới nhất: notebook hiển thị trực tiếp bảng và biểu đồ chi phí training khi **batch_size=1**, không chỉ thời gian/bộ nhớ batch 128 chia 128. Dùng `single_sample_training_benchmark` và run `training_performance_single_sample`; giữ benchmark batch 128 trước đó riêng để tránh nhầm hai điều kiện. `RESOURCE_MODE` điều khiển benchmark độc lập với MODE huấn luyện model. Bộ nhớ process vẫn gồm runtime/Python; batch size 1 không biến RSS thành bộ nhớ chỉ của riêng ảnh. RAM tăng so baseline cũng được hiển thị. CPU không có số VRAM đo được; không vẽ cột VRAM bằng 0.
