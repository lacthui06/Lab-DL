# Chẩn đoán validation Lab02

## Yêu cầu và đặc tả đã xác nhận
Giải thích lỗi validation cho sinh viên trước khi chọn regularization. Chỉ suy luận checkpoint fine-tune của ResNet18/DenseNet121 trong run `20261027T083044Z_cifar10_macintel_cpu_control64_db231b28`; không train, không dùng test, không sửa run gốc.

1. Đầu vào: checkpoint, config và selected_indices.npz; validation đúng 5000 index duy nhất thuộc training pool CIFAR-10.
2. Hợp lệ: không giao train; nhãn nguyên 0..9; xác suất hữu hạn trong [0,1], shape (N,10), tổng hàng gần 1. Lỗi thì raise, không fallback.
3. Xử lý: khôi phục eval/inference_mode, dự đoán, kiểm tra, đối chiếu history, xuất metric/hình.
4. Đầu ra: mỗi model một thư mục; probabilities, CSV từng ảnh, confusion counts/row percent, metric, cặp nhầm, lưới ảnh, learning curves. Báo cáo chẩn đoán riêng quan sát và giả thuyết.
5. Xếp loại: đúng/sai theo argmax; không đặt ngưỡng tùy ý cho 'quá tự tin'. Xếp ảnh sai theo loss giảm dần, đồng thời lưu xác suất nhãn dự đoán và margin top-2.
6. Thứ tự: cặp nhầm theo count giảm dần rồi true/pred class index tăng dần; ảnh hòa loss theo index tăng dần. So history: accuracy atol 1e-7, loss atol 1e-5 để chịu sai số cached/full float32. Sai khác vượt ngưỡng thì dừng kết luận.

## Test case viết trước implementation
| Mã | Nhóm | Đầu vào | Mong đợi | Lý do |
|---|---|---|---|---|
| V01 | Thường | Validation thật | 5000 index đúng thứ tự, không giao train | Đúng split |
| V02 | Thường | Nhãn 0,1; probs .9/.1 và .6/.4 | Counts[0,0]=1; [1,0]=1; loss=(-ln(.9)-ln(.4))/2 | Tính tay |
| V03 | Biên | Không dự đoán lớp 1 | Precision lớp 1 = 0, metric hữu hạn | Mẫu số 0 |
| V04 | Không hợp lệ | Thiếu checkpoint, index trùng/ngoài miền/giao train | Raise, không fallback | Bảo toàn dữ liệu |
| V05 | Không hợp lệ/đặc biệt | NaN, sai shape, tổng hàng sai, nhãn sai kiểu, rỗng | Raise | Không xuất kết luận giả |
| V06 | Thường/biên | History khớp hoặc lệch | Khớp: pass; lệch: raise | Kiểm tra checkpoint |

Chuỗi/HTML không áp dụng vì đây là tensor số nội bộ; sai kiểu số vẫn được kiểm tra.

## Pseudocode và trace
Kiểm tra file/index -> build model không download -> load weights -> eval -> predict validation -> kiểm tra probabilities -> tính metric/loss -> so selected epoch -> xuất bảng/hình -> giải thích.
Trace thường: cat dự đoán dog tăng ô [cat,dog]; cat p=.9 có loss .10536.
Trace biên: không mẫu nào dự đoán lớp 1, precision đặt 0 thay vì NaN.

## Prompt
CONTEXT: Python 3.12, PyTorch 2.2.2, Intel Mac CPU; tái sử dụng src.data/models/experiment.
TASK: Chẩn đoán validation cho hai checkpoint đã chọn.
INPUT: run cố định, config, index và best.pt; không đọc prediction test.
CONSTRAINTS: không train, không đổi preprocessing, không ghi run gốc; output chỉ trong thư mục task này.
PSEUDOCODE: theo mô tả trên.
EXAMPLES: true cat p=.9 -> loss .10536; index trùng -> ValueError.
OUTPUT: source, comments ở quyết định sai số, artifact và báo cáo tiếng Việt; không giả định thêm ngưỡng thành công.
CRITERIA: V01-V06; phân biệt dữ kiện, giả thuyết và đề xuất.

## Nhật ký
- 1–5: yêu cầu, đặc tả, test, pseudocode, prompt được người dùng xác nhận qua plan.
- 6: triển khai run_diagnostics.py và test_diagnostics.py trong thư mục này.
- 7: kết quả thực thi lưu execution.json, pytest output và báo cáo sau chạy.
- 8: quyết định cuối của người dùng sau khi review; chưa lựa chọn regularization.

## Kết quả thực thi ngày 27/10/2026

Chỉ validation, mỗi lớp 500 ảnh. Suy luận CPU 4 threads mất 73,77 giây cho cả hai model và xuất hình; không tính thời gian khởi động import. Checkpoint được chọn theo accuracy trong run gốc, không đổi sang checkpoint loss thấp nhất.

| Model | Epoch | Accuracy | Macro precision | Macro recall/TPR | Macro F1 | Loss | Số ảnh sai |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 9 | 84,20% | 84,13% | 84,20% | 84,14% | 0,492284 | 790 |
| DenseNet121 | 10 | 83,00% | 83,21% | 83,00% | 83,06% | 0,518046 | 850 |

Accuracy khớp history tuyệt đối; chênh loss lần lượt 6,14e-9 và 1,34e-9. Đã kiểm tra đúng index/nhãn của training pool, 5000 index duy nhất, không giao train, checkpoint đúng hash của split. SHA256 trước/sau của checkpoint, config, history, split và selected registry giữ nguyên; xem [execution.json](execution.json). Không đọc test_predictions, không gọi train/optimizer.

### Lớp động vật nào khó nhất?

F1 cân bằng precision và recall; recall cho biết tìm được bao nhiêu trong 500 ảnh thật của lớp đó.

| Lớp | ResNet recall | ResNet F1 | DenseNet recall | DenseNet F1 |
|---|---:|---:|---:|---:|
| Cat | 68,0% | 71,28% | 74,4% | 72,02% |
| Dog | 76,4% | 75,94% | 75,6% | 75,75% |
| Bird | 81,2% | 81,36% | 80,0% | 81,72% |
| Deer | 82,8% | 81,66% | 82,0% | 79,23% |
| Horse | 88,4% | 87,87% | 80,4% | 82,89% |
| Frog | 90,8% | 89,81% | 84,2% | 85,83% |

Cat có F1 thấp nhất, dog thứ hai ở cả hai model. DenseNet nhận được nhiều cat thật hơn (372 so với 340), nhưng precision cat thấp hơn (69,79% so với 74,89%): nó cũng gọi nhầm nhiều ảnh khác là cat hơn. Vì thế không thể đọc recall đơn lẻ rồi kết luận DenseNet tốt hơn toàn diện ở cat.

### Những cặp nhầm nổi bật

| Chiều nhầm | ResNet số ảnh (% lớp thật) | DenseNet số ảnh (% lớp thật) |
|---|---:|---:|
| Cat -> dog | 81 (16,2%) | 61 (12,2%) |
| Dog -> cat | 58 (11,6%) | 67 (13,4%) |
| Bird -> deer | 24 (4,8%) | 30 (6,0%) |
| Horse -> deer | 19 (3,8%) | 38 (7,6%) |

Cat/dog hai chiều chiếm 139/790 lỗi ResNet (17,59%) và 128/850 lỗi DenseNet (15,06%). Đây là trọng tâm phân tích nhưng vẫn còn nhiều lỗi khác: automobile/truck, ship/airplane. Không tăng trọng số riêng cat/dog hoặc bỏ các lớp khác chỉ từ bảng này.

- ResNet: [ma trận](resnet18/confusion_matrix.png), [top 10 cặp](resnet18/top_confusions.png), [bảng từng lớp](resnet18/per_class.csv).
- DenseNet: [ma trận](densenet121/confusion_matrix.png), [top 10 cặp](densenet121/top_confusions.png), [bảng từng lớp](densenet121/per_class.csv).

Ma trận: hàng là nhãn thật, cột là dự đoán. Ví dụ hàng cat/cột dog của ResNet là 81, tức 81/500 cat bị đoán dog; không phải 81% hoặc precision của dog. Bảng phải chuẩn hóa mỗi hàng về 100%.

### Sai do phân vân hay rất tự tin?

Có cả hai loại. Trung vị xác suất nhãn dự đoán trên các ảnh sai là 71,95% với ResNet, 67,15% với DenseNet; trung vị chênh lệch hai xác suất cao nhất là 51,03 và 42,69 điểm phần trăm. Đây là xác suất model xuất ra, chưa phải độ tin cậy đã hiệu chuẩn.

- ResNet #20883: horse -> deer, xác suất nhãn dự đoán 36,24%, margin chỉ 0,14 điểm phần trăm: gần như hòa giữa hai lựa chọn.
- DenseNet #42015: frog -> dog, xác suất 28,53%, margin 0,07 điểm phần trăm: phân vân.
- ResNet #32085: cat -> dog, p(dog) khoảng 99,8%, loss 8,17: sai tự tin.
- Cả hai sai #49425 (nhãn lưu là truck, dự đoán bird): ResNet p(bird) 99,995%, loss 12,92; DenseNet p(bird) 99,895%, loss 15,37.

790 ảnh sai đóng góp 85,52% tổng loss ResNet; 850 ảnh sai đóng góp 82,77% tổng loss DenseNet. 20 ảnh sai loss cao nhất đóng góp lần lượt 8,17% và 6,77% tổng loss. Chúng giải thích vì sao loss nhạy với một số lỗi tự tin, nhưng không đại diện toàn bộ validation.

Đã đọc trực tiếp [ảnh sai ResNet](resnet18/highest_loss_errors.png), [ảnh sai DenseNet](densenet121/highest_loss_errors.png), và hai lưới top_pair_examples. Nhiều ảnh cat/dog chỉ có đầu/mặt, góc nhìn hoặc ảnh mờ; đây là quan sát trực quan, chưa chứng minh nguyên nhân. Mẫu #49425 trông giống chim trên vật thể đứng dù nhãn là truck; #21538 nhãn automobile trông giống một con vật; #11562 nhãn deer cũng cần kiểm tra. Đây là **nghi vấn chất lượng nhãn/ảnh**, không kết luận dữ liệu sai chắc chắn. Không sửa nhãn, loại ảnh hoặc thay số liệu benchmark từ đánh giá trực quan này.

Các lưới chọn ảnh loss cao nhất trong mỗi cặp, nên có thiên lệch về lỗi khó/tự tin; không dùng chúng để ước lượng tần suất toàn bộ lỗi. Ảnh hiển thị là ảnh gốc 32x32, model nhận phiên bản resize 64x64 theo preprocessing cũ; phóng lớn không tạo thêm chi tiết.

### Bằng chứng overfitting và hướng tiếp theo

**ResNet18:** [đường cong](resnet18/learning_curves.png). Epoch 7 val loss 0,487652; epoch 10 là 0,504908 trong khi train loss giảm 0,190716 -> 0,100072. Validation accuracy vẫn đạt đỉnh ở epoch 9 (84,20%), nên checkpoint accuracy tốt nhất khác checkpoint loss tốt nhất. Có dấu hiệu overfit ở cuối, không có nghĩa toàn bộ fine-tuning thất bại. Gap accuracy epoch 10 là 13,71 điểm phần trăm.

**DenseNet121:** [đường cong](densenet121/learning_curves.png). Val loss giảm 0,570037 -> 0,518046 và accuracy tăng 80,78% -> 83,00% từ epoch 1 đến 10. Gap cuối 9,34 điểm phần trăm, nhưng chưa có validation xấu đi kéo dài. Không nên gọi mức độ overfit giống ResNet hoặc mặc định cắt sớm epoch 7 cho cả hai.

Train metric là trung bình online khi weights còn cập nhật, validation dùng weights cuối epoch; gap chỉ là chỉ báo, không phải phép so hai tập ở cùng một model cố định. Chưa đánh giá lại train cùng checkpoint và chưa có confusion matrix từng epoch, nên **chưa chứng minh cat/dog là nơi overfit tăng lên theo thời gian**. Chỉ biết đây là cặp lỗi lớn ở checkpoint đã chọn.

Thứ tự đề xuất, chưa triển khai training:
1. Người dùng kiểm tra lại các index nghi vấn nhãn; giữ nguyên benchmark. Không sửa nhãn validation để cải thiện điểm số.
2. Với ResNet, ưu tiên thiết kế thử nghiệm early stopping và quy tắc chọn checkpoint rõ ràng (accuracy hay loss); không tự đổi tiêu chí hiện tại. History có thể gợi ý epoch hợp lý nhưng checkpoint epoch 7 không được giữ riêng, không thể tái dựng dự đoán epoch 7 từ checkpoint epoch 9.
3. Sau đó chỉ thử một regularizer mỗi lần: weight decay cao hơn hoặc dropout classifier. Chưa có bằng chứng quyết định loại nào chắc chắn tốt hơn; chọn bằng validation loss, accuracy, macro F1 và các cặp lỗi, không chỉ gap nhỏ hơn.
4. Với DenseNet, giữ baseline riêng; nếu thử regularization, so trên cùng ngân sách/split/head initialization. Không ép giảm epoch chỉ vì cat/dog khó.
5. Augmentation nhẹ có thể kiểm chứng giả thuyết nhạy tư thế/nền, nhưng random augmentation mỗi epoch không tương thích với cache prefix cố định; cần plan khác về luồng dữ liệu và chi phí trước khi thử.

Một seed và một validation split chưa đủ kết luận hiệu quả tổng quát của regularization. Validation đã dùng chọn checkpoint; các chẩn đoán là phân tích phát triển, không phải kiểm định độc lập. Không dùng test để quyết định thay đổi tiếp theo.

### Kiểm chứng và code trọng tâm

16 test đạt; Ruff F đạt; git diff --check đạt. V01/V06 còn được kiểm tra trên hai model thật: 5000 ảnh, nhãn/index khớp, accuracy/loss khớp history. Trọng số và file đầu vào được bảo vệ bằng hash, không có cập nhật optimizer. Lần chạy sandbox đầu bị chặn ghi output; đã chạy lại với quyền phù hợp. Có chỉnh cách ghi xác suất nhỏ thành ký hiệu khoa học sau xem hình; dựng lại hình từ CSV đã lưu, không suy luận lại.

```python
model = build_model(name, pretrained=False)  # Không tải weights mới.
model.load_state_dict(checkpoint['state_dict'], strict=True)
p, y, indices = predict(model, loader, torch.device('cpu'))
# predict() đặt eval() và torch.inference_mode().
losses = -np.log(np.maximum(p[np.arange(len(y)), y].astype(np.float64),
                            np.finfo(float).tiny))
```

Điểm cần đọc: `p[i, y[i]]` lấy xác suất **nhãn thật** của ảnh i, không lấy xác suất nhãn model đoán. `strict=True` báo lỗi nếu checkpoint không khớp cấu trúc. `check_history()` báo lỗi nếu metric lệch trước khi xuất báo cáo model đó.

Mã không thay API/source Lab02 chính thức. Người dùng review kết quả trước khi quyết định thí nghiệm training tiếp theo.

## Rà soát nghi vấn nhãn đã thực hiện

Phạm vi: 10 ảnh được chọn từ các mẫu nghi vấn và lỗi loss cao; không phải kiểm toán tất cả 5000 ảnh. Không chạy lại model. Xem [ảnh gốc phóng lớn](label_review.png), [bảng truy vết](label_review.csv) và [biên bản kiểm tra](label_audit.json).

### Bằng chứng kỹ thuật

- Cả 5 data_batch train đều khớp MD5 chuẩn mà torchvision CIFAR10 cung cấp. Không phát hiện file batch bị sửa hoặc hỏng so với bản chuẩn đó.
- Đọc trực tiếp từng record từ pickle gốc; pixel, nhãn và thứ tự class khớp loader. Nhãn trong per_image.csv của hai model cũng khớp record gốc. Tất cả 10 index nằm trong validation.
- Ví dụ index 49425 là data_batch_5, hàng 9425 (đếm từ 0), source label 9 = truck, filename dump_truck_s_000294.png. Index 21538 là data_batch_3, hàng 1538, label automobile, filename automobile_s_001273.png.
- Điều này loại trừ lỗi lệch index/ghép nhãn cho 10 mẫu đã kiểm tra, nhưng **không chứng minh mọi nhãn gốc đều đúng về ngữ nghĩa**. Filename là metadata cùng nguồn, không phải xác minh độc lập bằng ảnh độ phân giải cao.

### Nhận xét sau xem ảnh trực tiếp

| Index | Nhãn gốc | Quan sát và kết luận hiện tại |
|---|---|---|
| 21538 | automobile | Hình tối giống đầu/thân một con vật hơn ô tô trong bản 32x32; cả hai đoán cat. Nghi vấn đáng ưu tiên kiểm tra thủ công, chưa xác nhận nhãn thay thế. |
| 12210 | automobile | Silhouette sáng trên nền đen, trông giống sinh vật/cá hơn xe khi phóng lớn. Hai model đoán khác nhau (airplane/frog); không được chọn nhãn bằng biểu quyết model. Giữ trạng thái nghi vấn. |
| 20178 | dog | Hình giống chim cổ dài trong bản nhỏ; cả hai đoán bird. Source filename maltese_s_001237.png lại gắn với label dog. Cần đối chiếu nguồn ảnh rõ hơn trước khi kết luận nhãn sai. |
| 49425 | truck | Vật thể đứng màu nâu/vàng trên nền trời dễ gợi hình chim, nhưng hình quá nhỏ để loại trừ cấu trúc thùng xe dựng. Metadata gốc dump_truck. Điều chỉnh nhận định ban đầu: ảnh mơ hồ, chưa có cơ sở xác nhận truck là nhãn sai. |
| 11562 | deer | Con vật nhỏ màu nâu, tai/mõm dễ nhầm dog; filename muntjac. Chưa có bằng chứng đủ mạnh bác bỏ deer. Không coi hai model đoán dog là nhãn chuẩn. |
| 8922 | ship | Thấy thân tàu/cột trên nền hoàng hôn; nhãn ship phù hợp quan sát. Cả hai đoán deer rất tự tin: ví dụ lỗi model, không phải bằng chứng nhãn sai. |
| 43123 | cat | Vật thể đỏ bao quanh vùng tối có chi tiết khó nhận; có thể đối tượng cat chỉ chiếm vùng nhỏ. Chưa xác định sai nhãn; cần xem vùng đối tượng, không đổi sang truck vì hai model cùng dự đoán vậy. |
| 49472 | dog | Thấy đầu con vật tối trong vật chứa đỏ, nhãn dog vẫn hợp lý. Model bất đồng horse/airplane; không có bằng chứng nhãn sai rõ. |
| 25869 | ship | Bối cảnh trắng/xanh, nhiều chi tiết mơ hồ; ResNet đúng ship còn DenseNet đoán dog. Chưa đủ bằng chứng nhãn sai. |
| 18566 | deer | Ảnh giống bố cục nhiều vùng/hình nhỏ, khó nhận đối tượng ở 32x32. Cả hai đoán truck; giữ nghi vấn chất lượng/khả năng nhận diện ảnh, chưa xác nhận sai nhãn. |

Các nhận xét về nội dung ảnh là đánh giá trực quan có giới hạn, không phải nhãn ground truth mới. Phóng lớn bằng nearest-neighbor giữ pixel gốc, không phục hồi chi tiết đã mất.

### Kết luận và ảnh hưởng đến hướng giảm overfit

**Chưa xác nhận được nhãn nào sai chắc chắn.** Đã xác minh nghi vấn xuất phát từ dữ liệu chuẩn và nội dung ảnh khó/mơ hồ, không thấy sai mapping trong 10 mẫu. Ưu tiên người dùng review #21538, #12210, #20178; #49425 và #11562 không nên bị kết luận sai nhãn từ dự đoán model trước đó.

Giữ nguyên 5000 validation và toàn bộ metric. Không loại các ảnh làm loss cao hoặc đổi nhãn để tăng điểm. Nếu sau này có nguồn ảnh/nhãn độc lập xác nhận lỗi, lưu một bản annotation audit riêng và báo cáo sensitivity riêng, không thay điểm benchmark gốc.

Nghi vấn nhãn không xóa bằng chứng đường cong ResNet train giảm/validation loss tăng ở cuối; cũng chưa đủ quy toàn bộ overfit cho label noise. Bước hợp lý vẫn là chốt tiêu chí checkpoint/early stopping cho ResNet trước khi thử regularization; nhầm cat/dog và chất lượng ảnh là hai nhánh chẩn đoán cần theo dõi riêng.

Kiểm chứng lượt này: script audit hoàn tất 10/10, 5/5 checksum chuẩn, class order khớp, pixel/label/prediction-record khớp; Ruff F đạt. Không chạy lại bộ test training hoặc inference, vì không thay engine/model. Code chỉ thêm audit_labels.py trong thư mục task; không sửa dataset, checkpoint hoặc nhãn.

## Cấu hình cải tiến được xác nhận: phương án 1

File [regularized.json](regularized.json) tạo run mới, không sửa control_proposed.json hoặc run cũ.

| Thành phần | Cấu hình |
|---|---|
| ResNet18 | Chỉ mở layer4.1 + classifier; 4.725.770 tham số trainable thay vì 8.398.858 |
| DenseNet121 | Chỉ mở denselayer13..16 của denseblock4 + norm5 + classifier; 651.658 thay vì 2.170.378 |
| Dropout | 0,2 trước Linear cuối, dùng cả head search và fine-tune; eval tắt dropout |
| Augmentation | Chỉ fine-tune train: RandomCrop(32,padding=4,reflect) và HorizontalFlip(0.5), sau đó resize/normalize cũ |
| Cache | Head vẫn dùng pooled features cố định trước dropout; fine-tune không dùng prefix cache |
| LR và weight decay | Giữ head search .001/.01, fine backbone 1e-5/head 1e-4, AdamW decay 1e-4 |
| Epoch/batch | Giữ 4 head trials, batch 64/128, head 10 và fine-tune 10 epoch/model |
| Checkpoint | Giữ best validation accuracy, tie loss; chưa thêm early stopping vì chưa chốt policy |

Không benchmark và không train full bằng agent. Không đảm bảo hoàn thành trong 2 giờ vì fresh augmentation chạy full image forward. Đây là một cấu hình kết hợp ba thay đổi; kết quả chỉ đánh giá cả cấu hình, không tách đóng góp từng biện pháp.

Từ bất kỳ thư mục nào:

```bash
cd /Users/mac/Desktop/DEEP-V2/Lab-DL
.venv/bin/python labs/lab02/scripts/run.py \
  --config /Users/mac/Desktop/DEEP-V2/Lab-DL/.agents/lab02-val-diagnostics/regularized.json \
  --device cpu --dry-run
```

Để train thật, bỏ `--dry-run`. Terminal khác mở TensorBoard bằng `.venv/bin/tensorboard --logdir /Users/mac/Desktop/DEEP-V2/Lab-DL/labs/lab02/outputs/runs --port 6006` rồi chọn run có profile cifar10_macintel_regularized64. Nếu server cũ đang chạy trên 6006 thì dùng lại server đó khi logdir chứa outputs/runs.

Code trọng tâm:

```python
model.fc = nn.Sequential(nn.Dropout(0.2), model.fc)
# Các tham số trước đó đều được freeze, chỉ bật lại layer4.1 và classifier.
for p in model.layer4[1].parameters():
    p.requires_grad = True
```

Metadata checkpoint lưu dropout và augmentation để infer/resume dựng lại đúng model/loader. Mặc định dropout=0 giữ schema checkpoint cũ. Cache + augmentation bị chặn bằng ValueError; augmentation trên val/test cũng báo lỗi.

Regression toàn suite lần đầu: 56 đạt, 5 lỗi. Một lỗi do mock loader chưa nhận augmentation đã được sửa; 4 lỗi còn lại gắn với fixture run lịch sử/selected run (hash EDA, thiếu resume snapshot, execution policy và số epoch 5/2 hard-code), không thay artifact để làm test xanh. Rerun tập test liên quan: **32 đạt trong 47,31 giây** (test_regularization, test_tuning, test_contracts, test_cached_prefix). Gồm smoke pipeline một epoch/4 ảnh giả lập, kiểm tra dropout checkpoint roundtrip và legacy checkpoint, gradient freeze, BatchNorm, augmentation train-only. Không gọi đây là full train hoặc benchmark.

## Cách chạy từ root repository
`MPLBACKEND=Agg .venv/bin/python .agents/lab02-val-diagnostics/run_diagnostics.py`

`MPLBACKEND=Agg .venv/bin/python -m pytest .agents/lab02-val-diagnostics/test_diagnostics.py -q`

## Tổ chức lại regression test

`test_regularization.py` đã chuyển sang `labs/lab02/tests/test_regularization.py` để được thu thập bởi `python -m pytest tests`. Nội dung kiểm tra giữ nguyên; bỏ bootstrap đường dẫn dành riêng cho thư mục task. Các script chẩn đoán và artifact ở đây tiếp tục lưu bằng chứng lịch sử.
