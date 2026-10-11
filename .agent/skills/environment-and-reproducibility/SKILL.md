---
name: environment-and-reproducibility
description: Quản lý môi trường thực thi và đảm bảo khả năng tái lập kết quả trong các dự án Data Science và Deep Learning, bao gồm cố định seed, ghi nhận phiên bản thư viện, deterministic training, và cấu hình môi trường giữa local và Kaggle.
---

# environment-and-reproducibility (Bản DS & AI)

## 1. Tổng quan

Skill này quản lý tính tái lập (reproducibility) - năng lực cốt lõi nhưng thường bị xem nhẹ nhất trong các dự án Machine Learning. Một kết quả không thể tái lập thì không phải là kết quả khoa học: nó không thể được kiểm chứng, không thể được so sánh công bằng giữa các run, và không thể bảo vệ trước hội đồng chấm.

Vấn đề gốc rễ của deep learning là tính ngẫu nhiên (stochasticity) xuất hiện ở rất nhiều nơi: khởi tạo trọng số, thứ tự shuffle dữ liệu, dropout, augmentation, và các thuật toán không xác định trên GPU. Nếu không kiểm soát từng nguồn ngẫu nhiên này, hai lần chạy cùng một cấu hình sẽ cho ra hai kết quả khác nhau, khiến mọi kết luận thí nghiệm trở nên vô nghĩa.

Skill này cũng quản lý khoảng cách môi trường giữa máy cục bộ và Kaggle GPU - nơi phiên bản thư viện khác nhau có thể âm thầm thay đổi hành vi mô hình.

Nguyên lý then chốt: mọi nguồn ngẫu nhiên phải được cố định, và mọi thành phần môi trường phải được ghi nhận, trước khi một kết quả được coi là hợp lệ.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Kết quả giữa hai lần chạy cùng cấu hình không giống nhau.
- Bắt đầu một lab mới và cần thiết lập nền tảng tái lập từ đầu.
- Chuyển thực thi giữa máy cục bộ và Kaggle GPU.
- Cần so sánh công bằng nhiều run khác nhau (baseline so với tuned).
- Chuẩn bị bàn giao codebase hoặc nộp bài cần kiểm chứng.
- Phát hiện mô hình hoạt động khác nhau giữa các máy.

**Điều kiện tiên quyết:**
- Đã có pipeline có thể chạy được tới bước huấn luyện.
- Đã xác định được các thư viện chính sử dụng trong dự án.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Pipeline huấn luyện hiện có | Bộ seed cố định được thiết lập ở điểm vào duy nhất |
| Danh sách thư viện sử dụng | Bản ghi phiên bản thư viện chính xác |
| Cấu hình môi trường local và Kaggle | Tài liệu ghi nhận khác biệt môi trường và cách xử lý |
| Hyperparameter và cấu hình run | Snapshot cấu hình cho từng run |
| Yêu cầu về mức độ tái lập | Bằng chứng chạy lại cho cùng kết quả |

## 4. Quy trình chi tiết

### Bước 1: Kiểm kê mọi nguồn ngẫu nhiên
Liệt kê tất cả nơi sinh ra tính ngẫu nhiên trong pipeline: khởi tạo trọng số, shuffle dữ liệu, split dữ liệu, dropout, augmentation, và các kernel không xác định trên GPU. Mục đích là để không bỏ sót nguồn nào; chỉ cần một nguồn không được cố định là toàn bộ tính tái lập sụp đổ.

### Bước 2: Cố định seed tại một điểm vào duy nhất
Thiết lập seed cho tất cả các thư viện liên quan (Python chuẩn, NumPy, PyTorch, CUDA) tại đúng một nơi trong module cấu hình, và gọi nó ngay đầu mỗi lần thực thi. Không rải lệnh đặt seed rải rác khắp code, vì như vậy khó kiểm soát và dễ bị bỏ sót. Ghi lại giá trị seed đã dùng cho từng run.

### Bước 3: Bật chế độ deterministic cho thuật toán GPU
Một số phép toán trên GPU không xác định theo mặc định. Bật chế độ deterministic để buộc các thuật toán này chạy theo cách có thể tái lập, và ghi nhận đánh đổi về tốc độ. Xác nhận rằng việc bật deterministic không gây lỗi trên môi trường mục tiêu.

### Bước 4: Ghi nhận phiên bản môi trường
Tạo và duy trì một bản ghi các thư viện chính cùng phiên bản chính xác (PyTorch, Torchvision, NumPy, Scikit-learn, và các thư viện phụ thuộc quan trọng). Bản ghi này là bằng chứng về môi trường đã tạo ra kết quả, và là cơ sở để tái lập về sau.

Bản ghi bắt buộc phải có cả **phiên bản trình thông dịch Python** (`python --version`), không chỉ phiên bản thư viện: cùng bộ thư viện nhưng khác phiên bản Python có thể cho hành vi khác nhau (ví dụ thay đổi mặc định của `random`, thay đổi trong thư viện chuẩn). Ghi phiên bản Python dưới dạng **comment** trong `requirements.txt` (ví dụ `# Local: Python 3.13.15`), vì dòng `python==x.y.z` không phải cú pháp pip hợp lệ và sẽ làm hỏng `pip install -r requirements.txt`.

Phải phân biệt hai thứ, và ghi cả hai:

- **Yêu cầu thật (derived requirement):** khoảng phiên bản Python mà project thực sự chạy được, suy ra từ trường `Requires-Python` của các thư viện đã pin (lấy giao của tất cả). Cách kiểm tra: `python -c "import importlib.metadata as md; print(md.metadata('numpy').get('Requires-Python'))"`. Ví dụ với Lab 01, giao của các pin là `>=3.12, !=3.14.1` (numpy 2.5.3 đặt sàn `>=3.12`, torchvision 0.29.1 loại trừ `3.14.1`). Đây mới là thông tin người tái lập cần.
- **Bản đã kiểm chứng (verified build):** đúng phiên bản trình thông dịch đã tạo ra kết quả, ví dụ 3.13.15.

Ghi cả hai dạng comment trong `requirements.txt`. Chỉ ghi mỗi "bản đã chạy" mà bỏ "yêu cầu thật" là ghi chú trang trí, không phải ràng buộc kỹ thuật.

### Bước 5: Đồng bộ môi trường giữa local và Kaggle
So sánh phiên bản thư viện giữa máy cục bộ và Kaggle. Ghi nhận mọi khác biệt có thể ảnh hưởng đến hành vi mô hình. Nếu có khác biệt lớn, điều chỉnh để hai môi trường gần nhau nhất có thể, hoặc ghi rõ khác biệt đó trong tài liệu.

### Bước 6: Ghi snapshot cấu hình cho từng run
Với mỗi lần huấn luyện, ghi lại đầy đủ: seed, hyperparameter, phiên bản thư viện, **phiên bản trình thông dịch Python**, và cấu hình thiết bị. Snapshot này được lưu cùng artifact của run để bất kỳ ai cũng có thể tái tạo lại đúng điều kiện; artifact chuẩn là `outputs/environment_snapshot.json` (hàm `collect_environment_snapshot()` trong `src/utils.py`).

### Bước 7: Kiểm chứng tái lập
Chạy lại pipeline hai lần với cùng cấu hình và so sánh kết quả. Nếu loss và metric khớp nhau (trong sai số cho phép của phép toán dấu phẩy động), tính tái lập đã được xác lập. Nếu không khớp, quay lại Bước 1 để tìm nguồn ngẫu nhiên chưa được cố định.

## 5. Deliverables

- Bộ seed cố định được thiết lập tại một điểm vào duy nhất trong module cấu hình.
- Bản ghi phiên bản thư viện chính xác của môi trường, kèm phiên bản trình thông dịch Python.
- Một snapshot môi trường dạng máy đọc được, sinh ra lúc chạy (`environment_snapshot.json`): phiên bản Python, platform, phiên bản thư viện, và thiết bị tính toán thực tế. Đây mới là bằng chứng của môi trường đã tạo ra kết quả; `requirements.txt` chỉ nêu môi trường mong muốn.
- Tài liệu ghi nhận khác biệt giữa local và Kaggle.
- Snapshot cấu hình cho từng run huấn luyện.
- Bằng chứng chạy lại cho ra cùng kết quả.
- Ghi chú về các đánh đổi khi bật chế độ deterministic.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `spec-driven-development` (cung cấp ràng buộc môi trường và mục tiêu tái lập) | `walkthrough-and-experiment-tracking` (nhận snapshot cấu hình để ghi vào changelog) |
| `planning-and-task-breakdown` (xác định các module cần cấu hình) | `kaggle-remote-execution` (nhận bản ghi phiên bản để đồng bộ môi trường) |
| `source-driven-development` (cung cấp phiên bản API hợp lệ) | `doubt-driven-development` (nhận bằng chứng tái lập để loại trừ artifact đánh giá) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Seed rải rác:** Đặt seed ở nhiều nơi khác nhau khiến một số nguồn ngẫu nhiên bị bỏ sót. Dấu hiệu: kết quả vẫn khác nhau dù đã "đặt seed". Phòng tránh: tập trung seed vào một hàm duy nhất và gọi ngay đầu chương trình.
- **Bỏ qua nguồn ngẫu nhiên của DataLoader:** Quên seed cho worker của DataLoader khi dùng nhiều tiến trình. Dấu hiệu: thứ tự dữ liệu thay đổi giữa các lần chạy. Phòng tránh: kiểm kê đầy đủ nguồn ngẫu nhiên ở Bước 1.
- **Lệch phiên bản thư viện:** Local và Kaggle dùng phiên bản PyTorch khác nhau gây hành vi khác nhau. Dấu hiệu: cùng code nhưng kết quả khác biệt rõ rệt giữa hai môi trường. Phòng tránh: so sánh phiên bản ở Bước 5.
- **Không ghi lại seed đã dùng:** Một tháng sau không biết run nào dùng seed nào. Phòng tránh: ghi snapshot cấu hình ở Bước 6.
- **Bật deterministic mà không kiểm tra lỗi:** Một số phép toán không hỗ trợ deterministic và gây lỗi runtime. Phòng tránh: xác nhận trên môi trường mục tiêu ở Bước 3.

## 8. Definition of Done

- [ ] Đã kiểm kê đầy đủ mọi nguồn ngẫu nhiên trong pipeline.
- [ ] Seed được cố định tại một điểm vào duy nhất và gọi ngay đầu chương trình.
- [ ] Chế độ deterministic đã được bật và xác nhận không gây lỗi.
- [ ] Phiên bản các thư viện chính đã được ghi nhận.
- [ ] Phiên bản trình thông dịch Python đã được ghi nhận (dạng comment trong `requirements.txt`).
- [ ] Snapshot môi trường sinh ra lúc chạy (`environment_snapshot.json`) tồn tại và khớp với `requirements.txt`.
- [ ] Khác biệt môi trường giữa local và Kaggle đã được ghi lại.
- [ ] Snapshot cấu hình tồn tại cho mỗi run huấn luyện.
- [ ] Đã chạy lại hai lần và xác nhận kết quả khớp nhau.
- [ ] Seed và cấu hình đã được ghi vào `walkthrough_lab_0X.md`.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Đặt `torch.manual_seed` là đủ để tái lập rồi"* -> **Bác bỏ:** Chỉ seed PyTorch là bỏ sót nguồn ngẫu nhiên từ NumPy, Python chuẩn, và DataLoader worker. Phải cố định toàn bộ các nguồn.
- [X] *"Kết quả hơi khác một chút cũng không sao, xu hướng vẫn đúng"* -> **Bác bỏ:** Nếu kết quả không tái lập được, không thể phân biệt cải thiện thật với nhiễu ngẫu nhiên. Mọi so sánh giữa các run mất giá trị.
- [X] *"Bật deterministic làm chậm huấn luyện, bỏ qua cho nhanh"* -> **Bác bỏ:** Đánh đổi tốc độ nhỏ so với giá trị của một kết quả có thể kiểm chứng. Có thể tắt deterministic khi đã xác lập xong kết quả cuối.
- [X] *"Ghi phiên bản thư viện mất thời gian, để sau"* -> **Bác bỏ:** Khi kết quả không tái lập được và cần truy nguyên, bản ghi phiên bản là manh mối duy nhất. Ghi ngay từ đầu rẻ hơn nhiều so với truy ngược về sau.
- [X] *"Local và Kaggle chắc chắn giống nhau, không cần so sánh"* -> **Bác bỏ:** Kaggle cập nhật phiên bản thư viện định kỳ và không đồng bộ với local. Khác biệt phiên bản là nguyên nhân phổ biến của kết quả lệch giữa hai môi trường.
