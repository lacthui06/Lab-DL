# Module utils — Quản Lý Thí Nghiệm, Đo Tài Nguyên & Đồ Thị Khoa Học

## 1. Thông Tin Mô-đun (Name & Metadata)
- **Tên mô-đun:** `src/utils` (Run Management, Hardware Profiling, Environment Snapshots & Scientific Visualization)
- **Ngày tạo (Created):** 2026-10-22 · **Tác giả (Author):** Thái Anh Lạc
- **Ngày cập nhật (Updated):** 2026-11-10 · **Người cập nhật (Author):** Thái Anh Lạc
- **Tệp tài liệu chính thức:** `labs/lab01/src/utils/README.md`

---

## 2. Mục Đích (Purpose)
Module này là "bộ đồ nghề hậu cần" phục vụ cho toàn bộ các khâu thực nghiệm của Lab 01:
- Quản lý vòng đời từng lần chạy thí nghiệm (`Run`): Mỗi lần thực thi đều được sinh ra một thư mục riêng biệt gắn ngày giờ và mã ngẫu nhiên, tự động ghi lại toàn bộ nhật ký terminal vào `console.log`, không bao giờ ghi đè làm mất kết quả của các lần chạy trước.
- Khóa chặt mọi nguồn ngẫu nhiên (`set_seed`) và chụp ảnh toàn bộ thư viện máy tính (`environment_snapshot`) để đảm bảo đem sang máy khác chạy vẫn y chang kết quả.
- Đo đạc chính xác thời gian và dung lượng RAM/GPU tiêu thụ khi học (ms/mẫu) bằng một tiến trình con độc lập (`subprocess`) để tránh rò rỉ bộ nhớ.
- Vẽ các biểu đồ khoa học đạt chuẩn báo cáo: đường cong học tập 2 khung, ma trận nhầm lẫn 10×10 có in số đếm thật từng ô, và 10 đường ROC/PR One-vs-Rest.

---

## 3. Khởi Tạo & Các Công Cụ Nền Tảng (Init Utils & Toolset)

### 3.1. Cấu Trúc 5 Nhóm Công Cụ Hỗ Trợ (Toolset Structure)

```text
1. Khởi tạo & Quản lý Thí nghiệm khép kín:
   └─ Run (runs.py#L33) ──> Tự sinh thư mục outputs/runs/<timestamp>_<name>_<hash>/
   └─ set_seed (helpers.py#L95) ──> Khóa chặt cả 4 nguồn ngẫu nhiên (Python, NumPy, PyTorch CPU/GPU)

2. Quản lý Checkpoint & Bàn giao:
   └─ helpers.py#L106 ──> save_checkpoint() lưu model_state_dict và metadata vào best.pt
   └─ helpers.py#L136 ──> promote_model_checkpoint() tự động sao lưu model tốt nhất vào labs/lab01/models/

3. Đo đạc chi phí tài nguyên (Hardware Profiling):
   └─ performance.py#L135 ──> Tách tiến trình con đo thời gian (ms/mẫu) và RAM RSS cho batch 128
   └─ performance.py#L200 ──> single_sample_training_benchmark() đo chi phí khi batch_size=1

4. Trực quan hóa tiến trình học (Learning Visualization):
   └─ plots.py#L92 ──> plot_loss_and_acc_curves() vẽ Loss và Accuracy (Train vs Val) qua 10 epoch

5. Chẩn đoán lỗi và Đánh giá đa lớp (Error Diagnostics):
   └─ plots.py#L13 ──> draw_confusion_counts() vẽ ma trận nhầm lẫn 10x10 có in số đếm thật
   └─ comparison_plots.py#L14 ──> build_comparison() vẽ 10 đường cong ROC và PR One-vs-Rest
```

---

### 3.2. Đầu Vào và Đầu Ra của Các Tiện Ích (Input / Output Utils)

| Công cụ / Tiện ích | Dữ liệu đầu vào tiếp nhận | Sản phẩm đầu ra (Artifact Path) | Ý nghĩa nghiệp vụ |
| :--- | :--- | :--- | :--- |
| **Quản lý Thí nghiệm (`Run`)** | Tên thực nghiệm + Dictionary tham số `config` | `outputs/runs/<run_id>/console.log`<br>`config.json`, `status.json` | Lưu vết toàn bộ những gì in ra màn hình terminal, kể cả khi quá trình chạy bị văng lỗi. |
| **Ảnh môi trường (`Snapshot`)** | Thông tin máy tính và thư viện hiện hành | `environment_snapshot.json` | Ghi nhận phiên bản OS, Python 3.13, PyTorch 2.14, CPU/CUDA để đối chiếu khi chuyển máy. |
| **Đo tài nguyên (`Benchmark`)** | Checkpoint mô hình + Mini-batch 128 ảnh | `training_performance.csv`<br>`training_resources.png` | Bảng đo thời gian (ms/mẫu) và dung lượng RAM thực tế của 1 bước cập nhật gradient. |
| **Đồ thị học tập (`Curves`)** | Mảng loss và accuracy từ `history.json` | `learning_curves.png` | Nhìn trực quan xem mô hình học mượt hay bị giật, phát hiện dấu hiệu học vẹt (overfit). |
| **Ma trận nhầm lẫn (`Confusion`)** | Mảng nhãn thật và mảng nhãn dự đoán test | `confusion_comparison.png` | Ma trận 10×10 có in số lượng ảnh bị đoán nhầm vào từng ô cụ thể kèm thanh màu sắc. |
| **Đường cong ROC & PR (`OvR`)** | Mảng xác suất Softmax `(10000, 10)` trên test | `*_ovr_curves.png`<br>`roc_ovr_curves.csv`, `pr_ovr_curves.csv` | 10 đường cong đánh giá độc lập năng lực phân loại của từng loại trang phục kèm điểm AUC/AP. |

---

### 3.3. Tệp Mã Nguồn & Bản Đồ Điều Hướng Code (Source Files)

Giữ phím **`Ctrl + Click`** để mở trực tiếp trong trình soạn thảo:

| Tệp mã nguồn (Source File) | Ký hiệu / Hàm chính (Ctrl+Click) | Dòng code | Việc đảm nhiệm |
| :--- | :--- | :--- | :--- |
| [`runs.py`](runs.py) | [`class Run`](runs.py#L33)<br>[`write_json()`](runs.py#L13) | [`runs.py#L13-L67`](runs.py#L13-L67) | Quản lý thư mục thực nghiệm khép kín, tự sinh mã hash và ngày giờ, lưu `console.log` và `status.json`. |
| [`helpers.py`](helpers.py) | [`set_seed()`](helpers.py#L95)<br>[`save_checkpoint()`](helpers.py#L106)<br>[`promote_model_checkpoint()`](helpers.py#L136) | [`helpers.py#L95-L160`](helpers.py#L95-L160) | Cố định seed 42, lưu/nạp checkpoint và cơ chế tự động bàn giao model tốt nhất vào `labs/lab01/models/`. |
| [`performance.py`](performance.py) | [`benchmark_models()`](performance.py#L135)<br>[`single_sample_training_benchmark()`](performance.py#L200) | [`performance.py#L135-L230`](performance.py#L135-L230) | Bật tiến trình con đo thời gian (ms/sample) và RAM RSS khi huấn luyện 1 bước (cả batch 128 và batch 1). |
| [`plots.py`](plots.py) | [`draw_confusion_counts()`](plots.py#L13)<br>[`plot_loss_and_acc_curves()`](plots.py#L92) | [`plots.py#L13-L120`](plots.py#L13-L120) | Vẽ biểu đồ tiến trình học 2 khung và vẽ ma trận nhầm lẫn kèm số đếm số lượng ảnh thật. |
| [`comparison_plots.py`](comparison_plots.py) | [`build_comparison()`](comparison_plots.py#L14) | [`comparison_plots.py#L14-L90`](comparison_plots.py#L14-L90) | Tổng hợp kết quả đối sánh giữa các mô hình, vẽ 10 đường cong ROC và PR One-vs-Rest kèm điểm Macro AUC. |

- **Kho checkpoint bàn giao tại [`labs/lab01/models/`](../../models/):**
  - Hàm `promote_model_checkpoint()` trong `helpers.py#L136` tự động quản lý các file `.pt` và sổ cái `manifest.json`.

---

## 4. Các Thảo Luận & Quyết Định Kỹ Thuật (Discussions from Description)

### 4.1. Tại sao phải đo tài nguyên bằng một tiến trình con riêng biệt (Subprocess)? ([Description Mục 7.12](../../description_lab_01.md#12-yêu-cầu-bổ-sung-về-kiến-trúc-performance-và-monitoring))
- Nếu ta thực hiện đo tài nguyên (RAM, thời gian) ngay trong luồng chính của Python, các biến đệm (cache), bộ thu gom rác (garbage collector) và các mảng NumPy của các bước huấn luyện trước đó vẫn còn nằm trong bộ nhớ RAM, làm con số RAM đo được bị phình to giả tạo.
- Việc kích hoạt một tiến trình mới tinh (`multiprocessing.Process`) giúp đo đạc hoàn toàn độc lập, phản ánh trung thực bộ nhớ RAM RSS tối thiểu mà mô hình thực sự chiếm dụng từ hệ điều hành.

### 4.2. Tại sao ma trận nhầm lẫn bắt buộc phải in số đếm thật lên từng ô? ([Description Mục 7.9](../../description_lab_01.md#9-notebook-phải-trình-bày-những-gì-và-tránh-lặp-ra-sao))
- Rất nhiều báo cáo chỉ vẽ ma trận nhầm lẫn bằng các ô màu xanh đỏ đậm nhạt. Tuy nhiên, mắt người không thể phân biệt được ô này là nhầm 5 ảnh hay nhầm 50 ảnh.
- Việc in rõ số lượng ảnh đếm thực tế lên từng ô (ví dụ: in rõ số lượng áo sơ mi bị nhầm thành áo thun hay áo khoác) cung cấp bằng chứng định lượng không thể chối cãi để giảng viên và người phản biện đánh giá chính xác nguyên nhân gây lỗi của mô hình.

### 4.3. Tại sao không bao giờ ghi đè lên thư mục run cũ? ([Description Mục 7.11](../../description_lab_01.md#11-môi-trường-chạy-quản-lý-artifact-và-báo-cáo))
- Trong nghiên cứu khoa học, mỗi lần chạy thí nghiệm là một tài sản dữ liệu. Bằng cách tự động tạo thư mục kèm dấu thời gian và mã băm ngẫu nhiên `outputs/runs/<timestamp>_<name>_<hash>/`, ta bảo toàn 100% lịch sử thực nghiệm. Nếu một lần chạy thử nghiệm mới bị thất bại, ta vẫn có thể mở lại kết quả tốt nhất của lần chạy trước mà không bị mất mát dữ liệu.
