# AGENTS.md — Hiến pháp vận hành và giao thức thực thi cho AI Agent trong DS & AI

Tài liệu này đóng vai trò là **hiến pháp kỹ thuật bắt buộc** mà mọi AI Coding Agent (Antigravity, Claude, Cursor, Copilot) phải tuân thủ nghiêm ngặt khi làm việc trên các dự án Data Science, Machine Learning và Deep Learning trong repository `Lab DL`.

---

## 1. Nguyên tắc cốt lõi

### 1.1. Spec và Baseline đi trước
Không bao giờ triển khai hay huấn luyện một kiến trúc phức tạp trước khi đã đặc tả rõ mô hình baseline đơn giản và định nghĩa metric đánh giá. Baseline là mốc neo để đo lường mọi tiến bộ về sau; thiếu baseline đồng nghĩa với việc không có căn cứ để chứng minh cải tiến.

### 1.2. Tham chiếu bắt buộc tới Playbook workflow
- Với bài toán **Tabular ML**: bắt buộc tham chiếu [`workflows/ML_flow.md`](../workflows/ML_flow.md) và [`workflows/eda_guide.md`](../workflows/eda_guide.md).
- Với bài toán **Deep Learning (CV, NLP, Audio)**: bắt buộc tham chiếu [`dl_workflows/DL_flow.md`](../dl_workflows/DL_flow.md), [`dl_workflows/dl_data_and_augmentation_guide.md`](../dl_workflows/dl_data_and_augmentation_guide.md) và [`dl_workflows/dl_training_and_optimization_guide.md`](../dl_workflows/dl_training_and_optimization_guide.md).

### 1.3. Tái lập kết quả nghiêm ngặt (Strict Reproducibility)
Luôn cố định random seed (`random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) xuyên suốt mọi lần thực thi pipeline. Không có seed cố định thì kết quả giữa các lần chạy không thể so sánh, và mọi kết luận thí nghiệm trở nên vô nghĩa.

### 1.4. Kiểm tra 1-Batch Sanity Overfit bắt buộc
Trước khi đẩy job huấn luyện lên môi trường GPU từ xa (ví dụ Kaggle), mô hình bắt buộc phải vượt qua bài test overfit 1 batch chạy cục bộ, kéo loss về sát 0 (khoảng 5 giây). Bài test này là "chốt chặn" cuối cùng phát hiện lỗi kiến trúc, lỗi loss hoặc lỗi optimizer trước khi đốt GPU quota.

### 1.5. Giao thức Zero Waste
Nghiêm cấm chạy huấn luyện full nhiều epoch trên máy CPU cục bộ. Làm vậy vừa tốn thời gian vừa đốt token quota của người dùng. Thực thi cục bộ chỉ dừng ở mức: hoàn thiện module trong `src/`, dựng notebook tương tác trong `notebooks/`, và chạy test 1-batch sanity. Huấn luyện đầy đủ được thực thi trên Kaggle GPU (Tesla T4) hoặc compute tương đương.

### 1.6. Tài liệu sống duy nhất (Hợp nhất 3 Rule)
Toàn bộ tài liệu kỹ thuật của một lab phải được hợp nhất vào đúng MỘT tài liệu sống duy nhất: `walkthrough_lab_0X.md`. File này duy trì liên kết tường minh với 3 rule cốt lõi:
- **Section 1 (Spec & Baseline Setup):** Chịu sự điều phối của `spec-driven-development` (problem statement, data contract, mô hình baseline, ngưỡng chấp nhận).
- **Section 2 (Full-Flow Experiment Changelog):** Chịu sự điều phối của `walkthrough-and-experiment-tracking` (dòng chảy vòng lặp, chênh lệch qua 5 giai đoạn pipeline, lý do điều chỉnh hyperparameter).
- **Section 3 (Results & Final Technical Report):** Chịu sự điều phối của `documentation-and-adrs` (ma trận so sánh, chẩn đoán confusion matrix, phân tích lỗi, kết luận).

Không được tạo file `SPEC.md` hay `REPORT.md` riêng biệt.

### 1.7. Chính sách không Emoji/Icon
Nghiêm cấm emoji hoặc icon đồ họa trong mọi source code (`.py`, `.sh`), câu lệnh `print()`, log formatter, docstring, comment code, hoặc markdown cell trong Jupyter Notebook (`.ipynb`). Mọi tiêu đề và chỉ báo trạng thái phải dùng text kỹ thuật thuần ASCII/UTF-8 sạch, tương thích trên mọi môi trường terminal và pipeline CI/CD.

---

## 2. Ánh xạ Ý định (Intent) sang Skill

Bảng dưới đây ánh xạ ý định của người dùng sang skill tương ứng và tài liệu workflow tham chiếu:

| Ý định người dùng | Skill được kích hoạt | Tài liệu workflow tham chiếu |
|---|---|---|
| Yêu cầu mơ hồ / phác thảo ý tưởng ban đầu | `idea-refine` -> `interview-me` | `DL_flow.md` Stage 1 |
| Bắt đầu lab mới / đặc tả bài toán | `spec-driven-development` | Section 1 trong `walkthrough_lab_0X.md` |
| Kiến trúc pipeline / phân rã công việc | `planning-and-task-breakdown` | `DL_flow.md` kiến trúc 7 giai đoạn |
| Triển khai Dataset, DataLoader, Model | `test-driven-development` | `dl_data_and_augmentation_guide.md` |
| Xác thực API PyTorch 2.x / Hugging Face | `source-driven-development` | Tài liệu chính thức PyTorch |
| GPU CUDA OOM, loss NaN, lỗi môi trường Kaggle | `debugging-and-error-recovery` | `dl_training_and_optimization_guide.md` |
| Tối ưu tốc độ huấn luyện, AMP FP16 | `performance-optimization` | `dl_training_and_optimization_guide.md` |
| Độ chính xác mô hình cao bất thường (> 98%) | `doubt-driven-development` | Kiểm toán Data Leakage |
| Đóng gói mô hình thành dịch vụ inference | `api-and-interface-design` | `DL_flow.md` Stage 7 |
| Refactor code, tách module trong `src/` | `code-simplification` | Hướng dẫn Clean Architecture |
| Ghi log các lần chạy, so sánh chênh lệch thí nghiệm | `walkthrough-and-experiment-tracking` | Section 2 trong `walkthrough_lab_0X.md` |
| Hoàn thành lab, tạo báo cáo nộp bài cuối | `documentation-and-adrs` | Section 3 trong `walkthrough_lab_0X.md` |

---

## 3. Chống biện minh (Anti-Rationalization)

Đây là danh sách các "lý lẽ ngụy biện" thường gặp và lý do bác bỏ chúng:

- [X] *"Cứ code trực tiếp, không cần viết spec"* -> **Bác bỏ:** Vi phạm giao thức kỹ thuật; không để lại baseline hay tiêu chí chấp nhận nào cho việc phân tích so sánh về sau.
- [X] *"Tách thành file SPEC.md và REPORT.md riêng"* -> **Bác bỏ:** Gây phân mảnh tài liệu và trùng lặp nội dung kỹ thuật. Hợp nhất cả 3 thành phần (Spec, Changelog, Report) vào `walkthrough_lab_0X.md`.
- [X] *"Bỏ qua test overfit 1 batch, train thẳng trên Kaggle"* -> **Bác bỏ:** Đốt GPU quota và hàng giờ compute nếu tồn tại lỗi dimension hoặc lỗi loss âm thầm.
- [X] *"Chạy huấn luyện full nhiều epoch trên CPU cục bộ"* -> **Bác bỏ:** Bào mòn tài nguyên máy cục bộ và token quota. Dừng sau khi xác thực 1-batch sanity và chuyển thực thi sang Kaggle GPU.
- [X] *"Dán log terminal thô vào chat thay vì ghi tài liệu"* -> **Bác bỏ:** Phải cập nhật đủ 5 giai đoạn pipeline và các chênh lệch metric định lượng trong `walkthrough_lab_0X.md`.
- [X] *"Thêm emoji hoặc icon vào notebook hay script cho đẹp"* -> **Bác bỏ:** Vi phạm chuẩn chuyên nghiệp và gây lỗi mã hóa ký tự trên nhiều môi trường runtime khác nhau. Chỉ dùng text kỹ thuật thuần túy.
