# AGENTS.md — Hiến pháp vận hành và giao thức thực thi cho AI Agent trong DS & AI

Tài liệu này đóng vai trò là **hiến pháp kỹ thuật bắt buộc** mà mọi AI Coding Agent (Antigravity, Claude, Cursor, Copilot) phải tuân thủ nghiêm ngặt khi làm việc trên các dự án Data Science, Machine Learning và Deep Learning trong repository `Lab DL`.

---

## 1. Nguyên tắc cốt lõi

### 1.1. Spec và Baseline đi trước
Không bao giờ triển khai hay huấn luyện một kiến trúc phức tạp trước khi đã đặc tả rõ mô hình baseline đơn giản và định nghĩa metric đánh giá. Baseline là mốc neo để đo lường mọi tiến bộ về sau; thiếu baseline đồng nghĩa với việc không có căn cứ để chứng minh cải tiến.

### 1.2. Tham chiếu bắt buộc tới Playbook workflow
- Tài liệu tham chiếu nền tảng chung: [`docs/eda.md`](../docs/eda.md) (Exploratory Data Analysis) và [`docs/feature.md`](../docs/feature.md) (Feature Engineering).
- Với bài toán **Deep Learning (CV, NLP, Audio)**: bắt buộc tham chiếu [`docs/dl_flow.md`](../docs/dl_flow.md), [`docs/dl_data.md`](../docs/dl_data.md) và [`docs/dl_training.md`](../docs/dl_training.md).

### 1.3. Tái lập kết quả nghiêm ngặt (Strict Reproducibility)
Luôn cố định random seed (`random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) xuyên suốt mọi lần thực thi pipeline. Không có seed cố định thì kết quả giữa các lần chạy không thể so sánh, và mọi kết luận thí nghiệm trở nên vô nghĩa.

### 1.4. Kiểm tra 1-Batch Sanity Overfit bắt buộc
Trước khi đẩy job huấn luyện lên môi trường GPU từ xa (ví dụ Kaggle), mô hình bắt buộc phải vượt qua bài test overfit 1 batch chạy cục bộ, kéo loss về sát 0 (khoảng 5 giây). Bài test này là "chốt chặn" cuối cùng phát hiện lỗi kiến trúc, lỗi loss hoặc lỗi optimizer trước khi đốt GPU quota.

### 1.5. Giao thức Zero Waste
Nghiêm cấm chạy huấn luyện full nhiều epoch trên máy CPU cục bộ. Làm vậy vừa tốn thời gian vừa đốt token quota của người dùng. Thực thi cục bộ chỉ dừng ở mức: hoàn thiện module trong `src/`, dựng notebook tương tác trong `notebooks/`, và chạy test 1-batch sanity. Huấn luyện đầy đủ được thực thi trên Kaggle GPU (Tesla T4) hoặc compute tương đương.

### 1.6. Tái lập nghiêm ngặt và ghi nhận môi trường
Mọi kết quả phải tái lập được. Cố định seed cho toàn bộ thư viện tại một điểm vào duy nhất, bật chế độ deterministic khi cần, ghi nhận phiên bản thư viện chính, và đồng bộ môi trường giữa local và Kaggle. Một kết quả không tái lập được thì không phải là kết quả khoa học. Skill chịu trách nhiệm: `environment-and-reproducibility`.

### 1.7. Cổng kiểm tra bắt buộc (Checklist Gates)
Các cổng quan trọng phải vượt qua checklist tương ứng trong `.agent/checklists/` trước khi đi tiếp: `new_lab_bootstrap.md` khi bắt đầu lab mới, `pre_commit.md` trước khi commit, `pre_kaggle_upload.md` trước khi lên Kaggle GPU, và `pre_submission.md` trước khi nộp bài. Checklist biến quy tắc "nên làm" thành bước "phải đủ mới đi tiếp".

### 1.8. Tái sử dụng khung mẫu (Template Reuse)
Khi bắt đầu lab mới, agent phải dùng khung mẫu trong `.agent/templates/` (`walkthrough_template.md`, `lab_readme_template.md`) thay vì tự phát minh lại cấu trúc. Mọi lab phải có cùng cấu trúc để đảm bảo tính đồng nhất và giảm công rà soát.

### 1.9. Tài liệu sống duy nhất (Hợp nhất 3 Rule)
Toàn bộ tài liệu kỹ thuật của một lab phải được hợp nhất vào đúng MỘT tài liệu sống duy nhất: `walkthrough_lab_0X.md`. File này duy trì liên kết tường minh với 3 rule cốt lõi:
- **Section 1 (Spec & Baseline Setup):** Chịu sự điều phối của `spec-driven-development` (problem statement, data contract, mô hình baseline, ngưỡng chấp nhận).
- **Section 2 (Full-Flow Experiment Changelog):** Chịu sự điều phối của `walkthrough-and-experiment-tracking` (dòng chảy vòng lặp, chênh lệch qua 5 giai đoạn pipeline, lý do điều chỉnh hyperparameter).
- **Section 3 (Results & Final Technical Report):** Chịu sự điều phối của `documentation-and-adrs` (ma trận so sánh, chẩn đoán confusion matrix, phân tích lỗi, kết luận).

Không được tạo file `SPEC.md` hay `REPORT.md` riêng biệt.

### 1.10. Chính sách không Emoji/Icon
Nghiêm cấm emoji hoặc icon đồ họa trong mọi source code (`.py`, `.sh`), câu lệnh `print()`, log formatter, docstring, comment code, hoặc markdown cell trong Jupyter Notebook (`.ipynb`). Mọi tiêu đề và chỉ báo trạng thái phải dùng text kỹ thuật thuần ASCII/UTF-8 sạch, tương thích trên mọi môi trường terminal và pipeline CI/CD.

---

## 2. Ánh xạ Ý định (Intent) sang Skill

Bảng dưới đây ánh xạ ý định của người dùng sang skill tương ứng và tài liệu workflow tham chiếu:

| Ý định người dùng | Skill được kích hoạt | Tài liệu workflow tham chiếu |
|---|---|---|
| Yêu cầu mơ hồ / phác thảo ý tưởng ban đầu | `idea-refine` -> `interview-me` | `docs/dl_flow.md` Stage 1 |
| Bắt đầu lab mới / đặc tả bài toán | `spec-driven-development` | Section 1 trong `walkthrough_lab_0X.md` |
| Kiến trúc pipeline / phân rã công việc | `planning-and-task-breakdown` | `docs/dl_flow.md` kiến trúc 7 giai đoạn |
| Triển khai Dataset, DataLoader, Model | `test-driven-development` | `docs/dl_data.md` |
| Xác thực API PyTorch 2.x / Hugging Face | `source-driven-development` | Tài liệu chính thức PyTorch |
| GPU CUDA OOM, loss NaN, lỗi môi trường Kaggle | `debugging-and-error-recovery` | `docs/dl_training.md` |
| Tối ưu tốc độ huấn luyện, AMP FP16 | `performance-optimization` | `docs/dl_training.md` |
| Độ chính xác mô hình cao bất thường (> 98%) | `doubt-driven-development` | Kiểm toán Data Leakage |
| Đóng gói mô hình thành dịch vụ inference | `api-and-interface-design` | `docs/dl_flow.md` Stage 7 |
| Refactor code, tách module trong `src/` | `code-simplification` | Hướng dẫn Clean Architecture |
| Ghi log các lần chạy, so sánh chênh lệch thí nghiệm | `walkthrough-and-experiment-tracking` | Section 2 trong `walkthrough_lab_0X.md` |
| Hoàn thành lab, tạo báo cáo nộp bài cuối | `documentation-and-adrs` | Section 3 trong `walkthrough_lab_0X.md` |
| Kết quả chạy hai lần khác nhau, cần tái lập | `environment-and-reproducibility` | Giao thức cố định seed và ghi nhận phiên bản |
| Chuyển huấn luyện lên Kaggle GPU, đồng bộ artifact | `kaggle-remote-execution` | Checklist `pre_kaggle_upload.md` |
| Commit, push, quyết định file nào thuộc Git | `version-control-and-artifacts` | Checklist `pre_commit.md` |
| Bắt đầu lab mới | Checklist `new_lab_bootstrap.md` | `.agent/templates/` |

---

## 3. Chống biện minh (Anti-Rationalization)

Đây là danh sách các "lý lẽ ngụy biện" thường gặp và lý do bác bỏ chúng:

- [X] *"Cứ code trực tiếp, không cần viết spec"* -> **Bác bỏ:** Vi phạm giao thức kỹ thuật; không để lại baseline hay tiêu chí chấp nhận nào cho việc phân tích so sánh về sau.
- [X] *"Tách thành file SPEC.md và REPORT.md riêng"* -> **Bác bỏ:** Gây phân mảnh tài liệu và trùng lặp nội dung kỹ thuật. Hợp nhất cả 3 thành phần (Spec, Changelog, Report) vào `walkthrough_lab_0X.md`.
- [X] *"Bỏ qua test overfit 1 batch, train thẳng trên Kaggle"* -> **Bác bỏ:** Đốt GPU quota và hàng giờ compute nếu tồn tại lỗi dimension hoặc lỗi loss âm thầm.
- [X] *"Chạy huấn luyện full nhiều epoch trên CPU cục bộ"* -> **Bác bỏ:** Bào mòn tài nguyên máy cục bộ và token quota. Dừng sau khi xác thực 1-batch sanity và chuyển thực thi sang Kaggle GPU.
- [X] *"Dán log terminal thô vào chat thay vì ghi tài liệu"* -> **Bác bỏ:** Phải cập nhật đủ 5 giai đoạn pipeline và các chênh lệch metric định lượng trong `walkthrough_lab_0X.md`.
- [X] *"Thêm emoji hoặc icon vào notebook hay script cho đẹp"* -> **Bác bỏ:** Vi phạm chuẩn chuyên nghiệp và gây lỗi mã hóa ký tự trên nhiều môi trường runtime khác nhau. Chỉ dùng text kỹ thuật thuần túy.
- [X] *"Đặt seed là đủ để tái lập rồi"* -> **Bác bỏ:** Còn nhiều nguồn ngẫu nhiên khác (NumPy, DataLoader worker, thuật toán GPU không xác định). Phải cố định toàn bộ và ghi nhận phiên bản môi trường.
- [X] *"Bỏ qua checklist cho nhanh"* -> **Bác bỏ:** Checklist là cổng chặn các lỗi tốn kém như lộ secret, đẩy file nặng, hoặc đốt GPU quota. Bỏ qua checklist tiết kiệm vài giây nhưng có thể tốn hàng giờ.
- [X] *"Commit checkpoint nặng lên Git cũng được"* -> **Bác bỏ:** Checkpoint làm repository phình to vĩnh viễn trong lịch sử Git. Chỉ commit source, tài liệu, và figure nhẹ.
- [X] *"Script Kaggle sẽ tự đẩy notebook lên Git"* -> **Bác bỏ:** File notebook tương tác không nằm trên đĩa container. Phải thu hồi notebook qua giao diện Kaggle rồi commit thủ công.
- [X] *"Mỗi lab tự thiết kế cấu trúc riêng cho phù hợp"* -> **Bác bỏ:** Cấu trúc không đồng nhất khiến việc rà soát tốn công và agent phải chỉnh sửa mỗi lần. Dùng khung mẫu trong `.agent/templates/`.
