---
name: using-ds-ai-skills
description: Điều phối và định tuyến các tác vụ Data Science, Machine Learning, Deep Learning tới đúng skill chuyên biệt và playbook workflow tương ứng trong Lab DL.
---

# using-ds-ai-skills (Bản DS & AI)

## Tổng quan

Skill này đóng vai trò **nhạc trưởng tổng thể (master conductor)**, định tuyến mọi tác vụ thuộc Data Science, Machine Learning và Deep Learning tới các skill kỹ thuật chuẩn hóa, đồng thời liên kết trực tiếp tới các playbook chuyên ngành trong [`workflows/`](../../workflows/) và [`dl_workflows/`](../../dl_workflows/).

Đây là điểm vào (entry point) của toàn bộ framework. Bất kỳ khi nào một phiên làm việc mới bắt đầu mà chưa xác định rõ cần dùng skill nào, skill này chịu trách nhiệm phân loại và điều phối.

## Khi nào sử dụng

- Bắt đầu một phiên phát triển DS/AI mới.
- Người dùng đưa ra một bài tập, dataset, hoặc yêu cầu huấn luyện mô hình.
- Cần quyết định bước đi tiếp theo trong vòng đời dự án.
- Có sự nhập nhằng giữa nhiều hướng tiếp cận và cần chọn đúng playbook.

## Luồng thực thi chuẩn

### Bước 1: Phân loại tác vụ
- Dataset dạng bảng (Tabular) -> Định tuyến tới `workflows/ML_flow.md`.
- Computer Vision / NLP / Deep Learning -> Định tuyến tới `dl_workflows/DL_flow.md`.
- Việc phân loại sai ngay từ bước này sẽ khiến toàn bộ quy trình phía sau lệch chuẩn, vì vậy cần xác định rõ modality của dữ liệu đầu vào trước tiên.

### Bước 2: Khởi tạo và Đặc tả
- Kích hoạt `idea-refine` / `interview-me` nếu yêu cầu còn mơ hồ, chưa rõ metric, chưa rõ dữ liệu.
- Kích hoạt `spec-driven-development` để dựng đặc tả kỹ thuật và baseline trong **Section 1** của `walkthrough_lab_0X.md`.

### Bước 3: Triển khai và Kiểm thử
- Kích hoạt `planning-and-task-breakdown` để phân rã source code module trong `src/` và notebook tương tác trong `notebooks/`.
- Kích hoạt `test-driven-development` để xác thực schema dữ liệu và chạy bài test 1-batch sanity overfit cục bộ (khoảng 5 giây).

### Bước 4: Giao thức thực thi và thu hoạch (Zero Waste Protocol)
- **Cục bộ:** Nghiêm cấm chạy huấn luyện full nhiều epoch trên CPU cục bộ. Dừng lại ngay khi các module và bài test 1-batch sanity đã pass.
- **Kaggle GPU:** Chuyển hướng thực thi sang Kaggle GPU (Tesla T4) để huấn luyện tăng tốc. Notebook lưu checkpoint (`best_model.pt`) và các hình ảnh trực quan (predictions, confusion matrix, loss curves) vào `/kaggle/working/outputs/`.
- **Thu hoạch (Harvest):** Đồng bộ hoặc tải artifact trở về thư mục `outputs/` cục bộ.

### Bước 5: Tài liệu hóa và Nộp bài
- Kích hoạt `walkthrough-and-experiment-tracking` để ghi nhận mọi vòng lặp thí nghiệm vào **Section 2** của `walkthrough_lab_0X.md`.
- Kích hoạt `doubt-driven-development` để kiểm toán rò rỉ dữ liệu trước khi kích hoạt `documentation-and-adrs` để hoàn thiện **Section 3** của `walkthrough_lab_0X.md`.

## Tiêu chí xác thực

- Mỗi giai đoạn thực thi đều khớp với playbook workflow phù hợp và tạo ra các bản ghi có thể kiểm chứng được trong tài liệu sống duy nhất.
- Không có giai đoạn nào bị bỏ qua hoặc thực thi không đúng thứ tự phụ thuộc.
