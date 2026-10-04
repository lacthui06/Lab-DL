---
name: using-ds-ai-skills
description: Điều phối và ánh xạ luồng làm việc của các dự án Data Science, Machine Learning và Deep Learning tới kỹ năng phù hợp và tài liệu Flow tương ứng trong Lab DL.
---

# using-ds-ai-skills

## Tổng quan
Kỹ năng này đóng vai trò là "Nhạc trưởng" định tuyến mọi yêu cầu trong lĩnh vực Khoa học Dữ liệu và AI vào đúng quy trình kỹ thuật chuẩn hóa, đồng thời kích hoạt việc tham chiếu trực tiếp vào các tài liệu nghiệp vụ [**`workflows/`**](../../workflows/) và [**`dl_workflows/`**](../../dl_workflows/).

## Khi nào sử dụng
- Bắt đầu một phiên làm việc mới về DS/AI.
- Người dùng đưa ra một bài toán, tập dữ liệu hoặc yêu cầu xây dựng mô hình.
- Cần quyết định bước tiếp theo trong vòng đời phát triển dự án.

## Quy trình điều phối chuẩn (Execution Flow)
1. **Phân loại bài toán (Task Classification):**
   - Xác định dạng bài: Dữ liệu bảng Tabular -> Trỏ vào `workflows/ML_flow.md`.
   - Dữ liệu Ảnh/Chữ/Học sâu -> Trỏ vào `dl_workflows/DL_flow.md`.
2. **Khởi tạo & Đặc tả:**
   - Kích hoạt `idea-refine` / `interview-me` nếu đề bài chưa rõ.
   - Kích hoạt `spec-driven-development` để lập bản Spec kỹ thuật có mốc Baseline.
3. **Thực thi & Kiểm thử:**
   - Kích hoạt `planning-and-task-breakdown` để chia nhỏ module `src/` và xây dựng file Notebook tương tác trong `notebooks/`.
   - Kích hoạt `test-driven-development` để kiểm thử dữ liệu và Sanity Overfit 1 batch trên Local (~5 giây).
4. **Quy trình Huấn luyện & Thu hoạch Tối ưu (Zero Waste Protocol):**
   - **Tại Local:** Tuyệt đối KHÔNG tự ý chạy full training nhiều epoch trên CPU local làm tốn quota và thời gian của người dùng. Dừng ở mức hoàn thiện code + test sanity 1 batch thành công.
   - **Tại Kaggle GPU:** Hướng dẫn người dùng upload Notebook lên Kaggle GPU (Tesla T4) chạy huấn luyện siêu tốc. Code notebook tự động xuất file trọng số (`best_model.pt`) và ảnh trực quan hóa (Input/Output predictions, Confusion Matrix, Loss curves) ra `/kaggle/working/`.
   - **Thu hoạch:** Người dùng tải file kết quả từ Kaggle về thư mục `outputs/` ở local.
5. **Ghi nhận & Nghiệm thu:**
   - Kích hoạt `walkthrough-and-experiment-tracking` để cập nhật toàn bộ số liệu và bằng chứng vào file `walkthrough_lab_0X.md`.
   - Kích hoạt `doubt-driven-development` kiểm tra rò rỉ dữ liệu trước khi `documentation-and-adrs`.

## Tiêu chí nghiệm thu (Verification)
- Mọi khâu thực thi đều bám sát đúng tài liệu Flow tương ứng và có bằng chứng ghi nhận trong file Walkthrough.
