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
   - Xác định dạng bài: Dữ liệu bảng Tabular ➔ Trỏ vào `workflows/ML_flow.md`.
   - Dữ liệu Ảnh/Chữ/Học sâu ➔ Trỏ vào `dl_workflows/DL_flow.md`.
2. **Khởi tạo & Đặc tả:**
   - Kích hoạt `idea-refine` / `interview-me` nếu đề bài chưa rõ.
   - Kích hoạt `spec-driven-development` để lập bản Spec kỹ thuật có mốc Baseline.
3. **Thực thi & Kiểm thử:**
   - Kích hoạt `planning-and-task-breakdown` để chia nhỏ module `src/`.
   - Kích hoạt `test-driven-development` để kiểm thử dữ liệu và Sanity Overfit 1 batch.
4. **Huấn luyện & Ghi nhận Thực nghiệm:**
   - Huấn luyện trên local hoặc hướng dẫn đẩy lên Kaggle GPU.
   - **Bắt buộc kích hoạt `walkthrough-and-experiment-tracking`** để cập nhật toàn bộ flow vào file `walkthrough_lab_0X.md`.
5. **Nghiệm thu:**
   - Kích hoạt `doubt-driven-development` kiểm tra rò rỉ dữ liệu trước khi `documentation-and-adrs`.

## Tiêu chí nghiệm thu (Verification)
- Mọi khâu thực thi đều bám sát đúng tài liệu Flow tương ứng và có bằng chứng ghi nhận trong file Walkthrough.
