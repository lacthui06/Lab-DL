# AGENTS.md — Quy tắc ứng xử và thực thi cho AI Agent trong DS & AI

File này là bản "Hiến pháp kỹ thuật" bắt buộc mọi AI Coding Agent (Antigravity, Claude, Cursor, Copilot) phải tuân thủ khi làm việc trên các dự án Data Science, Machine Learning và Deep Learning trong thư mục `Lab DL`.

---

## 1. Nguyên tắc cốt lõi (Core Principles)

1. **Spec & Baseline trước khi Code:** Tuyệt đối không nhảy vào train model phức tạp khi chưa có Baseline đơn giản và chưa chốt Metric đánh giá.
2. **Tham chiếu Tài liệu Quy trình Chuyên sâu (Flow References):**
   - Khi làm bài toán Dữ liệu bảng (Tabular ML) ➔ Bắt buộc tham chiếu [**`workflows/ML_flow.md`**](../workflows/ML_flow.md) và [**`workflows/eda_guide.md`**](../workflows/eda_guide.md).
   - Khi làm bài toán Học sâu (Deep Learning / CV / NLP) ➔ Bắt buộc tham chiếu [**`dl_workflows/DL_flow.md`**](../dl_workflows/DL_flow.md), [**`dl_workflows/dl_data_and_augmentation_guide.md`**](../dl_workflows/dl_data_and_augmentation_guide.md), và [**`dl_workflows/dl_training_and_optimization_guide.md`**](../dl_workflows/dl_training_and_optimization_guide.md).
3. **Mã nguồn phải tái lập được (Reproducibility):** Luôn cố định random seed (`random`, `numpy`, `torch.manual_seed`) trong mọi thử nghiệm.
4. **Kiểm thử Sanity Overfit 1 batch:** Trước khi mang code lên Kaggle train, bắt buộc phải test thử trên 1 batch nhỏ ở local xem mô hình có ép được loss về 0 hay không (~5 giây).
5. **Tiết kiệm tài nguyên & Hạn ngạch (Zero Waste Protocol):** TUYỆT ĐỐI KHÔNG tự ý chạy full training nhiều epoch trên máy Local CPU gây tốn thời gian và lãng phí quota chat của người dùng. Ở local chỉ hoàn thiện module `src/`, notebook `notebooks/`, chạy test Sanity 1 batch, rồi hướng dẫn người dùng đẩy lên Kaggle GPU (Tesla T4) huấn luyện.
6. **Cập nhật Nhật ký Flow tự động:** Sau mỗi lần chạy thử nghiệm có kết quả (tải từ Kaggle về hoặc kết quả đo lường), AI phải kích hoạt `walkthrough-and-experiment-tracking` để ghi nhận toàn bộ thay đổi vào file `walkthrough_lab_0X.md`.
7. **Nghiêm cấm dùng Icon / Emoji trong Mã nguồn và Notebook (No Icons/Emojis Policy):** TUYỆT ĐỐI KHÔNG đưa bất kỳ biểu tượng cảm xúc (emoji/icon) nào vào mã nguồn (`.py`, `.sh`), chuỗi `print()`, log, docstring, chú thích, hoặc các ô markdown/code của Jupyter Notebook (`.ipynb`). Mọi thông báo trạng thái, tiêu đề phải dùng văn bản kỹ thuật chuẩn mực, chuyên nghiệp, hiển thị tốt trên mọi terminal và môi trường kiểm thử CI/CD.

---

## 2. Bảng ánh xạ ý định người dùng sang Kỹ năng (Intent ➔ Skill Mapping)

| Ý định của người dùng | Kỹ năng phải kích hoạt | Tài liệu Flow tham chiếu |
|---|---|---|
| Đề bài chưa rõ / Mới có ý tưởng sơ khai | `idea-refine` &rarr; `interview-me` | `DL_flow.md` Giai đoạn 1 |
| Bắt đầu bài Lab mới / Yêu cầu giải đề | `spec-driven-development` | `ML_flow.md` hoặc `DL_flow.md` |
| Phân rã cấu trúc / Lên kế hoạch Pipeline | `planning-and-task-breakdown` | `DL_flow.md` Sơ đồ 7 giai đoạn |
| Viết hàm Dataset, DataLoader, Model | `test-driven-development` | `dl_data_and_augmentation_guide.md` |
| Tra cứu chuẩn API PyTorch 2.x / HuggingFace | `source-driven-development` | Tài liệu chính thức PyTorch |
| Lỗi tràn VRAM GPU, Loss NaN, lỗi Kaggle | `debugging-and-error-recovery` | `dl_training_and_optimization_guide.md` |
| Tối ưu tốc độ huấn luyện, AMP FP16 | `performance-optimization` | `dl_training_and_optimization_guide.md` |
| Model đạt Accuracy cao bất thường (>98%) | `doubt-driven-development` | Soi Data Leakage |
| Đóng gói mô hình thành API suy luận | `api-and-interface-design` | `DL_flow.md` Giai đoạn 7 |
| Dọn dẹp code sạch đẹp, tách `src/`, xóa emoji/icon | `code-simplification` | Chuẩn Clean Architecture |
| **Ghi nhận lịch sử chạy, so sánh các Run** | **`walkthrough-and-experiment-tracking`** | **`walkthrough_lab_0X.md`** |
| Hoàn thành bài Lab, viết Model Card | `documentation-and-adrs` | Model Card & Báo cáo nộp bài |

---

## 3. Các suy nghĩ sai trái bị nghiêm cấm (Anti-Rationalization)

- ❌ *"Code luôn không cần Spec"* ➔ **Bác bỏ:** Vi phạm nguyên tắc kỹ thuật, không có mốc Baseline để so sánh.
- ❌ *"Không cần test overfit 1 batch, cứ mang lên Kaggle train"* ➔ **Bác bỏ:** Làm lãng phí hạn ngạch 30 tiếng GPU Kaggle nếu mô hình bị bug ngầm.
- ❌ *"Tự ý chạy full training trên CPU local"* ➔ **Bác bỏ:** Làm tràn tài nguyên local và lãng phí quota chat của người dùng. Bắt buộc dừng ở mức hoàn thiện Notebook + Sanity test 1 batch, sau đó hướng dẫn mang lên Kaggle GPU train và download artifact về.
- ❌ *"Chạy xong chỉ copy paste vài dòng log ra chat"* ➔ **Bác bỏ:** Bắt buộc phải cập nhật toàn bộ 5 khâu và changelog vào `walkthrough_lab_0X.md`.
- ❌ *"Thêm icon/emoji vào notebook hoặc script cho sinh động"* ➔ **Bác bỏ:** Vi phạm phong cách kỹ thuật chuẩn mực, dễ gây lỗi encoding môi trường (như Windows cp1252), làm rối rác output log. Chỉ dùng văn bản chuẩn ASCII/UTF-8 thuần túy.
