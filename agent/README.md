# Workflow DS & AI (Data Science & Artificial Intelligence)

**Hệ thống quy trình và kỹ năng kỹ thuật chuẩn hóa dành riêng cho AI Coding Agents khi phát triển các dự án Data Science, Machine Learning và Deep Learning.**

---

## Tổng quan vòng đời phát triển DS & AI

Khác với phát triển phần mềm truyền thống, một dự án DS & AI đòi hỏi quy trình nghiêm ngặt từ khâu dữ liệu, kiểm chứng rò rỉ (leakage), kiểm soát tài nguyên GPU cho đến đóng gói mô hình. Hệ thống gồm **14 kỹ năng cốt lõi** trải dài qua 6 giai đoạn:

```
  KHỞI TẠO        ĐẶC TẢ & KẾ HOẠCH         XÂY DỰNG & TDD         GỠ LỖI & PHẢN BIỆN       TỐI ƯU & GIAO DIỆN       TỔNG KẾT
 ┌─────────┐     ┌───────────────────┐     ┌──────────────┐       ┌────────────────────┐   ┌───────────────────┐    ┌─────────┐
 │ Ý tưởng │ ──▶ │ Spec & Pipeline   │ ──▶ │ Code & Data  │ ───▶  │ Debug GPU / NaN    │ ─▶│ Vectorize / GPU   │ ──▶│ Clean   │
 │ Khảo sát│     │ Phân rã công việc │     │ Test Tensor  │       │ Soi Data Leakage   │   │ Build API Serving │    │ Báo cáo │
 └─────────┘     └───────────────────┘     └──────────────┘       └────────────────────┘   └───────────────────┘    └─────────┘
  interview-me     spec-driven-dev           tdd-for-ai             debug-error-recovery     performance-opt          simplify
  idea-refine      planning-breakdown        source-driven-dev      doubt-driven-dev         api-interface-design     doc-adrs
                                                                                                                      walkthrough
```

---

## Danh mục 14 Kỹ năng chuyên sâu cho DS & AI

### 0. Tầng điều phối (Meta)
- **`using-ds-ai-skills`**: Nhạc trưởng điều phối toàn bộ workflow, tự động nhận diện bài toán DS/AI và kích hoạt kỹ năng phù hợp.

### 1. Giai đoạn Khởi tạo & Định hình bài toán (Define)
- **`idea-refine`**: Mài giũa ý tưởng đề tài nghiên cứu, đồ án tốt nghiệp, bài toán Hackathon; gợi ý dataset mở và kiến trúc khả thi.
- **`interview-me`**: Phỏng vấn làm rõ yêu cầu dữ liệu, nhãn mất cân bằng, metric mục tiêu, môi trường huấn luyện (Colab / GPU local).
- **`spec-driven-development`**: Thiết lập đặc tả kỹ thuật ML: Baseline model, hàm mục tiêu (Loss function), Metric đo lường (F1, AUC, RMSE) và Schema dữ liệu.

### 2. Giai đoạn Kế hoạch & Kiến trúc Pipeline (Plan)
- **`planning-and-task-breakdown`**: Phân rã pipeline thành các module nguyên tử: Ingestion -> Preprocessing -> Feature Engineering -> Train -> Eval -> Serving.

### 3. Giai đoạn Xây dựng chuẩn mực (Build & Test)
- **`test-driven-development`**: TDD chuyên biệt cho Data & AI: Test schema dữ liệu (Pandera), test chiều Tensor (PyTorch), test rò rỉ dữ liệu (Data Leakage) và sanity test overfit 1 batch.
- **`source-driven-development`**: Buộc AI tra cứu và trích dẫn tài liệu chính thức từ PyTorch, HuggingFace, Scikit-learn; triệt tiêu hoàn toàn mã lỗi thời hoặc tham số bịa đặt.

### 4. Giai đoạn Gỡ lỗi & Kiểm chứng đối kháng (Verify & Doubt)
- **`debugging-and-error-recovery`**: Gỡ lỗi chuyên sâu Deep Learning: Xử lý tràn bộ nhớ GPU (`CUDA Out of Memory`), Loss ra `NaN/Inf`, đạo hàm bị triệt tiêu/bùng nổ, lệch chiều Tensor.
- **`doubt-driven-development`**: Cơ chế phản biện: Khi mô hình đạt Accuracy quá cao (99.9%), ép AI phải nghi ngờ rò rỉ dữ liệu, kiểm tra Overfitting và phân phối tập Test.

### 5. Giai đoạn Tối ưu & Triển khai (Optimize & Serve)
- **`performance-optimization`**: Tối ưu tốc độ huấn luyện: Vector hóa (NumPy), tối ưu DataLoader GPU (`num_workers`, `pin_memory`), huấn luyện độ chính xác hỗn hợp (Mixed Precision FP16/AMP).
- **`api-and-interface-design`**: Đóng gói mô hình thành dịch vụ API suy luận (Inference Service) với FastAPI / TorchServe, chuẩn hóa Schema Request/Response.

### 6. Giai đoạn Dọn dẹp & Nghiệm thu (Review & Ship)
- **`code-simplification`**: Tinh gọn và làm sạch mã nguồn: Biến các đoạn code nháp lộn xộn trong Jupyter Notebook thành module Python sạch, phẳng logic, dễ bảo trì.
- **`documentation-and-adrs`**: Tự động viết Model Card, Báo cáo thí nghiệm, phân tích Confusion Matrix, biểu đồ trực quan và hướng dẫn nạp checkpoint model.
- **`walkthrough-and-experiment-tracking`**: Ghi nhận và theo dõi nhật ký thực nghiệm tập trung theo từng Run trong file walkthrough duy nhất.

---

## Cách sử dụng

1. **Sử dụng trong Antigravity / IDE:** Đặt file `AGENTS.md` vào thư mục gốc dự án của bạn để AI tự động tuân thủ.
2. **Kích hoạt tự nhiên:** Bạn chỉ cần nói tự nhiên bằng tiếng Việt (ví dụ: *"Lên spec cho bài toán phân loại ảnh này"*, *"Kiểm tra xem có bị leak data không"*), AI sẽ tự động kích hoạt skill tương ứng.
