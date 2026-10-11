# Framework Quy Trình DS & AI

Bộ quy trình kỹ thuật chuẩn hóa và năng lực chuyên môn được thiết kế riêng cho các AI Coding Agent khi phát triển dự án Data Science, Machine Learning và Deep Learning.

---

## 1. Tổng quan vòng đời phát triển DS & AI

Khác với kỹ thuật phần mềm truyền thống, một dự án DS & AI đòi hỏi các giao thức nghiêm ngặt trải dài trên nhiều lĩnh vực: hợp đồng dữ liệu (data contract), kiểm toán rò rỉ dữ liệu (leakage audit), quản lý tài nguyên GPU, và đóng gói mô hình (model packaging). Framework này bao gồm 14 skill lõi trải đều trên 6 giai đoạn:

```text
 KHỞI TẠO  -->  ĐẶC TẢ & KẾ HOẠCH  -->  XÂY DỰNG & KIỂM THỬ  -->  GỠ LỖI & HOÀI NGHI  -->  TỐI ƯU & TRIỂN KHAI  -->  RÀ SOÁT & BÁO CÁO
```

Diễn giải từng giai đoạn:

| Giai đoạn | Mục tiêu chính | Skill phụ trách |
|---|---|---|
| 1. Khởi tạo | Làm rõ ý tưởng, phạm vi, dữ liệu và ràng buộc tính toán | `idea-refine`, `interview-me` |
| 2. Đặc tả & Kế hoạch | Chốt spec kỹ thuật, baseline, metric và phân rã công việc | `spec-driven-development`, `planning-and-task-breakdown` |
| 3. Xây dựng & Kiểm thử | Viết code module, viết test, xác thực API thư viện | `test-driven-development`, `source-driven-development` |
| 4. Gỡ lỗi & Hoài nghi | Xử lý OOM/NaN, kiểm toán rò rỉ dữ liệu, chống ảo tưởng kết quả | `debugging-and-error-recovery`, `doubt-driven-development` |
| 5. Tối ưu & Triển khai | Tăng throughput, giảm VRAM, đóng gói inference API | `performance-optimization`, `api-and-interface-design` |
| 6. Rà soát & Báo cáo | Refactor code, ghi vết thí nghiệm, viết báo cáo cuối | `code-simplification`, `walkthrough-and-experiment-tracking`, `documentation-and-adrs` |

Ngoài ra, skill `using-ds-ai-skills` đóng vai trò **nhạc trưởng (orchestrator)** đứng trên tất cả, phân loại bài toán và điều phối luồng thực thi xuyên suốt vòng đời.

---

## 2. Danh mục 14 Skill chuyên biệt cho DS & AI

### 0. Tầng điều phối (Meta)

- **`using-ds-ai-skills`**: Nhạc trưởng tổng thể, điều phối toàn bộ vòng đời, nhận diện loại bài toán (Tabular ML hay Deep Learning) và định tuyến công việc tới đúng skill cùng playbook tương ứng trong `workflows/` và `dl_workflows/`.

### 1. Khởi tạo và Xác định phạm vi (Define)

- **`idea-refine`**: Tinh chỉnh ý tưởng nghiên cứu, đề tài đồ án môn học, bài toán hackathon; xác định bộ dữ liệu công khai khả thi và kiến trúc mô hình phù hợp với năng lực tính toán hiện có.
- **`interview-me`**: Phỏng vấn có cấu trúc để làm rõ yêu cầu, tỷ lệ mất cân bằng lớp (class imbalance), metric đánh giá mục tiêu và ràng buộc hạ tầng tính toán trước khi viết bất kỳ dòng code nào.
- **`spec-driven-development`**: Soạn đặc tả kỹ thuật: mô hình baseline, hàm loss, metric mục tiêu (F1, AUC, RMSE) và hợp đồng dữ liệu (data contract). Ánh xạ trực tiếp vào **Section 1** của `walkthrough_lab_0X.md`.

### 2. Lập kế hoạch và Kiến trúc pipeline (Plan)

- **`planning-and-task-breakdown`**: Phân rã pipeline thành các module nguyên tử (atomic module): Ingestion -> Preprocessing -> Feature Engineering -> Train -> Eval -> Serving, kèm thứ tự phụ thuộc và tiêu chí nghiệm thu rõ ràng.

### 3. Triển khai theo chuẩn mực (Build & Test)

- **`test-driven-development`**: TDD chuyên biệt cho AI/DL: xác thực schema dữ liệu, kiểm tra chiều tensor (tensor dimension assertion), test chống rò rỉ dữ liệu, và bài kiểm tra 1-batch sanity overfit chạy cục bộ.
- **`source-driven-development`**: Neo mọi quyết định triển khai vào tài liệu chính thức (PyTorch, Hugging Face, Scikit-learn), loại bỏ các pattern đã bị deprecated và các keyword argument bị "ảo tưởng" (hallucinated).

### 4. Chẩn đoán và Kiểm toán đối kháng (Verify & Doubt)

- **`debugging-and-error-recovery`**: Khắc phục sự cố chuyên sâu trong huấn luyện Deep Learning: xử lý CUDA Out-of-Memory, loss bị NaN/Inf, gradient bùng nổ/tiêu biến, và các lỗi đặc thù trên môi trường Kaggle.
- **`doubt-driven-development`**: Kiểm toán đối kháng: khi mô hình đạt độ chính xác cao bất thường (> 98%), chủ động truy tìm rò rỉ nhãn (label leakage), nhiễm bẩn tập dữ liệu (split contamination) và overfitting tiềm ẩn.

### 5. Tối ưu và Triển khai (Optimize & Serve)

- **`performance-optimization`**: Tối đa hóa throughput: vector hóa bằng NumPy, tinh chỉnh GPU DataLoader (`num_workers`, `pin_memory`), và Automatic Mixed Precision (AMP FP16).
- **`api-and-interface-design`**: Đóng gói mô hình đã huấn luyện thành dịch vụ inference production bằng FastAPI hoặc TorchServe với schema request/response chặt chẽ.

### 6. Chất lượng code, Ghi vết và Nộp bài (Review & Ship)

- **`code-simplification`**: Refactor code notebook khám phá (exploratory) thành gói Python module sạch, dễ bảo trì trong `src/`.
- **`walkthrough-and-experiment-tracking`**: Ghi vết toàn bộ dòng chảy thí nghiệm của từng vòng lặp (hyperparameter, thay đổi kiến trúc, chênh lệch định lượng) trực tiếp vào **Section 2** của `walkthrough_lab_0X.md`.
- **`documentation-and-adrs`**: Tổng hợp model card, chẩn đoán confusion matrix, phân tích lỗi (error analysis) và kết luận sẵn sàng nộp bài, đặt trực tiếp trong **Section 3** của `walkthrough_lab_0X.md`.

---

## 3. Kiến trúc 3 Rule và tài liệu sống duy nhất

Một điểm đặc trưng của framework này là **tuyệt đối không phân mảnh tài liệu**. Mọi lab chỉ có đúng MỘT tài liệu sống duy nhất: `walkthrough_lab_0X.md`, được liên kết chặt chẽ với 3 rule cốt lõi:

| Rule | Tên rule | Vị trí trong `walkthrough_lab_0X.md` | Nội dung |
|---|---|---|---|
| Rule 1 | `spec-driven-development` | **Section 1: Spec & Baseline Setup** | Problem statement, data contract, baseline model, ngưỡng chấp nhận |
| Rule 2 | `walkthrough-and-experiment-tracking` | **Section 2: Full-Flow Experiment Changelog** | Dòng chảy vòng lặp, chênh lệch qua 5 giai đoạn pipeline, lý do điều chỉnh hyperparameter |
| Rule 3 | `documentation-and-adrs` | **Section 3: Results & Final Technical Report** | Ma trận so sánh, chẩn đoán confusion matrix, phân tích lỗi, kết luận |

Nghiêm cấm tạo file `SPEC.md` hoặc `REPORT.md` riêng biệt. Cả 3 thành phần phải được hợp nhất vào đúng tài liệu sống.

---

## 4. Cách sử dụng

1. **Tích hợp Antigravity / IDE:** Đặt file `AGENTS.md` ở thư mục gốc của project để áp đặt rule lên mọi tương tác của agent.
2. **Kích hoạt theo ngữ cảnh:** Giao tiếp tự nhiên bằng tiếng Việt hoặc tiếng Anh; agent sẽ tự động định tuyến ý định (intent) tới skill kỹ thuật và tài liệu workflow tương ứng.
3. **Đọc theo lớp:** Bắt đầu từ `README.md` (tổng quan) -> `AGENTS.md` (hiến pháp) -> file `SKILL.md` cụ thể của từng skill khi cần triển khai chi tiết.
