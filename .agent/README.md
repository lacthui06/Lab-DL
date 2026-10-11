# Framework Quy Trình DS & AI

Bộ quy trình kỹ thuật chuẩn hóa và năng lực chuyên môn được thiết kế riêng cho các AI Coding Agent khi phát triển dự án Data Science, Machine Learning và Deep Learning.

---

## 1. Tổng quan vòng đời phát triển DS & AI

Khác với kỹ thuật phần mềm truyền thống, một dự án DS & AI đòi hỏi các giao thức nghiêm ngặt trải dài trên nhiều lĩnh vực: hợp đồng dữ liệu (data contract), kiểm toán rò rỉ dữ liệu (leakage audit), quản lý tài nguyên GPU, và đóng gói mô hình (model packaging). Framework này bao gồm 17 skill lõi trải đều trên 7 giai đoạn:

```text
 KHỞI TẠO  -->  ĐẶC TẢ & KẾ HOẠCH  -->  XÂY DỰNG & KIỂM THỬ  -->  GỠ LỖI & HOÀI NGHI  -->  TỐI ƯU & TRIỂN KHAI  -->  THỰC THI TỪ XA  -->  RÀ SOÁT & BÁO CÁO
```

Diễn giải từng giai đoạn:

| Giai đoạn | Mục tiêu chính | Skill phụ trách |
|---|---|---|
| 1. Khởi tạo | Làm rõ ý tưởng, phạm vi, dữ liệu và ràng buộc tính toán | `idea-refine`, `interview-me` |
| 2. Đặc tả & Kế hoạch | Chốt spec kỹ thuật, baseline, metric và phân rã công việc | `spec-driven-development`, `planning-and-task-breakdown` |
| 3. Xây dựng & Kiểm thử | Viết code module, viết test, xác thực API, cố định môi trường | `test-driven-development`, `source-driven-development`, `environment-and-reproducibility` |
| 4. Gỡ lỗi & Hoài nghi | Xử lý OOM/NaN, kiểm toán rò rỉ dữ liệu, chống ảo tưởng kết quả | `debugging-and-error-recovery`, `doubt-driven-development` |
| 5. Tối ưu & Triển khai | Tăng throughput, giảm VRAM, đóng gói inference API | `performance-optimization`, `api-and-interface-design` |
| 6. Thực thi từ xa | Chạy trên Kaggle GPU, đồng bộ artifact, thu hồi notebook | `kaggle-remote-execution` |
| 7. Rà soát & Báo cáo | Refactor code, quản lý Git, ghi vết thí nghiệm, viết báo cáo cuối | `code-simplification`, `version-control-and-artifacts`, `walkthrough-and-experiment-tracking`, `documentation-and-adrs` |

Ngoài ra, skill `using-ds-ai-skills` đóng vai trò **nhạc trưởng (orchestrator)** đứng trên tất cả, phân loại bài toán và điều phối luồng thực thi xuyên suốt vòng đời.

---

## 2. Danh mục 17 Skill chuyên biệt cho DS & AI

### 0. Tầng điều phối (Meta)

- **`using-ds-ai-skills`**: Nhạc trưởng tổng thể, điều phối toàn bộ vòng đời, nhận diện loại bài toán (Computer Vision, NLP, Audio) và định tuyến công việc tới đúng skill cùng playbook tương ứng trong `docs/`.

### 1. Khởi tạo và Xác định phạm vi (Define)

- **`idea-refine`**: Tinh chỉnh ý tưởng nghiên cứu, đề tài đồ án môn học, bài toán hackathon; xác định bộ dữ liệu công khai khả thi và kiến trúc mô hình phù hợp với năng lực tính toán hiện có.
- **`interview-me`**: Phỏng vấn có cấu trúc để làm rõ yêu cầu, tỷ lệ mất cân bằng lớp, metric đánh giá mục tiêu và ràng buộc hạ tầng tính toán trước khi viết bất kỳ dòng code nào.
- **`spec-driven-development`**: Soạn đặc tả kỹ thuật: mô hình baseline, hàm loss, metric mục tiêu và hợp đồng dữ liệu. Ánh xạ trực tiếp vào **Section 1** của `walkthrough_lab_0X.md`.

### 2. Lập kế hoạch và Kiến trúc pipeline (Plan)

- **`planning-and-task-breakdown`**: Phân rã pipeline thành các module nguyên tử (Ingestion -> Preprocessing -> Feature Engineering -> Train -> Eval -> Serving), kèm thứ tự phụ thuộc và tiêu chí nghiệm thu rõ ràng.

### 3. Triển khai theo chuẩn mực (Build & Test)

- **`test-driven-development`**: TDD chuyên biệt cho AI/DL: xác thực schema dữ liệu, kiểm tra chiều tensor, test chống rò rỉ dữ liệu, và bài kiểm tra 1-batch sanity overfit chạy cục bộ.
- **`source-driven-development`**: Neo mọi quyết định triển khai vào tài liệu chính thức (PyTorch, Hugging Face, Scikit-learn), loại bỏ pattern deprecated và keyword argument ảo tưởng.
- **`environment-and-reproducibility`**: Quản lý tính tái lập: cố định seed, bật chế độ deterministic, ghi nhận phiên bản thư viện, và đồng bộ môi trường giữa local và Kaggle.

### 4. Chẩn đoán và Kiểm toán đối kháng (Verify & Doubt)

- **`debugging-and-error-recovery`**: Khắc phục sự cố chuyên sâu: xử lý CUDA Out-of-Memory, loss NaN/Inf, gradient bùng nổ/tiêu biến, và các lỗi đặc thù trên Kaggle.
- **`doubt-driven-development`**: Kiểm toán đối kháng: khi mô hình đạt độ chính xác cao bất thường, chủ động truy tìm rò rỉ nhãn, nhiễm bẩn tập dữ liệu và overfitting tiềm ẩn.

### 5. Tối ưu và Triển khai (Optimize & Serve)

- **`performance-optimization`**: Tối đa hóa throughput: vector hóa bằng NumPy, tinh chỉnh GPU DataLoader, và Automatic Mixed Precision (AMP FP16).
- **`api-and-interface-design`**: Đóng gói mô hình đã huấn luyện thành dịch vụ inference production bằng FastAPI với schema request/response chặt chẽ.

### 6. Thực thi từ xa (Remote Execution)

- **`kaggle-remote-execution`**: Quản lý huấn luyện trên Kaggle GPU: cấu hình đường dẫn đúng chuẩn, quản lý secret an toàn, đồng bộ artifact về cục bộ, và thu hồi notebook đã chạy có output.

### 7. Chất lượng, Ghi vết và Nộp bài (Review & Ship)

- **`code-simplification`**: Refactor code notebook khám phá thành gói Python module sạch, dễ bảo trì trong `src/`.
- **`version-control-and-artifacts`**: Quản lý phiên bản và artifact: quy ước commit, kỷ luật `.gitignore`, phân loại file thuộc Git và không thuộc Git, bảo vệ secret.
- **`walkthrough-and-experiment-tracking`**: Ghi vết toàn bộ dòng chảy thí nghiệm của từng vòng lặp trực tiếp vào **Section 2** của `walkthrough_lab_0X.md`.
- **`documentation-and-adrs`**: Tổng hợp model card, chẩn đoán confusion matrix, phân tích lỗi và kết luận sẵn sàng nộp bài trong **Section 3** của `walkthrough_lab_0X.md`.

---

## 3. Cấu trúc thư mục của framework

```text
.agent/
    README.md              <-- File này: tổng quan framework
    AGENTS.md              <-- Hiến pháp vận hành cho AI Agent
    skills/
        INDEX.md           <-- Manifest định tuyến nhanh 17 skill
        <tên-skill>/SKILL.md
    templates/             <-- Khung mẫu tái sử dụng
    checklists/            <-- Danh sách kiểm tra bắt buộc (các cổng)
```

- **`skills/INDEX.md`**: Bảng định tuyến theo ý định, theo triệu chứng, và theo artifact cần tạo. Agent đọc file này trước để chọn đúng skill mà không cần quét toàn bộ thư mục.
- **`templates/`**: Khung mẫu để agent copy thay vì tự phát minh lại cấu trúc cho mỗi lab mới.
- **`checklists/`**: Danh sách tick-box cho các cổng quan trọng, biến quy tắc "nên làm" thành bước "phải đủ mới đi tiếp".

### Khung mẫu trong `templates/`

| Template | Mục đích |
|---|---|
| `SKILL_TEMPLATE.md` | Khung chuẩn 9 mục để soạn một file SKILL.md |
| `walkthrough_template.md` | Khung 3 Section cho tài liệu sống của lab mới |
| `description_lab_template.md` | Khung ghi đề bài gốc của giảng viên cho từng lab |
| `lab_readme_template.md` | Khung README cho từng lab |

### Danh sách kiểm tra trong `checklists/`

| Checklist | Cổng kiểm tra |
|---|---|
| `new_lab_bootstrap.md` | Trước khi bắt đầu lab mới |
| `pre_commit.md` | Trước khi commit và push lên Git |
| `pre_kaggle_upload.md` | Trước khi đẩy lên Kaggle GPU |
| `pre_submission.md` | Trước khi nộp bài |

---

## 4. Kiến trúc 3 Rule và tài liệu sống duy nhất

Một điểm đặc trưng của framework này là **tuyệt đối không phân mảnh tài liệu**. Mọi lab chỉ có đúng MỘT tài liệu sống duy nhất: `walkthrough_lab_0X.md`, được liên kết chặt chẽ với 3 rule cốt lõi:

| Rule | Tên rule | Vị trí trong `walkthrough_lab_0X.md` | Nội dung |
|---|---|---|---|
| Rule 1 | `spec-driven-development` | **Section 1: Spec & Baseline Setup** | Problem statement, data contract, baseline model, ngưỡng chấp nhận |
| Rule 2 | `walkthrough-and-experiment-tracking` | **Section 2: Full-Flow Experiment Changelog** | Dòng chảy vòng lặp, chênh lệch qua 5 giai đoạn pipeline, lý do điều chỉnh hyperparameter |
| Rule 3 | `documentation-and-adrs` | **Section 3: Results & Final Technical Report** | Ma trận so sánh, chẩn đoán confusion matrix, phân tích lỗi, kết luận |

Nghiêm cấm tạo file `SPEC.md` hoặc `REPORT.md` riêng biệt. Cả 3 thành phần phải được hợp nhất vào đúng tài liệu sống.

---

## 5. Cách sử dụng

1. **Tích hợp Antigravity / IDE:** Đặt file `AGENTS.md` ở thư mục gốc của project để áp đặt rule lên mọi tương tác của agent.
2. **Định tuyến nhanh:** Đọc `skills/INDEX.md` để tra cứu skill theo ý định hoặc theo triệu chứng.
3. **Kích hoạt theo ngữ cảnh:** Giao tiếp tự nhiên bằng tiếng Việt hoặc tiếng Anh; agent sẽ tự động định tuyến ý định tới skill kỹ thuật và tài liệu workflow tương ứng.
4. **Tuân thủ các cổng kiểm tra:** Chạy checklist tương ứng trước mỗi cổng quan trọng (lab mới, commit, Kaggle, nộp bài).
5. **Đọc theo lớp:** Bắt đầu từ `README.md` (tổng quan) -> `AGENTS.md` (hiến pháp) -> `skills/INDEX.md` (định tuyến) -> file `SKILL.md` cụ thể khi cần triển khai chi tiết.
