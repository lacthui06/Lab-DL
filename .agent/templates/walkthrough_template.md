# Walkthrough & Full-Flow Technical Report — Lab 0X

> **Lab Title:** Lab 0X — <tên bài lab>
> **Status:** [ ] In Progress | [ ] Completed
> **Primary Objective:** <một câu mô tả mục tiêu chính và kết quả đạt được>
> **Workflow Playbook References:** `docs/dl_flow.md` | `docs/dl_training.md`
> **Unified 3-Rule Standard (Single Living Walkthrough Architecture):**
> - **Rule 1 (`spec-driven-development`):** Ánh xạ trực tiếp vào **Section 1 — Technical Specification & Flow Design**.
> - **Rule 2 (`walkthrough-and-experiment-tracking`):** Ánh xạ trực tiếp vào **Section 2 — Full-Flow Experiment Lineage & Changelog**.
> - **Rule 3 (`documentation-and-adrs`):** Ánh xạ trực tiếp vào **Section 3 — Comparative Results, Error Diagnostics & Final Submission Report**.

---

## 1. TECHNICAL SPECIFICATION & FLOW DESIGN (Rule: spec-driven-development)

### 1.0. Brief Interpretation and Clarifications
Bản tóm tắt cách hiểu đề bài gốc trong `description_lab_0X.md`, kèm các điểm đã được làm rõ và hướng giải quyết đã chốt. Mục này là vết truy vết cho mọi quyết định đặc tả phía sau.

- **Cách hiểu đề bài:** <tóm tắt mục tiêu, phạm vi, ràng buộc, sản phẩm cần nộp>
- **Điểm đã làm rõ:** <mỗi dòng nêu điểm mơ hồ và hướng giải quyết đã thống nhất>
- **Giả định còn lại:** <giả định chưa được xác nhận, nếu có>

### 1.1. Master Flow Design
Sơ đồ luồng end-to-end của pipeline, từ nạp dữ liệu tới quyết định chọn mô hình cuối.

### 1.2. Problem Formulation and Modality
- **Modality:** <Computer Vision / NLP / Tabular / Audio>
- **Task Formulation:** <loại bài toán và danh sách lớp mục tiêu nếu có>
- **Input Tensor Contract:** <shape và dtype của tensor đầu vào>
- **Output Contract:** <shape và ý nghĩa của đầu ra>

### 1.3. Data Contract and Partitioning Strategy
- **Raw Volume:** <số lượng mẫu, độ phân giải>
- **Class Distribution:** <cân bằng hay mất cân bằng, tỷ lệ cụ thể>
- **Partitioning Strategy:** <tỷ lệ chia Train/Val/Test, phương pháp chia, seed>
- **Feature Normalization:** <các bước chuẩn hóa và giá trị thống kê dùng để chuẩn hóa>

### 1.4. Baseline Model and Target Metrics
- **Baseline Model:** <kiến trúc baseline tối thiểu>
- **Primary Metric:** <metric chính>
- **Acceptance Threshold:** <ngưỡng định lượng tối thiểu để coi là đạt>

### 1.5. Loss Function and Optimization Design
- **Loss Function:** <hàm loss và lý do chọn>
- **Optimizer:** <optimizer và learning rate>
- **Scheduler:** <lịch điều chỉnh learning rate>

### 1.6. Execution Environment Constraints
- **Compute Target:** <local CPU/GPU hoặc Kaggle GPU Tesla T4>
- **Filesystem Compatibility:** <đường dẫn input chỉ đọc và working ghi được>
- **Memory Constraints:** <ngân sách VRAM và cấu hình AMP>

---

## 2. FULL-FLOW EXPERIMENT LINEAGE & CHANGELOG (Rule: walkthrough-and-experiment-tracking)

### 2.1. Flow Comparison Matrix

| Run | Data | Preprocessing & Augmentation | Model | Training Strategy | Test Loss | Test Acc | Macro F1 | Best Checkpoint |
|---|---|---|---|---|---|---|---|---|
| Run #1 Baseline | | | | | | | | |
| Run #2 Tuned | | | | | | | | |
| Run #3 Final | | | | | | | | |

Với bài toán phân loại, ngoài bảng trên phải có thêm bảng metric theo từng lớp (bắt buộc cho mọi run, không chỉ Accuracy):

| Run | Class | Precision | Recall (TPR) | F1 | FPR |
|---|---|---|---|---|---|
| | | | | | |

Ghi chú: mọi metric theo lớp ghi kèm tên lớp; chênh lệch giữa các run ghi dạng delta tuyệt đối; metric chưa có trong artifact ghi "not logged", không bịa số.

### 2.2. Changelog per Run

Với mỗi Run, trả lời đủ 3 câu hỏi bắt buộc:
- **Thay đổi gì?** <giai đoạn pipeline nào đã thay đổi so với run trước>
- **Vì sao?** <lý do kỹ thuật>
- **Tác động định lượng?** <chênh lệch metric cụ thể>

#### Run #1 Baseline
- Thay đổi gì:
- Vì sao:
- Tác động định lượng:

#### Run #2 Tuned
- Thay đổi gì:
- Vì sao:
- Tác động định lượng:

#### Run #3 Final
- Thay đổi gì:
- Vì sao:
- Tác động định lượng:

### 2.3. Artifact Harvesting Log
- **Checkpoint tốt nhất:** `outputs/checkpoints/best_model.pt`
- **Hình dự đoán:** `outputs/predictions_*.png`
- **Confusion matrix:** `outputs/confusion_matrix_*.png`
- **Đường cong huấn luyện:** `outputs/run*_curves.png`
- **Bảng kết quả tổng hợp:** `outputs/experiment_results.json`

Mỗi hình trong log phải kèm một dòng nhận xét đọc trực tiếp từ hình (nội dung thực tế của hình), không chỉ ghi đường dẫn:
- **Đường cong:** hội tụ ở epoch nào, val loss có tăng lại không (overfitting), khoảng cách train/val.
- **Confusion matrix:** cặp lớp nhầm lẫn nhiều nhất và tỷ lệ nhầm cụ thể.
- **Lưới dự đoán:** các mẫu dự đoán sai kèm nhãn thật, nhãn dự đoán, mức tự tin.
- **TPR/FPR:** lớp có TPR thấp nhất và lớp có FPR cao nhất.

---

## 3. COMPARATIVE RESULTS, ERROR DIAGNOSTICS & FINAL SUBMISSION REPORT (Rule: documentation-and-adrs)

### 3.1. Quantitative Evaluation & Comparison Matrix
Bảng so sánh trực tiếp giữa baseline và các vòng lặp nâng cao, kèm phần trăm cải thiện.

Với bài toán phân loại, bảng định lượng phải phủ đủ bộ metric (không chỉ Accuracy) và kèm bảng metric theo từng lớp:

| Run | Test Loss | Test Acc | Macro F1 | Precision (macro) | Recall (macro) |
|---|---|---|---|---|---|
| | | | | | |

| Run | Class | Precision | Recall (TPR) | F1 | FPR |
|---|---|---|---|---|---|
| | | | | | |

Mọi chênh lệch ghi dạng delta tuyệt đối; metric chưa có trong artifact ghi "not logged" kèm lý do, không bịa số.

### 3.2. Confusion Matrix Diagnostics
Phân tích sâu các cặp lớp nhầm lẫn cao và giải thích nguyên nhân trực quan. Phải mở ảnh confusion matrix và đọc giá trị từng ô: nêu tên cặp lớp cụ thể và tỷ lệ nhầm (ví dụ "hàng Shirt có 18% bị dự đoán thành Pullover"). Kết hợp biểu đồ TPR/FPR để chỉ ra lớp có TPR thấp nhất và lớp có FPR cao nhất.

### 3.3. Error Analysis Slices
Kiểm tra trực quan và thống kê các mẫu bị phân loại sai, xác định nguyên nhân gốc. Phải mở ảnh lưới dự đoán và chỉ ra các mẫu cụ thể bị dự đoán sai kèm nhãn thật, nhãn dự đoán, mức tự tin; nhóm lỗi thành lát cắt và nêu lát cắt lỗi nhiều nhất.

### 3.4. Key Technical Findings
Giải thích các đột phá về hiệu năng, đánh giá tác động của regularization, scheduler và augmentation.

### 3.5. Checkpoint Verification (Save/Load)
Xác nhận việc nạp lại checkpoint tốt nhất tái lập 100% metric inference.

### 3.6. Decision Traceability
Bảng truy vết mỗi quyết định kỹ thuật quan trọng về điểm thảo luận tương ứng ở Mục 1.0, kèm hướng giải quyết đã chốt. Dùng văn phong kỹ thuật trung tính, ngôi thứ ba; không dùng lối kể chuyện hay từ chỉ người như "bạn", "tôi", "chúng ta".

| Quyết định kỹ thuật | Điểm thảo luận (Mục 1.0) | Hướng giải quyết đã chốt |
|---|---|---|
| | | |

### 3.7. Conclusion & Lessons Learned
Kết luận cuối cùng và các bài học rút ra cho lab tiếp theo.
