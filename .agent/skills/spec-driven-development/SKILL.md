---
name: spec-driven-development
description: Soạn đặc tả kỹ thuật ML/DL trước khi triển khai code. Tự động liên kết và trích xuất chuẩn từ workflows/ML_flow.md (cho Tabular ML) hoặc dl_workflows/DL_flow.md (cho Deep Learning). Ánh xạ trực tiếp vào Section 1 của walkthrough_lab_0X.md.
---

# spec-driven-development (Bản DS & AI)

## Tổng quan

Skill này bắt buộc AI Agent phải soạn một **Đặc tả Machine Learning / Deep Learning (ML/DL Spec)** toàn diện trước khi viết code. Đặc tả này neo dự án vào các hợp đồng dữ liệu (data contract) chuẩn hóa, công thức loss khách quan, kiến trúc baseline và tiêu chí đánh giá định lượng.

Nguyên lý đằng sau là: code là hệ quả của đặc tả, không phải ngược lại. Khi đặc tả rõ ràng, việc triển khai chỉ còn là công việc cơ học; khi đặc tả mơ hồ, mọi quyết định code đều trở thành phỏng đoán.

## Tham chiếu workflow

Khi viết đặc tả, agent **BẮT BUỘC** phải trích xuất chuẩn từ playbook workflow tương ứng:
- **Tabular ML:** Tham chiếu [`workflows/ML_flow.md`](../../workflows/ML_flow.md) và [`workflows/eda_guide.md`](../../workflows/eda_guide.md) cho hướng dẫn EDA, chuẩn chuẩn hóa (normalization), mã hóa biến phân loại (categorical encoding) và các benchmark baseline.
- **Deep Learning (CV / NLP):** Tham chiếu [`dl_workflows/DL_flow.md`](../../dl_workflows/DL_flow.md) cho modality đầu vào, chiều tensor, lựa chọn backbone và các hàm loss chuyên biệt (ví dụ Focal Loss, Label Smoothing).

## Kiến trúc ML/DL Spec chuẩn (tích hợp vào Section 1 của `walkthrough_lab_0X.md`)

Đặc tả được soạn trực tiếp trong **Section 1 của tài liệu sống duy nhất `walkthrough_lab_0X.md`** trước khi viết code (chịu sự điều phối của Rule 1 trong Kiến trúc 3 Rule).

Nghiêm cấm tạo file `SPEC.md` riêng biệt. Mọi định hình bài toán và kế hoạch baseline đều được tích hợp trực tiếp vào Section 1 của `walkthrough_lab_0X.md`.

### 1. Định hình bài toán và Modality
- Loại tác vụ: Classification (Binary/Multiclass), Regression, Segmentation, Object Detection, Sequence Modeling.
- Modality dữ liệu đầu vào: Tabular DataFrame, tensor ảnh 2D `[B, C, H, W]`, chuỗi token văn bản `[B, Seq_Len]`.

### 2. Hợp đồng dữ liệu và Chiến lược chia tập
- Schema dữ liệu, kiểu dữ liệu, và tỷ lệ cân bằng lớp.
- Chiến lược chia tập không rò rỉ dữ liệu:
  - Mẫu Tabular/Image độc lập -> `StratifiedKFold` (bảo toàn tỷ lệ lớp).
  - Dữ liệu chuỗi thời gian -> `TimeSeriesSplit` (nghiêm ngặt theo thứ tự thời gian, không bao giờ shuffle).
  - Thực thể có nhóm (grouped entities) -> `GroupKFold` (ngăn chặn chồng lấn thực thể giữa train và val).
- Lựa chọn sai chiến lược chia tập là nguyên nhân rò rỉ dữ liệu phổ biến nhất, nên bước này cần đặc biệt cẩn trọng.

### 3. Mô hình Baseline và Metric mục tiêu
- **Mô hình Baseline:** Benchmark tối thiểu khả thi để neo tiến bộ so sánh (ví dụ Heuristic, Logistic Regression, hoặc MLP 2 lớp đơn giản).
- **Metric chính:** Macro F1-Score, ROC-AUC, mAP, IoU, RMSE.
- **Ngưỡng chấp nhận (Acceptance Threshold):** Biên định lượng tối thiểu mà mô hình nâng cao phải đạt được so với baseline để được coi là chấp nhận.

### 4. Hàm Loss và Thiết kế tối ưu
- Lựa chọn hàm loss:
  - Phân loại đa lớp chuẩn -> `nn.CrossEntropyLoss(label_smoothing=0.1)`.
  - Mất cân bằng lớp nghiêm trọng (> 1:10) -> `FocalLoss` hoặc `class_weights` có trọng số.
- Optimizer (AdamW) và Learning Rate Scheduler (CosineAnnealingLR hoặc OneCycleLR).

### 5. Ràng buộc môi trường thực thi
- Mục tiêu compute: CPU/GPU cục bộ so với **Kaggle GPU (Tesla T4)**.
- Tương thích hệ thống file: Xử lý `/kaggle/input/` (chỉ đọc) và `/kaggle/working/` (ghi được).
- Ràng buộc bộ nhớ: Áp đặt ngân sách VRAM (< 15GB, bật AMP FP16).

## Chống biện minh (Anti-Rationalization)

- [X] *"Bài lab này ngắn, code luôn cho nhanh"* -> **Bác bỏ:** Triển khai không có spec sẽ đánh mất benchmark baseline, có nguy cơ chọn sai metric, và làm giảm độ chặt chẽ của báo cáo cuối.
