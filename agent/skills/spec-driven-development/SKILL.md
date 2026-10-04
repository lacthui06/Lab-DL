---
name: spec-driven-development
description: Viết tài liệu đặc tả kỹ thuật (ML/DL Spec) cho bài toán Data Science & Deep Learning trước khi viết mã nguồn. Tự động liên kết và trích xuất chuẩn mực từ workflows/ML_flow.md (cho Tabular ML) hoặc dl_workflows/DL_flow.md (cho Deep Learning).
---

# spec-driven-development (DS & AI Edition)

## Tổng quan
Kỹ năng này bắt buộc AI Agent phải xây dựng một bản **Đặc tả Kỹ thuật Học máy / Học sâu (ML/DL Spec)** chi tiết trước khi lập trình. Kỹ năng tự động liên kết với các tài liệu quy trình chuyên sâu trong dự án để thiết lập đúng tiêu chuẩn dữ liệu, hàm mất mát và chỉ số đánh giá.

## Liên kết Quy trình Nghiệp vụ (Flow References)
Khi lập Spec, AI **BẮT BUỘC** phải mở và trích xuất chuẩn mực từ tài liệu nghiệp vụ tương ứng:
- **Bài toán Dữ liệu bảng (Tabular ML):** Tham chiếu [**`workflows/ML_flow.md`**](../../workflows/ML_flow.md) và [**`workflows/eda_guide.md`**](../../workflows/eda_guide.md) để chốt chuẩn EDA, chuẩn hóa số, mã hóa biến phân loại và mô hình cơ sở.
- **Bài toán Học sâu (Deep Learning / CV / NLP):** Tham chiếu [**`dl_workflows/DL_flow.md`**](../../dl_workflows/DL_flow.md) để chốt phương thái dữ liệu (Modality), kích thước ảnh/độ dài câu, kiến trúc Backbone và hàm mất mát chuyên biệt (Focal Loss, Label Smoothing).

---

## Cấu trúc chuẩn của bản ML/DL Spec (Ghi vào Mục 1 của walkthrough_lab_0X.md)

Bản đặc tả được khởi tạo ngay trong **Mục 1 của file duy nhất `walkthrough_lab_0X.md`** trước khi code, bao gồm 5 phần sau:

### 1. Phân loại Bài toán & Phương thái (Problem & Modality)
- Dạng bài: Classification (Binary/Multiclass), Regression, Segmentation, Object Detection, Sequence Modeling.
- Dạng dữ liệu đầu vào: Tabular DataFrame, Ảnh 2D `[B, C, H, W]`, Chuỗi văn bản `[B, Seq_Len]`.

### 2. Hợp đồng Dữ liệu & Phân tách (Data Contract & Split Strategy)
- Khảo sát Schema, kiểu dữ liệu và tỷ lệ phân bố nhãn (Class Balance Ratio).
- Chiến lược phân tách dữ liệu chống rò rỉ (Chống Data Leakage):
  - Tabular / Image độc lập -> `StratifiedKFold` (giữ nguyên tỷ lệ các lớp).
  - Time-series -> `TimeSeriesSplit` (tuyệt đối không shuffle).
  - Grouped data -> `GroupKFold` (không để cùng 1 đối tượng xuất hiện ở cả train và val).

### 3. Mô hình Cơ sở & Chỉ số Mục tiêu (Baseline & Target Metrics)
- **Baseline Model:** Thiết lập mô hình tối thiểu để làm mốc so sánh (Heuristic, Logistic Regression, hoặc Simple MLP/CNN 2 tầng).
- **Chỉ số đo lường chính (Primary Metric):** F1-Macro, ROC-AUC, mAP, IoU, RMSE.
- **Ngưỡng nghiệm thu (Acceptance Threshold):** Mô hình nâng cao phải vượt Baseline bao nhiêu % thì mới đạt yêu cầu.

### 4. Hàm Mất mát & Tối ưu (Loss & Optimization Design)
- Lựa chọn hàm mất mát:
  - Phân loại chuẩn -> `nn.CrossEntropyLoss(label_smoothing=0.1)`.
  - Mất cân bằng nhãn nặng ($> 1:10$) -> `FocalLoss` hoặc gán `class_weights`.
- Lựa chọn Optimizer (AdamW) và Learning Rate Scheduler (CosineAnnealing / OneCycleLR).

### 5. Ràng buộc Môi trường Thực thi (Execution Environment Constraints)
- Thiết bị chạy: Local CPU/GPU hay **Kaggle GPU (Tesla T4)**.
- Đường dẫn dữ liệu tương thích: Nhận diện `/kaggle/input/` (Read-Only) và `/kaggle/working/` (Writable).
- Giới hạn bộ nhớ: Kiểm soát VRAM $< 15\text{GB}$ (bật AMP FP16).

---

## Anti-Rationalization
- [X] *"Bài Lab này ngắn, code luôn cho kịp nộp"* -> **Bác bỏ:** Không có Spec đồng nghĩa với việc không có mốc Baseline để so sánh trong báo cáo, dễ chọn sai metric dẫn đến điểm số thấp.
