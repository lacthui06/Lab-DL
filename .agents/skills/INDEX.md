# SKILL INDEX — Bảng định tuyến nhanh 17 skill

File này là manifest định tuyến. Agent đọc file này trước để chọn đúng skill mà không cần quét toàn bộ thư mục. Mỗi dòng ánh xạ một tình huống kích hoạt sang skill tương ứng.

---

## 1. Tầng điều phối (Meta)

| Skill | Kích hoạt khi | Nằm ở giai đoạn |
|---|---|---|
| `using-ds-ai-skills` | Bắt đầu phiên làm việc, cần phân loại bài toán và điều phối luồng | Xuyên suốt vòng đời |

## 2. Khởi tạo và Đặc tả (Define)

| Skill | Kích hoạt khi | Nằm ở giai đoạn |
|---|---|---|
| `idea-refine` | Ý tưởng còn mơ hồ, cần chọn dataset và kiến trúc khả thi | Khởi tạo |
| `interview-me` | Yêu cầu chưa rõ metric, dữ liệu, hoặc ràng buộc compute | Khởi tạo |
| `spec-driven-development` | Cần chốt spec, baseline, metric trước khi code | Đặc tả |

## 3. Lập kế hoạch (Plan)

| Skill | Kích hoạt khi | Nằm ở giai đoạn |
|---|---|---|
| `planning-and-task-breakdown` | Cần phân rã pipeline thành module nguyên tử | Kế hoạch |

## 4. Xây dựng và Kiểm thử (Build & Test)

| Skill | Kích hoạt khi | Nằm ở giai đoạn |
|---|---|---|
| `test-driven-development` | Viết Dataset, DataLoader, Model; cần test chiều tensor và overfit 1 batch | Xây dựng |
| `source-driven-development` | Xác thực API PyTorch/HuggingFace, tránh argument ảo tưởng | Xây dựng |
| `environment-and-reproducibility` | Cần cố định seed, quản lý phiên bản thư viện, đảm bảo tái lập | Xây dựng |

## 5. Chẩn đoán và Kiểm toán (Verify & Doubt)

| Skill | Kích hoạt khi | Nằm ở giai đoạn |
|---|---|---|
| `debugging-and-error-recovery` | CUDA OOM, loss NaN, lỗi môi trường Kaggle, gradient tiêu biến | Gỡ lỗi |
| `doubt-driven-development` | Độ chính xác cao bất thường (> 95%), nghi ngờ rò rỉ dữ liệu | Kiểm toán |

## 6. Tối ưu và Triển khai (Optimize & Serve)

| Skill | Kích hoạt khi | Nằm ở giai đoạn |
|---|---|---|
| `performance-optimization` | GPU bị bỏ không, VRAM hạn chế, độ trễ inference cao | Tối ưu |
| `api-and-interface-design` | Đóng gói mô hình thành dịch vụ inference FastAPI | Triển khai |

## 7. Thực thi từ xa (Remote Execution)

| Skill | Kích hoạt khi | Nằm ở giai đoạn |
|---|---|---|
| `kaggle-remote-execution` | Chuyển huấn luyện lên Kaggle GPU, đồng bộ artifact về local | Thực thi |

## 8. Chất lượng, Ghi vết và Nộp bài (Review & Ship)

| Skill | Kích hoạt khi | Nằm ở giai đoạn |
|---|---|---|
| `code-simplification` | Refactor notebook thành module sạch, loại bỏ code chết | Rà soát |
| `version-control-and-artifacts` | Commit, push, quyết định file nào thuộc Git, quản lý artifact | Rà soát |
| `walkthrough-and-experiment-tracking` | Ghi vết các run, chênh lệch metric, và phân tích ảnh output vào Section 2 | Ghi vết |
| `documentation-and-adrs` | Tổng hợp kết quả, chẩn đoán, kết luận vào Section 3 | Nộp bài |

---

## Bảng tra cứu theo triệu chứng (Symptom -> Skill)

| Triệu chứng / Tình huống | Skill cần gọi |
|---|---|
| "Tôi muốn làm một dự án AI nhưng chưa biết làm gì" | `idea-refine` -> `interview-me` |
| "Yêu cầu mơ hồ, chưa rõ cần đo metric gì" | `interview-me` |
| "Cần bắt đầu code một lab mới" | `spec-driven-development` -> `planning-and-task-breakdown` |
| "Model không hội tụ sau nhiều epoch" | `test-driven-development` (overfit 1 batch) -> `debugging-and-error-recovery` |
| "RuntimeError: CUDA out of memory" | `debugging-and-error-recovery` |
| "Loss trở thành NaN" | `debugging-and-error-recovery` |
| "Kết quả chạy hai lần khác nhau" | `environment-and-reproducibility` |
| "Model đạt 99% ngay lần đầu, có đáng tin không" | `doubt-driven-development` |
| "Kaggle chạy xong nhưng không thấy kết quả về máy" | `kaggle-remote-execution` |
| "Notebook đã chạy nhưng không được push lên Git" | `version-control-and-artifacts` |
| "Lỡ commit file checkpoint nặng" | `version-control-and-artifacts` |
| "GPU bị bỏ không, huấn luyện chậm" | `performance-optimization` |
| "Cần đóng gói model thành API" | `api-and-interface-design` |
| "Code notebook lộn xộn, cần tách ra src/" | `code-simplification` |
| "Cần ghi lại các lần chạy và so sánh" | `walkthrough-and-experiment-tracking` |
| "Cần viết báo cáo cuối để nộp" | `documentation-and-adrs` |
| "Cần ghi đủ metric phân loại (Macro F1, Precision, Recall, TPR, FPR), không chỉ Accuracy" | `walkthrough-and-experiment-tracking` -> `documentation-and-adrs` |
| "Cần phân tích ảnh output (confusion matrix, lưới dự đoán, đường cong) chứ không chỉ liệt kê tên file" | `documentation-and-adrs` |
| "Cần ghi phiên bản Python và snapshot môi trường (`environment_snapshot.json`)" | `environment-and-reproducibility` |
| "Trước khi commit / push" | Checklist `pre_commit.md` |
| "Trước khi đẩy lên Kaggle" | Checklist `pre_kaggle_upload.md` |
| "Trước khi nộp bài" | Checklist `pre_submission.md` |
| "Bắt đầu lab mới" | Checklist `new_lab_bootstrap.md` |

---

## Bảng tra cứu theo artifact cần tạo (Deliverable -> Skill)

| Cần tạo ra | Skill chịu trách nhiệm |
|---|---|
| `walkthrough_lab_0X.md` Section 1 | `spec-driven-development` |
| `walkthrough_lab_0X.md` Section 2 | `walkthrough-and-experiment-tracking` |
| `walkthrough_lab_0X.md` Section 3 | `documentation-and-adrs` |
| Bảng metric phân loại theo lớp (Precision, Recall, TPR, FPR, Macro F1) | `documentation-and-adrs` |
| `src/` module sạch | `code-simplification` |
| `tests/test_pipeline.py` | `test-driven-development` |
| Inference API service | `api-and-interface-design` |
| Bản ghi seed, phiên bản thư viện và snapshot môi trường (`environment_snapshot.json`) | `environment-and-reproducibility` |
| Artifact đồng bộ từ Kaggle | `kaggle-remote-execution` |
| Commit và lịch sử Git sạch | `version-control-and-artifacts` |
