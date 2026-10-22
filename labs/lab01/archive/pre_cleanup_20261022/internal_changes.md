# Nhật ký nội bộ - không đưa vào báo cáo PDF

## 22/10/2026

- Ghi phạm vi đã thống nhất vào description trước triển khai; giữ tài liệu gốc và bổ sung walkthrough, không thay bằng bản rút gọn.
- Thêm hai notebook và module FE/preparation/comparison/run tracking. Notebook EDA đã thực thi; modeling chưa có benchmark GPU.
- Sửa fallback Optuna giả, tải normalization khi cần và tách đường dẫn source/output để hỗ trợ fresh runtime. Notebook lịch sử chỉ xóa output cell có Optuna và sửa thứ tự prediction/target.
- Sửa optimizer comparison trả weights tại best validation epoch thay vì epoch cuối; khi truyền save_fig_path lưu thêm histories/summary.
- Phát hiện lỗi console Windows cp1252 trong lượt EDA đầu. Run thất bại được giữ; Tee ghi UTF-8 trước và escape ký tự khi console không hỗ trợ. Lượt sau hoàn thành.
- Regression gồm feature geometry/ảnh rỗng, forward/backward và one-batch overfit ba cấu hình, exception logging, figure export và optimizer best epoch. Fixture synthetic chỉ nằm trong pytest temp và không được đưa vào báo cáo như kết quả thật.
- Chưa commit/push; chưa tạo PDF. Cần GPU/Kaggle để thu metric và hoàn thiện diễn giải modeling từ artifact thật.

- Theo yêu cầu mới: bỏ giới hạn full training GPU-only ở model runner và optimizer notebook; ghi device vào run config; sửa số epoch hiển thị. Hỗ trợ local CPU/CUDA và Kaggle cùng API.

- Bỏ F1 heatmap; thay lưới ROC/PR bằng 3 ảnh hai panel OvR theo model. Bỏ glob hiển thị mọi PNG và bản sao learning curves trong comparison. Giữ CSV và artifact phụ.

- Triển khai tuning.py: fixed LR, ba study Optuna độc lập ngân sách bằng nhau, SQLite và checkpoint/history từng trial, pruning có status riêng, validation-only selection, refit và diagnostics cuối. Notebook modeling có cell từng giai đoạn, không còn placeholder optimizer. Cài Optuna từ requirements vào môi trường local để thực thi study thật.

- Thêm Pipeline Execution Roadmap đầu hai notebook; sửa mô tả trạng thái notebook EDA. Thêm hiển thị confusion matrix baseline và refit trong cell riêng, cùng mục đọc lại artifact không train lại. Không khôi phục F1 heatmap hoặc thay đổi hyperparameter run đã hoàn thành.


Roadmap đầu hai notebook đã chuyển sang bảng 5 cột theo mẫu người dùng: Bước, Giai đoạn, Nội dung chi tiết, Module triển khai, Output/Artifact. Nội dung và đường dẫn khớp pipeline run-based hiện tại; không đổi code hoặc chạy lại thí nghiệm.
