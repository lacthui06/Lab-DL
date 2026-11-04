# Outputs — thí nghiệm 3 epoch

Chưa có kết quả full training. Không điền trước accuracy, loss, threshold, thời gian hoặc bộ nhớ.

Khi thực thi notebook, mỗi run tạo thư mục riêng trong `runs/`. Các hình hiển thị inline được đọc từ chính PNG trong thư mục run. Sau modeling, `published/index.md` chứa hình, bảng, log, nguồn và SHA256 để dùng cho report.

- `latest_preparation.json`: selection của phiên EDA đã hoàn thành.
- `latest_report.json`: selection của phiên modeling đã hoàn thành.
- `architecture/`: module tree dạng text và bảng tham số.
- `preflight/`: log kiểm thử trước training.
- `verification/`: bằng chứng kiểm tra phần mềm; không phải kết quả mô hình.

Review ZIP chứa artifact/log; checkpoint ZIP chứa model/tokenizer/policy, lưu riêng.
