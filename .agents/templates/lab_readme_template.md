# Lab 0X — <tên bài lab>

Mô tả mục tiêu, dataset và thiết kế thực nghiệm của lab.

## Cấu trúc thực tế

- description_lab_0X.md: đề gốc và phần yêu cầu mở rộng đã chốt.
- walkthrough_lab_0X.md: nguồn duy nhất cho báo cáo/PDF.
- config/: cấu hình thực tế bằng JSON hoặc YAML; liệt kê đúng file đang dùng.
- notebooks/: notebook chuẩn bị dữ liệu/EDA/FE và notebook modeling khi phù hợp; liệt kê tên thật.
- src/: thư viện tái sử dụng; roadmap tham chiếu module/hàm tồn tại.
- scripts/: CLI thực tế của lab, gồm publish nếu đã triển khai.
- tests/: kiểm thử dữ liệu, mô hình, training và artifact theo phạm vi.
- data/: dữ liệu local không đưa lên Git.
- outputs/runs/<run_id>/: artifact, config, môi trường và log gốc từng lần chạy, giữ local.
- outputs/published/: tất cả ảnh và evidence báo cáo của các run được chọn tường minh; không chứa bản sao báo cáo hay log sửa code.

Không liệt kê module/tệp chưa triển khai chỉ để lấp cấu trúc.

## Xem kết quả sau khi clone

Dẫn link walkthrough, notebook đã chạy có output và hướng dẫn bộ published. Người đọc không cần train lại để phân tích báo cáo. Nêu rõ notebook view cần run gốc/prepared data nếu chưa hỗ trợ đọc published.

## Cài đặt và tái lập

Ghi interpreter đã kiểm chứng và yêu cầu Python suy ra từ dependencies thực tế; không dùng python==... trong requirements.txt. Mỗi run lưu environment_snapshot.json riêng.

Liệt kê lệnh cài đặt, tests, train, eval, infer và publish đã kiểm chứng. Cho phép local CPU/GPU hoặc Kaggle theo workload; chỉ nhận đã xác thực môi trường thực sự chạy.

## Artifacts đầy đủ

Ghi nơi tải và checksum khi gói đầy đủ đã tồn tại; cách giải nén để lấy checkpoint/prepared data/predictions hoặc tiếp tục study. Nếu chưa có Release thì ghi chưa cung cấp, không tạo link giả.

Bộ published cần có ảnh, bảng, histories, metrics, cấu hình và provenance. Checkpoint/prepared data/database tải riêng khi cần thực thi. Console log, nhật ký kỹ thuật và secret không nằm trong bộ Git.

## Kết quả chính

Tóm tắt ngắn từ artifact thật và dẫn walkthrough cho bảng đầy đủ; không hard-code kết quả chưa kiểm chứng hoặc yêu cầu số test cố định.

## Giới hạn

Nêu seed, ngân sách tuning, môi trường chưa thử và điều kiện tái lập. Không đưa nhật ký sửa phần mềm vào báo cáo.
