# Bản đồ outputs của Lab02

Không đổi vị trí run/checkpoint để dọn cấu trúc; registry và provenance tiếp tục dùng đường dẫn thực. Các nhóm có vai trò khác nhau:

- `eda/`: split cố định, thống kê và ảnh dữ liệu của máy local. Không phải kết quả training.
- `runs/`: mỗi run có ID riêng; config, history, study, weights, log, manifest. Không tự lấy run mới nhất. Head feature cache là dữ liệu trung gian, không phải feature engineering được nối thêm vào model.
- `kaggle/group_00..02/`: artifact các phiên GPU cũ được thu hồi. `extracted/` là nội dung gói tải về; không phải code hiện hành để sửa hoặc chạy.
- `review/`: bảng/hình trình bày lại từ run đã lưu; không phải thí nghiệm mới, không dùng để thay trọng số.
- `logs/`: log kỹ thuật local. Không chèn vào báo cáo, không tự publish console log thô.
- `kaggle_upload/`: các ZIP source để upload. Đây là snapshot, không chứa dữ liệu/checkpoint hoặc kết quả thí nghiệm.

## Registry

- `selected_run.json`: các run GPU và CPU lịch sử cho notebook 02.
- `selected_optuna.json`: chỉ sinh sau khi hoàn tất thí nghiệm DenseNet shared/separate của notebook 03.
- `selected_standard.json`: run notebook 04 (hai kiến trúc, LR chung, 3 trial/model). Registry có thể trỏ run đang thực hiện; phải đọc `status.json`, checkpoint completion và selection trước khi coi là hoàn tất.

Notebook 04 trên Kaggle ghi vào `/kaggle/working/lab02_standard/outputs/`; tải `lab02_standard_outputs.zip` về sẽ thấy các thư mục tương ứng. Gói đầy đủ chứa log/study/checkpoint: giữ local, không push nguyên gói công khai. Việc đổi cấu trúc Python không đổi schema hoặc đường dẫn artifact cũ.

## Entry point của source hiện tại

- Bài chuẩn mới: `src/training/standard_suite.py`.
- DenseNet shared/separate trước đó: `src/training/optuna_dense.py`.
- Training/evaluation chung: `src/training/trainer.py`, `src/utils/plots.py`.

Chưa có benchmark mới ở local chỉ vì source đã refactor. Phiên Kaggle cũ đang chạy vẫn dùng ZIP đã upload; không tự chuyển sang source mới.
