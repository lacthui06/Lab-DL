# Checklist — Khởi tạo Lab mới (New Lab Bootstrap)

Danh sách kiểm tra khi bắt đầu một lab mới, đảm bảo mọi lab tuân theo cùng một cấu trúc chuẩn ngay từ đầu.

---

## 1. Khung thư mục (Directory Scaffold)
- [ ] Tạo thư mục `labs/lab0X/` theo đúng cấu trúc chuẩn.
- [ ] Tạo các thư mục con: `config/`, `notebooks/`, `src/`, `scripts/`, `tests/`, `data/`, `outputs/`.
- [ ] Thêm file `.gitkeep` vào `data/`, `notebooks/`, `outputs/` để giữ cấu trúc trên Git.
- [ ] Tạo các thư mục con trong `outputs/`: `checkpoints/`, `logs/`, `predictions/`, `eda/tables/`, `eda/figures/`.

## 2. Tài liệu (Documentation)
- [ ] Tạo `walkthrough_lab_0X.md` từ template `walkthrough_template.md`.
- [ ] Tạo `description_lab_0X.md` từ template `description_lab_template.md` và dán đề bài gốc nguyên văn.
- [ ] Xác nhận đã làm rõ đề bài (Bước 0 của `spec-driven-development`) và ghi bản tóm tắt vào Mục 1.0 trước khi viết Section 1 (Spec).
- [ ] Tạo `README.md` cho lab từ template `lab_readme_template.md`.
- [ ] Điền Section 1 (Spec) trước khi viết bất kỳ dòng code nào.
- [ ] Không tạo file `SPEC.md` hay `REPORT.md` riêng lẻ.

## 3. Module source (Source Modules & Configs)
- [ ] Tạo các file cấu hình YAML trong `config/` (`config.yaml`, `augmentations.yaml`, `experiments.yaml`).
- [ ] Tạo `src/__init__.py` và 4 phân hệ chuẩn: `src/datasets/` (`base.py`, `classification.py`, `eda.py`), `src/models/`, `src/training/` (`trainer.py`, `experiments.py`), `src/utils/` (`helpers.py`, `metrics.py`, `plots.py`).
- [ ] Tạo các script dòng lệnh trong `scripts/` (`train.py`, `eval.py`, `infer.py`).
- [ ] Đảm bảo mọi module có type hint và docstring mô tả shape tensor, không dùng hằng số thống kê giả lập (phải đọc từ artifact EDA).

## 4. Kiểm thử (Testing)
- [ ] Tạo bộ test TDD trong `tests/` (`test_datasets.py`, `test_models.py`, `test_training.py`).
- [ ] Chạy `pytest` và xác nhận tất cả pass trước khi viết thêm code.
- [ ] Xác nhận bài test 1-batch sanity overfit hoạt động.

## 5. Đặc tả trước code (Spec First)
- [ ] Đã xác định modality, loại bài toán, và hợp đồng tensor.
- [ ] Đã chọn dataset công khai khả dụng và ghi nguồn.
- [ ] Đã chốt metric chính và ngưỡng chấp nhận so với baseline.
- [ ] Đã định nghĩa mô hình baseline tối thiểu.

## 6. Xác nhận cuối
- [ ] Đã commit khung ban đầu lên Git với thông điệp rõ ràng.
- [ ] Đã cập nhật `labs/README.md` nếu cần liệt kê lab mới.
- [ ] Đã xác nhận `.gitignore` bao phủ đúng `data/` và checkpoint của lab mới.
