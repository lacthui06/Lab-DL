# Thư Mục Jupyter Notebooks (labs/lab03/notebooks)

## 1. Thông tin chung
- **Tác giả:** Thái Anh Lạc
- **Ngày cập nhật:** 2026-11-11
- **Vị trí:** `labs/lab03/notebooks/`

## 2. Danh mục 3 Notebook Chuyên Biệt (Modular Architecture)

Hệ thống notebook của Lab 03 được tách nhỏ thành 3 notebook chuyên sâu, nhẹ nhàng và độc lập:

| Notebook | Tên File | Vai Trò & Playbook Tham Chiếu | Chế Độ Mặc Định |
| :---: | :--- | :--- | :---: |
| **01** | [`01_eda.ipynb`](01_eda.ipynb) | Khám phá dữ liệu chuyên sâu 6 bước theo [`docs/eda.md`](../../../docs/eda.md) (Schema, Stats 5 số, Phân phối, **Box plot Negative vs Positive**, **Dò Outlier IQR**) | `view` (đổi `prepare` để chạy lại) |
| **02** | [`02_feature_engineering.ipynb`](02_feature_engineering.ipynb) | Kỹ nghệ đặc trưng theo [`docs/feature.md`](../../../docs/feature.md) (Capping Outliers `max_length=128`, TF-IDF Scaling, **Feature Selection Chi-Square $\chi^2$**, Text Augmentation) | `view` (đổi `prepare` để chạy lại) |
| **03** | [`03_modeling.ipynb`](03_modeling.ipynb) | Huấn luyện mô hình Transformer & Đánh giá theo [`docs/dl_training.md`](../../../docs/dl_training.md) (Baseline TF-IDF, DistilBERT Fine-tune, Frozen Backbone, Quét 19 ngưỡng tối ưu) | `view` (đổi `train` để chạy lại) |

---

## 3. Quy Chuẩn Trình Bày Bắt Buộc (Mandatory Notebook Standard)

Mọi notebook trong dự án bắt buộc phải tuân thủ nghiêm ngặt 3 nguyên tắc sau:

### Nguyên tắc 1: Bảng Lộ Trình Thực Thi (Pipeline Execution Roadmap) ở Cell Đầu Tiên
Cell Markdown đầu tiên của mỗi notebook phải chứa bảng Roadmap định hướng người đọc, gồm 5 cột chuẩn:
1. **Bước (Step):** Số thứ tự các bước thực hiện.
2. **Giai Đoạn (Phase):** Tên giai đoạn kỹ thuật.
3. **Nội Dung Cốt Lõi (Core Description):** Giải thích ngắn gọn câu hỏi nghiên cứu hoặc mục tiêu.
4. **Mô-đun Triển Khai (Source Module):** Đường dẫn hàm/lớp trong `src/` thực thi logic.
5. **Sản Phẩm Đầu Ra (Artifact):** Tên file ảnh `.png`, bảng `.csv` hoặc file cấu hình `.json` sinh ra.

### Nguyên tắc 2: Không Viết Code Logic Nặng Trực Tiếp Trong Cell
- Mọi giải thuật, hàm toán học, vòng lặp train hoặc logic vẽ biểu đồ phức tạp phải được đóng gói sạch sẽ trong thư mục [`src/`](../src/).
- Cell trong notebook chỉ đóng vai trò **gọi hàm (Orchestrator)** và hiển thị kết quả trực quan (ảnh inline, bảng Pandas).

### Nguyên tắc 3: Chế Độ Kép (Dual Mode `view` vs `train`/`prepare`)
- Mặc định mở notebook ở chế độ `MODE="view"`: đọc nhanh kết quả và hiển thị ảnh/bảng từ các run đã kiểm chứng mà không bắt người đọc phải nạp GPU huấn luyện lại.
- Khi người dùng muốn chạy thực nghiệm mới, chỉ cần đổi biến cờ thành `MODE="train"` (hoặc `prepare`).
