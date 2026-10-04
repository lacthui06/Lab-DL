# Quản Lý Dự Án Các Bài Lab Deep Learning

Thư mục này quản lý toàn bộ các bài thực hành Lab Deep Learning. Cấu trúc mẫu đã được thiết lập chuẩn hóa tại **`lab01/`**:

```text
labs/
└── lab01/                      <-- Cấu trúc mẫu chuẩn cho một bài Lab
    ├── walkthrough_lab_01.md   <-- Nhật ký theo dõi toàn bộ Flow (Data -> Preprocessing -> Model -> Training -> Eval -> Changelog)
    ├── notebooks/              <-- Thư mục chứa Jupyter Notebook (trống)
    ├── src/                    <-- Thư mục chứa mã nguồn module hóa (trống)
    ├── data/                   <-- Thư mục chứa dữ liệu đầu vào (trống)
    └── outputs/                <-- Thư mục lưu checkpoint, metrics, biểu đồ (trống)
```

---

## Hướng Dẫn Sử Dụng File `walkthrough_lab_01.md`:
File `walkthrough_lab_01.md` được thiết kế theo tư duy **Full-Flow Experiment Lineage** (Nhật ký tiến hóa toàn bộ quy trình):
- Ghi lại từng lần chạy (Run #1 Baseline, Run #2 Iteration 1, Run #3 Iteration 2,...).
- Mỗi lần chạy ghi nhận đầy đủ cả 5 khâu: **Data -> Preprocessing -> Model -> Training Strategy -> Evaluation**.
- Chỉ rõ **thay đổi ở khâu nào**, **lý do thay đổi (Why)** và **tác động lên toàn bộ Flow**.
- Bảng ma trận so sánh các phiên bản Flow giúp trích xuất trực tiếp số liệu và kết luận vào báo cáo môn học.
