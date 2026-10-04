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

## Hướng Dẫn Sử Dụng File `walkthrough_lab_0X.md` (Bản Tài Liệu Sống Hợp Nhất 3 Rule):
File `walkthrough_lab_0X.md` hợp nhất toàn bộ 3 quy chuẩn (3 Rule) thành một tài liệu sống duy nhất, tuyệt đối không tạo file `SPEC.md` hay `REPORT.md` riêng lẻ:
- **Mục 1 (Spec & Baseline Setup):** Dẫn dắt bởi Rule `spec-driven-development` — Đặc tả bài toán, schema dữ liệu, Baseline và ngưỡng nghiệm thu.
- **Mục 2 (Changelog & Lineage):** Dẫn dắt bởi Rule `walkthrough-and-experiment-tracking` — Ghi lại từng lần chạy (Run #1, Run #2, Run #3,...), đủ 5 khâu trong Flow, lý do thay đổi (Why) và tác động định lượng của việc tinh chỉnh siêu tham số.
- **Mục 3 (Results & Final Report):** Dẫn dắt bởi Rule `documentation-and-adrs` — Bảng ma trận đối đầu các phiên bản Flow, chẩn đoán ma trận nhầm lẫn, phân tích lỗi sai thực tế và kết luận hoàn chỉnh để nộp bài.
