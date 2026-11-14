# Lab 02 — Kết quả đã chạy

[Tài liệu sống](../../../walkthrough_lab_02.md)

Run: `20261030T052121Z_cifar10_scheduled64_b895bfc1`. Profile: **cifar10_scheduled64**. Số mẫu xem trong data_summary.json và báo cáo.

- [Notebook EDA và tiền xử lý](../../../notebooks/01_lab02_eda.ipynb)
- [Notebook huấn luyện và đánh giá](../../../notebooks/02_lab02_modeling.ipynb)
- [Báo cáo có hình](report_vi.html)
- [Báo cáo Markdown](report_vi.md)
- [So sánh kết quả](comparison.csv)
- [Kiểm chứng](verification.json)

Dữ liệu: CIFAR-10; đầu vào 64×64. Kết quả chỉ áp dụng cho cấu hình đã ghi. Đọc README để tái lập.

| Model | Stage | Test Accuracy | Macro F1 |
|---|---|---:|---:|
| resnet18 | head | 76.35% | 0.7630 |
| resnet18 | finetune | 84.14% | 0.8406 |
| densenet121 | head | 78.79% | 0.7879 |
| densenet121 | finetune | 91.80% | 0.9180 |
