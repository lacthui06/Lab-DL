# Lab 02 — Kết quả đã chạy

[Tài liệu sống](../../../walkthrough_lab_02.md)

Run: `20261027T083044Z_cifar10_macintel_cpu_control64_db231b28`. Profile: **cifar10_macintel_cpu_control64**. Số mẫu xem trong data_summary.json và báo cáo.

- [Notebook EDA và tiền xử lý](../../../notebooks/01_lab02_eda.ipynb)
- [Notebook huấn luyện và đánh giá](../../../notebooks/02_lab02_modeling.ipynb)
- [Báo cáo có hình](report_vi.html)
- [Báo cáo Markdown](report_vi.md)
- [So sánh kết quả](comparison.csv)
- [Kiểm chứng](verification.json)

Dữ liệu: CIFAR-10; đầu vào 64×64. Kết quả chỉ áp dụng cho cấu hình đã ghi. Đọc README để tái lập.

| Model | Stage | Test Accuracy | Macro F1 |
|---|---|---:|---:|
| resnet18 | head | 75.93% | 0.7581 |
| resnet18 | finetune | 83.59% | 0.8349 |
| densenet121 | head | 79.04% | 0.7907 |
| densenet121 | finetune | 82.99% | 0.8301 |
