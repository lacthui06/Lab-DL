# Lab 02 — Kết quả đã chạy

[Tài liệu sống](../../../walkthrough_lab_02.md)

Run: `20261027T042305Z_cifar10_gpu_tuning_group_00_c444e2bc`. Profile: **cifar10_gpu_tuning_group_00**. Số mẫu xem trong data_summary.json và báo cáo.

- [Notebook EDA và tiền xử lý](../../../notebooks/01_lab02_eda.ipynb)
- [Notebook huấn luyện và đánh giá](../../../notebooks/02_lab02_modeling.ipynb)
- [Báo cáo có hình](report_vi.html)
- [Báo cáo Markdown](report_vi.md)
- [So sánh kết quả](comparison.csv)
- [Kiểm chứng](verification.json)

Dữ liệu: CIFAR-10; đầu vào 224×224. Kết quả chỉ áp dụng cho cấu hình đã ghi. Đọc README để tái lập.

| Model | Stage | Test Accuracy | Macro F1 |
|---|---|---:|---:|
| resnet18 | head | 85.86% | 0.8590 |
| resnet18 | finetune | 91.37% | 0.9137 |
| densenet121 | head | 87.90% | 0.8791 |
| densenet121 | finetune | 91.11% | 0.9112 |
