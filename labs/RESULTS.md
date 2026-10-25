# Lab 02 — Kết quả đã chạy

[Báo cáo Lab 02 (Markdown)](Lab2_CIFAR10.md)

Run: `20261024T075133Z_cifar10_cpu_full64_71863a0e`. Profile: **cifar10_cpu_full64**. Số mẫu xem trong data_summary.json và báo cáo.

- [Notebook EDA và kiến trúc](notebooks/01_cifar10_eda_architecture.ipynb)
- [Notebook huấn luyện và đánh giá](notebooks/lab_02_transfer_learning.ipynb)
- [Báo cáo có hình](outputs/runs/20261024T075133Z_cifar10_cpu_full64_71863a0e/report_vi.html)
- [Báo cáo Markdown](outputs/runs/20261024T075133Z_cifar10_cpu_full64_71863a0e/report_vi.md)
- [So sánh kết quả](outputs/runs/20261024T075133Z_cifar10_cpu_full64_71863a0e/comparison.csv)
- [Kiểm chứng](outputs/runs/20261024T075133Z_cifar10_cpu_full64_71863a0e/verification.json)

Dữ liệu: CIFAR-10; đầu vào 64×64. Kết quả chỉ áp dụng cho cấu hình đã ghi. Đọc README để tái lập.

| Model | Stage | Test Accuracy | Macro F1 |
|---|---|---:|---:|
| resnet18 | head | 76.09% | 0.7611 |
| resnet18 | finetune | 81.49% | 0.8146 |
| densenet121 | head | 78.21% | 0.7811 |
| densenet121 | finetune | 81.53% | 0.8151 |
