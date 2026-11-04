# Lab03 — Artifacts cho report

Nguồn run được chọn tường minh trong source_runs.json (report.json hoặc latest_report.json). Không huấn luyện hoặc chọn lại threshold khi xuất.

| Policy | Accuracy | Precision | Recall | Binary F1 | Macro F1 | Test loss |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 78.9869% | 79.2060% | 78.6116% | 78.9077% | 78.9866% | 0.517371 |
| Frozen DistilBERT threshold 0.5 | 82.1764% | 82.4197% | 81.8011% | 82.1092% | 82.1761% | 0.414510 |
| Frozen DistilBERT validation threshold | 82.1764% | 82.4197% | 81.8011% | 82.1092% | 82.1761% | 0.414510 |

TF-IDF loss được tính lại từ saved test probabilities; Transformer loss là số đo test_runtime_metrics của checkpoint.

Bảng chính thuộc frozen trial: hoàn thành 3 epoch, checkpoint epoch 3 theo validation loss; threshold 0.5 theo validation Macro-F1. Checkpoint và threshold được khóa trước khi đo test.

## Nguồn thí nghiệm

| Vai trò | Run |
|---|---|
| frozen_trial | `20261104T110654Z_frozen_trial_00eccdd1` |
| exercise1 | `20261104T110637Z_exercise1_6d992d02` |
| exercise1_analysis | `20261104T110646Z_exercise1_analysis_a061550c` |

## Hiệu năng và phạm vi đo

Test evaluation: **0.496248 ms/mẫu**; Trainer.predict full test, includes batching/transfer/evaluation overhead; not single-request latency.
RAM/VRAM training: **measured**.
[Bảng từng epoch](frozen_trial/performance_history.json) · [Phạm vi/đơn vị](frozen_trial/performance_summary.json)

Peak/batch slots là bộ nhớ dùng chung được phân bổ theo batch, không phải bộ nhớ độc lập cho mỗi câu. Không thay số thiếu bằng 0.

## Kiến trúc hai model

[Toàn bộ module tree dạng text](architecture/model_architectures.txt) · [Thông số và provenance](architecture/summary.json)

Đây là cấu trúc từ config đã pin, không phải phép xác thực reload checkpoint. Exercise1 chỉ inference, không tối ưu weights.

## Log nguồn đã publish

Các log dưới đây được copy nguyên byte từ run nguồn và có SHA256 trong catalog/manifest. Log thiếu trong nguồn giữ khai báo artifact_availability; không sinh log thay thế.

- [frozen_trial: console.log](logs/frozen_trial/console.log)
- [exercise1: console.log](logs/exercise1/console.log)
- [exercise1_analysis: console.log](logs/exercise1_analysis/console.log)

## Hình và bảng

Hình nguồn được copy nguyên byte. Mục result_analysis được dẫn xuất từ predictions/history đã audit; JSON ghi nguồn và phương pháp.

### frozen_trial/attention_mask.png

![frozen_trial](frozen_trial/attention_mask.png)

### frozen_trial/augmentation_counts.png

![frozen_trial](frozen_trial/augmentation_counts.png)

### frozen_trial/augmentation_diagnostics.png

![frozen_trial](frozen_trial/augmentation_diagnostics.png)

### frozen_trial/augmentation_examples.png

![frozen_trial](frozen_trial/augmentation_examples.png)

### frozen_trial/calibration_confidence.png

![frozen_trial](frozen_trial/calibration_confidence.png)

### frozen_trial/class_distribution.png

![frozen_trial](frozen_trial/class_distribution.png)

### frozen_trial/confusion_matrix.png

![frozen_trial](frozen_trial/confusion_matrix.png)

### frozen_trial/data_quality.png

![frozen_trial](frozen_trial/data_quality.png)

### frozen_trial/learning_curves.png

![frozen_trial](frozen_trial/learning_curves.png)

### frozen_trial/lengths_by_class.png

![frozen_trial](frozen_trial/lengths_by_class.png)

### frozen_trial/metrics_bar.png

![frozen_trial](frozen_trial/metrics_bar.png)

### frozen_trial/performance.png

![frozen_trial](frozen_trial/performance.png)

### frozen_trial/preprocessing_diagnostics.png

![frozen_trial](frozen_trial/preprocessing_diagnostics.png)

### frozen_trial/preprocessing_examples.png

![frozen_trial](frozen_trial/preprocessing_examples.png)

### frozen_trial/roc_pr_curves.png

![frozen_trial](frozen_trial/roc_pr_curves.png)

### frozen_trial/sample_predictions.png

![frozen_trial](frozen_trial/sample_predictions.png)

### frozen_trial/tfidf_features.png

![frozen_trial](frozen_trial/tfidf_features.png)

### frozen_trial/token_length_distribution.png

![frozen_trial](frozen_trial/token_length_distribution.png)

### frozen_trial/train_examples.png

![frozen_trial](frozen_trial/train_examples.png)

### frozen_trial/validation_threshold_curves.png

![frozen_trial](frozen_trial/validation_threshold_curves.png)

### exercise1_analysis/pipeline_manual_comparison.png

![exercise1_analysis](exercise1_analysis/pipeline_manual_comparison.png)

### exercise1_analysis/sentiment_predictions.png

![exercise1_analysis](exercise1_analysis/sentiment_predictions.png)

### exercise1_analysis/tokenizer_table.png

![exercise1_analysis](exercise1_analysis/tokenizer_table.png)

### result_analysis/learning_detail.png

![result_analysis](result_analysis/learning_detail.png)

### result_analysis/per_class_metrics.png

![result_analysis](result_analysis/per_class_metrics.png)
