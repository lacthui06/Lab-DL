# Lab03 — Artifacts cho report

Nguồn run được chọn tường minh trong `config/report.json`. Không huấn luyện hoặc chọn lại threshold khi xuất.

| Policy | Accuracy | Precision | Recall | Binary F1 | Macro F1 | Test loss |
|---|---:|---:|---:|---:|---:|---:|
| Argmax | 83.9587% | 83.5185% | 84.6154% | 84.0634% | 83.9580% | 0.544268 |
| Validation threshold 0.8 | 84.0525% | 86.5191% | 80.6754% | 83.4951% | 84.0343% | 0.544268 |

## Nguồn thí nghiệm

| Vai trò | Run |
|---|---|
| exercise1 | `20261026T155744Z_exercise1_65dfe784` |
| exercise1_analysis | `20261026T161956Z_exercise1_analysis_46ca085f` |
| finetuning | `20261026T155822Z_finetuning_2c2f0401` |
| analysis | `20261026T161957Z_analysis_b336c410` |
| evaluation | `20261026T160058Z_evaluation_07c24854` |
| threshold_analysis | `20261026T162909Z_threshold_analysis_f8b3fdf7` |
| inference | `20261026T160111Z_inference_7f2b16f0` |

## Hình và bảng

Các PNG dưới đây là bản sao của hình đã lưu trong run nguồn. JSON giữ số liệu để tính lại.

### exercise1_analysis/pipeline_manual_comparison.png

![exercise1_analysis](exercise1_analysis/pipeline_manual_comparison.png)

### exercise1_analysis/sentiment_predictions.png

![exercise1_analysis](exercise1_analysis/sentiment_predictions.png)

### exercise1_analysis/tokenizer_table.png

![exercise1_analysis](exercise1_analysis/tokenizer_table.png)

### finetuning/confusion_matrix.png

![finetuning](finetuning/confusion_matrix.png)

### finetuning/learning_curves.png

![finetuning](finetuning/learning_curves.png)

### finetuning/metrics_bar.png

![finetuning](finetuning/metrics_bar.png)

### finetuning/sample_predictions.png

![finetuning](finetuning/sample_predictions.png)

### finetuning/test_threshold/confusion_matrix.png

![finetuning](finetuning/test_threshold/confusion_matrix.png)

### finetuning/token_length_distribution.png

![finetuning](finetuning/token_length_distribution.png)

### finetuning/validation_threshold/threshold_curves.png

![finetuning](finetuning/validation_threshold/threshold_curves.png)

### analysis/calibration_curve.png

![analysis](analysis/calibration_curve.png)

### analysis/class_distribution.png

![analysis](analysis/class_distribution.png)

### analysis/metric_comparison.png

![analysis](analysis/metric_comparison.png)

### analysis/per_class_metrics.png

![analysis](analysis/per_class_metrics.png)

### analysis/probability_distribution.png

![analysis](analysis/probability_distribution.png)

### analysis/review_length_distribution.png

![analysis](analysis/review_length_distribution.png)

### analysis/roc_pr_curves.png

![analysis](analysis/roc_pr_curves.png)

### analysis/threshold_test_diagnostic/threshold_curves.png

![analysis](analysis/threshold_test_diagnostic/threshold_curves.png)

### analysis/training_threshold_confusion.png

![analysis](analysis/training_threshold_confusion.png)

### analysis/training_threshold_metrics.png

![analysis](analysis/training_threshold_metrics.png)

### analysis/training_threshold_per_class.png

![analysis](analysis/training_threshold_per_class.png)

### evaluation/confusion_matrix.png

![evaluation](evaluation/confusion_matrix.png)

### evaluation/metrics_bar.png

![evaluation](evaluation/metrics_bar.png)

### evaluation/sample_predictions.png

![evaluation](evaluation/sample_predictions.png)

### evaluation/test_threshold/confusion_matrix.png

![evaluation](evaluation/test_threshold/confusion_matrix.png)

### threshold_analysis/changed_predictions.png

![threshold_analysis](threshold_analysis/changed_predictions.png)

### threshold_analysis/confusion_comparison.png

![threshold_analysis](threshold_analysis/confusion_comparison.png)

### threshold_analysis/error_tradeoff.png

![threshold_analysis](threshold_analysis/error_tradeoff.png)

### threshold_analysis/inference_result.png

![threshold_analysis](threshold_analysis/inference_result.png)

### threshold_analysis/operating_points.png

![threshold_analysis](threshold_analysis/operating_points.png)

### threshold_analysis/per_class_comparison.png

![threshold_analysis](threshold_analysis/per_class_comparison.png)

### threshold_analysis/test_comparison.png

![threshold_analysis](threshold_analysis/test_comparison.png)

### threshold_analysis/validation_comparison.png

![threshold_analysis](threshold_analysis/validation_comparison.png)

### threshold_analysis/validation_threshold_curves.png

![threshold_analysis](threshold_analysis/validation_threshold_curves.png)

### inference/inference.png

![inference](inference/inference.png)
