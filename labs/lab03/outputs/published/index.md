# Lab03 — Artifacts cho report

Nguồn run được chọn tường minh trong `config/report.json`. Không huấn luyện hoặc chọn lại threshold khi xuất.

| Policy | Accuracy | Precision | Recall | Binary F1 | Macro F1 | Test loss |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression (0.5) | 78.2364% | 78.6667% | 77.4859% | 78.0718% | 78.2352% | Not logged |
| TF-IDF + validation threshold | 78.7054% | 75.4153% | 85.1782% | 80.0000% | 78.6158% | Not logged |
| DistilBERT (0.5) | 83.3959% | 83.8403% | 82.7392% | 83.2861% | 83.3952% | 0.798076 |
| DistilBERT + validation threshold | 83.3959% | 83.8403% | 82.7392% | 83.2861% | 83.3952% | 0.798076 |

Bảng chính thuộc study 10 epoch; các role finetuning/evaluation/analysis cũ giữ bằng chứng lịch sử 3 epoch. Không gán số liệu hai protocol cho cùng một run.

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
| study | `20261026T184219Z_study_94f2ba7b` |
| candidate_lr_1e5 | `20261026T184223Z_candidate_4b1da858` |
| candidate_lr_2e5 | `20261026T185017Z_candidate_202c0434` |
| candidate_lr_3e5 | `20261026T185804Z_candidate_8308037d` |
| study_exercise1 | `20261026T184202Z_exercise1_de76a80c` |
| study_exercise1_analysis | `20261026T184207Z_exercise1_analysis_5406bc6c` |
| study_review | `20261026T191751Z_study_review_d66d4edb` |

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

### study/baseline_model_comparison.png

![study](study/baseline_model_comparison.png)

### study/calibration_confidence.png

![study](study/calibration_confidence.png)

### study/candidate_validation_comparison.png

![study](study/candidate_validation_comparison.png)

### study/class_distribution.png

![study](study/class_distribution.png)

### study/confusion_matrix.png

![study](study/confusion_matrix.png)

### study/inference_predictions.png

![study](study/inference_predictions.png)

### study/metrics_bar.png

![study](study/metrics_bar.png)

### study/roc_pr_curves.png

![study](study/roc_pr_curves.png)

### study/sample_predictions.png

![study](study/sample_predictions.png)

### study/test_threshold/confusion_matrix.png

![study](study/test_threshold/confusion_matrix.png)

### candidate_lr_1e5/learning_curves.png

![candidate_lr_1e5](candidate_lr_1e5/learning_curves.png)

### candidate_lr_1e5/step_learning_curves.png

![candidate_lr_1e5](candidate_lr_1e5/step_learning_curves.png)

### candidate_lr_1e5/token_length_distribution.png

![candidate_lr_1e5](candidate_lr_1e5/token_length_distribution.png)

### candidate_lr_1e5/validation_threshold/threshold_curves.png

![candidate_lr_1e5](candidate_lr_1e5/validation_threshold/threshold_curves.png)

### candidate_lr_2e5/learning_curves.png

![candidate_lr_2e5](candidate_lr_2e5/learning_curves.png)

### candidate_lr_2e5/step_learning_curves.png

![candidate_lr_2e5](candidate_lr_2e5/step_learning_curves.png)

### candidate_lr_2e5/token_length_distribution.png

![candidate_lr_2e5](candidate_lr_2e5/token_length_distribution.png)

### candidate_lr_2e5/validation_threshold/threshold_curves.png

![candidate_lr_2e5](candidate_lr_2e5/validation_threshold/threshold_curves.png)

### candidate_lr_3e5/learning_curves.png

![candidate_lr_3e5](candidate_lr_3e5/learning_curves.png)

### candidate_lr_3e5/step_learning_curves.png

![candidate_lr_3e5](candidate_lr_3e5/step_learning_curves.png)

### candidate_lr_3e5/token_length_distribution.png

![candidate_lr_3e5](candidate_lr_3e5/token_length_distribution.png)

### candidate_lr_3e5/validation_threshold/threshold_curves.png

![candidate_lr_3e5](candidate_lr_3e5/validation_threshold/threshold_curves.png)

### study_exercise1_analysis/pipeline_manual_comparison.png

![study_exercise1_analysis](study_exercise1_analysis/pipeline_manual_comparison.png)

### study_exercise1_analysis/sentiment_predictions.png

![study_exercise1_analysis](study_exercise1_analysis/sentiment_predictions.png)

### study_exercise1_analysis/tokenizer_table.png

![study_exercise1_analysis](study_exercise1_analysis/tokenizer_table.png)

### study_review/candidate_summary.png

![study_review](study_review/candidate_summary.png)

### study_review/historical_paired_comparison.png

![study_review](study_review/historical_paired_comparison.png)

### study_review/overfitting_diagnostics.png

![study_review](study_review/overfitting_diagnostics.png)

### study_review/per_class_metrics.png

![study_review](study_review/per_class_metrics.png)
