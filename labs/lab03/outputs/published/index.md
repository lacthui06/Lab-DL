# Lab03 — Artifacts cho report

Nguồn run được chọn tường minh trong `config/report.json`. Không huấn luyện hoặc chọn lại threshold khi xuất.

| Policy | Accuracy | Precision | Recall | Binary F1 | Macro F1 | Test loss |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 78.9869% | 79.2060% | 78.6116% | 78.9077% | 78.9866% | Not logged |
| Frozen DistilBERT threshold 0.5 | 82.8330% | 82.5279% | 83.3021% | 82.9132% | 82.8326% | 0.412560 |
| Frozen DistilBERT validation threshold | 82.8330% | 81.3620% | 85.1782% | 83.2264% | 82.8236% | 0.412560 |

Bảng chính thuộc frozen trial: checkpoint epoch 9 theo validation loss, threshold 0.45 theo validation Macro-F1. Ưu tiên giảm overfitting; không khẳng định accuracy cao nhất lịch sử. Các study khác giữ vai trò so sánh.

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
| epoch_presentation | `20261027T061914Z_epoch_presentation_9a0cfccc` |
| frozen_trial | `20261027T095655Z_frozen_trial_d8d09116` |
| augmented_study | `20261027T083010Z_study_99bce559` |
| augmented_lr_5e6 | `20261027T083013Z_candidate_a2759300` |
| augmented_lr_1e5 | `20261027T083713Z_candidate_d00acd03` |
| frozen_exercise1 | `20261027T095627Z_exercise1_80baa48e` |
| frozen_exercise1_analysis | `20261027T095630Z_exercise1_analysis_dc3ef4d7` |
| frozen_dataset_preview | `20261027T095637Z_dataset_preview_e2b4d582` |
| frozen_augmentation_preview | `20261027T095647Z_augmentation_preview_acb26523` |
| final_review | `20261027T112616Z_final_review_82e3f77d` |

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

### epoch_presentation/20261026T184223Z_candidate_4b1da858/learning_curves.png

![epoch_presentation](epoch_presentation/20261026T184223Z_candidate_4b1da858/learning_curves.png)

### epoch_presentation/20261026T185017Z_candidate_202c0434/learning_curves.png

![epoch_presentation](epoch_presentation/20261026T185017Z_candidate_202c0434/learning_curves.png)

### epoch_presentation/20261026T185804Z_candidate_8308037d/learning_curves.png

![epoch_presentation](epoch_presentation/20261026T185804Z_candidate_8308037d/learning_curves.png)

### frozen_trial/augmentation_counts.png

![frozen_trial](frozen_trial/augmentation_counts.png)

### frozen_trial/augmentation_examples.png

![frozen_trial](frozen_trial/augmentation_examples.png)

### frozen_trial/calibration_confidence.png

![frozen_trial](frozen_trial/calibration_confidence.png)

### frozen_trial/confusion_matrix.png

![frozen_trial](frozen_trial/confusion_matrix.png)

### frozen_trial/learning_curves.png

![frozen_trial](frozen_trial/learning_curves.png)

### frozen_trial/roc_pr_curves.png

![frozen_trial](frozen_trial/roc_pr_curves.png)

### frozen_trial/validation_threshold_curves.png

![frozen_trial](frozen_trial/validation_threshold_curves.png)

### augmented_study/augmentation_counts.png

![augmented_study](augmented_study/augmentation_counts.png)

### augmented_study/augmentation_examples.png

![augmented_study](augmented_study/augmentation_examples.png)

### augmented_study/baseline_model_comparison.png

![augmented_study](augmented_study/baseline_model_comparison.png)

### augmented_study/calibration_confidence.png

![augmented_study](augmented_study/calibration_confidence.png)

### augmented_study/candidate_validation_comparison.png

![augmented_study](augmented_study/candidate_validation_comparison.png)

### augmented_study/class_distribution.png

![augmented_study](augmented_study/class_distribution.png)

### augmented_study/confusion_matrix.png

![augmented_study](augmented_study/confusion_matrix.png)

### augmented_study/inference_predictions.png

![augmented_study](augmented_study/inference_predictions.png)

### augmented_study/metrics_bar.png

![augmented_study](augmented_study/metrics_bar.png)

### augmented_study/roc_pr_curves.png

![augmented_study](augmented_study/roc_pr_curves.png)

### augmented_study/sample_predictions.png

![augmented_study](augmented_study/sample_predictions.png)

### augmented_study/test_threshold/confusion_matrix.png

![augmented_study](augmented_study/test_threshold/confusion_matrix.png)

### augmented_lr_5e6/augmentation_counts.png

![augmented_lr_5e6](augmented_lr_5e6/augmentation_counts.png)

### augmented_lr_5e6/augmentation_examples.png

![augmented_lr_5e6](augmented_lr_5e6/augmentation_examples.png)

### augmented_lr_5e6/learning_curves.png

![augmented_lr_5e6](augmented_lr_5e6/learning_curves.png)

### augmented_lr_5e6/step_learning_curves.png

![augmented_lr_5e6](augmented_lr_5e6/step_learning_curves.png)

### augmented_lr_5e6/token_length_distribution.png

![augmented_lr_5e6](augmented_lr_5e6/token_length_distribution.png)

### augmented_lr_5e6/validation_threshold/threshold_curves.png

![augmented_lr_5e6](augmented_lr_5e6/validation_threshold/threshold_curves.png)

### augmented_lr_1e5/augmentation_counts.png

![augmented_lr_1e5](augmented_lr_1e5/augmentation_counts.png)

### augmented_lr_1e5/augmentation_examples.png

![augmented_lr_1e5](augmented_lr_1e5/augmentation_examples.png)

### augmented_lr_1e5/learning_curves.png

![augmented_lr_1e5](augmented_lr_1e5/learning_curves.png)

### augmented_lr_1e5/step_learning_curves.png

![augmented_lr_1e5](augmented_lr_1e5/step_learning_curves.png)

### augmented_lr_1e5/token_length_distribution.png

![augmented_lr_1e5](augmented_lr_1e5/token_length_distribution.png)

### augmented_lr_1e5/validation_threshold/threshold_curves.png

![augmented_lr_1e5](augmented_lr_1e5/validation_threshold/threshold_curves.png)

### frozen_exercise1_analysis/pipeline_manual_comparison.png

![frozen_exercise1_analysis](frozen_exercise1_analysis/pipeline_manual_comparison.png)

### frozen_exercise1_analysis/sentiment_predictions.png

![frozen_exercise1_analysis](frozen_exercise1_analysis/sentiment_predictions.png)

### frozen_exercise1_analysis/tokenizer_table.png

![frozen_exercise1_analysis](frozen_exercise1_analysis/tokenizer_table.png)

### frozen_dataset_preview/class_distribution.png

![frozen_dataset_preview](frozen_dataset_preview/class_distribution.png)

### frozen_dataset_preview/token_length_distribution.png

![frozen_dataset_preview](frozen_dataset_preview/token_length_distribution.png)

### frozen_dataset_preview/train_examples.png

![frozen_dataset_preview](frozen_dataset_preview/train_examples.png)

### frozen_augmentation_preview/augmentation_counts.png

![frozen_augmentation_preview](frozen_augmentation_preview/augmentation_counts.png)

### frozen_augmentation_preview/augmentation_examples.png

![frozen_augmentation_preview](frozen_augmentation_preview/augmentation_examples.png)

### final_review/experiment_comparison.png

![final_review](final_review/experiment_comparison.png)

### final_review/generalization_gaps.png

![final_review](final_review/generalization_gaps.png)

### final_review/per_class_metrics.png

![final_review](final_review/per_class_metrics.png)

### final_review/sample_predictions.png

![final_review](final_review/sample_predictions.png)
