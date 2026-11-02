# Lab03 — Báo cáo thí nghiệm 10 epoch

Run: `20261027T083010Z_study_99bce559`

## Thiết kế và lựa chọn

Full official splits: 8.530 train / 1.066 validation / 1.066 test. Seed 42.
Mỗi DistilBERT candidate chạy đủ 10 epoch, không early stopping. Checkpoint chọn bằng validation Macro-F1.
TF-IDF và Logistic Regression fit chỉ trên train; C chọn bằng validation.
Learning rate chọn bằng validation Macro-F1; hòa ưu tiên loss thấp hơn rồi LR nhỏ hơn.
Threshold chốt từ 19 điểm validation trước test; không tối ưu threshold trên test.

| Candidate | LR | Epoch hoàn tất | Best step | Val Macro-F1 | Val loss |
|---|---:|---:|---:|---:|---:|
| 20261027T083013Z_candidate_a2759300 | 5e-06 | 10 | 2204 | 0.84428 | 0.38041 |
| 20261027T083713Z_candidate_d00acd03 | 1e-05 | 10 | 3306 | 0.84894 | 0.58491 |

Winner: `20261027T083713Z_candidate_d00acd03`.

![Validation candidates](candidate_validation_comparison.png)

## Test với các policy đã chốt

| Model / policy | Accuracy | Precision (+) | Recall (+) | Macro-F1 | Binary-F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression (0.5) | 0.7899 | 0.7921 | 0.7861 | 0.7899 | 0.7891 | 110 | 114 |
| TF-IDF + validation threshold | 0.7899 | 0.7921 | 0.7861 | 0.7899 | 0.7891 | 110 | 114 |
| DistilBERT (0.5) | 0.8368 | 0.8270 | 0.8518 | 0.8367 | 0.8392 | 95 | 79 |
| DistilBERT + validation threshold | 0.8415 | 0.8396 | 0.8443 | 0.8415 | 0.8419 | 86 | 83 |

![Baseline comparison](baseline_model_comparison.png)

## Xếp hạng, confidence và bất định

ROC-AUC: 0.91869; AP: 0.92361; Brier: 0.14236; argmax-confidence ECE: 0.13060.
95% percentile bootstrap CI: `{'accuracy': [0.8189493433395872, 0.8630393996247655], 'macro_f1': [0.8189412492170806, 0.8630007238788416]}`.
CI phân tầng trên test cho model/policy cố định; không bao gồm bất định do seed train.

![ROC/PR](roc_pr_curves.png)

![Calibration](calibration_confidence.png)

## Ma trận nhầm lẫn và mẫu lỗi

![Argmax confusion](confusion_matrix.png)

![Frozen threshold confusion](test_threshold/confusion_matrix.png)

![Sample predictions](sample_predictions.png)

Xem report.md cho mẫu lỗi argmax và threshold_errors.json cho toàn bộ lỗi của policy chốt.
Giả thuyết ngôn ngữ không phải causal attribution. Không sửa policy dựa trên ví dụ test.

## Giới hạn

Test lịch sử 3 epoch từng được xem. Đây là mở rộng thí nghiệm, không phải benchmark hoàn toàn chưa đụng tới.
Một seed không chứng minh cấu hình ổn định qua các lần khởi tạo. Ba learning rate không chứng minh tối ưu toàn cục.
Chạy lâu hơn không đảm bảo test accuracy/F1 cao hơn. Báo cáo cả cải thiện và suy giảm.
Nếu overlap audit có giao nhau giữa split, phải nêu rõ trong kết luận; không tự đổi official splits.

## Artifact theo từng candidate

### 20261027T083013Z_candidate_a2759300: learning_curves.png

![learning_curves.png](../20261027T083013Z_candidate_a2759300/learning_curves.png)

### 20261027T083013Z_candidate_a2759300: step_learning_curves.png

![step_learning_curves.png](../20261027T083013Z_candidate_a2759300/step_learning_curves.png)

### 20261027T083013Z_candidate_a2759300: validation_threshold/threshold_curves.png

![validation_threshold/threshold_curves.png](../20261027T083013Z_candidate_a2759300/validation_threshold/threshold_curves.png)

### 20261027T083713Z_candidate_d00acd03: learning_curves.png

![learning_curves.png](../20261027T083713Z_candidate_d00acd03/learning_curves.png)

### 20261027T083713Z_candidate_d00acd03: step_learning_curves.png

![step_learning_curves.png](../20261027T083713Z_candidate_d00acd03/step_learning_curves.png)

### 20261027T083713Z_candidate_d00acd03: validation_threshold/threshold_curves.png

![validation_threshold/threshold_curves.png](../20261027T083713Z_candidate_d00acd03/validation_threshold/threshold_curves.png)

