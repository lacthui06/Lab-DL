# Nhật ký nội bộ - không đưa vào báo cáo PDF

## 22/10/2026

- Ghi phạm vi đã thống nhất vào description trước triển khai; giữ tài liệu gốc và bổ sung walkthrough, không thay bằng bản rút gọn.
- Thêm hai notebook và module FE/preparation/comparison/run tracking. Notebook EDA đã thực thi; modeling chưa có benchmark GPU.
- Sửa fallback Optuna giả, tải normalization khi cần và tách đường dẫn source/output để hỗ trợ fresh runtime. Notebook lịch sử chỉ xóa output cell có Optuna và sửa thứ tự prediction/target.
- Sửa optimizer comparison trả weights tại best validation epoch thay vì epoch cuối; khi truyền save_fig_path lưu thêm histories/summary.
- Phát hiện lỗi console Windows cp1252 trong lượt EDA đầu. Run thất bại được giữ; Tee ghi UTF-8 trước và escape ký tự khi console không hỗ trợ. Lượt sau hoàn thành.
- Regression gồm feature geometry/ảnh rỗng, forward/backward và one-batch overfit ba cấu hình, exception logging, figure export và optimizer best epoch. Fixture synthetic chỉ nằm trong pytest temp và không được đưa vào báo cáo như kết quả thật.
- Chưa commit/push; chưa tạo PDF. Cần GPU/Kaggle để thu metric và hoàn thiện diễn giải modeling từ artifact thật.

- Theo yêu cầu mới: bỏ giới hạn full training GPU-only ở model runner và optimizer notebook; ghi device vào run config; sửa số epoch hiển thị. Hỗ trợ local CPU/CUDA và Kaggle cùng API.

- Bỏ F1 heatmap; thay lưới ROC/PR bằng 3 ảnh hai panel OvR theo model. Bỏ glob hiển thị mọi PNG và bản sao learning curves trong comparison. Giữ CSV và artifact phụ.

- Triển khai tuning.py: fixed LR, ba study Optuna độc lập ngân sách bằng nhau, SQLite và checkpoint/history từng trial, pruning có status riêng, validation-only selection, refit và diagnostics cuối. Notebook modeling có cell từng giai đoạn, không còn placeholder optimizer. Cài Optuna từ requirements vào môi trường local để thực thi study thật.

- Thêm Pipeline Execution Roadmap đầu hai notebook; sửa mô tả trạng thái notebook EDA. Thêm hiển thị confusion matrix baseline và refit trong cell riêng, cùng mục đọc lại artifact không train lại. Không khôi phục F1 heatmap hoặc thay đổi hyperparameter run đã hoàn thành.


Roadmap đầu hai notebook đã chuyển sang bảng 5 cột theo mẫu người dùng: Bước, Giai đoạn, Nội dung chi tiết, Module triển khai, Output/Artifact. Nội dung và đường dẫn khớp pipeline run-based hiện tại; không đổi code hoặc chạy lại thí nghiệm.


## Đợt hợp nhất cấu trúc

- Snapshot trước dọn: archive/pre_cleanup_20261022 (tài liệu, ba notebook, config YAML và CLI cũ). Không xóa hoặc sửa bất kỳ artifact run lịch sử nào.
- config/workflow.json là cấu hình hiện hành và danh mục run đang dùng; YAML cũ giữ tương thích, đã đánh dấu legacy. Không tự lấy run mới nhất.
- Notebook modeling gộp train/view chung cell, bỏ phần đọc lại/tiêu đề optimizer thừa; notebook lịch sử thành trang chỉ dẫn. Hai notebook chính được thực thi view lại để giữ output thật.
- README chỉ hướng dẫn; description giữ đề bài và phạm vi hiện hành; walkthrough hợp nhất phương pháp và kết quả có bằng chứng. Lịch sử K-fold/sweep chưa xác minh chỉ còn trong bản lưu.
- CLI train/eval/infer chuyển sang prepared data và run metadata như notebook. Eval/infer nhận --run, hỗ trợ --data khi đổi máy; bản CLI/YAML trước dọn vẫn lưu archive.
- EDA và modeling dùng chung foreground_bbox/measure_intensity. Công thức modeling không đổi. Bảng EDA mới sử dụng edge density cùng định nghĩa modeling; bảng EDA lịch sử không bị sửa hoặc đổi cách diễn giải.
- Đối chiếu 256 mẫu với prepared artifact trước dọn: max absolute error feature sau scaler 4.77e-7, đạt tolerance 2e-6. Log ở outputs/runs/20261022T141936Z_cleanup_feature_equivalence_cb10dc49.
- Lượt regression đầu phát hiện EDA hỗ trợ ảnh 6x7 trong khi extractor modeling chỉ nhận 28x28. Đã sửa bằng helper cường độ độc lập kích thước; không sửa/xóa test để che lỗi. Log thất bại được giữ.

- Sau sửa hồi quy: 44 passed, 1 deselected; log `outputs/runs/20261022T142053Z_cleanup_regression_e56cc9c1/`. Hai notebook chạy view đủ từ đầu tới cuối; bản executed và log ở `outputs/runs/20261022T142021Z_cleanup_notebook_view_execution_aab01226/`.

- Viết lại README đầy đủ theo mẫu người dùng: cây thư mục chú thích, pipeline, kiến trúc, notebook/CLI/local/Kaggle, artifact, kiểm thử và benchmark đọc từ run đã đăng ký. Sửa việc rút ngắn quá mức ở đợt dọn trước. Không đổi source hoặc train lại.


## Rà soát theo yêu cầu chỉ bỏ file ma — 22/10/2026

- Đối chiếu 22 module source với import trong src/scripts/tests/notebook; giữ toàn bộ module vì còn được dùng trực tiếp hoặc thuộc API tương thích. Giữ archive, YAML legacy, notebook điều hướng lab01.ipynb, checkpoint, log và mọi run lịch sử. Chỉ xóa notebooks/.gitkeep: thư mục đã có notebook thật; bản sao trước xóa ở outputs/runs/20261022T144702Z_project_audit_ad0aa453/before/notebooks/.gitkeep. Không rút gọn README/notebook hay xóa bài làm.
- Confusion matrix trước đây chỉ imshow: thêm helper draw_confusion_counts dùng chung cho comparison và optimizer results. Số trong ô là count nguyên; màu là tỷ lệ theo hàng, cùng thang 0–1, colorbar có nhãn; ô 0 cũng được ghi. Dựng mới từ probabilities/labels đã lưu, không huấn luyện lại benchmark. Đã xem trực tiếp hai hình để kiểm tra chữ/số.
- Registry trỏ comparison mới outputs/runs/20261022T145033Z_comparison_c83937e3 và optimizer results mới outputs/runs/20261022T145040Z_optimizer_results_da3a081a. Tất cả CSV chung khớp bản trước. SHA256 của metrics, checkpoint và test_predictions ở cả ba model và refit giữ nguyên; bằng chứng tại outputs/runs/20261022T145033Z_audit_visual_refresh_fa060532/unchanged_model_evidence.json.
- Roadmap hai notebook liên kết file thực, ghi đủ src.workflow và hàm tính toán phía sau; tách ý nghĩa train/view, setup chưa sinh artifact, refit wrapper/model/results là các run riêng. Xác thực file và thuộc tính/hàm bằng import trước khi chạy notebook.
- Thêm kiểm tra MODE trong notebook 01; comparison_stage ở view từ chối bảng so sánh không thuộc các model run đang chọn. Run ID vẫn đối chiếu được khi chuyển Windows/Linux/Kaggle. Comparison kiểm tra thêm weight_decay/momentum/scheduler và sample indices khi có.
- Sửa CLI --config: model_stage chuyển cấu hình kiến trúc vào run_model; metadata lưu đúng in_features/hidden_dims. Test thực sự train fixture với hidden_dims=[32], lưu rồi load checkpoint đúng kiến trúc. Benchmark FashionMNIST không thay đổi.
- Metadata tuning ghi search/sampler/pruner thực tế thay vì literals. Ngân sách và thuật toán thí nghiệm không thay đổi.
- Boxplot feature mới ghi nhãn class 0–9. Hình của preparation run lịch sử vẫn nguyên bản; notebook/walkthrough giải thích vị trí 1–10 tương ứng class 0–9 để không đọc sai. Không sửa ảnh hoặc manifest của run đã hoàn thành.
- Sửa đoạn walkthrough còn nhắc F1 heatmap, ROC/PR micro-average, phần readback cuối notebook cũ; thay tuyên bố mọi kết quả từ một Kaggle run bằng nguồn artifact từng phần. README cấp repo/labs cập nhật registry và hướng dẫn hiện hành. Bỏ bốn import không sử dụng, giữ API.
- File/module audit: outputs/runs/20261022T145409Z_audit_file_references_6aad6c33/ (module_references.json, deletions.json, retained_files_rationale.json); kiểm tra 68 liên kết local không có liên kết hỏng. remaining_findings.json ghi trạng thái lúc phát hiện; trạng thái sau sửa ở finding_resolutions.json của run final checks.
- Hai notebook chạy view toàn bộ, không error output; bản executed và log ở outputs/runs/20261022T145250Z_audit_notebook_execution_6fcbd376/. Ghi chú lịch sử boxplot được bổ sung sau lượt thực thi, không đổi code cell.
- Kiểm chứng cuối: 47 passed, 1 deselected; bỏ test signature lịch sử vì nó train nhiều candidate toàn dataset. Optuna hiện hành đã được kiểm thử tích hợp bằng fixture. Ruff F401/F821 và git diff --check qua; log và finding_resolutions.json tại outputs/runs/20261022T145701Z_audit_final_checks_2707f5d0/.
- Giới hạn còn lại thuộc thiết kế thí nghiệm: 10 trial bằng 10 startup trials; chưa enqueue baseline, weight_decay=0 nằm ngoài search log; một seed chưa đủ kết luận thống kê; chưa kiểm chứng Kaggle end-to-end; requirements chưa là lockfile đầy đủ. Không tự thay ngân sách/tune lại trong lượt sửa trình bày và dọn file này.
- Thay đổi ở máy local, chưa commit/push; chưa tạo PDF.


## Bổ sung loss/accuracy refit và save/load cuối notebook

- Description ghi yêu cầu trước triển khai. Notebook 02 hiển thị learning_curves.png và history.csv của FINAL_RUN ngay sau search/refit: loss và accuracy train/validation theo epoch.
- Cell cuối save_load_model xuất best.pt + config kiến trúc, tạo instance mới, load state_dict ở eval mode, so sánh xác suất trước/sau trên 16 ảnh test. Trả về SAVED_MODEL_RUN và LOADED_MODEL; mỗi lần chạy lưu log, environment, status, manifest, model_architecture.txt, roundtrip.json và roundtrip_predictions.npz.
- Chạy với checkpoint thật, không train lại. Lượt đầu phát hiện metadata run cũ thiếu in_features/hidden_dims; đã bổ sung fallback tương thích với loader hiện hành, không sửa artifact gốc. Run thất bại vẫn lưu: 20261022T151013Z_model_save_load_6a1fdd28.
- Kiểm chứng thành công: outputs/runs/20261022T151050Z_model_save_load_34645627/roundtrip.json; 16 ảnh, max_probability_error=0, predicted_labels_unchanged=true. Roadmap, README và walkthrough đồng bộ; chưa tạo PDF.


## Làm rõ ba optimizer và test sau load model

- Thêm plot_best_optimizer_histories trong tuning.py: bốn panel train/validation loss/accuracy, mỗi panel ba đường SGD/RMSprop/Adam từ best completed trial của từng optimizer. Không chạy lại search hoặc giả lập ba refit. Run hình: outputs/runs/20261022T151640Z_best_optimizer_learning_curves_cbe88b2e; CSV giữ best parameters, trial number, toàn history.
- save_load_model đánh giá chính instance vừa restored trên toàn test; lưu test_metrics.json, classification_report.json, test_predictions.npz, loaded_model_test_samples.csv/png. Lưới chọn tối đa 8 đúng/8 sai theo index và bổ sung nếu thiếu, có true/predicted/probability. Notebook hiển thị metric, bảng và hình.
- Chạy notebook đầy đủ không training mới: outputs/runs/20261022T151630Z_three_optimizer_curves_loaded_test_validation_9ecd79e7. Save/load+full test thực: outputs/runs/20261022T151641Z_model_save_load_2f8d78d9; 10.000 mẫu, accuracy 0,8914, Macro F1 0,891219120478717, loss 0,30367913842201233; roundtrip 16 mẫu sai lệch 0.
- Đã xem trực tiếp hình ba optimizer và lưới dự đoán; description/roadmap/README/walkthrough cập nhật. Regression kiểm tra history ba optimizer và save/load/full-test/export trên fixture ở tests/test_tuning.py và tests/test_workflow.py. Không tạo PDF, không xóa file.

- Theo phản hồi không thấy FE trong notebook EDA: thêm mục 2.1 hiển thị source thực của features.py và 10 hàng train_features.csv với đủ 10 cột feature, index, label. Cell tạo run feature_code_preview giữ feature_source.txt, train_feature_preview.csv, log/status/manifest. Notebook thực thi view để lưu output; không train lại hoặc sửa dữ liệu.

- Sửa lệch roadmap notebook EDA: trước đây chạy gộp prepare_data rồi in source/CSV và dồn hình cuối; nay trình bày đủ 7 giai đoạn với output tại chỗ. Backend chạy gộp được giải thích rõ, không giả định mỗi cell tạo một giai đoạn độc lập. FE cell thực thi extractor trên 10 ảnh thật, kiểm tra CSV, nhãn/index, scaler và NPZ; giữ snapshot source, extracted_features.csv, verification.json cùng log/manifest trong run riêng. Bản notebook trước sửa được lưu trong run eda_roadmap_restructure.

## Gỡ archive — 22/10/2026

- Xóa toàn bộ labs/lab01/archive/ (local và GitHub) theo yêu cầu người dùng. Không có source/test/config nào import hoặc đọc archive; chỉ tài liệu và docstring tham chiếu.
- Cập nhật tham chiếu trong README, walkthrough, description, notebooks/lab01.ipynb, docstring scripts/eval.py và scripts/infer.py sang lịch sử Git. Các mục nhật ký phía trên nhắc tới archive được giữ nguyên như lịch sử.
- Khôi phục khi cần: `git checkout e0fd041 -- labs/lab01/archive`.

## Dọn module, config và file ma — 22/10/2026

- Xóa 3 module legacy chỉ còn được gọi từ test cũ: `src/datasets/classification.py`, `src/training/experiments.py`, `src/utils/metrics.py`.
- Xóa 3 file config YAML cũ không thuộc pipeline `workflow.json`: `config/config.yaml`, `config/augmentations.yaml`, `config/experiments.yaml`.
- Xóa notebook điều hướng trống `notebooks/lab01.ipynb`, các file `outputs/predictions/*.png`, `outputs/logs/training_history.json` và thư mục local `outputs/checkpoints/`. Giữ nguyên `outputs/eda/` (được walkthrough tham chiếu) và toàn bộ `outputs/runs/`.
- Loại bỏ các hàm/class/hằng số không còn dùng trong `src/datasets/base.py` (`TransformedSubset`), `src/datasets/eda.py` (`verify_pipeline_handoff`), `src/training/trainer.py` (`train_model`, wrappers `train_one_epoch`/`evaluate`), `src/utils/helpers.py` (`NUM_CHANNELS`, `NUM_CLASSES`, `_resolve_normalization`, `CHECKPOINTS_DIR`, `PREDICTIONS_DIR`, `AUGMENTATIONS_CONFIG_PATH`, `load_yaml_config`), `src/utils/plots.py` (`unnormalize`, `plot_confusion_matrix`, `plot_prediction_grid`, `plot_optimizer_comparison`, `plot_roc_curves_fig`) và tinh gọn các file `__init__.py`.
- Cập nhật test suite (`39 passed` với `MPLBACKEND=Agg`) và cây thư mục trong `README.md`.
