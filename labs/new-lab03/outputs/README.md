# Outputs — thí nghiệm 3 epoch đã xác minh

Model run: `20261104T110654Z_frozen_trial_00eccdd1`. Hoàn thành3epoch, checkpoint epoch3, validation threshold0,5. Test accuracy82,18%, Binary F182,11%; không thay đổi số đo hoặc bytes raw run.

- [Published report](published/index.md): toàn bộ hình/bảng/log, nguồn và SHA256.
- [EDA/feature preparation](preparation/index.md): quality, nhãn/độ dài, tokenization, TF-IDF và augmentation.
- [Model selection](latest_report.json), [preparation selection](latest_preparation.json).
- `runs/`: năm run nguồn đã kiểm tra manifest; checkpoint files được chủ động loại khỏi review ZIP.
- `architecture/`: module tree dạng text từ pinned configuration; không phải bằng chứng reload weights tại máy review.
- `imports/`: bundle manifests, archive hashes và import records.
- `preflight/`:95tests trên Kaggle trước training; source commit a759119.
- `verification/`: kiểm tra phần mềm sau khi nhập kết quả, khác kết quả huấn luyện.

Review ZIP không có weights. Reload consistency đã được script kiểm trên Kaggle và lưu reload_check.json; người review không load lại weights không có trong ZIP. Giữ checkpoint ZIP riêng để inference/evaluate về sau.
