# Checklist — Trước khi Commit / Push lên Git

Danh sách kiểm tra bắt buộc phải tick đủ trước mỗi lần commit. Checklist này ngăn chặn các lỗi Git phổ biến: lộ secret, đẩy file nặng, và commit thiếu artifact.

---

## 1. An toàn bảo mật (Security)
- [ ] Không có token, API key, mật khẩu, hay secret nào bị hard-code trong bất kỳ file nào.
- [ ] Không có file `.env`, `credentials.json`, hay `kaggle.json` bị đưa vào staging.
- [ ] Mọi secret đều được đọc qua biến môi trường hoặc Kaggle Secret, không nằm trong source.
- [ ] Đã kiểm tra `git diff --staged` để chắc chắn không có dòng nào chứa chuỗi giống token.

## 2. Kỷ luật file nặng (Artifact Hygiene)
- [ ] Không có checkpoint mô hình (`*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.pkl`) bị commit.
- [ ] Không có dataset thô hoặc bảng dữ liệu nặng (`*.csv`, `*.zip`, `*.parquet`, `*.h5`) bị commit (ngoại trừ các bảng thống kê tổng hợp EDA nhỏ trong `outputs/eda/tables/*.csv` đã được whitelist trong `.gitignore`).
- [ ] Các file trong `outputs/` được commit chỉ là figure (`.png`), báo cáo (`.txt`), bảng kết quả (`.json`), và bảng tổng hợp EDA nhỏ (`outputs/eda/tables/*.csv`) — không phải trọng số.
- [ ] `.gitignore` đã bao phủ đúng các loại file nặng mới phát sinh.

## 3. Toàn vẹn nội dung (Content Integrity)
- [ ] Không có emoji, icon, hay ký tự vẽ khung trong bất kỳ file văn bản nào.
- [ ] Không có file tạm, file nháp, hay `__pycache__` bị đưa vào commit.
- [ ] Comment code và workflow giữ tiếng Việt; tài liệu agent giữ đúng quy ước ngôn ngữ.
- [ ] Tất cả đường dẫn tương đối trong tài liệu vẫn đúng sau khi di chuyển file.

## 4. Đồng bộ tài liệu sống (Living Document Sync)
- [ ] Nếu có thay đổi kết quả thí nghiệm, `walkthrough_lab_0X.md` đã được cập nhật với số liệu mới.
- [ ] Không tạo file `SPEC.md` hay `REPORT.md` riêng lẻ.
- [ ] Bảng so sánh (Flow Comparison Matrix) đã được cập nhật nếu có run mới.

## 5. Xác nhận cuối
- [ ] `git status` chỉ hiển thị đúng những file dự kiến thay đổi.
- [ ] Thông điệp commit mô tả rõ loại thay đổi (docs, feat, fix, refactor).
- [ ] Đã chạy `git diff --staged --stat` để rà soát tổng quan trước khi commit.
