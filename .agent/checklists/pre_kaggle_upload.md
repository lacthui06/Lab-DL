# Checklist — Trước khi đẩy lên Kaggle GPU

Danh sách kiểm tra bắt buộc trước khi chuyển thực thi từ máy cục bộ lên Kaggle GPU. Mục đích là tránh đốt GPU quota vào một pipeline còn lỗi.

---

## 1. Xác thực cục bộ (Local Verification)
- [ ] Toàn bộ 5 bài test TDD đã pass cục bộ (`pytest` xanh 100%).
- [ ] Bài test 1-batch sanity overfit đã pass: loss giảm về sát 0 và accuracy đạt 100% trên batch nhỏ.
- [ ] Không còn chạy full nhiều epoch trên CPU cục bộ.

## 2. Tái lập (Reproducibility)
- [ ] Random seed đã được cố định cho `random`, `numpy`, `torch`, và CUDA.
- [ ] Đã xác nhận chạy lại hai lần cho ra cùng kết quả.
- [ ] Phiên bản thư viện (PyTorch, Torchvision) đã được ghi nhận và khớp giữa local và Kaggle.
- [ ] Yêu cầu Python thật (`Requires-Python` suy ra từ các pin) đã được ghi, không chỉ phiên bản đã chạy.
- [ ] Phiên bản trình thông dịch Python đã được ghi nhận (comment trong `requirements.txt`), kèm chênh lệch giữa local và Kaggle nếu có.

## 3. Đường dẫn và môi trường (Paths & Environment)
- [ ] Đường dẫn dữ liệu dùng đúng `/kaggle/input/` (chỉ đọc) và không cố ghi vào đó.
- [ ] Mọi mục tiêu ghi (checkpoint, log, figure) đều trỏ về `/kaggle/working/`.
- [ ] Notebook có cơ chế tự động clone `src/` nếu chưa được gắn như Kaggle Dataset.
- [ ] Đã xác nhận bật Internet trong cài đặt notebook nếu cần tải trọng số pretrained.

## 4. Tài nguyên (Resources)
- [ ] `num_workers` của DataLoader được đặt ở mức an toàn (2 hoặc 0) để tránh bus error do `/dev/shm`.
- [ ] `pin_memory` được bật khi chạy trên GPU.
- [ ] Batch size và AMP FP16 đã được cân nhắc để nằm trong ngân sách VRAM 15GB.

## 5. Đồng bộ kết quả (Artifact Harvesting)
- [ ] Đã có kế hoạch đồng bộ artifact: qua GitHub Secret token hoặc tải `outputs.zip`.
- [ ] Đã hiểu rằng file `.ipynb` tương tác không nằm trên đĩa container; phải tải qua File -> Download Notebook hoặc Save Version.
- [ ] Đã chuẩn bị sẵn quy trình cập nhật `walkthrough_lab_0X.md` với số liệu thu về.

## 6. Xác nhận cuối
- [ ] Đã commit và push phiên bản code mới nhất lên Git trước khi mở phiên Kaggle.
- [ ] Đã ghi lại cấu hình run hiện tại (hyperparameter, kiến trúc) để đối chiếu sau.
