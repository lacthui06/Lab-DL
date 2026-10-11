---
name: kaggle-remote-execution
description: Quản lý quy trình huấn luyện trên GPU từ xa của Kaggle, bao gồm quản lý secret an toàn, xử lý đường dẫn /kaggle/input và /kaggle/working, đồng bộ artifact về máy cục bộ, và thu hồi notebook đã chạy có đầy đủ output.
---

# kaggle-remote-execution (Bản DS & AI)

## 1. Tổng quan

Skill này quản lý toàn bộ vòng đời thực thi trên Kaggle GPU - môi trường huấn luyện chính của dự án theo Giao thức Zero Waste. Nó giải quyết một khoảng trống quan trọng: trong khi các skill khác tập trung vào code và mô hình, skill này tập trung vào hạ tầng và vận hành - nơi nhiều vấn đề thực tế phát sinh mà không liên quan đến thuật toán.

Các vấn đề điển hình mà skill này giải quyết: secret bị lộ khi push code, ghi nhầm vào thư mục chỉ đọc, DataLoader bị kill do cạn `/dev/shm`, không tìm thấy kết quả sau khi chạy xong, và đặc biệt là hiểu lầm về việc file notebook tương tác nằm ở đâu.

Điểm gây nhầm lẫn lớn nhất cần làm rõ: trong phiên Kaggle tương tác, file `.ipynb` được quản lý bởi ứng dụng web của Kaggle và giữ trạng thái trong trình duyệt, **không** nằm trên đĩa container tại `/kaggle/working`. Vì vậy, các script chạy trong container không thể đọc được file notebook đang mở trên trình duyệt. Chỉ những file mà code Python ghi ra đĩa (hình ảnh, checkpoint, log) mới tồn tại thực sự trên máy ảo.

Nguyên lý then chốt: mọi thứ ghi ra đĩa phải nằm trong `/kaggle/working`, và notebook đã chạy phải được thu hồi qua giao diện Kaggle, không qua script container.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Chuẩn bị chuyển huấn luyện từ local lên Kaggle GPU.
- Cần đồng bộ kết quả từ Kaggle về máy cục bộ.
- Gặp lỗi runtime đặc thù Kaggle (bus error, permission denied, connection refused).
- Cần đưa notebook đã chạy có output lên Git.
- Cần cấu hình secret để push code từ trong Kaggle.
- Cần hiểu tại sao kết quả không được đẩy về sau khi chạy xong.

**Điều kiện tiên quyết:**
- Pipeline đã pass toàn bộ test TDD cục bộ (xem checklist `pre_kaggle_upload.md`).
- Đã có tài khoản Kaggle và quyền bật GPU.
- Đã hiểu cấu trúc thư mục của lab trên Git.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Pipeline đã pass test cục bộ | Job huấn luyện chạy trên Kaggle GPU |
| Cấu hình secret (token truy cập) | Cơ chế đồng bộ an toàn không lộ secret |
| Đường dẫn dữ liệu trên Kaggle | Code dùng đúng đường dẫn input/working |
| Yêu cầu về artifact cần thu hồi | Artifact được đưa về `outputs/` cục bộ |
| Notebook tương tác đã chạy | File `.ipynb` đầy đủ output được đưa lên Git |

## 4. Quy trình chi tiết

### Bước 1: Xác thực cục bộ trước khi lên Kaggle
Chạy checklist `pre_kaggle_upload.md` để đảm bảo test TDD pass, seed đã cố định, và đường dẫn đã đúng. Việc này ngăn chặn việc đốt GPU quota vào một pipeline còn lỗi cơ bản.

### Bước 2: Cấu hình đường dẫn đúng chuẩn Kaggle
Phân biệt rõ hai vùng hệ thống file: `/kaggle/input/` là chỉ đọc (chứa dataset được gắn), và `/kaggle/working/` là vùng ghi được (nơi lưu checkpoint, log, hình). Mọi mục tiêu ghi phải trỏ về `/kaggle/working/`. Code cần có cơ chế tự động thích ứng đường dẫn giữa môi trường local và Kaggle.

### Bước 3: Cấu hình DataLoader an toàn cho môi trường Kaggle
Đặt `num_workers` ở mức thấp (2 hoặc 0) để tránh làm cạn phân vùng bộ nhớ chia sẻ `/dev/shm`, nguyên nhân gây bus error và kill DataLoader đột ngột. Bật `pin_memory` khi chạy trên GPU để tăng tốc truyền dữ liệu.

### Bước 4: Quản lý secret an toàn
Khi cần push code từ trong Kaggle lên Git, đọc token truy cập qua Kaggle Secret (UserSecretsClient), tuyệt đối không hard-code token vào notebook. Bất kỳ đoạn code nào nhúng token phải đảm bảo token không xuất hiện trong output cell, vì output cell sẽ được lưu vào file notebook và có thể bị đẩy lên Git.

### Bước 5: Ghi artifact ra đĩa trong lúc huấn luyện
Đảm bảo code Python ghi tường minh mọi artifact ra `/kaggle/working/`: checkpoint tốt nhất, hình dự đoán, confusion matrix, đường cong huấn luyện, và bảng kết quả. Chỉ những file được ghi ra đĩa theo cách này mới tồn tại sau khi phiên kết thúc và mới có thể đồng bộ về.

### Bước 6: Đồng bộ artifact về máy cục bộ
Chọn một trong hai cơ chế: đồng bộ qua Git bằng Secret token, hoặc đóng gói `outputs/` thành file nén rồi tải thủ công. Sau khi đồng bộ, đưa artifact về đúng vị trí `labs/lab0X/outputs/` cục bộ.

### Bước 7: Thu hồi notebook đã chạy có output
Đây là bước quan trọng và hay bị hiểu sai. File `.ipynb` tương tác không nằm trên đĩa container. Để đưa notebook đã chạy có đầy đủ output lên Git, phải thu hồi qua giao diện Kaggle: dùng chức năng tải notebook về máy (File -> Download Notebook) hoặc lưu phiên bản trên Kaggle (Save Version). Sau đó chép file đã tải vào `labs/lab0X/notebooks/` và commit lên Git.

### Bước 8: Cập nhật tài liệu sống với số liệu thu về
Sau khi thu hồi artifact, đọc các con số thực nghiệm và cập nhật Section 2 của `walkthrough_lab_0X.md`. Ghi lại cấu hình run và chênh lệch metric so với run trước.

## 5. Deliverables

- Job huấn luyện hoàn tất trên Kaggle GPU với log đầy đủ.
- Checkpoint tốt nhất được lưu và đồng bộ về `outputs/checkpoints/`.
- Hình trực quan (dự đoán, confusion matrix, đường cong) được đồng bộ về `outputs/`.
- Bảng kết quả thực nghiệm được đồng bộ về `outputs/`.
- File `.ipynb` đầy đủ output được đưa vào `notebooks/` và commit.
- Section 2 của `walkthrough_lab_0X.md` được cập nhật với số liệu mới.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `test-driven-development` (xác nhận pipeline đủ điều kiện chạy) | `walkthrough-and-experiment-tracking` (nhận artifact và số liệu để ghi changelog) |
| `environment-and-reproducibility` (cung cấp seed và phiên bản môi trường) | `version-control-and-artifacts` (nhận notebook đã chạy để commit) |
| `performance-optimization` (tối ưu DataLoader và AMP trước khi chạy) | `documentation-and-adrs` (nhận số liệu cuối để viết báo cáo) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Ghi vào `/kaggle/input/`:** Cố tạo file trong vùng chỉ đọc gây lỗi permission denied. Dấu hiệu: lỗi hệ thống file khi lưu checkpoint. Phòng tránh: trỏ mọi ghi về `/kaggle/working/`.
- **Bus error do `num_workers` cao:** DataLoader bị kill đột ngột khi cạn `/dev/shm`. Dấu hiệu: tiến trình dừng giữa chừng không rõ nguyên nhân. Phòng tránh: đặt `num_workers` bằng 2 hoặc 0.
- **Không tìm thấy notebook sau khi chạy:** Kỳ vọng file `.ipynb` nằm trên đĩa container. Dấu hiệu: lệnh tìm file notebook trả về rỗng. Phòng tránh: hiểu rằng notebook do trình duyệt quản lý, phải thu hồi qua giao diện Kaggle.
- **Chỉ output được push, notebook thì không:** Script container chỉ thấy file trên đĩa nên chỉ push được `outputs/`. Phòng tránh: dùng Bước 7 để thu hồi notebook thủ công.
- **Lộ token trong output cell:** Token bị in ra output rồi lưu vào notebook và đẩy lên Git. Phòng tránh: không in token, dùng Secret và đảm bảo không xuất hiện trong output.
- **Quên bật Internet:** Không tải được trọng số pretrained. Dấu hiệu: connection refused khi tải model. Phòng tránh: bật Internet trong cài đặt notebook.

## 8. Definition of Done

- [ ] Đã chạy checklist `pre_kaggle_upload.md` và tất cả mục đều đạt.
- [ ] Mọi mục tiêu ghi đều trỏ về `/kaggle/working/`.
- [ ] `num_workers` và `pin_memory` được cấu hình phù hợp.
- [ ] Secret được đọc qua Kaggle Secret, không hard-code.
- [ ] Token không xuất hiện trong bất kỳ output cell nào.
- [ ] Artifact đã được đồng bộ về `outputs/` cục bộ.
- [ ] Notebook đã chạy có output đã được thu hồi và commit lên Git.
- [ ] Section 2 của `walkthrough_lab_0X.md` đã cập nhật số liệu mới.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Cứ chạy thẳng trên Kaggle, lỗi thì sửa sau"* -> **Bác bỏ:** Vi phạm Giao thức Zero Waste. Mỗi lần chạy lỗi trên GPU từ xa là một lần đốt quota. Test cục bộ trước là bắt buộc.
- [X] *"Nhúng token trực tiếp cho nhanh, không ai thấy đâu"* -> **Bác bỏ:** Token nhúng trong notebook sẽ được lưu vào file và có thể bị đẩy lên Git công khai. Luôn dùng Kaggle Secret.
- [X] *"Script sẽ tự đẩy notebook lên Git cùng output"* -> **Bác bỏ:** File notebook tương tác không nằm trên đĩa container. Script chỉ đẩy được những gì ghi ra đĩa. Phải thu hồi notebook qua giao diện Kaggle.
- [X] *"Đặt `num_workers` cao cho nhanh, Kaggle mạnh mà"* -> **Bác bỏ:** Kaggle giới hạn dung lượng `/dev/shm`, đặt cao gây bus error và kill DataLoader. Đặt thấp là an toàn và đủ nhanh.
- [X] *"Lưu checkpoint vào `/kaggle/input/` cho gần dataset"* -> **Bác bỏ:** `/kaggle/input/` là chỉ đọc. Mọi ghi phải vào `/kaggle/working/`.
- [X] *"Kết quả trên Kaggle cứ để đó, cần thì mở lại xem"* -> **Bác bỏ:** Phiên Kaggle có thời hạn và sẽ bị xóa. Artifact phải được đồng bộ về cục bộ ngay sau khi chạy xong.
