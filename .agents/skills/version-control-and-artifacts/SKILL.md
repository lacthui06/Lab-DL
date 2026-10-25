---
name: version-control-and-artifacts
description: Quản lý phiên bản và artifact cho các dự án Data Science và Deep Learning, bao gồm quy ước commit, kỷ luật .gitignore, phân loại file thuộc Git và không thuộc Git, và quy trình đưa notebook đã chạy lên repository.
---

# version-control-and-artifacts (Bản DS & AI)

## 1. Tổng quan

Skill này quản lý phiên bản (version control) và artifact cho dự án DS & AI. Đây là lĩnh vực mà các dự án Machine Learning thường thất bại theo cách đặc thù: repository bị phình to vì checkpoint nặng, lịch sử Git vô nghĩa vì commit chung chung, secret bị lộ, và artifact quan trọng bị mất vì không ai quyết định rõ cái gì thuộc Git.

Vấn đề gốc rễ là dự án ML sinh ra hai loại file có bản chất hoàn toàn khác nhau: **file source** (code, cấu hình, tài liệu - nhỏ, cần theo dõi phiên bản) và **file artifact** (checkpoint, dataset thô, hình nặng - lớn, không nên theo dõi phiên bản). Trộn lẫn hai loại này là nguyên nhân của hầu hết vấn đề Git trong ML.

Skill này cũng làm rõ một hiểu lầm phổ biến: notebook đã chạy có output **nên** được đưa lên Git (vì output là bằng chứng kết quả), nhưng nó phải được thu hồi đúng cách từ môi trường thực thi, không phải qua script container.

Nguyên lý then chốt: Git theo dõi những gì nhỏ, có thể đọc được, và cần thiết để tái lập; mọi thứ lớn hoặc sinh ra được thì để ngoài Git và quản lý riêng.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Chuẩn bị commit và push code lên Git.
- Cần quyết định file nào thuộc Git và file nào không.
- Repository bị phình to do checkpoint hoặc dataset nặng.
- Cần đưa notebook đã chạy có output lên Git.
- Cần viết thông điệp commit mô tả rõ ràng.
- Phát hiện secret hoặc file nặng lỡ bị commit.
- Cần khôi phục hoặc dọn dẹp lịch sử Git.

**Điều kiện tiên quyết:**
- Đã có repository Git được khởi tạo và kết nối remote.
- Đã có cấu trúc thư mục lab theo chuẩn.
- Đã xác định được danh sách artifact sinh ra trong quá trình huấn luyện.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Tập thay đổi cần lưu vào Git | Commit với thông điệp mô tả rõ loại thay đổi |
| Danh sách artifact sinh ra | Phân loại rõ file thuộc Git và không thuộc Git |
| Notebook đã chạy từ môi trường thực thi | Notebook có output được đưa vào repository |
| Các loại file nặng mới phát sinh | Cập nhật `.gitignore` tương ứng |
| Trạng thái repository hiện tại | Lịch sử Git sạch, không secret, không file nặng |

## 4. Quy trình chi tiết

### Bước 1: Phân loại file thành hai nhóm
Trước khi commit, phân loại mọi thay đổi vào hai nhóm. **Nhóm thuộc Git:** source code, module cấu hình, test, tài liệu, notebook đã chạy có output, figure kết quả (`.png`), báo cáo kết quả (`.txt`, `.json`), và file cấu hình môi trường. **Nhóm không thuộc Git:** checkpoint mô hình (`*.pt`, `*.pth`, `*.onnx`, `*.pkl`), dataset thô (`*.csv`, `*.zip`, `*.parquet`, `*.h5`), và mọi file tạm. Việc phân loại rõ ràng ngay từ đầu ngăn chặn việc vô tình commit file nặng.

### Bước 2: Duy trì `.gitignore` chặt chẽ
Đảm bảo `.gitignore` bao phủ toàn bộ các loại file nặng: checkpoint, dataset, archive, log, và thư mục cache. Khi phát sinh loại file nặng mới trong quá trình làm việc, cập nhật `.gitignore` ngay. Nguyên tắc là giữ cấu trúc thư mục (qua `.gitkeep`) nhưng không đẩy nội dung nặng lên.

### Bước 3: Bảo vệ secret tuyệt đối
Trước mỗi commit, kiểm tra chắc chắn không có token, API key, mật khẩu, hay file cấu hình bí mật nào bị đưa vào staging. Secret phải được đọc qua biến môi trường hoặc Kaggle Secret, không bao giờ nằm trong source. Nếu nghi ngờ, rà soát diff của staging trước khi commit.

### Bước 4: Đưa notebook đã chạy lên Git đúng cách
Notebook đã chạy có output là bằng chứng quan trọng và nên được đưa lên Git. Tuy nhiên, notebook tương tác không nằm trên đĩa container của môi trường thực thi, nên phải thu hồi qua giao diện (tải notebook về máy hoặc lưu phiên bản), sau đó chép vào `labs/lab0X/notebooks/` rồi commit. Không kỳ vọng script chạy trong container tự đẩy được notebook.

### Bước 5: Viết thông điệp commit mô tả rõ
Mỗi thông điệp commit phải nêu rõ loại thay đổi (docs, feat, fix, refactor, test, chore) và nội dung ngắn gọn. Thông điệp rõ ràng giúp lịch sử Git có thể đọc được và truy nguyên khi cần. Tránh các thông điệp chung chung như "update" hoặc "fix bug".

### Bước 6: Rà soát trước khi commit
Chạy checklist `pre_commit.md` để xác nhận: không có secret, không có file nặng, không có emoji, không có file tạm, và chỉ những file dự kiến mới được thay đổi. Rà soát tổng quan bằng cách xem thống kê diff của staging trước khi commit.

### Bước 7: Xử lý khi lỡ commit sai
Nếu phát hiện secret hoặc file nặng đã bị commit nhưng chưa push, hoàn tác commit đó và loại file khỏi staging. Nếu đã push, cần xử lý ngay: thu hồi secret (đổi token), và gỡ file nặng khỏi lịch sử. Ghi lại sự cố và cách phòng tránh để không tái diễn.

### Bước 8: Đồng bộ với remote và xác nhận trạng thái sạch
Push lên remote và xác nhận trạng thái làm việc sạch. Kiểm tra rằng nhánh cục bộ khớp với nhánh remote. Đây là điểm kết thúc của một chu kỳ làm việc gọn gàng.

## 5. Deliverables

- `.gitignore` bao phủ đầy đủ các loại file nặng.
- Danh sách phân loại rõ file thuộc Git và không thuộc Git.
- Lịch sử commit với thông điệp mô tả rõ ràng.
- Notebook đã chạy có output được đưa vào `notebooks/`.
- Xác nhận không có secret trong bất kỳ commit nào.
- Trạng thái repository sạch và đồng bộ với remote.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `code-simplification` (cung cấp source đã refactor sạch) | `walkthrough-and-experiment-tracking` (nhận cam kết lịch sử thay đổi rõ ràng) |
| `kaggle-remote-execution` (cung cấp notebook đã chạy và artifact) | `documentation-and-adrs` (nhận repository sạch để viết báo cáo) |
| `environment-and-reproducibility` (cung cấp bản ghi phiên bản để commit) | `using-ds-ai-skills` (báo cáo trạng thái hoàn tất chu kỳ) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Commit checkpoint nặng:** Đẩy file `.pt` lên Git làm repository phình to vĩnh viễn. Dấu hiệu: repository tăng kích thước đột ngột. Phòng tránh: `.gitignore` bao phủ checkpoint và rà soát staging trước commit.
- **Lộ secret trong commit:** Token bị đẩy lên Git công khai. Dấu hiệu: thấy token trong diff hoặc trong notebook output. Phòng tránh: dùng Secret, rà soát diff, và không in token ra output.
- **Notebook không được đẩy lên:** Kỳ vọng script container tự đẩy notebook. Dấu hiệu: chỉ output lên Git, notebook thiếu output. Phòng tránh: thu hồi notebook qua giao diện và commit thủ công.
- **Thông điệp commit vô nghĩa:** Lịch sử Git không đọc được, không truy nguyên được. Phòng tránh: tuân thủ quy ước loại thay đổi ở Bước 5.
- **File tạm lọt vào commit:** `__pycache__`, file nháp, file hệ điều hành bị đẩy lên. Phòng tránh: `.gitignore` và rà soát trước commit.
- **Đẩy file `.ipynb` chưa chạy:** Notebook không có output, mất bằng chứng kết quả. Phòng tránh: xác nhận notebook đã chạy đủ output trước khi commit.

## 8. Definition of Done

- [ ] `.gitignore` bao phủ checkpoint, dataset, archive, log, và cache.
- [ ] Đã phân loại rõ file thuộc Git và không thuộc Git.
- [ ] Không có secret trong bất kỳ file nào được commit.
- [ ] Notebook đã chạy có output đã được đưa vào `notebooks/`.
- [ ] Thông điệp commit mô tả rõ loại và nội dung thay đổi.
- [ ] Đã chạy checklist `pre_commit.md` và tất cả mục đều đạt.
- [ ] Repository đã push và trạng thái làm việc sạch.
- [ ] Nhánh cục bộ đồng bộ với nhánh remote.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Checkpoint nhỏ thôi, commit luôn cho tiện"* -> **Bác bỏ:** Checkpoint tích lũy qua nhiều run sẽ làm repository phình to nhanh chóng và vĩnh viễn nằm trong lịch sử Git. Không bao giờ commit checkpoint.
- [X] *"Token chỉ dùng một lần, để trong notebook cũng được"* -> **Bác bỏ:** Token trong notebook sẽ được lưu và có thể bị đẩy lên Git công khai. Luôn dùng biến môi trường hoặc Secret.
- [X] *"Script tự đẩy notebook lên rồi, không cần làm gì"* -> **Bác bỏ:** Notebook tương tác không nằm trên đĩa container. Script chỉ đẩy được file trên đĩa. Phải thu hồi notebook qua giao diện.
- [X] *"Commit 'update' cho nhanh, đọc lại lịch sử cũng hiểu"* -> **Bác bỏ:** Ba tháng sau sẽ không ai hiểu commit "update" thay đổi cái gì. Thông điệp mô tả rõ là bắt buộc để truy nguyên.
- [X] *"File tạm để đó, commit sau dọn cũng được"* -> **Bác bỏ:** File tạm một khi đã vào lịch sử Git thì rất khó gỡ sạch. Ngăn chặn từ đầu bằng `.gitignore`.
- [X] *"Đã lỡ push secret rồi, thôi để vậy"* -> **Bác bỏ:** Secret đã lộ phải được thu hồi ngay lập tức (đổi token) và gỡ khỏi lịch sử. Không được bỏ mặc.
