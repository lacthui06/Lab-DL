---
name: planning-and-task-breakdown
description: Phân rã đặc tả kỹ thuật thành các giai đoạn pipeline nguyên tử, có thể kiểm thử, với thứ tự phụ thuộc, hợp đồng rõ ràng và tiêu chí nghiệm thu đã xác thực cho các dự án Data Science và AI.
---

# planning-and-task-breakdown (Bản DS & AI)

## 1. Tổng quan

Skill này chuyển hóa một đặc tả kỹ thuật thành một **Kế hoạch triển khai Machine Learning** có thể hành động được. Nó phân rã các workflow phức tạp thành những giai đoạn module độc lập có thể kiểm thử riêng, trước khi kết nối chúng lại thành một pipeline end-to-end.

Nguyên lý: một pipeline lớn không thể kiểm thử được. Một pipeline được chia thành các module nhỏ, mỗi module có input/output rõ ràng, thì mỗi phần đều có thể xác thực độc lập, và lỗi được cô lập về đúng module gây ra nó.

Skill cũng xác lập thứ tự phụ thuộc: không thể huấn luyện trước khi có dữ liệu sạch, không thể đánh giá trước khi có checkpoint. Việc sắp xếp sai thứ tự gây ra việc phải quay lại làm lại nhiều lần.

Nguyên lý then chốt: mỗi task phải có input, output, và tiêu chí nghiệm thu tường minh trước khi bắt đầu thực hiện.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Ngay sau khi hoàn thành `spec-driven-development`.
- Trước khi tạo hoặc chỉnh sửa gói Python module trong `src/`.
- Cần chia công việc thành các bước có thể kiểm thử.
- Cần xác định thứ tự phụ thuộc giữa các thành phần pipeline.

**Điều kiện tiên quyết:**
- Đã có Section 1 của `walkthrough_lab_0X.md` (đặc tả đầy đủ).
- Đã biết cấu trúc thư mục chuẩn của lab.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Đặc tả kỹ thuật từ Section 1 | Kế hoạch 6 giai đoạn với các task nguyên tử |
| Cấu trúc thư mục lab chuẩn | Danh sách module cần tạo trong `src/` |
| Ràng buộc thời gian và compute | Thứ tự phụ thuộc giữa các task |
| | Tiêu chí nghiệm thu cho từng task |

## 4. Quy trình chi tiết

### Bước 1: Xác định các giai đoạn pipeline
Chia toàn bộ công việc thành 6 giai đoạn chuẩn: nạp dữ liệu và xác thực, tiền xử lý và đặc trưng, baseline model, mô hình lõi, đánh giá và phân tích lỗi, đóng gói module. Việc dùng khung 6 giai đoạn cố định đảm bảo không bỏ sót giai đoạn nào.

### Bước 2: Phân rã giai đoạn 1 - Nạp dữ liệu và Xác thực
Xác định các task: nạp dữ liệu thô và kiểm tra chiều kích thước, triển khai assertion về schema và kiểu dữ liệu, kiểm tra khoảng giá trị nhãn và ràng buộc null. Mỗi task phải có tiêu chí xác thực rõ ràng.

### Bước 3: Phân rã giai đoạn 2 - Tiền xử lý và Đặc trưng
Xác định các task: xử lý giá trị thiếu và ngoại lai, áp dụng mã hóa biến phân loại hoặc augmentation. Quy tắc bắt buộc: mọi thao tác fit biến đổi chỉ tính trên tập train, chỉ transform sang tập validation và test để ngăn rò rỉ dữ liệu.

### Bước 4: Phân rã giai đoạn 3 và 4 - Baseline và Mô hình lõi
Giai đoạn 3: triển khai baseline tối thiểu và ghi nhận metric benchmark. Giai đoạn 4: định nghĩa kiến trúc mô hình, xây dựng vòng lặp huấn luyện, tích hợp checkpointing và early stopping. Tách rõ baseline và mô hình lõi để so sánh được cải thiện.

### Bước 5: Phân rã giai đoạn 5 - Đánh giá và Phân tích lỗi
Xác định các task: đo metric cuối cùng trên tập test độc lập, kết xuất artifact trực quan (đường cong, confusion matrix), kiểm tra lát cắt mẫu bị phân loại sai để xác định chế độ thất bại.

### Bước 6: Phân rã giai đoạn 6 - Đóng gói và Module hóa
Xác định các task: tổ chức module tái sử dụng trong `src/`, đảm bảo ranh giới giao diện sạch và script có thể tái lập.

### Bước 7: Xác định thứ tự phụ thuộc và tiêu chí nghiệm thu
Sắp xếp các task theo thứ tự phụ thuộc và gán tiêu chí nghiệm thu cho từng task. Mỗi tiêu chí phải kiểm chứng được (ví dụ: shape tensor đúng, test pass, metric đạt ngưỡng). Không chuyển sang giai đoạn sau khi giai đoạn trước chưa đạt tiêu chí.

## 5. Deliverables

- Kế hoạch 6 giai đoạn với danh sách task nguyên tử.
- Sơ đồ thứ tự phụ thuộc giữa các task.
- Danh sách module cần tạo trong `src/`.
- Tiêu chí nghiệm thu cho từng task.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `spec-driven-development` (cung cấp đặc tả) | `test-driven-development` (nhận danh sách module để viết test) |
| `interview-me` (cung cấp ràng buộc) | `code-simplification` (nhận cấu trúc module đích) |
| | `environment-and-reproducibility` (nhận yêu cầu cấu hình) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Task quá lớn:** Một task bao gồm nhiều bước không thể kiểm thử riêng. Dấu hiệu: không biết task đã xong chưa. Phòng tránh: phân rã đến mức mỗi task có một tiêu chí nghiệm thu duy nhất.
- **Bỏ sót giai đoạn tiền xử lý:** Nhảy thẳng từ dữ liệu thô sang huấn luyện. Phòng tránh: tuân thủ khung 6 giai đoạn ở Bước 1.
- **Fit trên toàn bộ dữ liệu:** Vi phạm quy tắc chống rò rỉ ở Bước 3. Dấu hiệu: metric validation cao bất thường. Phòng tránh: chỉ fit trên tập train.
- **Không có tiêu chí nghiệm thu:** Task hoàn thành mơ hồ. Phòng tránh: gán tiêu chí kiểm chứng được ở Bước 7.

## 8. Definition of Done

- [ ] Đã chia công việc thành đủ 6 giai đoạn chuẩn.
- [ ] Mỗi giai đoạn đã được phân rã thành các task nguyên tử.
- [ ] Mỗi task có input và output tường minh.
- [ ] Đã xác định thứ tự phụ thuộc giữa các task.
- [ ] Mỗi task có tiêu chí nghiệm thu kiểm chứng được.
- [ ] Đã xác định danh sách module cần tạo trong `src/`.
- [ ] Quy tắc chống rò rỉ dữ liệu đã được ghi trong giai đoạn tiền xử lý.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Kế hoạch dài dòng, cứ code tuần tự từ đầu đến cuối"* -> **Bác bỏ:** Code tuần tự không có tiêu chí nghiệm thu từng bước khiến lỗi tích lũy và khó cô lập. Phân rã giúp phát hiện lỗi sớm.
- [X] *"Gộp tất cả vào một file cho gọn"* -> **Bác bỏ:** Một file lớn không thể kiểm thử từng phần và vi phạm cấu trúc module chuẩn. Phân rã theo 6 giai đoạn.
- [X] *"Fit scaler trên toàn bộ dữ liệu cho tiện"* -> **Bác bỏ:** Gây rò rỉ dữ liệu nghiêm trọng và cho metric ảo. Chỉ fit trên tập train.
- [X] *"Không cần tiêu chí nghiệm thu, nhìn kết quả là biết"* -> **Bác bỏ:** Không có tiêu chí thì không biết task đã đạt yêu cầu chưa, dẫn đến việc chuyển giai đoạn khi nền tảng còn lỗi.
