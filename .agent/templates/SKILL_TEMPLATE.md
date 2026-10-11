# SKILL_TEMPLATE.md — Khung chuẩn để soạn một file SKILL.md

File này là khuôn mẫu nội bộ. Mọi `SKILL.md` trong `.agent/skills/` phải tuân theo đúng cấu trúc 9 mục dưới đây để đảm bảo agent có thể đọc, định tuyến và thực thi nhất quán.

---

## Quy ước bắt buộc

- Ngôn ngữ: tiếng Việt. Giữ nguyên thuật ngữ kỹ thuật tiếng Anh (DataLoader, loss, checkpoint, leakage, baseline, metric...).
- Không emoji, không icon, không ký tự vẽ khung.
- Không chèn code vào file markdown. Mô tả bằng văn xuôi và bảng.
- Mỗi file phải bắt đầu bằng frontmatter YAML gồm đúng 2 khóa: `name` và `description`.
- `name` phải trùng khớp tuyệt đối với tên thư mục chứa file.
- `description` phải là một câu tiếng Việt mô tả mục đích và phạm vi kích hoạt của skill.

---

## Cấu trúc 9 mục chuẩn

### Phần frontmatter
Hai khóa `name` và `description` nằm giữa hai dòng `---`.

### 1. Tổng quan
Từ 2 đến 3 đoạn văn. Nêu rõ: skill này giải quyết vấn đề gì, triết lý cốt lõi đằng sau nó là gì, và vì sao nó quan trọng trong bối cảnh DS & AI. Kết thúc bằng một câu nêu nguyên lý then chốt.

### 2. Khi nào kích hoạt (Triggers)
Chia làm hai phần:
- **Dấu hiệu kích hoạt:** liệt kê các tình huống, cụm từ, hoặc trạng thái dự án khiến skill này phải được gọi.
- **Điều kiện tiên quyết:** liệt kê những gì phải có sẵn trước khi skill này chạy (ví dụ: đã có spec, đã pass test 1-batch, đã có checkpoint).

### 3. Hợp đồng Input / Output
Bảng hai cột:
- **Input (nhận vào):** những gì skill cần để bắt đầu.
- **Output (trả ra):** những gì skill phải tạo ra khi kết thúc.

### 4. Quy trình chi tiết
Phần thân chính. Trình bày theo các bước hoặc các giai đoạn có thứ tự. Mỗi bước nêu rõ: hành động cần làm, lý do kỹ thuật, và kết quả kỳ vọng. Có thể dùng bảng nếu nhiều nhánh song song.

### 5. Deliverables
Liệt kê tường minh các artifact phải sinh ra (file, bảng, hình, báo cáo, ghi chú). Mỗi deliverable nêu rõ vị trí lưu.

### 6. Bàn giao (Handoff)
Bảng hai cột:
- **Skill trước (upstream):** skill nào thường chạy trước và bàn giao gì cho skill này.
- **Skill sau (downstream):** skill nào thường nhận bàn giao từ skill này và nhận gì.

### 7. Chế độ thất bại thường gặp (Failure Modes)
Liệt kê các lỗi đặc thù của riêng skill này, kèm dấu hiệu nhận biết và cách phòng tránh.

### 8. Definition of Done
Danh sách tiêu chí đo lường được, dạng tick-box, xác nhận skill đã hoàn thành đúng nghĩa.

### 9. Chống biện minh (Anti-Rationalization)
Danh sách các "lý lẽ ngụy biện" đi kèm lý do bác bỏ, theo định dạng: trích dẫn ngụy biện, mũi tên, lý do bác bỏ.

---

## Tiêu chí chất lượng một file SKILL.md đạt chuẩn

- Có đủ 9 mục, đúng thứ tự.
- Mỗi mục đều có nội dung cụ thể, không để trống hoặc viết chung chung.
- Độ dài tối thiểu hợp lý để bao phủ hết 9 mục (thường từ 90 dòng trở lên).
- Không có câu văn xuôi tiếng Anh (trừ tên skill, tên thư viện, và thuật ngữ kỹ thuật).
- Không có emoji, icon, hoặc ký tự vẽ khung.
