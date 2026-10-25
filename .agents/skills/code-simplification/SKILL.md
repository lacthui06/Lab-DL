---
name: code-simplification
description: Refactor và tinh gọn codebase Data Science và Deep Learning, chuyển code Jupyter Notebook khám phá thành gói Python module sẵn sàng production, loại bỏ biến toàn cục, hằng số hard-code và ký tự phi chuẩn.
---

# code-simplification (Bản DS & AI)

## 1. Tổng quan

Skill này refactor các script notebook khám phá, biến toàn cục rải rác, và logic tùy hứng thành **code Python sạch, module hóa, chuẩn production**. Nó chuẩn bị codebase cho việc nộp bài môn học hoặc tích hợp kỹ thuật, đồng thời đảm bảo tái lập chính xác kết quả thí nghiệm.

Điểm then chốt: refactor không được làm thay đổi kết quả. Một refactor thành công phải cho ra cùng loss và metric với code gốc khi chạy lại với seed cố định. Đây là ràng buộc bất biến (invariant) quan trọng nhất.

Skill cũng xử lý vấn đề train/serve skew bằng cách đảm bảo chỉ có một nguồn chân lý duy nhất cho logic tiền xử lý.

Nguyên lý then chốt: refactor để dễ đọc, dễ kiểm thử, dễ tái sử dụng, nhưng không thay đổi hành vi.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Dọn dẹp và tổ chức codebase sau khi huấn luyện thành công, trước khi nộp bài.
- Cần tách thành phần tái sử dụng từ notebook `.ipynb` sang file `.py` dưới `src/`.
- Phát hiện hyperparameter hard-code hoặc hằng số phân mảnh.
- Code có nhiều biến toàn cục và trạng thái rải rác.
- Cần loại bỏ dead code và import không dùng.

**Điều kiện tiên quyết:**
- Pipeline đã chạy đúng và cho kết quả hợp lệ.
- Đã có seed cố định để xác minh hành vi không đổi sau refactor.
- Đã biết cấu trúc module đích trong `src/`.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Notebook hoặc code khám phá lộn xộn | Module Python sạch trong `src/` |
| Danh sách hyperparameter và đường dẫn | File cấu hình tập trung |
| Logic tiền xử lý trùng lặp | Pipeline tiền xử lý thống nhất |
| Bằng chứng kết quả trước refactor | Bằng chứng kết quả không đổi sau refactor |

## 4. Quy trình chi tiết

### Bước 1: Loại bỏ hằng số và đường dẫn hard-code
Tập trung mọi hyperparameter, đường dẫn, và cài đặt thực thi vào các lớp cấu hình có cấu trúc hoặc file YAML config. Một dataclass cấu hình huấn luyện nên chứa: đường dẫn dữ liệu, batch size, learning rate, số epoch, và random seed. Việc tập trung hóa giúp thay đổi cấu hình mà không phải sửa code rải rác.

### Bước 2: Loại bỏ biến toàn cục
Thay thế trạng thái toàn cục kiểu notebook bằng các hàm thuần với chữ ký tường minh. Ví dụ: hàm nạp dữ liệu nhận đường dẫn và trả về DataFrame; hàm tiền xử lý nhận DataFrame và cờ báo train hay eval; hàm huấn luyện một epoch nhận model, loader và trả về loss. Hàm thuần dễ kiểm thử và không có tác dụng phụ ẩn.

### Bước 3: Duy trì nguồn chân lý duy nhất cho tiền xử lý
Không duy trì logic tiền xử lý trùng lặp cho huấn luyện và đánh giá. Đóng gói các biến đổi trong pipeline class thống nhất hoặc transforms để đảm bảo hành vi giống hệt nhau trong lúc inference. Đây là nguồn gốc của lỗi train/serve skew.

### Bước 4: Thêm type hint và docstring chính xác
Bao gồm chú thích kiểu và docstring mô tả chiều tensor để loại bỏ nhập nhằng. Docstring của hàm forward cần nêu rõ shape của tensor đầu vào và shape của logits đầu ra. Type hint giúp bắt lỗi sớm và làm code tự tài liệu hóa.

### Bước 5: Loại bỏ dead code và import không dùng
Dọn sạch các import không dùng, câu lệnh print nháp, và các cell thí nghiệm đã deprecated. Code chết gây nhầm lẫn và làm chậm việc đọc hiểu.

### Bước 6: Xóa script sinh code sau khi dùng xong
Script sinh code (generator notebook, scaffolder, patch script) là script tạo ra file khác chứ không trực tiếp phân tích dữ liệu. Sau khi output đã sinh, đã chạy sạch, và đã commit, phải xóa script đó.

Lý do: giữ cả generator và output tạo ra hai nguồn sự thật cho cùng một nội dung. Sửa notebook thì generator lệch; sửa generator thì notebook lệch; không có cơ chế nào phát hiện. Ngoài ra generator vẫn chiếm dung lượng và thời gian đọc trên máy người dùng dù không còn cần chạy.

Ngoại lệ duy nhất: khi output cần tái sinh định kỳ từ một nguồn thay đổi. Khi đó generator là nguồn sự thật, và output phải được đánh dấu rõ là generated (không sửa tay). Trước khi xóa, xác nhận output đã nằm trong Git để có thể phục hồi từ lịch sử.

### Bước 7: Notebook - code tối thiểu, output tối đa
Notebook là tài liệu trình bày, không phải nơi chứa logic. Hai quy tắc đi kèm nhau:

- **Code tối thiểu:** mỗi cell chỉ nên 3 đến 8 dòng gọi hàm từ `src/`. Logic dài, vòng lặp nhiều bước, hay hàm phụ trợ phải chuyển vào module. Cell trên 20 dòng là dấu hiệu logic đang nằm sai chỗ.
- **Output tối đa:** mỗi cell tính toán phải hiển thị kết quả ngay trong notebook. Hàm vẽ phải vừa ghi file vừa gọi `plt.show()`; hàm thống kê phải in bảng tóm tắt. Ghi file mà không hiển thị làm notebook mất chức năng chính: người đọc không thấy gì ngoài text log.

Nội dung giải thích trong markdown cell phải viết bằng tiếng Việt có dấu, như mọi tài liệu khác của dự án. Không hạ cấp sang tiếng Việt không dấu để né lỗi encoding; lỗi encoding phải chặn bằng hook, không bằng cách bỏ dấu.

### Bước 8: Áp đặt chính sách không Emoji nghiêm ngặt
Nghiêm cấm emoji và icon trang trí trong file source code, câu lệnh xuất terminal, comment, docstring, và markdown cell của notebook. Thể hiện mọi trạng thái bằng text kỹ thuật chuẩn (ví dụ `[INFO]`, `[SUCCESS]`, `[ERROR]`). Điều này ngăn ngừa sự cố `UnicodeEncodeError` trên các hệ điều hành khác nhau như Windows CP1252.

### Bước 9: Xác minh hành vi không đổi
Chạy lại các module đã refactor với seed cố định và so sánh loss và metric với kết quả trước refactor. Nếu kết quả khớp, refactor thành công. Nếu không khớp, có lỗi logic được đưa vào trong quá trình refactor và cần sửa.

## 5. Deliverables

- Module Python sạch trong `src/` với type hint và docstring đầy đủ.
- File cấu hình tập trung cho hyperparameter và đường dẫn.
- Pipeline tiền xử lý thống nhất dùng chung cho train và eval.
- Notebook có cell ngắn gọi `src/`, hiển thị đầy đủ output, markdown tiếng Việt có dấu.
- Không còn script sinh code đã dùng xong.
- Bằng chứng kết quả không đổi sau refactor.
- Xác nhận không còn emoji hoặc ký tự phi chuẩn.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| `planning-and-task-breakdown` (cung cấp cấu trúc module đích) | `api-and-interface-design` (nhận module sạch để đóng gói) |
| `debugging-and-error-recovery` (cung cấp code đã sửa lỗi) | `version-control-and-artifacts` (nhận code để commit) |
| | `documentation-and-adrs` (nhận cấu trúc code để viết tài liệu) |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Refactor làm thay đổi kết quả:** Vô tình thay đổi logic trong quá trình dọn dẹp. Dấu hiệu: metric sau refactor khác trước. Phòng tránh: xác minh ở Bước 9.
- **Vẫn còn hằng số hard-code:** Bỏ sót một số giá trị rải rác. Dấu hiệu: phải sửa nhiều nơi để đổi một hyperparameter. Phòng tránh: rà soát toàn bộ ở Bước 1.
- **Tiền xử lý trùng lặp:** Giữ hai bản logic cho train và eval. Dấu hiệu: kết quả inference khác kỳ vọng. Phòng tránh: thống nhất ở Bước 3.
- **Emoji lọt vào code:** Gây lỗi encoding trên Windows. Phòng tránh: tuân thủ chính sách ở Bước 8.
- **Để lại dead code:** Không dọn sạch import và cell cũ. Phòng tránh: rà soát ở Bước 5.
- **Hai nguồn sự thật:** Giữ cả generator và output nó sinh ra. Dấu hiệu: sửa một bên, bên kia không đổi theo. Phòng tránh: xóa generator ở Bước 6.
- **Notebook rỗng output:** Cell chỉ `savefig` hoặc `to_csv` mà không hiển thị. Dấu hiệu: mở notebook chỉ thấy text log, không thấy bảng hay hình. Phòng tránh: áp dụng Bước 7.
- **Logic nằm trong notebook:** Cell dài trên 20 dòng, có vòng lặp hoặc hàm phụ trợ. Dấu hiệu: không thể test cell đó bằng pytest. Phòng tránh: chuyển vào `src/` theo Bước 7.

## 8. Definition of Done

- [ ] Mọi hyperparameter và đường dẫn đã được tập trung hóa.
- [ ] Không còn biến toàn cục, tất cả là hàm thuần có chữ ký rõ ràng.
- [ ] Logic tiền xử lý chỉ có một nguồn chân lý duy nhất.
- [ ] Tất cả hàm có type hint và docstring mô tả shape.
- [ ] Đã loại bỏ dead code và import không dùng.
- [ ] Không còn script sinh code đã hoàn thành nhiệm vụ.
- [ ] Notebook có cell ngắn, hiển thị output, markdown tiếng Việt có dấu.
- [ ] Không còn emoji hoặc ký tự phi chuẩn.
- [ ] Kết quả sau refactor khớp với kết quả trước refactor.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Code chạy được rồi, refactor làm gì cho mất công"* -> **Bác bỏ:** Code lộn xộn khó kiểm thử, khó tái sử dụng, và dễ phát sinh lỗi khi mở rộng. Refactor chuẩn bị cho bảo trì và nộp bài.
- [X] *"Hằng số để rải rác cũng được, đọc hiểu được là được"* -> **Bác bỏ:** Hằng số rải rác khiến việc thay đổi cấu hình phải sửa nhiều nơi và dễ bỏ sót. Tập trung hóa là bắt buộc.
- [X] *"Viết lại tiền xử lý cho eval riêng cho gọn"* -> **Bác bỏ:** Duy trì hai bản logic dẫn đến train/serve skew. Dùng chung một pipeline.
- [X] *"Emoji cho đẹp thôi, không ảnh hưởng gì"* -> **Bác bỏ:** Emoji gây lỗi `UnicodeEncodeError` trên môi trường như Windows CP1252 và vi phạm chuẩn chuyên nghiệp. Chỉ dùng text thuần.
- [X] *"Giữ generator để sau này còn sinh lại notebook"* -> **Bác bỏ:** Nếu output đã trong Git thì phục hồi được từ lịch sử. Giữ generator khi không dùng tạo hai nguồn sự thật và khiến lần sửa sau không biết phải sửa bên nào.
- [X] *"Notebook chỉ cần lưu hình ra file, không cần hiển thị"* -> **Bác bỏ:** Notebook là tài liệu trình bày. Nếu không hiển thị thì người đọc phải mở từng file PNG bên ngoài, và notebook mất lý do tồn tại.
- [X] *"Viết tiếng Việt không dấu cho an toàn encoding"* -> **Bác bỏ:** Né lỗi bằng cách hạ cấp nội dung là sai hướng. Lỗi encoding phải chặn bằng pre-commit hook, không bằng cách bỏ dấu tiếng Việt.
