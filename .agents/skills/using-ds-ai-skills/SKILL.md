---
name: using-ds-ai-skills
description: Điều phối và định tuyến các tác vụ Data Science, Machine Learning, Deep Learning tới đúng skill chuyên biệt và playbook workflow tương ứng trong docs/ của Lab DL, đóng vai trò nhạc trưởng của toàn bộ vòng đời.
---

# using-ds-ai-skills (Bản DS & AI)

## 1. Tổng quan

Skill này đóng vai trò **nhạc trưởng tổng thể (master conductor)** của toàn bộ framework. Nó định tuyến mọi tác vụ thuộc Data Science, Machine Learning và Deep Learning tới các skill kỹ thuật chuẩn hóa, đồng thời liên kết trực tiếp tới các playbook chuyên ngành trong `docs/`.

Đây là điểm vào (entry point) của framework. Khi một phiên làm việc mới bắt đầu mà chưa rõ cần dùng skill nào, skill này chịu trách nhiệm phân loại bài toán, xác định giai đoạn hiện tại, và điều phối chuỗi skill phù hợp.

Skill chọn local CPU/GPU hoặc Kaggle theo workload, ngân sách và yêu cầu đã thống nhất; không bắt buộc chuyển môi trường chỉ vì thiếu CUDA.

Nguyên lý then chốt: phân loại đúng ngay từ đầu quyết định toàn bộ chuỗi hành động phía sau; định tuyến sai sẽ khiến mọi bước tiếp theo lệch chuẩn.

## 2. Khi nào kích hoạt (Triggers)

**Dấu hiệu kích hoạt:**
- Bắt đầu một phiên phát triển DS/AI mới.
- Người dùng đưa ra một bài tập, dataset, hoặc yêu cầu huấn luyện mô hình.
- Cần quyết định bước đi tiếp theo trong vòng đời dự án.
- Có sự nhập nhằng giữa nhiều hướng tiếp cận và cần chọn đúng playbook.
- Người dùng hỏi "bây giờ làm gì tiếp".

**Điều kiện tiên quyết:**
- Đã có quyền truy cập vào cấu trúc repository và các thư mục `docs/`, `.agents/skills/`.
- Đã đọc `.agents/skills/INDEX.md` để nắm bảng định tuyến.

## 3. Hợp đồng Input / Output

| Input (nhận vào) | Output (trả ra) |
|---|---|
| Yêu cầu của người dùng (mô tả bài toán) | Phân loại bài toán và giai đoạn hiện tại |
| Loại dữ liệu và modality | Playbook workflow tương ứng được chỉ định |
| Trạng thái dự án hiện tại | Chuỗi skill cần kích hoạt theo thứ tự |
| Ràng buộc tính toán | Quyết định thực thi cục bộ hay Kaggle GPU |

## 4. Quy trình chi tiết

### Bước 1: Phân loại tác vụ
Xác định modality của bài toán dựa trên dữ liệu đầu vào. Computer Vision, NLP, Audio định tuyến tới `docs/dl_flow.md`; phân tích và kỹ thuật đặc trưng tham chiếu `docs/eda.md` và `docs/feature.md`. Phân loại sai ngay bước này khiến toàn bộ quy trình phía sau lệch chuẩn.

### Bước 2: Khởi tạo và Đặc tả
Nếu yêu cầu còn mơ hồ, kích hoạt `idea-refine` và `interview-me`. Sau khi đã rõ yêu cầu, kích hoạt `spec-driven-development` để dựng đặc tả và baseline trong Section 1 của `walkthrough_lab_0X.md`.

### Bước 3: Lập kế hoạch
Kích hoạt `planning-and-task-breakdown` để phân rã pipeline thành các module trong `src/` và notebook trong `notebooks/`. Bước này biến đặc tả thành kế hoạch công việc cụ thể.

### Bước 4: Triển khai và Kiểm thử
Kích hoạt `test-driven-development` để xác thực schema dữ liệu và chạy test 1-batch sanity overfit. Kích hoạt `source-driven-development` để xác thực API thư viện. Kích hoạt `environment-and-reproducibility` để cố định seed và ghi nhận phiên bản môi trường.

### Bước 5: Thực thi và thu hoạch theo môi trường đã chọn
Sau sanity/smoke test, cho phép full training trên local CPU/GPU hoặc Kaggle theo cấu hình và ngân sách đã chốt. Chỉ kích hoạt `kaggle-remote-execution` khi dùng Kaggle. Mỗi lần chạy lưu config, môi trường, output và console log tại outputs/runs/<run_id>; publish evidence có chọn lọc sang outputs/published, giữ log thô local.

### Bước 6: Chẩn đoán và Kiểm toán
Nếu gặp lỗi kỹ thuật, kích hoạt `debugging-and-error-recovery`. Nếu kết quả cao bất thường, kích hoạt `doubt-driven-development` để kiểm toán rò rỉ dữ liệu.

### Bước 7: Tối ưu và Triển khai
Kích hoạt `performance-optimization` nếu cần tăng throughput. Kích hoạt `api-and-interface-design` nếu cần đóng gói mô hình thành dịch vụ inference.

### Bước 8: Rà soát, Ghi vết và Nộp bài
Kích hoạt `code-simplification` để dọn code. Kích hoạt `version-control-and-artifacts` để commit sạch. Kích hoạt `walkthrough-and-experiment-tracking` cho Section 2 và `documentation-and-adrs` cho Section 3.

## 5. Deliverables

- Bản phân loại bài toán và giai đoạn hiện tại.
- Chuỗi skill được kích hoạt theo đúng thứ tự.
- Tài liệu sống `walkthrough_lab_0X.md` được cập nhật qua cả 3 Section.
- Quyết định thực thi cục bộ hay Kaggle được ghi nhận.

## 6. Bàn giao (Handoff)

| Skill trước (upstream) | Skill sau (downstream) |
|---|---|
| Không có (đây là điểm vào) | `idea-refine`, `interview-me`, `spec-driven-development` |
| `version-control-and-artifacts` (báo cáo chu kỳ hoàn tất) | Phiên làm việc tiếp theo |

## 7. Chế độ thất bại thường gặp (Failure Modes)

- **Phân loại sai modality:** Chọn sai playbook cho modality thực tế (ví dụ áp kiến trúc chuỗi cho dữ liệu ảnh). Dấu hiệu: tiền xử lý và backbone không khớp với dạng dữ liệu đầu vào. Phòng tránh: xác định modality trước tiên ở Bước 1.
- **Bỏ qua bước đặc tả:** Nhảy thẳng vào code khi chưa có spec. Dấu hiệu: không có baseline và metric rõ ràng. Phòng tránh: luôn chạy `spec-driven-development` trước.
- **Chọn compute không phù hợp:** Dự kiến thời gian vượt ngân sách hoặc tự ép Kaggle khi người dùng chọn local. Phòng tránh: đánh giá workload và giữ lựa chọn đã chốt.
- **Bỏ sót ghi vết:** Không cập nhật walkthrough sau mỗi run. Phòng tránh: kích hoạt `walkthrough-and-experiment-tracking` ngay sau mỗi lần chạy.

## 8. Definition of Done

- [ ] Bài toán đã được phân loại đúng modality.
- [ ] Playbook workflow tương ứng đã được chỉ định.
- [ ] Chuỗi skill đã được kích hoạt theo đúng thứ tự.
- [ ] Compute và ngân sách thực thi đã được chọn phù hợp; thiết bị thực tế được ghi vào run.
- [ ] `walkthrough_lab_0X.md` đã được cập nhật qua cả 3 Section.
- [ ] Trạng thái dự án và bước tiếp theo đã được xác định rõ.

## 9. Chống biện minh (Anti-Rationalization)

- [X] *"Cứ code trước, phân loại sau"* -> **Bác bỏ:** Định tuyến sai khiến toàn bộ quy trình lệch chuẩn và phải làm lại. Phân loại đúng tốn vài giây, tiết kiệm hàng giờ.
- [X] *"Bài này nhỏ, bỏ qua đặc tả và kế hoạch"* -> **Bác bỏ:** Không có spec và kế hoạch thì không có baseline để so sánh và không có tiêu chí nghiệm thu. Mọi kết quả sau đó thiếu căn cứ.
- [X] *"Không có GPU thì không train được"* -> **Bác bỏ:** CPU được phép nếu workload và ngân sách phù hợp; chạy sanity trước và ghi thiết bị thực tế.
- [X] *"Ghi vết để sau, giờ tập trung chạy thí nghiệm"* -> **Bác bỏ:** Ghi vết ngay sau mỗi run rẻ hơn nhiều so với truy ngược về sau. Ký ức về lý do thay đổi sẽ phai nhạt nhanh chóng.
