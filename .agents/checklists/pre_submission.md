# Checklist — Trước khi nộp bài (Submission Gate)

Danh sách kiểm tra bắt buộc trước khi nộp lab. Mục đích là đảm bảo tài liệu sống đầy đủ, số liệu khớp log, và không còn nội dung chưa xác thực.

---

## 1. Tài liệu sống (Living Document Completeness)
- [ ] `walkthrough_lab_0X.md` có đủ 3 Section: Bài toán, Phương pháp/thiết kế thí nghiệm, Kết quả.
- [ ] Section 1 mô tả đầy đủ bản tóm tắt cách hiểu đề bài (Mục 1.0), problem formulation, data contract, baseline, metric, ngưỡng chấp nhận.
- [ ] Section 2 giải thích EDA, preprocessing, feature engineering và thiết kế thí nghiệm; bảng kết quả đầy đủ đặt ở Section 3, không chép lặp.
- [ ] Section 3 có ma trận đánh giá định lượng, chẩn đoán confusion matrix, phân tích lỗi, truy vết quyết định về Mục 1.0, kết luận.
- [ ] Không tồn tại file `SPEC.md` hay `REPORT.md` riêng lẻ.

## 2. Tính chính xác của số liệu (Numerical Accuracy)
- [ ] Mọi con số trong tài liệu khớp 100% với artifact thực nghiệm đã chọn và bảng metrics thực tế của lab.
- [ ] Không còn chỗ trống placeholder hoặc giá trị chưa điền.
- [ ] Các metric được ghi rõ đơn vị và số chữ số thập phân nhất quán.
- [ ] Không có tuyên bố định lượng nào thiếu bằng chứng từ artifact.
- [ ] Với bài toán phân loại, đã ghi đủ bộ metric: Test Loss, Test Accuracy, Macro F1, Precision/Recall/F1 và support theo lớp. Recall là TPR; FPR có thể ở bảng. Không bắt buộc biểu đồ TPR/FPR; Lab01 dùng ROC/PR OvR.

## 3. Artifact và hình ảnh (Artifacts)
- [ ] Tất cả hình trực quan được nhúng hoặc tham chiếu đúng đường dẫn trong `outputs/published/` và có sẵn sau clone.
- [ ] Confusion matrix, đường cong huấn luyện, và lưới dự đoán đều hiện diện.
- [ ] Mỗi hình đã được mở và diễn giải nội dung thực tế (cặp lớp nhầm lẫn, mẫu dự đoán sai, lớp có TPR thấp/FPR cao), không chỉ liệt kê đường dẫn.
- [ ] Checkpoint tốt nhất đã được xác thực save/load và ghi rõ đường dẫn.

## 4. Kiểm toán chất lượng (Quality Audit)
- [ ] Đã kiểm tra leakage/split/fit-on-train; tăng mức kiểm toán theo dataset/baseline khi đáng ngờ. Ghi rõ kiểm tra chưa chạy, không bắt buộc permutation test cho mọi bài.
- [ ] Không có emoji, icon, hay ký tự phi chuẩn trong bất kỳ file nào.
- [ ] Code trong `src/` đã được refactor sạch, không còn biến toàn cục hay hằng số hard-code.
- [ ] Notebook đã chạy có đầy đủ output đã được đưa lên Git.

## 5. Tái lập (Reproducibility)
- [ ] Có hướng dẫn một lệnh để tái lập kết quả từ checkpoint tốt nhất.
- [ ] Hướng dẫn cài đặt môi trường đầy đủ.
- [ ] Phiên bản thư viện chính và phiên bản trình thông dịch Python đã được ghi lại.
- [ ] Yêu cầu Python thật (`Requires-Python` suy ra từ các pin) đã được ghi, không chỉ phiên bản đã chạy.
- [ ] Mỗi run được dùng có `outputs/runs/<run_id>/environment_snapshot.json`, bản publish đã khử thông tin riêng tư; giải thích khác biệt giữa phiên bản đã chạy và constraint requirements.

- [ ] Không có nhật ký sửa code/lỗi môi trường hay console log thô trong báo cáo hoặc bộ publish; nhận xét mô hình nhầm lớp vẫn được giữ.
- [ ] PDF khi được yêu cầu đã render kiểm tra số trang, caption, bảng và ảnh; Lab01 tối thiểu 25 trang, không suy ra từ độ dài Markdown.
- [ ] Clone đủ để đọc báo cáo; hướng dẫn full artifacts để thực thi nói đúng khả năng hiện có và không có link Release giả.

## 6. Xác nhận cuối
- [ ] Repository đã push lên remote và trạng thái làm việc sạch.
- [ ] Đã xác nhận link notebook Kaggle (nếu có) hoạt động và hiển thị đúng output.
- [ ] Đã đọc lại toàn bộ walkthrough một lượt như người chấm sẽ đọc.
