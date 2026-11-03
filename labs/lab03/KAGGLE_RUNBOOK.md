# Đưa Lab 03 lên Kaggle GPU

Hướng dẫn vận hành để train frozen DistilBERT (tối đa5 epoch, early stopping và chọn checkpoint theo validation loss)
trên Kaggle. Gói kernel được sinh từ notebook modeling gốc, không sửa tay hai lần.

## 0. Import notebook trực tiếp từ GitHub

Kaggle: New Notebook → File → Import Notebook → GitHub, dán URL:

```text
https://github.com/lacthui06/Lab-DL/blob/lab03/labs/lab03/notebooks/02_modeling.ipynb
```

Bật GPU và Internet trong Settings. Đổi `MODE = "view"` thành `MODE = "train"`, chọn `DEVICE = "cuda"`, rồi Run All từ đầu. Không cần upload checkpoint cũ: pipeline bắt đầu từ pretrained base. Nếu repo cũ còn trong session, dùng phiên Kaggle mới để clone đúng code đã cập nhật.

Max5epoch là ngân sách tối đa; early stopping patience2/min_delta0.001 có thể kết thúc sớm. Bảng/đường học có đúng số epoch đã chạy, không điền thêm điểm. Run view hiện có là bằng chứng10epoch lịch sử; không dùng số đó làm kết quả profile mới.

Các bước CLI dưới đây là lựa chọn khác; không bắt buộc cài Kaggle CLI khi import bằng giao diện.

## 1. Điều kiện tiên quyết

- Code đã push lên nhánh `lab03` của `https://github.com/lacthui06/Lab-DL.git`. Cell đầu của
  notebook clone đúng nhánh này, nên nếu code local chưa push thì Kaggle sẽ chạy bản cũ.
- Tài khoản Kaggle đã xác minh số điện thoại (bắt buộc để bật GPU và Internet).
- Đã cài `kaggle` CLI: `python -m pip install "kaggle>=1.7"`.

## 2. Đăng nhập Kaggle (một lần)

Chọn một trong hai cách:

- OAuth: `python -m kaggle auth login` (mở trình duyệt, không cần quản lý token).
- API token: tạo tại <https://www.kaggle.com/settings/api>, lưu vào `~/.kaggle/access_token`
  hoặc đặt biến môi trường `KAGGLE_API_TOKEN`.

Kiểm tra: `python -m kaggle quota` phải trả về hạn mức GPU của tài khoản.

Không commit token hoặc `kaggle.json` vào Git.

## 3. Sinh gói kernel

```bash
python labs/lab03/scripts/export_kaggle_kernel.py --username <username_kaggle>
```

Lệnh này tạo `labs/lab03/kaggle/` gồm:

- `02_modeling_train.ipynb` — bản sao notebook modeling với `MODE = "train"`, phần còn lại giống hệt bản gốc.
- `kernel-metadata.json` — bật GPU, bật Internet, để chế độ private.

Notebook trong `notebooks/` giữ nguyên `MODE = "view"`; test `test_kaggle_kernel_export.py`
chặn mọi khác biệt ngoài dòng `MODE`.

## 4. Kiểm tra trước khi đẩy (cổng Zero Waste)

```bash
python -m pytest labs/lab03/tests -q
```

Phải xanh 100% trước khi đẩy, để không đốt GPU quota vào pipeline còn lỗi. Test chạy fixture
CPU nhỏ; không train full cục bộ.

## 5. Đẩy và chạy

```bash
python -m kaggle kernels push -p labs/lab03/kaggle
python -m kaggle kernels status lacthui06/lab03-frozen-distilbert-train
python -m kaggle kernels logs lacthui06/lab03-frozen-distilbert-train -f
```

Trạng thái `KernelWorkerStatus.COMPLETE` là chạy xong. Chạy lại từ đầu:
`python -m kaggle kernels push -p labs/lab03/kaggle` (tạo version mới).

Cell đầu cài `uv` và tạo venv Python 3.12 tại `/kaggle/working/lab03-py312`, sau đó mọi bước
ML chạy bằng interpreter con đó; kernel cha chỉ hiển thị ảnh.

## 6. Thu hồi artifact

Sau khi chạy xong, `/kaggle/working` có hai ZIP:

- `lab03_review_<run>.zip` — config, log, predictions, epoch history, ảnh, manifest SHA256.
  Không chứa weights; có cả published hình/bảng/log và nguồn selection mới.
- `lab03_checkpoint_<run>.zip` — `best_model` (weights, tokenizer, `decision_threshold.json`).

Tải về:

```bash
python -m kaggle kernels output lacthui06/lab03-frozen-distilbert-train -p labs/lab03/outputs/kaggle
```

Nếu link trong notebook không tải được, dùng giao diện Kaggle: **Output > /kaggle/working >
Refresh > dấu ba chấm cạnh ZIP > Download**.

Notebook đã chạy có output không nằm trên đĩa container. Tải riêng bằng **File > Download
Notebook** rồi chép vào `labs/lab03/notebooks/` nếu cần lưu bản có output.

## 7. Sau khi có kết quả

1. Giải nén `lab03_review_<run>.zip` và đọc `test_comparison.json`, `selection_summary.json`,
   `probability_diagnostics.json`.
2. Cập nhật Section 2 của `walkthrough_lab_03.md` với run ID mới và chênh lệch metric so với
   `20261027T095655Z_frozen_trial_d8d09116`.
3. Review ZIP chứa `outputs/runs/<run>/`, `outputs/published/`, `outputs/latest_report.json` và manifest. Kiểm tra hash trước khi nhập. Dùng run mới cập nhật report selection/báo cáo sau khi kiểm toán, không đổi nhãn run cũ.
4. Không commit ZIP, weights, hay dataset thô. Giữ checkpoint ZIP riêng để reload/evaluate/infer; commit figure, bảng, log nhẹ và notebook có output. Có thể gửi review ZIP để kiểm tra, hoặc giải nén và commit các artifact nhẹ trong `labs/lab03/`.
5. Accuracy/loss/time/RAM/VRAM mới phải đọc từ run mới. CPU tests không thay phép đo GPU; nếu một trường chưa đo thì ghi rõ thiếu.

## 8. Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| `AssertionError: Hãy bật Settings > Accelerator > GPU` | Chưa bật GPU | Bật GPU trong Settings, dùng `enable_gpu: true` |
| Connection refused khi tải model/dataset | Chưa bật Internet | Bật Internet, dùng `enable_internet: true` |
| Bus error giữa chừng | `/dev/shm` cạn do `num_workers` cao | Giữ `dataloader_num_workers: 0` như config |
| FileNotFoundError ở cell đầu | Repo chưa có code mới trên nhánh `lab03` | Push code rồi tạo version mới |
| Session chạy lại từ đầu | Restart kernel làm mất biến | Run All theo thứ tự, không chỉ chạy cell lẻ |
