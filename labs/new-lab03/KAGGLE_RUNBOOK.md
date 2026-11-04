# Chạy New Lab03 trên Kaggle GPU

## 1. Import notebook từ GitHub

Kaggle → New Notebook → File → Import Notebook → GitHub, dán:

```text
https://github.com/lacthui06/Lab-DL/blob/new-lab03/labs/new-lab03/notebooks/02_modeling.ipynb
```

Notebook và code trên branch phải đồng bộ: cell setup clone `lacthui06/Lab-DL`, branch `new-lab03`.
Dùng một phiên Kaggle mới để clone đầy đủ source.

## 2. Bật tài nguyên

Trong Settings, bật GPU và Internet. Tài khoản Kaggle cần hoàn thành xác minh theo yêu cầu dịch vụ.
Notebook 01 mặc định MODE=prepare/DEVICE=cpu; notebook 02 mặc định MODE=train/DEVICE=cuda. Import xong có thể Run All; cấu hình modeling đã đặt:

```python
MODE = "train"
DEVICE = "cuda"
```

Bấm Run All. Không cần upload dataset hoặc checkpoint: pipeline tải official dataset/pretrained base đã pin.
Kaggle tạo Python3.12 riêng; mọi lệnh ML chạy qua interpreter đó, kernel notebook dùng để hiển thị bảng/ảnh.

## 3. Các giai đoạn thực thi

1. Setup và CPU fixture preflight/sanity overfit.
2. Pretrained sentiment và pipeline/manual comparison.
3. Official data contract, augmentation và TF-IDF baseline.
4. DistilBERT fine-tuning tối đa 3 epoch, early stopping validation loss.
5. Reload checkpoint, threshold sweep validation, khóa policy.
6. Test, confusion/ROC/PR/calibration và error analysis.
7. Performance, save/load verification, publish và ZIP.

Mỗi giai đoạn có bảng/ảnh inline đồng thời lưu file. Early stopping patience1/min_delta0.001 có thể dừng trước 3 epoch; không thêm điểm giả lên learning curves.

## 4. Lựa chọn dùng Kaggle CLI

Cài Kaggle CLI và đăng nhập theo hướng dẫn của Kaggle. Không đặt token vào source hoặc commit token.
Sinh gói notebook/metadata:

```bash
python labs/new-lab03/scripts/export_kaggle_kernel.py --username YOUR_KAGGLE_USERNAME
python -m kaggle kernels push -p labs/new-lab03/kaggle
```

`YOUR_KAGGLE_USERNAME` là username Kaggle của người chạy.
Gói được sinh từ notebook modeling, bật GPU/Internet và xóa cached outputs.
Không cần CLI nếu dùng giao diện import GitHub.

## 5. Tải kết quả

Khi hoàn thành, cell cuối hiện đường dẫn và link tải:

- `lab03_review_<run_id>.zip`: artifact nhẹ và published report.
- `lab03_checkpoint_<run_id>.zip`: model/tokenizer/policy riêng.

Nếu link không tải được: Output → `/kaggle/working` → Refresh → menu cạnh file → Download.
Dùng Save Version để lưu phiên chạy; File → Download Notebook để lấy notebook có cell outputs.

Review ZIP có bundle_manifest SHA256. Không sửa config/log/predictions rồi giữ manifest như chưa thay đổi.
Checkpoint cần giữ để đánh giá lại hoặc infer; không đưa weights/ZIP vào Git.

## 6. Đọc lại report không cần training

`outputs/latest_report.json` chỉ rõ run đã chọn.
`outputs/published/index.md` nhúng hình và liên kết log/bảng.
`artifact_catalog.json` và `manifest.json` xác nhận file/hash.
MODE=view chỉ đọc các selection này sau khi có kết quả.

Nếu muốn render lại notebook từ artifact, dùng:

```bash
python labs/new-lab03/scripts/render_notebooks.py --kernel python3
```

Lệnh render chuyển MODE sang view trong bộ nhớ khi thực thi, không train hoặc tải dữ liệu.
Source notebook mặc định prepare/train. Muốn xem lại kết quả, chọn MODE=view trước Run All; gói Kaggle CLI giữ MODE=train và xóa cached outputs.

## 7. Troubleshooting

| Hiện tượng | Kiểm tra / xử lý |
|---|---|
| CUDA unavailable | Bật GPU, chờ session khởi động, chạy lại từ đầu |
| GPU/Internet không chọn được | Kiểm tra xác minh tài khoản/hạn mức Kaggle |
| Dataset/model download lỗi | Bật Internet, đọc hostname/revision trong log |
| Preflight failed | Đọc outputs/preflight/.../pytest.log; sửa lỗi trước training |
| NameError | Run All từ cell setup; không chạy riêng cell phụ thuộc biến trước đó |
| Bus error | Giữ dataloader_num_workers=0 |
| CUDA out of memory | Dừng, giảm batch trong config trước phiên train; ghi lại cấu hình |
| View thiếu selection | Chạy prepare/train để tạo artifact trước khi view |
| Không thấy ZIP | Refresh đúng gốc /kaggle/working, xem đường dẫn cell cuối |

Notebook setup kiểm tra đúng `labs/new-lab03/config/frozen_trial.json`, không coi một thư mục clone bất kỳ là đủ. Trên Kaggle dùng clone riêng `/kaggle/working/Lab-DL-new-lab03`; nếu đường dẫn đó chưa đầy đủ, tạo clone có hậu tố riêng và bảo toàn dữ liệu hiện có.
