"""Patches notebooks/lab01_fashionmnist_walkthrough.ipynb for split_v2 / preproc_v2.

The training notebook is patched rather than regenerated because it carries the
executed Kaggle outputs, which are evidence and must not be thrown away. This
script performs three surgical edits:

1. Insert a guard cell after the environment setup that fails loudly when the EDA
   split artifact or the Feature Engineering normalization artifact is missing.
2. Replace the symbolic "EDA" cell with one that reads the real EDA outputs.
3. Correct two descriptions that had drifted from the code (the Kaggle source is
   a mounted dataset, not a GitHub clone; the zip step is PHAN 12, not PHAN 11).

Running the script twice is a no-op because every edit is guarded by a marker
string that only exists after the edit has been applied.
"""

import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NB_PATH = HERE.parent / "notebooks" / "lab01_fashionmnist_walkthrough.ipynb"

GUARD_MARKER = "## PHẦN 1B: GUARD - XÁC NHẬN ARTIFACT EDA & FEATURE ENGINEERING"
EDA_MARKER = "## PHẦN 2: NẠP DỮ LIỆU & KHẢO SÁT NHANH TỪ ARTIFACT EDA"

GUARD_CODE = '''
# Guard bat buoc: training chi duoc phep bat dau khi da co artifact cua EDA va
# Feature Engineering. Neu thieu, notebook dung ngay thay vi am tham roi ve
# mean/std cu (legacy std = mean cua per-image std, khong phai global std) va
# mot split khac voi split ma thong ke duoc do tren do.
from pathlib import Path as _Path
import json as _json

EDA_DIR = OUTPUTS_DIR / "eda"
_required = {
    "split_indices.npz": EDA_DIR / "split_indices.npz",
    "split_metadata.json": EDA_DIR / "split_metadata.json",
    "preprocessing_config.json": EDA_DIR / "preprocessing_config.json",
}
_missing = [name for name, path in _required.items() if not path.exists()]
assert not _missing, (
    "Thieu artifact bat buoc: " + ", ".join(_missing) + ".\\n"
    "Thu tu bat buoc: chay notebooks/lab01_eda.ipynb, roi "
    "notebooks/lab01_feature_engineering.ipynb, roi moi chay notebook nay.\\n"
    "Ca hai notebook do phai duoc dong goi vao outputs/ va mount cung src/."
)

_split_meta = _json.loads(_required["split_metadata.json"].read_text(encoding="utf-8"))
_prep = _json.loads(_required["preprocessing_config.json"].read_text(encoding="utf-8"))

print(f"[GUARD] split_version     : {_split_meta['split_version']}")
print(f"[GUARD] split seed        : {_split_meta['seed']} | stratified={_split_meta['stratified']}")
print(f"[GUARD] train/val         : {_split_meta['train_size']} / {_split_meta['val_size']}")
print(f"[GUARD] train per-class   : {_split_meta['train_class_counts']}")
print(f"[GUARD] val   per-class   : {_split_meta['val_class_counts']}")
print(f"[GUARD] preproc_version   : {_prep['preproc_version']}")
print(f"[GUARD] normalization     : mean={_prep['normalization']['mean']} "
      f"std={_prep['normalization']['std']}")
print(f"[GUARD] fitted_on         : {_prep['normalization']['fitted_on']}")

assert _prep["split_version"] == _split_meta["split_version"], (
    f"Feature Engineering artifact was produced for split "
    f"{_prep['split_version']} but the mounted split is {_split_meta['split_version']}"
)
assert _prep["normalization"]["mean"] == list(FASHION_MNIST_MEAN), (
    f"src/config.py loaded mean {FASHION_MNIST_MEAN} but the artifact records "
    f"{_prep['normalization']['mean']}; the module did not pick up the artifact"
)
assert _prep["normalization"]["std"] == list(FASHION_MNIST_STD), (
    f"src/config.py loaded std {FASHION_MNIST_STD} but the artifact records "
    f"{_prep['normalization']['std']}; the module did not pick up the artifact"
)
assert NORMALIZATION_SOURCE.startswith("artifact:"), (
    f"Normalization came from '{NORMALIZATION_SOURCE}' instead of the measured "
    f"artifact. Training would use unverified statistics."
)
print(f"[GUARD] normalization source: {NORMALIZATION_SOURCE}")
print("[GUARD] OK - artifact EDA va Feature Engineering khop voi src/config.py")
'''

EDA_CODE = '''
# Doc lai artifact EDA thay vi trinh bay lai bang mot vai anh mau.
# So lieu o day la so lieu da do trong notebooks/lab01_eda.ipynb, khong phai
# so lieu go lai bang tay.
import pandas as pd

eda_dir = OUTPUTS_DIR / "eda"
tables_dir = eda_dir / "tables"

split_meta = _json.loads((eda_dir / "split_metadata.json").read_text(encoding="utf-8"))
eda_findings = _json.loads((eda_dir / "eda_findings.json").read_text(encoding="utf-8"))

print("=" * 70)
print("SPLIT (split_v2_stratified)")
print("=" * 70)
print(f"  train={split_meta['train_size']} val={split_meta['val_size']} "
      f"pool={split_meta['pool_size']}")
print(f"  seed={split_meta['seed']} stratified={split_meta['stratified']}")
print(f"  train per-class: {split_meta['train_class_counts']}")
print(f"  val   per-class: {split_meta['val_class_counts']}")

print("\\n" + "=" * 70)
print("NORMALIZATION (preproc_v2_train_global_std)")
print("=" * 70)
print(f"  mean={FASHION_MNIST_MEAN[0]:.6f}  std={FASHION_MNIST_STD[0]:.6f}")
print(f"  source: {NORMALIZATION_SOURCE}")
print(f"  legacy std {_prep['normalization']['rejected_alternative']['value']} da bi loai:")
print(f"    ly do -> {_prep['normalization']['rejected_alternative']['reason']}")

print("\\n" + "=" * 70)
print("INTEGRITY (khong loai bo mau nao)")
print("=" * 70)
_integrity = pd.read_csv(tables_dir / "integrity_report.csv")
print(_integrity[["check", "num_flagged"]].to_string(index=False))

print("\\n" + "=" * 70)
print("CAP LOP TUONG DONG NHAT (gia thuyet, chua phai ket luan)")
print("=" * 70)
_top_pairs = pd.read_csv(tables_dir / "most_similar_pairs_cosine.csv").head(5)
print(_top_pairs.to_string(index=False))
print("\\n  Day la may tao gia thuyet: mean image lam mat phuong sai trong lop,")
print("  nen ket luan ve cap lop kho chi duoc dua ra tu confusion matrix ben duoi.")

# Kiem tra lai tren loader that: split phai dung la split da do
_train_loader_eda, _val_loader_eda, _test_loader_eda, _ = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=64,
    val_split=VAL_SPLIT,
    augment=False,
    seed=RANDOM_SEED,
)
assert len(_train_loader_eda.dataset) == split_meta["train_size"], (
    f"Train loader has {len(_train_loader_eda.dataset)} samples but the split "
    f"artifact records {split_meta['train_size']}"
)
assert len(_val_loader_eda.dataset) == split_meta["val_size"], (
    f"Val loader has {len(_val_loader_eda.dataset)} samples but the split "
    f"artifact records {split_meta['val_size']}"
)
print(f"\\n[OK] get_dataloaders() doc dung artifact: "
      f"train={len(_train_loader_eda.dataset)} val={len(_val_loader_eda.dataset)}")

# Mot batch that de xac nhan contract shape/dtype truoc khi train
sample_images, sample_targets = next(iter(_train_loader_eda))
print(f"[OK] batch: images={tuple(sample_images.shape)} {sample_images.dtype} | "
      f"targets={tuple(sample_targets.shape)} {sample_targets.dtype}")
'''


def find_cell(cells, marker, start=0):
    """Returns the index of the first cell whose source contains `marker`."""
    for i in range(start, len(cells)):
        if marker in "".join(cells[i]["source"]):
            return i
    return -1


def main() -> None:
    nb = nbf.read(str(NB_PATH), as_version=4)
    cells = nb.cells
    changes = []

    # --- 1. Insert the guard cell after PHAN 1 ------------------------------
    if find_cell(cells, GUARD_MARKER) == -1:
        setup_idx = find_cell(cells, "PHẦN 1: THIẾT LẬP MÔI TRƯỜNG")
        if setup_idx == -1:
            raise RuntimeError("Could not find the PHAN 1 header cell")

        # The guard must run after the environment cell (the last code cell of
        # PHAN 1) and before PHAN 2.
        insert_at = setup_idx + 2
        while insert_at < len(cells) and cells[insert_at]["cell_type"] != "markdown":
            insert_at += 1

        guard_md = nbf.v4.new_markdown_cell(
            "## PHẦN 1B: GUARD - XÁC NHẬN ARTIFACT EDA & FEATURE ENGINEERING\n\n"
            "Training chỉ được phép bắt đầu khi đã có artifact của EDA và Feature Engineering.\n"
            "Nếu thiếu một trong hai, cell dưới đây dừng notebook thay vì âm thầm rơi về\n"
            "`FALLBACK_MEAN`/`FALLBACK_STD` — cặp giá trị cũ có `std = 0.3205` là **trung bình\n"
            "của std từng ảnh**, không phải global std, nên sau Normalize dữ liệu có `std ≈ 1.10`\n"
            "chứ không phải 1.0.\n\n"
            "Guard cũng kiểm tra artifact Feature Engineering được đo trên **đúng split** đang mount:\n"
            "một split khác nghĩa là mean/std được đo trên tập khác với tập đang train."
        )
        cells.insert(insert_at, guard_md)
        cells.insert(insert_at + 1, nbf.v4.new_code_cell(GUARD_CODE.strip()))
        changes.append("inserted guard cell (PHAN 1B)")

    # --- 2. Replace the symbolic EDA cell ----------------------------------
    if find_cell(cells, EDA_MARKER) == -1:
        eda_md_idx = find_cell(cells, "PHẦN 2: NẠP DỮ LIỆU")
        if eda_md_idx == -1:
            raise RuntimeError("Could not find the PHAN 2 header cell")

        eda_md = nbf.v4.new_markdown_cell(
            "## PHẦN 2: NẠP DỮ LIỆU & KHẢO SÁT NHANH TỪ ARTIFACT EDA\n\n"
            "FashionMNIST gồm ảnh xám $28 \\times 28$ thuộc 10 danh mục trang phục.\n"
            "Split được tạo **một lần** trong `notebooks/lab01_eda.ipynb` bằng\n"
            "`train_test_split(stratify=...)` và lưu tại `outputs/eda/split_indices.npz`.\n"
            "`get_dataloaders()` đọc lại artifact đó, không chia lại:\n\n"
            "- **Train:** 54,000 ảnh — 5,400 ảnh mỗi lớp (dùng để cập nhật gradient)\n"
            "- **Validation:** 6,000 ảnh — 600 ảnh mỗi lớp (dùng chọn checkpoint)\n"
            "- **Test:** 10,000 ảnh official (chỉ mở sau khi khoá cấu hình)\n\n"
            "Cell dưới đây đọc lại bảng thống kê đã đo trong notebook EDA và xác nhận rằng\n"
            "loader thực sự dùng đúng split đó. Không vẽ lại 16 ảnh mẫu: việc khảo sát trực quan\n"
            "đã nằm ở `outputs/eda/figures/sample_images.png` với 20 ảnh mỗi lớp."
        )
        cells.insert(eda_md_idx, eda_md)
        cells.insert(eda_md_idx + 1, nbf.v4.new_code_cell(EDA_CODE.strip()))

        # Remove the old PHAN 2 markdown + its code cell (now shifted by 2).
        old_md_idx = find_cell(cells, "FashionMNIST gồm các ảnh đen trắng", start=eda_md_idx + 2)
        if old_md_idx != -1:
            del cells[old_md_idx]
            if old_md_idx < len(cells) and cells[old_md_idx]["cell_type"] == "code":
                del cells[old_md_idx]
        changes.append("replaced symbolic EDA cell with artifact-driven readout")

    # --- 3. Correct the two drifted descriptions ---------------------------
    for cell in cells:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        if "Tự động tải repository `Lab-DL` từ GitHub" in src:
            cell["source"] = src.replace(
                "Nếu đang chạy trên **Kaggle GPU**: Tự động tải repository `Lab-DL` từ GitHub "
                "về `/kaggle/working/Lab-DL` và import trực tiếp `src/`!",
                "Nếu đang chạy trên **Kaggle GPU**: copy `src/` từ Kaggle Dataset đã mount "
                "(`/kaggle/input/**/src`) vào `/kaggle/working/src`, rồi import từ đó. "
                "Cùng Dataset đó phải chứa `outputs/eda/` do hai notebook EDA và Feature "
                "Engineering sinh ra, vì guard ở PHẦN 1B cần cả hai artifact.",
            )
            changes.append("fixed Kaggle source description in PHAN 1")

        if "PHẦN 11" in src and "outputs.zip" in src:
            cell["source"] = src.replace("**PHẦN 11**", "**PHẦN 12**")
            if cell["source"] != src:
                changes.append("fixed zip step reference (PHAN 11 -> PHAN 12)")

    nbf.write(nb, str(NB_PATH))
    print(f"Patched {NB_PATH}")
    for change in changes:
        print(f"  - {change}")
    if not changes:
        print("  (no changes needed; already patched)")


if __name__ == "__main__":
    main()
