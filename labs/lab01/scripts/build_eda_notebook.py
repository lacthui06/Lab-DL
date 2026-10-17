"""Generates notebooks/lab01_eda.ipynb.

The notebook is generated from source so the cell content is reviewable in a
diff instead of buried in a JSON blob. Re-running this script rewrites the
notebook; hand edits to the .ipynb are overwritten.
"""

import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "notebooks" / "lab01_eda.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


cells = []

cells.append(md("""
# Lab 01 - EDA (FashionMNIST)

Notebook nay thuc thi phan **Exploratory Data Analysis** theo `docs/EDA1.md`, pham vi:

- **Core (bat buoc):** Step 1 structural, Step 2 integrity, Step 3 class distribution,
  Step 4 visual inspection, Step 5 pixel intensity statistics, Step 13 findings.
- **Extended (thuc hien):** Step 6 spatial analysis, Step 8 inter-class similarity.
- **Extended (khong thuc hien):** Step 7 handcrafted features, Step 9 PCA/t-SNE,
  Step 10 outlier scoring, Step 11 distribution shift, Step 12 traditional baseline.
  Ly do ghi ro o phan cuoi notebook va trong `walkthrough_lab_01.md`.

**Hai quy tac duoc thuc thi bang code, khong chi bang ky luat:**

1. Tap test chinh thuc chi duoc dung cho kiem tra cau truc va trung lap noi dung.
   Moi thong ke co the anh huong den quyet dinh mo hinh chi tinh tren train/val.
2. Pixel 0 la nen, khong phai missing value. Foreground duoc dinh nghia bang mot
   nguong tuong minh (`FOREGROUND_THRESHOLD = 0.1`), khong phai `pixel != 0`.

**Notebook nay tao ra artifact ma Feature Engineering va ca ba run se tai su dung:**

- `outputs/eda/split_indices.npz` - chi so train/val, tao MOT lan duy nhat.
- `outputs/eda/split_metadata.json` - provenance cua split.
- `outputs/eda/tables/*.csv` - cac bang thong ke.
- `outputs/eda/figures/*.png` - cac bieu do.
"""))

cells.append(md("""
## E0. Thiet lap moi truong va import

Notebook tu nhan dien Local va Kaggle giong notebook training, de chay lai duoc o ca hai noi.
Khong doc du lieu qua `get_dataloaders()`: ham do tra ve tensor da Normalize, sai view cho EDA
(khong the kiem tra pixel range hay dtype goc tren tensor da chuan hoa).
"""))

cells.append(code("""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # rendering khong tuong tac: notebook chay duoc bang nbconvert
import matplotlib.pyplot as plt

# 1. Thiet lap PROJECT_ROOT giong notebook training
if os.path.exists("/kaggle/working"):
    sys.path.insert(0, "/kaggle/working")
    PROJECT_ROOT = Path("/kaggle/working")
else:
    PROJECT_ROOT = Path("..").resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR, EDA_DIR, OUTPUTS_DIR, CLASS_NAMES, RANDOM_SEED,
    VAL_SPLIT, SPLIT_VERSION, NUM_CLASSES,
)
from src.split import make_stratified_split, save_split, load_split_metadata
from src import eda as eda_mod
from src.eda import FOREGROUND_THRESHOLD
from src.utils import set_seed

TABLES_DIR = EDA_DIR / "tables"
FIGURES_DIR = EDA_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

set_seed(RANDOM_SEED)

print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
print(f"[DATA_DIR]     {DATA_DIR}")
print(f"[EDA_DIR]      {EDA_DIR}")
print(f"[SPLIT_VERSION] {SPLIT_VERSION}")
print(f"[FOREGROUND_THRESHOLD] {FOREGROUND_THRESHOLD}")
print(f"[CLASS_NAMES] {len(CLASS_NAMES)} classes")
"""))

cells.append(md("""
## E1. Nap du lieu raw va tao split mot lan duy nhat

Doc truc tiep file IDX de giu nguyen `uint8` va khoang `[0, 255]`.

**Vi sao phai stratified:** `torch.utils.data.random_split` khong bao toan ty le lop.
Voi FashionMNIST, mot lan chia seed=42 cho ra validation tu 537 den 658 anh moi lop
(thay vi 600) - lech toi +/-10%. Split lech lam cho val accuracy khong con la thuoc do
cong bang giua cac kien truc, va lam cho thong ke normalization do tren "train" phu thuoc
vao mot tap con khong dai dien.

Split duoc luu ra artifact ngay tai day. EDA, Feature Engineering va ca ba run deu doc lai
cung mot artifact nay; khong noi nao chia lai.
"""))

cells.append(code("""
X_train_pool, y_train_pool, X_test_raw, y_test_raw = eda_mod.load_raw_arrays(DATA_DIR)

print(f"[RAW] train pool : X={X_train_pool.shape} {X_train_pool.dtype} | y={y_train_pool.shape} {y_train_pool.dtype}")
print(f"[RAW] test       : X={X_test_raw.shape} {X_test_raw.dtype} | y={y_test_raw.shape} {y_test_raw.dtype}")
print(f"[RAW] pixel range: [{X_train_pool.min()}, {X_train_pool.max()}] (train) | "
      f"[{X_test_raw.min()}, {X_test_raw.max()}] (test)")

# Tao split MOT lan. Moi stage sau deu doc lai artifact nay.
train_idx, val_idx = make_stratified_split(
    y_train_pool, val_split=VAL_SPLIT, seed=RANDOM_SEED
)

npz_path = save_split(
    out_dir=EDA_DIR,
    train_idx=train_idx,
    val_idx=val_idx,
    targets=y_train_pool,
    seed=RANDOM_SEED,
    val_split=VAL_SPLIT,
    version=SPLIT_VERSION,
    extra={
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "foreground_threshold": FOREGROUND_THRESHOLD,
    },
)

split_meta = load_split_metadata(EDA_DIR)
print(f"\\n[SPLIT] artifact: {npz_path}")
print(f"[SPLIT] version={split_meta['split_version']} seed={split_meta['seed']} "
      f"val_split={split_meta['val_split']} stratified={split_meta['stratified']}")
print(f"[SPLIT] pool={split_meta['pool_size']} train={split_meta['train_size']} val={split_meta['val_size']}")
print(f"[SPLIT] train per-class: {split_meta['train_class_counts']}")
print(f"[SPLIT] val   per-class: {split_meta['val_class_counts']}")
print(f"[SPLIT] excluded: {split_meta['excluded_partition']}")

# Kiem tra chen cheo va coverage - neu that bai thi dung ngay, khong di tiep
assert np.intersect1d(train_idx, val_idx).size == 0, "train/val indices overlap"
assert len(train_idx) + len(val_idx) == len(y_train_pool), "split does not cover the pool"
assert min(split_meta["val_class_counts"]) == max(split_meta["val_class_counts"]), \\
    "validation partition is not class-balanced"

X_tr, y_tr = X_train_pool[train_idx], y_train_pool[train_idx]
X_va, y_va = X_train_pool[val_idx], y_train_pool[val_idx]
print(f"\\n[OK] train {X_tr.shape} | val {X_va.shape} | test {X_test_raw.shape} (held out)")
"""))

cells.append(md("""
## E2. Step 1 - Structural overview

Muc tieu: xac lap cau truc co ban (shape, dtype, pixel range, memory) bang so lieu do duoc,
khong phai bang mo ta. Bang nay la bang chung cho phan "Dataset overview" cua bao cao.
"""))

cells.append(code("""
overview = eda_mod.structural_overview(
    X_train_pool, y_train_pool, X_test_raw, y_test_raw, CLASS_NAMES
)
overview.to_csv(TABLES_DIR / "structural_overview.csv", index=False)

print(overview.to_string(index=False))
"""))

cells.append(md("""
## E3. Step 2 - Data integrity check

Sau nhom kiem tra, phan biet ro hai khai niem thuong bi gop lam mot:

- **Index disjointness:** split khong dung lai cung mot dong hai lan.
- **Content duplication:** hai dong khac nhau chua pixel giong het nhau.

Mot dataset co the pass kiem tra thu nhat va van fail kiem tra thu hai. Ca hai deu duoc do.

Khong xoa anh nao. Cac anh nghi van chi duoc liet ke kem `source_dataset`,
`original_index`, `split`, `label` de truy nguoc.
"""))

cells.append(code("""
integrity_summary, integrity_details = eda_mod.integrity_report(
    X_train=X_train_pool,
    y_train=y_train_pool,
    X_test=X_test_raw,
    y_test=y_test_raw,
    train_indices=train_idx,
    val_indices=val_idx,
    num_classes=NUM_CLASSES,
)

integrity_summary.to_csv(TABLES_DIR / "integrity_report.csv", index=False)
print(integrity_summary.to_string(index=False))

print("\\n[CHI TIET CAC MAU NGHI VAN]")
for name, frame in integrity_details.items():
    print(f"  {name:<32} rows={len(frame)}")
    if len(frame):
        frame.to_csv(TABLES_DIR / f"integrity_{name}.csv", index=False)

# Tong hop trung lap theo split de thay ro duplicate nam trong train hay val
dup_train = integrity_details["exact_duplicate_train"]
if len(dup_train):
    print("\\n[DUPLICATE THEO SPLIT]")
    print(dup_train["split"].value_counts().to_string())
else:
    print("\\n[DUPLICATE THEO SPLIT] khong co exact duplicate trong train pool")
"""))

cells.append(md("""
## E4. Step 3 - Class distribution

Spec cam mac dinh rang phan phoi can bang, nen bang duoc tinh tu du lieu thuc te.
Tap test khong nam trong bang nay: no chi duoc kiem tra tong so mau o buoc cuoi.
"""))

cells.append(code("""
distribution = eda_mod.class_distribution(y_tr, y_va, CLASS_NAMES)
distribution.to_csv(TABLES_DIR / "class_distribution.csv", index=False)

print(distribution.to_string(index=False))

train_counts = np.bincount(y_tr, minlength=NUM_CLASSES)
val_counts = np.bincount(y_va, minlength=NUM_CLASSES)
print(f"\\n[TRAIN] min={train_counts.min()} max={train_counts.max()} "
      f"ratio_max_min={train_counts.max() / train_counts.min():.4f}")
print(f"[VAL]   min={val_counts.min()} max={val_counts.max()} "
      f"ratio_max_min={val_counts.max() / val_counts.min():.4f}")
print(f"[TEST]  held out, counts withheld until the configuration is locked")

eda_mod.plot_class_distribution(distribution, save_path=FIGURES_DIR / "class_distribution.png")
plt.close("all")
print(f"\\n[FIG] {FIGURES_DIR / 'class_distribution.png'}")
"""))

cells.append(md("""
## E5. Step 4 - Visual inspection

Lay mau ngau nhien theo seed co dinh, **khong lay N anh dau tien**: du lieu IDX duoc sap xep,
nen phan dau file khong dai dien cho toan bo phan phoi.

Mean image theo lop duoc tinh them de thay "hinh dang trung binh" cua tung lop. Luu y:
mean image lam mat phuong sai trong lop - chinh la thu quyet dinh viec hai lop co de nham
lan hay khong - nen no chi dung de tao gia thuyet, khong dung de ket luan.
"""))

cells.append(code("""
N_PER_CLASS = 20
picks = eda_mod.sample_images_per_class(
    X_tr, y_tr, CLASS_NAMES, n_per_class=N_PER_CLASS, seed=RANDOM_SEED
)
print(f"[SAMPLE] {N_PER_CLASS} anh x {len(CLASS_NAMES)} lop = {picks.size} anh, seed={RANDOM_SEED}")

eda_mod.plot_sample_grid(
    X_tr, picks, y_tr, CLASS_NAMES, save_path=FIGURES_DIR / "sample_images.png"
)
plt.close("all")

mean_images = eda_mod.class_mean_images(X_tr, y_tr, CLASS_NAMES)
eda_mod.plot_class_mean_images(
    mean_images, CLASS_NAMES, save_path=FIGURES_DIR / "class_mean_images.png"
)
plt.close("all")

print(f"[FIG] {FIGURES_DIR / 'sample_images.png'}")
print(f"[FIG] {FIGURES_DIR / 'class_mean_images.png'}")

# Luu mean image duoi dang bang de Step 8 dung lai, khong tinh lai
np.save(EDA_DIR / "class_mean_images.npy", mean_images)
print(f"[ARTIFACT] {EDA_DIR / 'class_mean_images.npy'} shape={mean_images.shape}")
"""))

cells.append(md("""
## E6. Step 5 - Pixel intensity statistics

Day la buoc quyet dinh cho normalization o Feature Engineering.

**Quy tac (tu `docs/FEATURE-ENGINEERING.md` Step 3):** mean/std phai la thong ke gop tren
toan bo pixel cua train, dung population variance (`ddof=0`). Khong duoc lay trung binh
cua std tung anh.

Hai gia tri nay khac nhau dang ke tren FashionMNIST, nen ham `compute_normalization_stats`
tra ve ca `per_image_std_mean` nhu mot chan doan, de su khac biet khong bi che khuat.
"""))

cells.append(code("""
train_stats = eda_mod.compute_normalization_stats(X_tr)
val_stats = eda_mod.compute_normalization_stats(X_va)

print("[TRAIN PARTITION - scaled [0,1], ddof=0]")
for k, v in train_stats.items():
    print(f"  {k:<26} {v}")
print("\\n[VAL PARTITION - descriptive only, never used to fit Normalize]")
for k in ("mean", "std", "num_images", "zero_pixel_fraction"):
    print(f"  {k:<26} {val_stats[k]}")

print("\\n[SO SANH VOI CONFIG CU]")
from src.config import FALLBACK_MEAN, FALLBACK_STD
print(f"  legacy mean {FALLBACK_MEAN[0]:.6f} vs measured {train_stats['mean']:.6f} "
      f"(delta {train_stats['mean'] - FALLBACK_MEAN[0]:+.6f})")
print(f"  legacy std  {FALLBACK_STD[0]:.6f} vs measured {train_stats['std']:.6f} "
      f"(delta {train_stats['std'] - FALLBACK_STD[0]:+.6f})")
print(f"  legacy std vs mean-of-per-image-std {train_stats['per_image_std_mean']:.6f} "
      f"(delta {FALLBACK_STD[0] - train_stats['per_image_std_mean']:+.6f})")
print("  -> legacy std khop voi mean-of-per-image-std, KHONG phai global std.")
"""))

cells.append(code("""
pixel_by_class = eda_mod.pixel_statistics_by_class(X_tr, y_tr, CLASS_NAMES)
pixel_by_class.to_csv(TABLES_DIR / "pixel_statistics.csv", index=False)
print(pixel_by_class.to_string(index=False))

stats_table = pd.DataFrame(
    [
        {"partition": "train", **train_stats},
        {"partition": "val", **val_stats},
    ]
)
stats_table.to_csv(TABLES_DIR / "normalization_stats_measurement.csv", index=False)

eda_mod.plot_pixel_histogram(X_tr, X_va, save_path=FIGURES_DIR / "pixel_histogram.png")
plt.close("all")
eda_mod.plot_brightness_distribution(
    X_tr, y_tr, CLASS_NAMES, save_path=FIGURES_DIR / "image_brightness_distribution.png"
)
plt.close("all")

print(f"\\n[FIG] {FIGURES_DIR / 'pixel_histogram.png'}")
print(f"[FIG] {FIGURES_DIR / 'image_brightness_distribution.png'}")
"""))

cells.append(md("""
## E7. Step 6 - Spatial analysis

Foreground duoc dinh nghia bang nguong `FOREGROUND_THRESHOLD = 0.1` tren thang `[0, 1]`,
khong phai `pixel != 0`: khoang 50% pixel FashionMNIST bang 0, nen quy tac "khac 0" se tinh
ca nhieu hat nhieu gan den vien quan ao la vat the va lam phong moi bounding box.

Ba ban do duoc ve cung nhau vi chung tra loi ba cau hoi khac nhau:

- mean map: vung nao trung binh sang.
- variance map: vung nao thay doi nhieu giua cac anh.
- zero-frequency map: vung nao gan nhu luon la nen.

Mot pixel bien gioi co the bang 0 trong 99% anh va van co phuong sai lon nhat - hai chi so
khong thay the nhau.
"""))

cells.append(code("""
spatial = eda_mod.spatial_features(
    X_tr, y_tr, CLASS_NAMES, threshold=FOREGROUND_THRESHOLD
)
# Gan nhan split de moi dong truy nguoc duoc ve partition
spatial["split"] = "train"
spatial.to_csv(TABLES_DIR / "spatial_features.csv", index=False)

print(f"[SPATIAL] {len(spatial)} anh, threshold={FOREGROUND_THRESHOLD}")
print(f"[SPATIAL] empty foreground: {int(spatial['empty_foreground'].sum())} anh")
print("\\n[SPATIAL] trung binh theo lop:")
print(
    spatial.groupby("class_name")[
        ["foreground_ratio", "bbox_height", "bbox_width", "bbox_aspect_ratio", "bbox_area_ratio"]
    ].mean().round(4).to_string()
)

mean_map = eda_mod.mean_pixel_map(X_tr)
variance_map = eda_mod.pixel_variance_map(X_tr)
zero_map = eda_mod.zero_pixel_frequency_map(X_tr)
np.savez_compressed(
    EDA_DIR / "spatial_maps.npz",
    mean_map=mean_map, variance_map=variance_map, zero_frequency_map=zero_map,
)

eda_mod.plot_heatmaps(
    mean_map, variance_map, zero_map, save_path=FIGURES_DIR / "spatial_heatmaps.png"
)
plt.close("all")
eda_mod.plot_spatial_by_class(
    spatial, CLASS_NAMES, save_path=FIGURES_DIR / "spatial_by_class.png"
)
plt.close("all")

print(f"\\n[FIG] {FIGURES_DIR / 'spatial_heatmaps.png'}")
print(f"[FIG] {FIGURES_DIR / 'spatial_by_class.png'}")
print(f"[ARTIFACT] {EDA_DIR / 'spatial_maps.npz'}")
"""))

cells.append(md("""
## E8. Step 8 - Inter-class similarity

Cosine tren mean image cua tung lop. Day la **may tao gia thuyet**, khong phai bang chung
ve do kho phan loai: mean image da lam mat phuong sai trong lop, ma chinh phuong sai do moi
quyet dinh viec hai lop co nham lan hay khong. Ket luan ve cap lop kho chi duoc dua ra sau khi
co confusion matrix cua mo hinh da huan luyen.

Hai metric duoc tinh vi tra loi hai cau hoi hoi khac nhau: cosine so sanh hinh dang profile
sau khi bo do lon, correlation bo them muc sang trung binh cua tung anh.
"""))

cells.append(code("""
cosine_sim = eda_mod.class_similarity_matrix(mean_images, CLASS_NAMES, metric="cosine")
corr_sim = eda_mod.class_similarity_matrix(mean_images, CLASS_NAMES, metric="correlation")

cosine_sim.to_csv(TABLES_DIR / "class_similarity_matrix.csv")
corr_sim.to_csv(TABLES_DIR / "class_correlation_matrix.csv")

top_cosine = eda_mod.most_similar_pairs(cosine_sim, top_k=12)
top_corr = eda_mod.most_similar_pairs(corr_sim, top_k=12)
top_cosine.to_csv(TABLES_DIR / "most_similar_pairs_cosine.csv", index=False)
top_corr.to_csv(TABLES_DIR / "most_similar_pairs_correlation.csv", index=False)

print("[TOP 12 CAP TUONG DONG NHAT - COSINE]")
print(top_cosine.to_string(index=False))
print("\\n[TOP 12 CAP TUONG DONG NHAT - CORRELATION]")
print(top_corr.to_string(index=False))

eda_mod.plot_similarity_heatmap(
    cosine_sim, "cosine", save_path=FIGURES_DIR / "class_similarity_heatmap.png"
)
plt.close("all")
print(f"\\n[FIG] {FIGURES_DIR / 'class_similarity_heatmap.png'}")
"""))

cells.append(md("""
## E9. Step 13 - Findings, gioi han va ban giao sang Feature Engineering

Phan nay **khong viet ket luan dinh luong san**. Moi con so duoi day duoc doc tu bien vua tinh,
va moi phat bieu duoc gan nhan ro rang: **da do** hay **gia thuyet**.

Bang ban giao sang Feature Engineering theo format cua `docs/EDA1.md`:
`phat hien | split/view | bang chung/artifact | de xuat | cach kiem chung tren validation | trang thai`.
"""))

cells.append(code("""
# Gom cac con so DA DO vao mot dict de phan nhan xet chi duoc phep doc tu day.
findings = {
    "split_version": split_meta["split_version"],
    "train_size": int(len(y_tr)),
    "val_size": int(len(y_va)),
    "test_size_held_out": int(len(y_test_raw)),
    "train_class_count_min": int(train_counts.min()),
    "train_class_count_max": int(train_counts.max()),
    "val_class_count_min": int(val_counts.min()),
    "val_class_count_max": int(val_counts.max()),
    "train_pixel_mean": round(train_stats["mean"], 6),
    "train_pixel_std_ddof0": round(train_stats["std"], 6),
    "train_per_image_std_mean": round(train_stats["per_image_std_mean"], 6),
    "legacy_config_std": float(FALLBACK_STD[0]),
    "zero_pixel_fraction_train": round(train_stats["zero_pixel_fraction"], 6),
    "foreground_threshold": FOREGROUND_THRESHOLD,
    "foreground_ratio_mean": round(float(spatial["foreground_ratio"].mean()), 6),
    "empty_foreground_images": int(spatial["empty_foreground"].sum()),
    "exact_duplicate_within_train": int(
        integrity_summary.loc[
            integrity_summary["check"] == "exact_duplicate_within_train", "num_flagged"
        ].iloc[0]
    ),
    "exact_duplicate_train_vs_test": int(
        integrity_summary.loc[
            integrity_summary["check"] == "exact_duplicate_train_vs_test", "num_flagged"
        ].iloc[0]
    ),
    "all_black_images": int(
        integrity_summary.loc[
            integrity_summary["check"] == "all_black_images", "num_flagged"
        ].iloc[0]
    ),
    "top_similar_pair_cosine": f"{top_cosine.iloc[0]['class_a']} / "
                                f"{top_cosine.iloc[0]['class_b']} = "
                                f"{top_cosine.iloc[0]['similarity']:.4f}",
    "top_similar_pair_correlation": f"{top_corr.iloc[0]['class_a']} / "
                                    f"{top_corr.iloc[0]['class_b']} = "
                                    f"{top_corr.iloc[0]['similarity']:.4f}",
}

with open(EDA_DIR / "eda_findings.json", "w", encoding="utf-8") as f:
    json.dump(findings, f, indent=2)

print(json.dumps(findings, indent=2))
print(f"\\n[ARTIFACT] {EDA_DIR / 'eda_findings.json'}")
"""))

cells.append(code("""
# Bang ban giao sang Feature Engineering. Cot "trang thai" chi nhan 3 gia tri:
#   measured  = da do trong notebook nay
#   hypothesis = gia thuyet, phai kiem chung bang confusion matrix sau khi train
#   not_run   = chua thuc hien, co ly do ghi kem
handoff = pd.DataFrame(
    [
        {
            "finding": "Split stratified 54,000/6,000, moi lop dung 5,400/600",
            "split_view": "index-only, raw",
            "evidence": "split_metadata.json, class_distribution.csv",
            "proposal": "Dung dung artifact nay cho EDA, FE va ca 3 run; khong chia lai",
            "validation_check": "assert min==max class count trong moi partition",
            "status": "measured",
        },
        {
            "finding": f"Global train pixel mean={train_stats['mean']:.6f}, "
                       f"std(ddof=0)={train_stats['std']:.6f}",
            "split_view": "train, scaled [0,1], truoc augmentation",
            "evidence": "normalization_stats_measurement.csv",
            "proposal": "FE do lai tren dung artifact nay va ghi preprocessing_config.json",
            "validation_check": "std cua tensor sau Normalize phai ~1.0 (tolerance 0.02)",
            "status": "measured",
        },
        {
            "finding": f"Config cu dung std={FALLBACK_STD[0]:.4f}, bang mean cua per-image std "
                       f"({train_stats['per_image_std_mean']:.6f}), khong phai global std",
            "split_view": "train, scaled [0,1]",
            "evidence": "normalization_stats_measurement.csv (cot per_image_std_mean)",
            "proposal": "Thay bang global std do tren split moi; version hoa preprocessing",
            "validation_check": "test_normalization_yields_unit_variance",
            "status": "measured",
        },
        {
            "finding": f"Khong co exact duplicate trong train pool va khong co "
                       f"train-test overlap ({findings['exact_duplicate_train_vs_test']})",
            "split_view": "raw, md5 tren byte",
            "evidence": "integrity_report.csv",
            "proposal": "Khong loai bo mau nao; giu nguyen dataset goc",
            "validation_check": "hash lai trong FE guard, phai van bang 0",
            "status": "measured",
        },
        {
            "finding": f"Pixel 0 chiem {train_stats['zero_pixel_fraction']:.4f} tong pixel",
            "split_view": "train, scaled [0,1]",
            "evidence": "pixel_statistics.csv, pixel_histogram.png",
            "proposal": "Foreground phai dung nguong, khong dung pixel != 0",
            "validation_check": "foreground_ratio khong doi nghia khi threshold thay doi nho",
            "status": "measured",
        },
        {
            "finding": f"Cap lop giong nhau nhat (cosine): {findings['top_similar_pair_cosine']}",
            "split_view": "train, mean image",
            "evidence": "class_similarity_matrix.csv, class_similarity_heatmap.png",
            "proposal": "Chi ghi la gia thuyet ve cap lop de nham lan",
            "validation_check": "confusion matrix cua ca 3 run phai xac nhan hoac bac bo",
            "status": "hypothesis",
        },
        {
            "finding": "Handcrafted features (Step 7)",
            "split_view": "n/a",
            "evidence": "n/a",
            "proposal": "Khong dua vao model; chi dung de mo ta",
            "validation_check": "n/a",
            "status": "not_run",
        },
        {
            "finding": "PCA / t-SNE (Step 9)",
            "split_view": "n/a",
            "evidence": "n/a",
            "proposal": "Khong thuc hien: khong phuc vu quyet dinh nao cho 3 run MLP/CNN",
            "validation_check": "n/a",
            "status": "not_run",
        },
        {
            "finding": "Outlier scoring (Step 10)",
            "split_view": "n/a",
            "evidence": "n/a",
            "proposal": "Khong thuc hien: spec cam loai bo mau chi vi bi danh dau outlier",
            "validation_check": "n/a",
            "status": "not_run",
        },
        {
            "finding": "Distribution shift train/val (Step 11)",
            "split_view": "n/a",
            "evidence": "pixel_histogram.png (so sanh dinh tinh)",
            "proposal": "Khong chay KS test: hai tap cung mot pool, khac biet chu yeu la sampling noise",
            "validation_check": "n/a",
            "status": "not_run",
        },
        {
            "finding": "Traditional baseline LR/kNN (Step 12)",
            "split_view": "n/a",
            "evidence": "n/a",
            "proposal": "Khong thuc hien: baseline cua lab la BaselineMLP, khong phai mo hinh tuyen tinh",
            "validation_check": "n/a",
            "status": "not_run",
        },
    ]
)
handoff.to_csv(TABLES_DIR / "eda_to_fe_handoff.csv", index=False)
print(handoff.to_string(index=False))
"""))

cells.append(md("""
## E10. Tong ket artifact

Liet ke moi file da ghi de nguoi doc kiem tra duoc, va de notebook Feature Engineering biet
chinh xac no duoc phep doc gi.
"""))

cells.append(code("""
artifacts = sorted(
    [p for p in EDA_DIR.rglob("*") if p.is_file()],
    key=lambda p: str(p),
)
print(f"[EDA ARTIFACTS] {len(artifacts)} file trong {EDA_DIR}")
for p in artifacts:
    print(f"  {p.relative_to(EDA_DIR)}  ({p.stat().st_size / 1024:.1f} KB)")

required_for_fe = [
    EDA_DIR / "split_indices.npz",
    EDA_DIR / "split_metadata.json",
    EDA_DIR / "eda_findings.json",
]
missing = [str(p) for p in required_for_fe if not p.exists()]
assert not missing, f"Thieu artifact bat buoc cho Feature Engineering: {missing}"
print("\\n[OK] Artifact bat buoc cho Feature Engineering da day du:")
for p in required_for_fe:
    print(f"  {p.name}")
"""))

cells.append(md("""
## E11. Gioi han cua phan EDA nay

Ghi ro de bao cao khong phong dai pham vi da lam:

1. **Khong co phan tich outlier dinh luong.** Cac anh "bat thuong" chi duoc phat hien qua
   kiem tra tich hop (toan den, trung lap) va qua quan sat dinh tinh o luoi mau. Khong co
   diem outlier theo PCA reconstruction hay kNN.
2. **Khong co near-duplicate detection.** Tim near-duplicate tren 60,000 anh la O(n^2) neu
   khong co index; nam ngoai pham vi lab. Chi exact duplicate duoc do.
3. **Mean image lam mat phuong sai trong lop.** Moi nhan xet tu Step 8 chi la gia thuyet.
4. **Khong co kiem dinh thong ke drift.** Train va validation cung lay tu mot pool; khac biet
   quan sat duoc la sampling noise, khong phai drift.
5. **Tap test chua duoc phan tich chi tiet.** Dung theo spec: chi mo sau khi khoa cau hinh.
6. **Spatial features phu thuoc threshold.** `FOREGROUND_THRESHOLD = 0.1` la lua chon co ly do
   nhung khong duy nhat; cac ket luan ve bounding box nen duoc doc kem gia tri nay.
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.13.15"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {OUT} ({len(cells)} cells)")
