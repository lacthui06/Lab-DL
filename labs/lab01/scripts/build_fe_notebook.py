"""Generates notebooks/lab01_feature_engineering.ipynb.

The notebook is generated from source so the cell content is reviewable in a
diff instead of buried in a JSON blob. Re-running this script rewrites the
notebook; hand edits to the .ipynb are overwritten.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "notebooks" / "lab01_feature_engineering.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


cells = []

cells.append(md("""
# Lab 01 - Feature Engineering

Notebook nay thuc thi phan **Feature Engineering** theo `docs/FEATURE-ENGINEERING.md`, pham vi core:

| Step | Noi dung | Trang thai |
|---|---|---|
| Step 0 | Guard: xac nhan EDA artifact, split, label mapping, seed, dependency | Bat buoc |
| Step 1 | Data integrity review: doc lai ket qua EDA, khong xoa mau nao | Bat buoc |
| Step 2 | Basic transformation: `uint8` -> `float32`, `[0,255]` -> `[0,1]`, giu `[B,1,28,28]` | Bat buoc |
| Step 3 | Scaling: do mean/std **tren train split**, ghi artifact, `dataset.py` doc lai | Bat buoc |
| Step 4 | Data augmentation: kiem tra truc quan + ghi cau hinh theo run | Bat buoc |
| Step 11 | Final selection: khoa cau hinh, ghi `preprocessing_config.json` cho training | Bat buoc |
| Step 5, 6B, 6C, 9, 10 | PCA baseline, Conv AE, pretrained, representation comparison, drift | Khong thuc hien |

**Notebook nay khong huan luyen.** No chot dung mot thu: cac tham so tien xu ly ma ca ba
run se dung, va bang chung rang cac tham so do duoc do tren dung split ma training se dung.

**Vi sao normalization nam o day chu khong nam trong `dataset.py`:** gia tri mean/std la mot
ket qua do, khong phai mot hang so. Neu `dataset.py` tu tinh lai moi lan khoi tao loader thi
(a) khong co artifact de truy vet, (b) khong ai kiem tra duoc no tinh tren split nao, va
(c) hai tien trinh co the tinh ra hai gia tri khac nhau. Notebook nay do, ghi ra file;
`src/config.py` doc file do. Mot nguon duy nhat, co the audit.
"""))

cells.append(md("""
## F0. Guard - kiem tra dieu kien truoc khi tinh bat cu thu gi

Spec Step 0 yeu cau dung lai neu dieu kien quan trong chua duoc xac minh. Cac guard duoi day
that bai thi notebook dung, khong fallback im lang.
"""))

cells.append(code("""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision

if os.path.exists("/kaggle/working"):
    sys.path.insert(0, "/kaggle/working")
    PROJECT_ROOT = Path("/kaggle/working")
else:
    PROJECT_ROOT = Path("..").resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR, EDA_DIR, OUTPUTS_DIR, CLASS_NAMES, RANDOM_SEED, NUM_CLASSES,
    VAL_SPLIT, SPLIT_VERSION, PREPROC_VERSION,
)
from src.split import load_split, load_split_metadata
from src import eda as eda_mod
from src.utils import set_seed

FIGURES_DIR = EDA_DIR / "figures"
TABLES_DIR = EDA_DIR / "tables"

set_seed(RANDOM_SEED)

guard_failures = []

def guard(name, condition, detail=""):
    \"\"\"Records a guard result. A failed guard is collected, not raised here,
    so the notebook can report every unmet condition in one pass instead of
    stopping at the first one.\"\"\"
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" | {detail}" if detail else ""))
    if not condition:
        guard_failures.append(f"{name}: {detail}")
    return condition

print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
print(f"[EDA_DIR]      {EDA_DIR}")
print(f"[SPLIT_VERSION] {SPLIT_VERSION}")
print(f"[PREPROC_VERSION] {PREPROC_VERSION}")
"""))

cells.append(code("""
# Guard 1: EDA artifact phai ton tai. Khong co split thi khong the biet train la tap nao.
split_npz = EDA_DIR / "split_indices.npz"
split_json = EDA_DIR / "split_metadata.json"
findings_json = EDA_DIR / "eda_findings.json"

guard("EDA split artifact exists", split_npz.exists(), str(split_npz))
guard("EDA split metadata exists", split_json.exists(), str(split_json))
guard("EDA findings exists", findings_json.exists(), str(findings_json))

if guard_failures:
    raise RuntimeError(
        "Guard failed before any computation: chay notebooks/lab01_eda.ipynb truoc.\\n  - "
        + "\\n  - ".join(guard_failures)
    )

# Guard 2: dependencies
guard("torch available", True, torch.__version__)
guard("torchvision available", True, torchvision.__version__)
guard("numpy available", True, np.__version__)

# Guard 3: label mapping nhat quan voi so lop
guard("class name count matches NUM_CLASSES", len(CLASS_NAMES) == NUM_CLASSES,
      f"{len(CLASS_NAMES)} names vs NUM_CLASSES={NUM_CLASSES}")

# Guard 4: seed co dinh
guard("random seed is fixed", RANDOM_SEED == 42, f"RANDOM_SEED={RANDOM_SEED}")

print("\\n[OK] Guard co ban da qua.")
"""))

cells.append(code("""
# Guard 5: doc lai split va kiem tra no van dung la split ma EDA da tao
split_meta = load_split_metadata(EDA_DIR)
train_idx, val_idx = load_split(split_dir=EDA_DIR, expected_pool_size=60_000)

X_pool, y_pool, X_test_raw, y_test_raw = eda_mod.load_raw_arrays(DATA_DIR)

print(f"[SPLIT] version={split_meta['split_version']} seed={split_meta['seed']} "
      f"stratified={split_meta['stratified']}")
print(f"[SPLIT] train={len(train_idx)} val={len(val_idx)} pool={len(y_pool)}")

guard("split version matches config", split_meta["split_version"] == SPLIT_VERSION,
      f"artifact={split_meta['split_version']} config={SPLIT_VERSION}")
guard("split covers the pool", len(train_idx) + len(val_idx) == len(y_pool),
      f"{len(train_idx)} + {len(val_idx)} vs {len(y_pool)}")
guard("train/val indices disjoint", np.intersect1d(train_idx, val_idx).size == 0)
guard("validation size matches VAL_SPLIT",
      len(val_idx) == int(len(y_pool) * VAL_SPLIT),
      f"{len(val_idx)} vs {int(len(y_pool) * VAL_SPLIT)}")

train_counts = np.bincount(y_pool[train_idx], minlength=NUM_CLASSES)
val_counts = np.bincount(y_pool[val_idx], minlength=NUM_CLASSES)
guard("train partition is class-balanced", train_counts.min() == train_counts.max(),
      f"min={train_counts.min()} max={train_counts.max()}")
guard("val partition is class-balanced", val_counts.min() == val_counts.max(),
      f"min={val_counts.min()} max={val_counts.max()}")

print(f"\\n[TRAIN per-class] {train_counts.tolist()}")
print(f"[VAL   per-class] {val_counts.tolist()}")

if guard_failures:
    raise RuntimeError("Guard failed on the split artifact:\\n  - " + "\\n  - ".join(guard_failures))
"""))

cells.append(md("""
## F1. Step 1 - Data integrity review

Doc lai ket qua EDA va chot quyet dinh: **khong loai bo mau nao**.

Spec cam loai mau chi vi chung hiem hoac kho phan loai. EDA da do exact duplicate va
train/test overlap; neu ca hai bang 0 thi khong co co so nao de xoa, va official test set
tuyet doi khong bi thay doi.
"""))

cells.append(code("""
integrity = pd.read_csv(TABLES_DIR / "integrity_report.csv")
print(integrity.to_string(index=False))

def _flagged(check_name):
    row = integrity.loc[integrity["check"] == check_name, "num_flagged"]
    return int(row.iloc[0]) if len(row) else None

dup_train = _flagged("exact_duplicate_within_train")
dup_cross = _flagged("exact_duplicate_train_vs_test")
all_black = _flagged("all_black_images")
split_overlap = _flagged("split_indices_disjoint")

print(f"\\n[REVIEW] exact duplicate within train : {dup_train}")
print(f"[REVIEW] exact duplicate train vs test: {dup_cross}")
print(f"[REVIEW] all-black images             : {all_black}")
print(f"[REVIEW] split index overlap          : {split_overlap}")

review_rows = []
review_rows.append({
    "issue": "exact_duplicate_within_train",
    "count": dup_train,
    "decision": "keep" if dup_train == 0 else "keep_and_flag",
    "rationale": "Khong co duplicate nao bi phat hien; khong co gi de loai.",
})
review_rows.append({
    "issue": "exact_duplicate_train_vs_test",
    "count": dup_cross,
    "decision": "keep",
    "rationale": "Khong co overlap. Official test set khong duoc thay doi trong moi truong hop.",
})
review_rows.append({
    "issue": "all_black_images",
    "count": all_black,
    "decision": "keep" if all_black == 0 else "flag_for_manual_review",
    "rationale": "Anh toan den se khong hoc duoc, nhung chi loai sau khi kiem tra thu cong.",
})
review_rows.append({
    "issue": "near_duplicates",
    "count": None,
    "decision": "not_checked",
    "rationale": "Chua thuc hien: O(n^2) tren 60,000 anh, ngoai pham vi lab.",
})

data_review = pd.DataFrame(review_rows)
data_review.to_csv(TABLES_DIR / "data_review.csv", index=False)
print("\\n" + data_review.to_string(index=False))

cleaning_log = pd.DataFrame([
    {
        "step": "integrity_review",
        "action": "no_samples_removed",
        "num_affected": 0,
        "reason": "Khong co duplicate, khong co anh toan den, khong co overlap train/test.",
        "applies_to": "train, val, test",
    }
])
cleaning_log.to_csv(TABLES_DIR / "data_cleaning_log.csv", index=False)
print("\\n" + cleaning_log.to_string(index=False))
"""))

cells.append(md("""
## F2. Step 2 - Basic transformation

Chuyen `uint8 [0, 255]` -> `float32 [0, 1]`, giu nguyen cau truc anh `[1, 28, 28]` de
`nn.Flatten()` trong model tu xu ly. Khong resize, khong pad: chua co ly do thuc nghiem nao,
va `FashionCNN` khong con trong pham vi lab.

Kiem tra ca ba view duoc tach bach, vi day la cho nhieu pipeline nham lan:

| View | Khoang gia tri | Dung de lam gi |
|---|---|---|
| raw | `uint8 [0, 255]` | kiem tra integrity, hash, thong ke goc |
| scaled | `float32 [0, 1]` | do mean/std, kiem tra pixel range |
| model | `float32`, khong con gioi han | dau vao model, KHONG dung de kiem tra pixel range |
"""))

cells.append(code("""
def to_scaled(images_uint8):
    \"\"\"uint8 [0,255] -> float32 [0,1]. Tuong duong ToTensor() cua torchvision.\"\"\"
    if images_uint8.dtype != np.uint8:
        raise TypeError(f"Expected uint8 input, got {images_uint8.dtype}")
    return images_uint8.astype(np.float32) / 255.0

X_tr_raw, X_va_raw = X_pool[train_idx], X_pool[val_idx]
X_tr_scaled = to_scaled(X_tr_raw)
X_va_scaled = to_scaled(X_va_raw)

print("[RAW VIEW]")
print(f"  train dtype={X_tr_raw.dtype} shape={X_tr_raw.shape} "
      f"range=[{X_tr_raw.min()}, {X_tr_raw.max()}]")
print("[SCALED VIEW]")
print(f"  train dtype={X_tr_scaled.dtype} shape={X_tr_scaled.shape} "
      f"range=[{X_tr_scaled.min():.4f}, {X_tr_scaled.max():.4f}]")
print(f"  val   dtype={X_va_scaled.dtype} shape={X_va_scaled.shape} "
      f"range=[{X_va_scaled.min():.4f}, {X_va_scaled.max():.4f}]")

guard("scaled view is float32", X_tr_scaled.dtype == np.float32, str(X_tr_scaled.dtype))
guard("scaled view within [0, 1]",
      X_tr_scaled.min() >= 0.0 and X_tr_scaled.max() <= 1.0,
      f"[{X_tr_scaled.min()}, {X_tr_scaled.max()}]")
guard("scaling preserves shape", X_tr_scaled.shape == X_tr_raw.shape,
      f"{X_tr_scaled.shape} vs {X_tr_raw.shape}")

# ToTensor() cua torchvision phai cho ra dung ket qua nay; kiem tra bang mot anh that
from torchvision import transforms
sample_tensor = transforms.ToTensor()(X_tr_raw[0])
guard("ToTensor matches manual scaling",
      bool(np.allclose(sample_tensor.numpy().squeeze(), X_tr_scaled[0], atol=1e-6)),
      f"max abs diff={np.abs(sample_tensor.numpy().squeeze() - X_tr_scaled[0]).max():.2e}")
print(f"  ToTensor output shape={tuple(sample_tensor.shape)} dtype={sample_tensor.dtype}")
"""))

cells.append(md("""
## F3. Step 3 - Scaling: do mean/std va chot artifact

**Quy tac (spec Step 3):** mean/std la thong ke gop tren toan bo pixel cua **train split**,
truoc augmentation, dung population variance (`ddof=0`). Khong tinh lai rieng cho val/test.
Khong lay trung binh std tung anh.

Hai gia tri nay quyet dinh truc tiep chat luong huan luyen, nen chung duoc do lai o day tren
dung split ma training se dung, chu khong duoc chep tu config cu.
"""))

cells.append(code("""
# Chi train split duoc dung de fit. Val duoc do de doi chieu, khong dung de fit.
train_norm = eda_mod.compute_normalization_stats(X_tr_scaled)
val_norm = eda_mod.compute_normalization_stats(X_va_scaled)

measured_mean = train_norm["mean"]
measured_std = train_norm["std"]

print("[MEASURED ON TRAIN SPLIT - scaled [0,1], ddof=0]")
print(f"  num_images              {train_norm['num_images']}")
print(f"  num_pixels              {train_norm['num_pixels']}")
print(f"  mean                    {measured_mean:.6f}")
print(f"  std (ddof=0)            {measured_std:.6f}")
print(f"  std (ddof=1, rejected)  {float(X_tr_scaled.reshape(-1).std(ddof=1)):.6f}")
print(f"  mean of per-image std   {train_norm['per_image_std_mean']:.6f}  <- KHONG duoc dung")
print(f"  zero pixel fraction     {train_norm['zero_pixel_fraction']:.6f}")

print("\\n[VAL - descriptive only, never used to fit]")
print(f"  mean={val_norm['mean']:.6f}  std={val_norm['std']:.6f}  n={val_norm['num_images']}")

guard("train mean is finite", np.isfinite(measured_mean))
guard("train std is finite and positive", np.isfinite(measured_std) and measured_std > 0,
      f"std={measured_std}")
guard("all train pixels counted",
      train_norm["num_pixels"] == len(X_tr_scaled) * X_tr_scaled.shape[1] * X_tr_scaled.shape[2],
      f"{train_norm['num_pixels']} pixels")
guard("global std differs from mean-of-per-image-std",
      abs(measured_std - train_norm["per_image_std_mean"]) > 1e-3,
      f"global={measured_std:.6f} vs per-image-mean={train_norm['per_image_std_mean']:.6f}")
"""))

cells.append(code("""
# Kiem tra thuc nghiem: sau Normalize, tensor phai co mean ~0 va std ~1.
# Day la phep kiem chung ma config cu (std=0.3205) se truot.
def normalize_array(images, mean, std):
    return (images - mean) / std

normalized_train = normalize_array(X_tr_scaled, measured_mean, measured_std)
normalized_val = normalize_array(X_va_scaled, measured_mean, measured_std)

norm_mean = float(normalized_train.mean())
norm_std = float(normalized_train.std(ddof=0))

print("[AFTER NORMALIZE - train split]")
print(f"  mean = {norm_mean:+.6f}")
print(f"  std  = {norm_std:.6f}")
print(f"  range = [{normalized_train.min():.4f}, {normalized_train.max():.4f}]  "
      f"(ngoai [0,1] la dung, day la model view)")

# Doi chieu: config CU se cho std bao nhieu?
from src.config import FALLBACK_MEAN, FALLBACK_STD
legacy = normalize_array(X_tr_scaled, FALLBACK_MEAN[0], FALLBACK_STD[0])
print("\\n[AFTER NORMALIZE - legacy config, de doi chieu]")
print(f"  mean = {float(legacy.mean()):+.6f}")
print(f"  std  = {float(legacy.std(ddof=0)):.6f}  <- lech {float(legacy.std(ddof=0)) - 1.0:+.4f} so voi 1.0")

guard("normalized train mean is ~0", abs(norm_mean) < 1e-4, f"mean={norm_mean:+.6f}")
guard("normalized train std is ~1", abs(norm_std - 1.0) < 1e-3, f"std={norm_std:.6f}")
guard("legacy config would FAIL the unit-variance check",
      abs(float(legacy.std(ddof=0)) - 1.0) > 0.05,
      f"legacy std={float(legacy.std(ddof=0)):.4f}")
"""))

cells.append(code("""
# Ghi artifact ma src/config.py se doc.
# File nay la nguon duy nhat cua mean/std cho moi run.
preprocessing_config = {
    "preproc_version": PREPROC_VERSION,
    "split_version": split_meta["split_version"],
    "split_seed": split_meta["seed"],
    "val_split": split_meta["val_split"],
    "train_size": int(len(train_idx)),
    "val_size": int(len(val_idx)),
    "normalization": {
        "mean": [round(float(measured_mean), 6)],
        "std": [round(float(measured_std), 6)],
        "method": "global pooled statistics over all pixels of the train split",
        "variance": "population (ddof=0)",
        "fitted_on": "train split only, scaled [0,1], before augmentation",
        "num_images": int(train_norm["num_images"]),
        "num_pixels": int(train_norm["num_pixels"]),
        "rejected_alternative": {
            "value": round(float(train_norm["per_image_std_mean"]), 6),
            "reason": "mean of per-image std is not the global std; spec forbids using it",
        },
    },
    "transform": {
        "train": ["RandomHorizontalFlip(p=0.5)", "RandomRotation(degrees=10)",
                  "ToTensor()", "Normalize(mean, std)"],
        "eval": ["ToTensor()", "Normalize(mean, std)"],
        "augmentation_applied_to": "train only",
    },
    "runs": {
        "Run1_Baseline_MLP": {"augmentation": False, "epochs": 5, "scheduler": False},
        "Run2_Tuned_MLP": {"augmentation": True, "epochs": 10, "scheduler": True},
    },
    "library_versions": {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
}

config_path = EDA_DIR / "preprocessing_config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(preprocessing_config, f, indent=2)

print(f"[ARTIFACT] {config_path}")
print(json.dumps(preprocessing_config, indent=2))
"""))

cells.append(md("""
## F4. Step 4 - Data augmentation

Augmentation chi ap dung cho train. Val/test dung transform tat dinh.

`RandomHorizontalFlip` duoc dung vi quan ao gan nhu doi xung trai-phai: mot chiec ao bi lat
van la mot chiec ao cung lop. `RandomVerticalFlip` **khong** duoc dung: lat doc bien mot chiec
giay thanh hinh dang khong ton tai trong du lieu that, nen no day model hoc mot bat bien sai.
`RandomRotation(degrees=10)` la goc nho, du de mo phong tu the chup hoi nghieng ma khong lam
thay doi ngu nghia lop.

Spec cam mac dinh rang augmentation luon cai thien ket qua. O day no chi duoc mo ta va kiem tra
truc quan; tac dong cua no duoc do bang validation trong notebook training, khong duoc ket luan
truoc.
"""))

cells.append(code("""
from torchvision import transforms
from src.config import FASHION_MNIST_MEAN, FASHION_MNIST_STD

augment_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
])

print(f"[TRANSFORM] train mean={FASHION_MNIST_MEAN} std={FASHION_MNIST_STD}")
print(f"[TRANSFORM] eval  mean={FASHION_MNIST_MEAN} std={FASHION_MNIST_STD}")

# Chon mot anh that cua lop Shirt (label 6) - lop yeu nhat trong bao cao truoc
from PIL import Image
shirt_idx = int(np.flatnonzero(y_pool[train_idx] == 6)[0])
shirt_image = Image.fromarray(X_tr_raw[shirt_idx], mode="L")

set_seed(RANDOM_SEED)
fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
for i, ax in enumerate(axes.ravel()):
    augmented = augment_transform(shirt_image)
    from src.utils import unnormalize
    ax.imshow(unnormalize(augmented).squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"sample {i + 1}", fontsize=9)
    ax.axis("off")

fig.suptitle(
    "RandomHorizontalFlip(p=0.5) + RandomRotation(10 deg) applied 8 times to one Shirt image\\n"
    "Train only. Rotation stays small so the garment class is unchanged.",
    fontsize=11, fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.90))
aug_path = FIGURES_DIR / "augmentation_preview.png"
fig.savefig(str(aug_path), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\\n[FIG] {aug_path}")
"""))

cells.append(code("""
# Eval transform phai tat dinh: cung mot anh, hai lan goi, ket qua giong het.
eval_a = eval_transform(shirt_image)
eval_b = eval_transform(shirt_image)
guard("eval transform is deterministic",
      bool(torch.equal(eval_a, eval_b)),
      f"max abs diff={(eval_a - eval_b).abs().max().item():.2e}")

# Augment transform phai ngau nhien: goi nhieu lan phai ra ket qua khac nhau.
set_seed(RANDOM_SEED)
aug_samples = [augment_transform(shirt_image) for _ in range(6)]
distinct = len({tuple(s.flatten().tolist()[:32]) for s in aug_samples})
guard("augment transform is stochastic", distinct > 1, f"{distinct} distinct of 6 draws")

# Shape/dtype contract cua ca hai transform
guard("eval transform yields [1, 28, 28] float32",
      tuple(eval_a.shape) == (1, 28, 28) and eval_a.dtype == torch.float32,
      f"{tuple(eval_a.shape)} {eval_a.dtype}")
"""))

cells.append(md("""
## F5. Data contract - kiem tra dau ra cua pipeline truoc khi training

Day la hop dong ma model va loss dua vao. Sai o day thi training hong am tham chu khong bao loi.
"""))

cells.append(code("""
from src.dataset import get_dataloaders
from src.model import build_model, count_parameters

train_loader, val_loader, test_loader, class_names = get_dataloaders(
    data_dir=DATA_DIR, batch_size=64, val_split=VAL_SPLIT,
    augment=True, seed=RANDOM_SEED, num_workers=0,
)

images, targets = next(iter(train_loader))
print(f"[BATCH] images {tuple(images.shape)} {images.dtype} | targets {tuple(targets.shape)} {targets.dtype}")
print(f"[BATCH] images finite={bool(torch.isfinite(images).all())} "
      f"range=[{images.min():.4f}, {images.max():.4f}]")

guard("batch shape is [B, 1, 28, 28]", tuple(images.shape[1:]) == (1, 28, 28),
      str(tuple(images.shape)))
guard("images are float32", images.dtype == torch.float32, str(images.dtype))
guard("images are finite", bool(torch.isfinite(images).all()))
guard("targets are int64", targets.dtype == torch.int64, str(targets.dtype))
guard("targets within [0, 9]", int(targets.min()) >= 0 and int(targets.max()) < NUM_CLASSES,
      f"[{int(targets.min())}, {int(targets.max())}]")

# Logits contract cho ca hai kien truc
for model_type in ("baseline_mlp", "tuned_mlp"):
    model = build_model(model_type).eval()
    with torch.no_grad():
        logits = model(images[:4])
    params = count_parameters(model)
    print(f"  {model_type:<14} logits={tuple(logits.shape)} "
          f"params={params['total']:,} finite={bool(torch.isfinite(logits).all())}")
    guard(f"{model_type} logits shape is [B, 10]", tuple(logits.shape) == (4, NUM_CLASSES),
          str(tuple(logits.shape)))
    guard(f"{model_type} logits are finite", bool(torch.isfinite(logits).all()))

# Loader sizes: train loader dung drop_last=True nen so mau mot vong < 54,000.
print(f"\\n[LOADER] train batches={len(train_loader)} "
      f"(drop_last=True, mot vong chi thay {len(train_loader) * 64} / {len(train_idx)} mau)")
print(f"[LOADER] val batches={len(val_loader)} | test batches={len(test_loader)}")
guard("val loader covers every validation sample",
      len(val_loader.dataset) == len(val_idx),
      f"{len(val_loader.dataset)} vs {len(val_idx)}")
"""))

cells.append(md("""
## F6. Step 11 - Final selection: khoa cau hinh

Ghi lai cau hinh da khoa va kiem tra lan cuoi rang `src/config.py` doc dung artifact vua tao.

Day la buoc cuoi truoc khi training: sau cell nay, cau hinh tien xu ly khong doi nua. Neu doi,
phai version hoa va chay lai cac run lien quan.
"""))

cells.append(code("""
# Import lai config de xac nhan no doc artifact vua ghi (module da import truoc do
# nen can reload de thay file moi).
import importlib
import src.config as config_module
importlib.reload(config_module)

print(f"[CONFIG] FASHION_MNIST_MEAN = {config_module.FASHION_MNIST_MEAN}")
print(f"[CONFIG] FASHION_MNIST_STD  = {config_module.FASHION_MNIST_STD}")
print(f"[CONFIG] NORMALIZATION_SOURCE = {config_module.NORMALIZATION_SOURCE}")
print(f"[CONFIG] PREPROC_VERSION = {config_module.PREPROC_VERSION}")
print(f"[CONFIG] SPLIT_VERSION = {config_module.SPLIT_VERSION}")

guard("config reads the artifact, not the fallback",
      config_module.NORMALIZATION_SOURCE.startswith("artifact:"),
      config_module.NORMALIZATION_SOURCE)
guard("config mean matches the measurement",
      abs(config_module.FASHION_MNIST_MEAN[0] - measured_mean) < 1e-6,
      f"{config_module.FASHION_MNIST_MEAN[0]} vs {measured_mean}")
guard("config std matches the measurement",
      abs(config_module.FASHION_MNIST_STD[0] - measured_std) < 1e-6,
      f"{config_module.FASHION_MNIST_STD[0]} vs {measured_std}")

# Kiem tra lan cuoi bang chinh transform cua dataset (khong phai transform tu che)
from src.dataset import get_transforms
final_transform = get_transforms(augment=False)
final_tensor = final_transform(shirt_image)
final_mean = float(final_tensor.mean())
final_std = float(final_tensor.std(correction=0))
print(f"\\n[FINAL CHECK] mot anh qua get_transforms(augment=False): "
      f"mean={final_mean:+.4f} std={final_std:.4f}")
print("  (std cua MOT anh khong phai 1.0 - chi kiem tra tensor finite va dung shape)")
guard("final transform yields finite tensor", bool(torch.isfinite(final_tensor).all()))
guard("final transform yields [1, 28, 28]",
      tuple(final_tensor.shape) == (1, 28, 28), str(tuple(final_tensor.shape)))
"""))

cells.append(code("""
# Bang khoa cau hinh cuoi cung, de nguoi doc doi chieu voi walkthrough
locked = pd.DataFrame([
    {"item": "split_version", "value": split_meta["split_version"]},
    {"item": "split_seed", "value": split_meta["seed"]},
    {"item": "train_size", "value": int(len(train_idx))},
    {"item": "val_size", "value": int(len(val_idx))},
    {"item": "test_size", "value": int(len(y_test_raw))},
    {"item": "preproc_version", "value": PREPROC_VERSION},
    {"item": "normalization_mean", "value": round(float(measured_mean), 6)},
    {"item": "normalization_std", "value": round(float(measured_std), 6)},
    {"item": "normalization_source", "value": config_module.NORMALIZATION_SOURCE},
    {"item": "train_transform", "value": "ToTensor + Normalize (+ flip/rotation when augment=True)"},
    {"item": "eval_transform", "value": "ToTensor + Normalize, deterministic"},
    {"item": "runs", "value": "Run1_Baseline_MLP (no aug), Run2_Tuned_MLP (aug)"},
])
locked.to_csv(TABLES_DIR / "preprocessing_locked.csv", index=False)
print(locked.to_string(index=False))

summary = {
    "guard_failures": guard_failures,
    "guards_passed": len(guard_failures) == 0,
    "preprocessing_config": str(config_path),
    "split_version": split_meta["split_version"],
    "preproc_version": PREPROC_VERSION,
    "normalization_mean": round(float(measured_mean), 6),
    "normalization_std": round(float(measured_std), 6),
    "legacy_std_rejected": float(FALLBACK_STD[0]),
    "legacy_normalized_std": round(float(legacy.std(ddof=0)), 6),
}
with open(EDA_DIR / "feature_engineering_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\\n" + "=" * 68)
if guard_failures:
    print(f"[FAILED] {len(guard_failures)} guard khong dat:")
    for item in guard_failures:
        print(f"  - {item}")
    print("KHONG duoc dua pipeline nay len Kaggle.")
else:
    print("[PASSED] Tat ca guard deu dat. Cau hinh tien xu ly da khoa.")
    print(f"  normalization: mean={measured_mean:.6f} std={measured_std:.6f}")
    print(f"  legacy std {FALLBACK_STD[0]} da bi loai (cho std={float(legacy.std(ddof=0)):.4f} sau normalize)")
print("=" * 68)
print(f"\\n[ARTIFACT] {EDA_DIR / 'feature_engineering_summary.json'}")

assert not guard_failures, "Feature Engineering guard failed; do not proceed to training."
"""))

cells.append(md("""
## F7. Ban giao sang notebook training

Notebook training tren Kaggle chi can hai dieu kien:

1. `outputs/eda/split_indices.npz` + `split_metadata.json` ton tai (de `get_dataloaders` doc lai).
2. `outputs/eda/preprocessing_config.json` ton tai (de `src/config.py` doc mean/std).

Ca hai deu nam trong `outputs/eda/`, duoc dong goi vao `outputs.zip` cung cac artifact khac.
Cell guard trong notebook training se dung lai neu thieu, thay vi am tham roi ve gia tri cu.

### Nhung gi KHONG duoc lam o buoc nay

- Khong dung test set de chon mean/std, augmentation hay bat cu tham so nao.
- Khong fit PCA hay bat cu bien doi hoc nao (nhanh tuy chon, chua duoc xac nhan).
- Khong loai mau nao: EDA da chung minh khong co duplicate va khong co overlap.
- Khong ket luan augmentation co cai thien ket qua hay khong: dieu do chi duoc tra loi bang
  validation trong notebook training.
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
