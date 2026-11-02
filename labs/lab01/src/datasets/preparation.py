"""Prepare train-only preprocessing and feature artifacts for all three models."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from .eda import run_eda_and_fe_pipeline, load_raw_arrays
from .base import load_split
from .features import extract_features, FEATURE_NAMES
from ..utils.helpers import DATA_DIR, set_seed
from ..utils.runs import Run, write_json


def prepare_data(seed=42):
    with Run("preparation", {"seed":seed, "foreground_threshold":0.1,
                            "feature_names": FEATURE_NAMES, "augmentation":False}) as run:
        set_seed(seed)
        eda = run.path / "eda"
        run_eda_and_fe_pipeline(data_dir=DATA_DIR, eda_dir=eda, seed=seed)
        pool, labels, test, test_labels = load_raw_arrays(DATA_DIR)
        tr, va = load_split(eda, expected_pool_size=len(pool))
        cfg = __import__("json").loads((eda / "preprocessing_config.json").read_text())
        mean, std = cfg["normalization"]["mean"][0], cfg["normalization"]["std"][0]
        raw = {"train": (pool[tr], labels[tr], tr), "val": (pool[va], labels[va], va),
               "test": (test, test_labels, np.arange(len(test)))}
        arrays, scaler = {}, StandardScaler()
        for split, (images, y, indices) in raw.items():
            scaled = images.astype(np.float32)/255
            features = extract_features(scaled)
            if split == "train":
                scaler.fit(features)
                write_json(run.path / "feature_scaler.json", {"names": FEATURE_NAMES,
                           "mean":scaler.mean_.tolist(), "scale":scaler.scale_.tolist(),
                           "fit_split":"train", "threshold":0.1})
            frame = pd.DataFrame(features, columns=FEATURE_NAMES)
            frame.insert(0,"label",y)
            frame.insert(0,"original_index",indices)
            frame.insert(0,"split",split)
            frame.to_csv(run.path / f"{split}_features.csv",index=False)
            arrays[split+"_pixels"] = ((scaled-mean)/std).reshape(len(images),784).astype(np.float32)
            arrays[split+"_features"] = scaler.transform(features).astype(np.float32)
            arrays[split+"_labels"] = y
            arrays[split+"_indices"] = indices
            if split == "train":
                fig, axes = plt.subplots(2,5,figsize=(18,8))
                for j, ax in enumerate(axes.flat):
                    ax.boxplot([features[y==c,j] for c in range(10)], showfliers=False)
                    ax.set_xticks(range(1,11), range(10))
                    ax.set_title(FEATURE_NAMES[j]); ax.set_xlabel("Class index")
                fig.tight_layout(); fig.savefig(run.path / "feature_distributions.png"); plt.close(fig)
                fig, ax=plt.subplots(figsize=(10,8))
                corr=frame[list(FEATURE_NAMES)].corr()
                corr.to_csv(run.path / "feature_correlations.csv")
                im=ax.imshow(corr, vmin=-1,vmax=1,cmap="coolwarm")
                ax.set_xticks(range(10), FEATURE_NAMES, rotation=90)
                ax.set_yticks(range(10), FEATURE_NAMES); fig.colorbar(im,ax=ax)
                fig.tight_layout(); fig.savefig(run.path / "feature_correlations.png"); plt.close(fig)
                fig,axes=plt.subplots(2,5,figsize=(15,6))
                for c, ax in enumerate(axes.flat):
                    image=scaled[np.flatnonzero(y==c)[0]]
                    ys,xs=np.where(image>0.1)
                    ax.imshow(image,cmap="gray",vmin=0,vmax=1)
                    if len(xs):
                        from matplotlib.patches import Rectangle
                        ax.add_patch(Rectangle((xs.min()-.5,ys.min()-.5),np.ptp(xs)+1,np.ptp(ys)+1,fill=False,edgecolor="red"))
                    f=extract_features(image[None])[0]
                    ax.scatter(f[6]*27,f[7]*27,c="orange",s=15)
                    ax.set_title(f"Class {c}"); ax.axis("off")
                fig.tight_layout(); fig.savefig(run.path / "feature_geometry.png"); plt.close(fig)
        np.savez_compressed(run.path / "prepared.npz", **arrays)
        print("Prepared:", {k:v.shape for k,v in arrays.items()})
    return run.path
