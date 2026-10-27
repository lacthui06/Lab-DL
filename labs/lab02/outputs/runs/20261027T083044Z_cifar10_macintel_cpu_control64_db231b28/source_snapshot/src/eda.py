"""Raw CIFAR-10 analysis and handcrafted features, independent of training.

Only checksum-verified official batches are deserialized. Statistics and feature
distributions use the train partition; held-out pixels only enter integrity checks.
"""
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from .delivery import resolve_data_dir, write_json

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CHECKSUMS = ['c99cafc152244af753f735de768cd75f', 'd4bba439e000b95fd0a9bffe97cbabec',
             '54ebc095f3ab1f0389bbae665268c751', '634d18415352ddfa80567beed471001a',
             '482c414d41f54cd18b22e5b47cb7c3cb', '40351d587109b95175f43aff81a1287e']


def read_raw(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return official train/test uint8 NHWC arrays and integer labels."""
    folder = Path(data_dir) / 'cifar-10-batches-py'
    arrays, labels = [], []
    for name, expected in zip([f'data_batch_{i}' for i in range(1, 6)] + ['test_batch'], CHECKSUMS):
        content = (folder / name).read_bytes()
        if hashlib.md5(content).hexdigest() != expected:
            raise ValueError(f'Official CIFAR-10 checksum mismatch: {name}')
        batch = pickle.loads(content, encoding='bytes')
        arrays.append(batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        labels.append(np.asarray(batch[b'labels'], dtype=np.int64))
    return np.concatenate(arrays[:5]), np.concatenate(labels[:5]), arrays[5], labels[5]


def locked_split(output: Path, labels: np.ndarray, seed: int) -> dict:
    """Create once, then validate and reuse persisted indices; never silently resplit."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'split_indices.npz'
    metadata = output / 'split_metadata.json'
    identity = hashlib.sha256(labels.tobytes()).hexdigest()
    if path.exists():
        meta = json.loads(metadata.read_text(encoding='utf-8'))
        if meta['label_sha256'] != identity or meta['seed'] != seed:
            raise ValueError('Split metadata does not match raw labels or configured seed')
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['indices_sha256']:
            raise ValueError('Saved split was modified')
        indices = dict(np.load(path, allow_pickle=False))
    else:
        train, val = train_test_split(np.arange(len(labels)), test_size=5000,
                                      random_state=seed, stratify=labels)
        indices = dict(train=np.sort(train), val=np.sort(val), test=np.arange(10000))
        np.savez(path, **indices)
        metadata.write_text(json.dumps(dict(seed=seed, label_sha256=identity,
            indices_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            split_version=f'cifar10-stratified-{seed}-45000-5000-v1',
            provenance='New EDA split reconstructed from documented algorithm; original run indices unavailable'), indent=2), encoding='utf-8')
    if (np.intersect1d(indices['train'], indices['val']).size or
            not np.array_equal(np.union1d(indices['train'], indices['val']), np.arange(len(labels)))):
        raise ValueError('Invalid or overlapping train/validation indices')
    for split in ['train', 'val']:
        if len(indices[split]) != len(np.unique(indices[split])):
            raise ValueError('Repeated index in split')
    return indices


def rgb_statistics(images: np.ndarray, indices: np.ndarray) -> dict:
    """Population RGB moments, histogram and spatial statistics; train only.

    Used by build() and export_summary(), so run reports and EDA share one formula.
    """
    sums, squares, hist = np.zeros(3), np.zeros(3), np.zeros((3,256))
    spatial_sum, spatial_sq = np.zeros((32,32)), np.zeros((32,32))
    if not len(indices):
        raise ValueError('Cannot measure an empty training partition')
    for start in range(0,len(indices),256):
        raw = images[indices[start:start+256]]
        x = raw.astype(np.float64)/255
        sums += x.sum((0,1,2)); squares += (x*x).sum((0,1,2))
        gray = x.mean(3); spatial_sum += gray.sum(0); spatial_sq += (gray*gray).sum(0)
        for c in range(3):
            hist[c] += np.bincount(raw[:,:,:,c].ravel(),minlength=256)
    n = len(indices); pixels = n*32*32
    mean = sums/pixels
    return dict(mean=mean,std=np.sqrt(np.maximum(squares/pixels-mean**2,0)),hist=hist,
                spatial_sum=spatial_sum,spatial_sq=spatial_sq,n=n,pixels=pixels)


def class_counts(labels: np.ndarray, test_labels: np.ndarray, selected: dict) -> pd.DataFrame:
    """Count the exact selected partitions; shared by EDA and per-run reports."""
    return pd.DataFrame({s:np.bincount((test_labels if s=='test' else labels)[ids],minlength=10)
                         for s,ids in selected.items()},index=CLASSES)


def export_summary(data_dir: Path, run_dir: Path, selected: dict) -> None:
    """Small per-run data evidence for reporting/finalize; never loads a network."""
    pool, labels, _, test_labels = read_raw(data_dir)
    destination = Path(run_dir)/'eda'
    destination.mkdir(parents=True,exist_ok=True)
    class_counts(labels,test_labels,selected).to_csv(destination/'class_counts.csv')
    stats = rgb_statistics(pool,selected['train'])
    write_json(destination/'summary.json',dict(dataset='CIFAR10',original_shape=[32,32,3],
        classes=CLASSES,train_mean_rgb=stats['mean'].tolist(),train_std_rgb=stats['std'].tolist(),
        normalization_policy='Descriptive train statistics only; ImageNet normalization defines input'))
    fig,axes=plt.subplots(10,5,figsize=(9,15))
    for label,row in enumerate(axes):
        ids=selected['train'][labels[selected['train']]==label][:5]
        for ax in row: ax.axis('off')
        for ax,i in zip(row,ids): ax.imshow(pool[i])
        row[0].set_title(CLASSES[label],loc='left')
    fig.suptitle('Original RGB training images by class')
    save(fig,destination,'training_examples')


def handcrafted(images: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    """Describe NHWC images without learned weights; edge threshold is fixed at 0.1."""
    chunks = []
    for start in range(0, len(indices), 256):
        ids = indices[start:start+256]
        scaled = images[ids].astype(np.float32) / 255
        gray = scaled.mean(axis=3)
        dx = np.diff(gray, axis=2)
        dy = np.diff(gray, axis=1)
        edge = np.sqrt(dx[:, :-1] ** 2 + dy[:, :, :-1] ** 2) > .1
        chunks.append(pd.DataFrame(dict(original_index=ids, label=labels[ids],
            brightness=gray.mean((1, 2)), contrast=gray.std((1, 2)),
            zero_fraction=(scaled == 0).mean((1, 2, 3)), edge_density=edge.mean((1, 2)),
            red_mean=scaled[:, :, :, 0].mean((1, 2)), green_mean=scaled[:, :, :, 1].mean((1, 2)),
            blue_mean=scaled[:, :, :, 2].mean((1, 2)))))
    result = pd.concat(chunks, ignore_index=True)
    result['class'] = [CLASSES[i] for i in result.label]
    result['split'] = 'train'
    return result


def duplicate_audit(images: dict, labels: dict, indices: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hash exact raw bytes across all splits; record both original source indices."""
    seen, suspects, checks = {}, [], []
    for split in ['train', 'val', 'test']:
        raw = images[split]
        ids = indices[split]
        selected = raw[ids]
        checks.extend([dict(split=split, check='invalid_labels', count=int(((labels[split][ids] < 0) | (labels[split][ids] >= 10)).sum())),
                       dict(split=split, check='nonfinite_pixels', count=int((~np.isfinite(selected)).sum())),
                       dict(split=split, check='constant_images', count=int((selected.max((1,2,3)) == selected.min((1,2,3))).sum()))])
        for i in ids:
            key = hashlib.sha256(raw[i].tobytes()).hexdigest()
            current = (split, int(i), int(labels[split][i]))
            if key in seen:
                previous = seen[key]
                suspects.append(dict(split=split, original_index=int(i), label=current[2],
                    previous_split=previous[0], previous_original_index=previous[1],
                    previous_label=previous[2], sha256=key))
            else:
                seen[key] = current
    columns = ['split', 'original_index', 'label', 'previous_split', 'previous_original_index', 'previous_label', 'sha256']
    return pd.DataFrame(checks), pd.DataFrame(suspects, columns=columns)


def save(fig, output: Path, name: str) -> None:
    """Save artifact; notebook gallery displays it separately without duplicate output."""
    fig.tight_layout()
    fig.savefig(output / f'{name}.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def build(root: Path, data_dir: Path | None = None) -> Path:
    """Measure raw data, preserve the locked split, and save compact figure artifacts."""
    output = root / 'outputs/eda'
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / 'config/eda.json').read_text(encoding='utf-8'))
    pool, labels, test, test_labels = read_raw(resolve_data_dir(root,data_dir))
    indices = locked_split(output, labels, cfg['seed'])
    split = json.loads((output / 'split_metadata.json').read_text(encoding='utf-8'))
    images = dict(train=pool, val=pool, test=test)
    targets = dict(train=labels, val=labels, test=test_labels)
    structural = pd.DataFrame([dict(split=s, samples=len(ids), shape='32 x 32 x 3',
        dtype=str(images[s].dtype), pixel_min=int(images[s][ids].min()),
        pixel_max=int(images[s][ids].max()), memory_mib=images[s][ids].nbytes/2**20)
        for s, ids in indices.items()])
    structural.to_csv(output / 'structural_overview.csv', index=False)
    integrity, suspects = duplicate_audit(images, targets, indices)
    integrity.to_csv(output / 'integrity_report.csv', index=False)
    suspects.to_csv(output / 'duplicate_images.csv', index=False)
    counts = class_counts(labels,test_labels,indices)
    counts.to_csv(output / 'class_distribution.csv')
    fig, ax = plt.subplots(figsize=(10, 3.4))
    counts[['train', 'val']].plot.bar(ax=ax, color=['#2563eb', '#0d9488'], rot=30)
    ax.set(ylabel='Images', title='Locked stratified split: training / validation')
    save(fig, output, 'class_distribution')
    rng = np.random.default_rng(cfg['seed'])
    fig, axes = plt.subplots(2, 10, figsize=(13, 3.5))
    class_means = []
    for c in range(10):
        ids = indices['train'][labels[indices['train']] == c]
        class_means.append(pool[ids].mean(0) / 255)
        for ax, i in zip(axes[:, c], rng.choice(ids, 2, replace=False)):
            ax.imshow(pool[i]); ax.axis('off'); ax.set_title(f'{CLASSES[c]}\n#{i}', fontsize=8)
    save(fig, output, 'sample_images')
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, name, mean in zip(axes.flat, CLASSES, class_means):
        ax.imshow(mean); ax.axis('off'); ax.set_title(name)
    save(fig, output, 'class_mean_images')
    measured = rgb_statistics(pool,indices['train'])
    mean,std,hist = measured['mean'],measured['std'],measured['hist']
    spatial_sum,spatial_sq = measured['spatial_sum'],measured['spatial_sq']
    n,pixels = measured['n'],measured['pixels']
    pd.DataFrame(dict(channel=list('RGB'), mean=mean, std=std,
        zero_fraction=hist[:,0]/pixels)).to_csv(output / 'pixel_statistics.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11,3.4))
    for c, color in enumerate(['#dc2626','#16a34a','#2563eb']):
        axes[0].plot(np.arange(256)/255, hist[c]/pixels, color=color, label='RGB'[c])
    axes[0].set(title='Measured training RGB histogram', xlabel='Scaled intensity', ylabel='Pixel fraction')
    axes[0].legend()
    axes[1].bar(np.arange(3)-.18, mean, .36, label='Mean')
    axes[1].bar(np.arange(3)+.18, std, .36, label='Std (population)')
    axes[1].set(xticks=range(3), xticklabels=list('RGB'), title='All training pixels'); axes[1].legend()
    save(fig, output, 'pixel_statistics')
    fig, axes = plt.subplots(1, 2, figsize=(9,3.6))
    for ax, values, title in zip(axes, [spatial_sum/n, spatial_sq/n-(spatial_sum/n)**2], ['Mean brightness', 'Pixel variance']):
        im=ax.imshow(values, cmap='viridis'); ax.set_title(title); fig.colorbar(im, ax=ax)
    save(fig, output, 'spatial_statistics')
    frame = handcrafted(pool, labels, indices['train'])
    frame.to_csv(output / 'handcrafted_features.csv', index=False)
    fig, axes = plt.subplots(1,3,figsize=(13,3.5))
    for ax, col in zip(axes, ['brightness','contrast','edge_density']):
        ax.boxplot([frame.loc[frame.label==c,col] for c in range(10)], tick_labels=CLASSES, showfliers=False)
        ax.set_title(col); ax.tick_params(axis='x', rotation=60)
    save(fig, output, 'handcrafted_distributions')
    fig, ax = plt.subplots(figsize=(7,4.8))
    corr = frame.select_dtypes('number').drop(columns=['original_index','label']).corr()
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set(xticks=range(len(corr)), yticks=range(len(corr)), xticklabels=corr.columns, yticklabels=corr.columns)
    ax.tick_params(axis='x', rotation=70); fig.colorbar(im, ax=ax)
    save(fig, output, 'feature_correlation')
    flat = np.asarray(class_means).reshape(10,-1); flat -= flat.mean(1, keepdims=True)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True)
    similarity = flat @ flat.T
    pd.DataFrame(similarity, index=CLASSES, columns=CLASSES).to_csv(output / 'class_similarity_matrix.csv')
    fig, ax = plt.subplots(figsize=(8,5))
    im=ax.imshow(similarity, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set(xticks=range(10),yticks=range(10),xticklabels=CLASSES,yticklabels=CLASSES,title='Correlation of class mean images (train only)')
    ax.tick_params(axis='x', rotation=50); fig.colorbar(im,ax=ax)
    save(fig,output,'class_similarity')
    preprocessing_preview(pool[indices['train'][0]], cfg, output)
    config = {**cfg, 'split_version':split['split_version'], 'preproc_version':'rgb-resize64-imagenet-v1',
              'train_mean_rgb':mean.tolist(), 'train_std_rgb':std.tolist(),
              'statistics_images':n, 'statistics_pixels_per_channel':pixels,
              'foreground_bbox':'not applicable: natural RGB images have no foreground mask; zero pixel is not a background label',
              'handcrafted_usage':'descriptive analysis only; not appended to input tensors',
              'duplicate_records':len(suspects), 'split_provenance':split['provenance']}
    (output / 'preprocessing_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    return output


def preprocessing_preview(raw: np.ndarray, cfg: dict, output: Path) -> None:
    """Show raw/resize plus real channel histograms after configured normalization."""
    resized = np.asarray(PILImage.fromarray(raw).resize((cfg['image_size'],)*2, PILImage.Resampling.BILINEAR))
    scaled = resized.astype(np.float32)/255
    normalized = (scaled-np.asarray(cfg['normalization_mean']))/np.asarray(cfg['normalization_std'])
    fig, axes = plt.subplots(1,3,figsize=(11,3.3))
    axes[0].imshow(raw); axes[0].set_title('Raw uint8 HWC: 32 x 32'); axes[0].axis('off')
    axes[1].imshow(resized); axes[1].set_title(f'Resize RGB: {cfg["image_size"]} x {cfg["image_size"]}'); axes[1].axis('off')
    for c,color in enumerate(['r','g','b']):
        axes[2].hist(normalized[:,:,c].ravel(),bins=32,alpha=.4,color=color,label='RGB'[c])
    axes[2].set(title='Normalized values: this image',xlabel='(pixel/255 - mean) / std',ylabel='Pixels'); axes[2].legend()
    save(fig,output,'preprocessing_preview')


def context(root: Path) -> Path:
    """Require measured EDA artifacts without consulting experiments or checkpoints."""
    output = root / 'outputs/eda'
    if not (output / 'preprocessing_config.json').exists():
        raise FileNotFoundError('Run scripts/analyze_data.py first; set LAB02_DATA to the raw dataset folder')
    return output


def gallery(output: Path, *names: str) -> None:
    """Display measured figures inline at a readable width."""
    for name in names:
        display(Image(filename=str(output / f'{name}.png'), width=1100))


def overview(output: Path) -> None:
    display(pd.read_csv(output / 'structural_overview.csv').round(2))


def integrity_summary(output: Path) -> None:
    table = pd.read_csv(output / 'integrity_report.csv')
    display(table.pivot(index='check', columns='split', values='count'))
    duplicate = pd.read_csv(output / 'duplicate_images.csv')
    if len(duplicate):
        display(duplicate.groupby(['previous_split','split']).size().rename('duplicate_records').to_frame())
    else:
        display(pd.DataFrame({'check':['exact duplicate images across all partitions'], 'count':[0]}))


def handoff(output: Path) -> None:
    cfg = json.loads((output / 'preprocessing_config.json').read_text(encoding='utf-8'))
    display(pd.DataFrame([['Split',cfg['split_version']], ['Preprocessing',cfg['preproc_version']],
        ['Input','float32 CHW: 3 x 64 x 64'],['Normalization','ImageNet constants; measured train RGB statistics are descriptive'],
        ['Handcrafted features','Brightness, contrast, RGB means, edge density: diagnostic only'],
        ['Foreground / bounding box',cfg['foreground_bbox']]], columns=['Contract','Value']))
