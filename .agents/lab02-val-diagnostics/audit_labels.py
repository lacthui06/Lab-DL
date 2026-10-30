"""Verify suspected labels against official batch bytes; never relabel data."""
import hashlib
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10

from run_diagnostics import LAB, OUT, RUN, CLASSES, digest, save


def main():
    raw = CIFAR10(str(LAB / 'data'), train=True, download=False)
    source = LAB / 'data/cifar-10-batches-py'
    batches = {}
    integrity = {}
    for filename, expected in CIFAR10.train_list:
        data = (source / filename).read_bytes()
        actual = hashlib.md5(data).hexdigest()
        integrity[filename] = dict(expected_md5=expected, actual_md5=actual)
        if actual != expected:
            raise ValueError(f'Official batch checksum mismatch: {filename}')
        batches[filename] = pickle.loads(data, encoding='latin1')
    with np.load(RUN / 'selected_indices.npz', allow_pickle=False) as split:
        validation = set(split['val'].tolist())
    frames = {name: pd.read_csv(OUT / name / 'per_image.csv').set_index('index')
              for name in ('resnet18', 'densenet121')}
    # Focused audit of the flagged examples, not an estimate of dataset label noise.
    indices = [49425, 21538, 11562, 12210, 20178, 8922, 43123, 49472, 25869, 18566]
    rows = []
    fig, axes = plt.subplots(2, 5, figsize=(18, 9))
    for ax, idx in zip(axes.flat, indices):
        filename = f'data_batch_{idx // 10000 + 1}'
        offset = idx % 10000
        batch = batches[filename]
        label = int(batch['labels'][offset])
        original = batch['data'][offset].reshape(3, 32, 32).transpose(1, 2, 0)
        if idx not in validation or label != raw.targets[idx] or not np.array_equal(original, raw.data[idx]):
            raise ValueError(f'Loader/raw batch disagreement for index {idx}')
        row = dict(index=idx, batch=filename, row_zero_based=offset,
                   original_filename=batch['filenames'][offset], source_label_id=label,
                   source_label=CLASSES[label], pixel_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                   loader_pixels_match=True, validation_member=True)
        for name, frame in frames.items():
            record = frame.loc[idx]
            if int(record.true_id) != label:
                raise ValueError(f'Prediction label differs from source: {name} {idx}')
            row[f'{name}_prediction'] = record.predicted_class
            row[f'{name}_p_predicted'] = float(record.p_predicted)
        rows.append(row)
        ax.imshow(original, interpolation='nearest')
        ax.set_title(f'#{idx}: source={CLASSES[label]}\n{batch["filenames"][offset]}\n'
                     f'ResNet={row["resnet18_prediction"]}; DenseNet={row["densenet121_prediction"]}', fontsize=9)
        ax.axis('off')
    fig.suptitle('Label review: original CIFAR-10 pixels enlarged without smoothing', fontsize=14)
    save(fig, OUT / 'label_review.png')
    pd.DataFrame(rows).to_csv(OUT / 'label_review.csv', index=False)
    result = dict(audited_indices=indices, official_train_batch_integrity=integrity,
                  class_names_match=raw.classes == CLASSES,
                  split_sha256=digest(RUN / 'selected_indices.npz'),
                  loader_and_prediction_labels_match_original=True,
                  labels_modified=False, note='Filename and label agree by source; semantic correctness requires human review.')
    if not result['class_names_match']:
        raise ValueError('Class order differs from dataset metadata')
    (OUT / 'label_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('10 images checked; all 5 official batch checksums match; no labels changed.')


if __name__ == '__main__':
    main()
