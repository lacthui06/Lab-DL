"""Meaningful safeguards independent of downloads or benchmark metrics."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from src.datasets.data import balanced_subset, image_transform
from src.models.models import build_model, classifier, configure_trainable, features, training_mode
from PIL import Image


@pytest.mark.parametrize("name", ["resnet18", "densenet121"])
def test_feature_classifier_equivalence_and_freezing(name):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model(name, pretrained=False)
    configure_trainable(model, name, "head")
    model.eval()
    image = image_transform(name)(Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))).unsqueeze(0)
    assert image.shape == (1, 3, 224, 224)
    with torch.no_grad():
        direct = model(image)
        cached = classifier(model, name)(features(model, name, image))
    torch.testing.assert_close(direct, cached)
    assert direct.shape == (1, 10)
    head_ids = {id(p) for p in classifier(model, name).parameters()}
    assert all(p.requires_grad == (id(p) in head_ids) for p in model.parameters())
    configure_trainable(model, name, "finetune")
    training_mode(model)
    assert all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    frozen_name, frozen = next((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    before = frozen.detach().clone()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.SGD([p for _, p in trainable], lr=.01)
    loss = model(image).sum()
    loss.backward()
    assert frozen.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in trainable)
    optimizer.step()
    torch.testing.assert_close(before, dict(model.named_parameters())[frozen_name], rtol=0, atol=0)
    restored = build_model(name, pretrained=False)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)


def test_balanced_pilot_has_no_leakage():
    labels = np.repeat(np.arange(10), 100)
    train_pool = np.concatenate([np.arange(c*100, c*100+80) for c in range(10)])
    val_pool = np.setdiff1d(np.arange(1000), train_pool)
    train = balanced_subset(train_pool, labels, 200, 42)
    val = balanced_subset(val_pool, labels, 100, 42)
    assert not set(train) & set(val)
    assert np.bincount(labels[train]).tolist() == [20] * 10
    assert np.bincount(labels[val]).tolist() == [10] * 10
    np.testing.assert_array_equal(train, balanced_subset(train_pool, labels, 200, 42))


@pytest.mark.parametrize('size', [64, 224])
def test_rgb_preserved_and_normalization(size):
    from src.models.models import SPECS
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    result = image_transform('resnet18', size)(Image.fromarray(rgb))
    assert result.shape == (3, size, size)
    transform = SPECS['resnet18'][1].transforms()
    recovered = result * torch.tensor(transform.std)[:, None, None] + torch.tensor(transform.mean)[:, None, None]
    torch.testing.assert_close(recovered[0], torch.ones(size, size))
    torch.testing.assert_close(recovered[1:], torch.zeros(2, size, size), atol=1e-6, rtol=0)


def test_cifar_split_and_labels(tmp_path):
    from src.datasets.data import prepare_indices
    from src.utils.reporting import CLASSES
    from torchvision.datasets import CIFAR10
    from pathlib import Path
    import os
    data = Path(os.getenv('LAB02_DATA', str(Path(__file__).resolve().parents[1] / 'data')))
    if not (data / 'cifar-10-batches-py').exists():
        pytest.skip('Official CIFAR10 not downloaded yet')
    cfg = dict(seed=42, train_limit=0, val_limit=0, test_limit=0)
    selected = prepare_indices(data, tmp_path, cfg)
    pool = CIFAR10(str(data), train=True)
    test = CIFAR10(str(data), train=False)
    assert pool.classes == CLASSES
    assert [len(selected[s]) for s in ['train', 'val', 'test']] == [45000, 5000, 10000]
    assert not np.intersect1d(selected['train'], selected['val']).size
    np.testing.assert_array_equal(np.union1d(selected['train'], selected['val']), np.arange(50000))
    for split, count in [('train', 4500), ('val', 500)]:
        np.testing.assert_array_equal(np.bincount(np.asarray(pool.targets)[selected[split]]), [count]*10)
    np.testing.assert_array_equal(np.bincount(test.targets), [1000]*10)
