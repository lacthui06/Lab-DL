"""Feature cache extraction preserves sample identity and full-model predictions."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.models import build_model, features, last_block, CachedTail
from src.training import feature_cache as cache


@pytest.mark.parametrize('name', ['resnet18', 'densenet121'])
@pytest.mark.parametrize('prefix_enabled', [False, True])
def test_extraction_preserves_features_labels_indices_and_prefix(tmp_path, monkeypatch, name, prefix_enabled):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model(name, pretrained=False)
    images = torch.randn(3, 3, 64, 64)
    labels = torch.tensor([8, 1, 4])
    indices = torch.tensor([19, 2, 11])  # Deliberately not sorted.
    loader = DataLoader(TensorDataset(images, labels, indices), batch_size=2)
    calls = []

    def image_loader(data_dir, split, selected, model_name, batch, seed, **kwargs):
        calls.append((split, model_name, batch, seed, kwargs))
        np.testing.assert_array_equal(selected, indices.numpy())
        return loader

    monkeypatch.setattr(cache, 'image_loader', image_loader)
    cfg = dict(feature_batch_size=2, seed=42, image_size=64, cache_frozen_prefix=prefix_enabled)
    destination = tmp_path / 'train_features.pt'
    actual, seconds = cache.cached_features(model, name, tmp_path, 'train', indices.numpy(),
                                            cfg, torch.device('cpu'), destination)
    assert seconds >= 0 and not model.training
    assert calls == [('train', name, 2, 42, {'image_size': 64})]
    saved = torch.load(destination, map_location='cpu', weights_only=True)
    assert set(saved) == {'features', 'labels', 'indices'}
    for key, value in zip(('features', 'labels', 'indices'), actual):
        torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], labels, rtol=0, atol=0)
    torch.testing.assert_close(actual[2], indices, rtol=0, atol=0)
    captured = []
    handle = last_block(model, name).register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():
            expected = torch.cat([features(model, name, batch[0]) for batch in loader])
    finally:
        handle.remove()
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    prefix_path = tmp_path / 'train_prefix.npy'
    assert prefix_path.exists() == prefix_enabled
    if prefix_enabled:
        prefix = torch.from_numpy(np.load(prefix_path))
        torch.testing.assert_close(prefix, torch.cat(captured), rtol=0, atol=0)
        with torch.inference_mode():
            direct = model(images)
            cached = CachedTail(model, name)(prefix)
        torch.testing.assert_close(cached, direct, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('shuffle', [False, True])
def test_tensor_loader_keeps_alignment_and_seeded_order(shuffle):
    indices = torch.arange(9)
    data = (indices[:, None].float() * 10, indices % 3, indices)

    def collect():
        batches = list(cache.tensor_loader(data, batch=4, seed=42, shuffle=shuffle))
        return tuple(torch.cat([batch[column] for batch in batches]) for column in range(3))

    first, second = collect(), collect()
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert sorted(first[2].tolist()) == indices.tolist()
    torch.testing.assert_close(first[0][:, 0], first[2].float() * 10)
    torch.testing.assert_close(first[1], first[2] % 3)
    if not shuffle:
        torch.testing.assert_close(first[2], indices, rtol=0, atol=0)


def test_extraction_failure_removes_hook_and_allows_retry(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model('resnet18', pretrained=False)
    data = (torch.randn(3, 3, 64, 64), torch.tensor([0, 1, 2]), torch.tensor([7, 1, 9]))
    monkeypatch.setattr(cache, 'image_loader', lambda *a, **k: DataLoader(TensorDataset(*data), batch_size=2))
    cfg = dict(feature_batch_size=2, seed=42, image_size=64, cache_frozen_prefix=True)
    original = cache.features

    def fail_after_forward(*args):
        original(*args)
        raise RuntimeError('fixture extraction interrupted')

    block = last_block(model, 'resnet18')
    existing_hooks = set(block._forward_pre_hooks)
    destination = tmp_path / 'train_features.pt'
    monkeypatch.setattr(cache, 'features', fail_after_forward)
    with pytest.raises(RuntimeError, match='fixture extraction interrupted'):
        cache.cached_features(model, 'resnet18', tmp_path, 'train', data[2].numpy(),
                              cfg, torch.device('cpu'), destination)
    assert set(block._forward_pre_hooks) == existing_hooks
    assert not destination.exists()
    monkeypatch.setattr(cache, 'features', original)
    recovered, _ = cache.cached_features(model, 'resnet18', tmp_path, 'train', data[2].numpy(),
                                         cfg, torch.device('cpu'), destination)
    assert destination.exists()
    torch.testing.assert_close(recovered[2], data[2], rtol=0, atol=0)


def test_existing_exports_resolve_to_cache_module():
    import src.training as training
    from src.training import experiment
    assert training.cached_features is experiment.cached_features is cache.cached_features
    assert training.tensor_loader is experiment.tensor_loader is cache.tensor_loader
