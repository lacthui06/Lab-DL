"""Verify the approved smaller tail, dropout compatibility and data boundaries."""
import pytest
import torch
from PIL import Image
import numpy as np

from src.models.backbones import build_model, configure_trainable, parameter_counts, training_mode
from src.datasets.classification import IndexedImages
from src.training.tuning import fine_candidates


@pytest.mark.parametrize('name', ['resnet18', 'densenet121'])
def test_smaller_tail_and_dropout_roundtrip(name):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model(name, pretrained=False, dropout=.2)
    configure_trainable(model, name, 'finetune', 'last_block')
    previous = parameter_counts(model)['trainable']
    configure_trainable(model, name, 'finetune', 'last_units')
    assert parameter_counts(model)['trainable'] < previous
    allowed = (('layer4.1.', 'fc.') if name == 'resnet18' else
               tuple(f'features.denseblock4.denselayer{i}.' for i in range(13,17)) + ('features.norm5.', 'classifier.'))
    assert all(n.startswith(allowed) for n,p in model.named_parameters() if p.requires_grad)
    assert any(p.requires_grad for n,p in model.named_parameters() if n.startswith(allowed[0]))
    training_mode(model)
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert all(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))
    x = torch.randn(2,3,64,64)
    before = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    torch.nn.functional.cross_entropy(model(x), torch.tensor([0,1])).backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all(torch.equal(before[n], p) for n,p in model.named_parameters() if not p.requires_grad)
    restored = build_model(name, pretrained=False, dropout=.2)
    restored.load_state_dict(model.state_dict())
    model.eval(); restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)
    legacy = build_model(name, pretrained=False)
    build_model(name, pretrained=False, dropout=0).load_state_dict(legacy.state_dict())


def test_augmentation_train_only_and_stable_validation(monkeypatch):
    class FakeCIFAR:
        def __init__(self, *args, **kwargs):
            self.image = Image.fromarray(np.arange(32*32*3, dtype=np.uint8).reshape(32,32,3))
        def __getitem__(self, index):
            return self.image, 3
    monkeypatch.setattr('src.datasets.classification.datasets.CIFAR10', FakeCIFAR)
    train = IndexedImages('unused', 'train', [0], 'resnet18', 64, True)
    val = IndexedImages('unused', 'val', [0], 'resnet18', 64)
    assert torch.equal(val[0][0], val[0][0])
    samples = [train[0][0] for _ in range(6)]
    assert samples[0].shape == (3,64,64)
    assert any(not torch.equal(samples[0], s) for s in samples[1:])
    with pytest.raises(ValueError):
        IndexedImages('unused', 'val', [0], 'resnet18', 64, True)
    with pytest.raises(ValueError):
        IndexedImages('unused', 'test', [0], 'resnet18', 64, True)


def test_invalid_cache_and_dropout():
    cfg = dict(finetune_epochs=10, finetune_backbone_lr=1e-5, finetune_head_lr=1e-4,
               unfreeze='last_units', dropout=.2, train_augmentation=True, cache_frozen_prefix=False)
    assert fine_candidates(cfg)[0]['unfreeze'] == 'last_units'
    with pytest.raises(ValueError):
        fine_candidates({**cfg, 'cache_frozen_prefix':True})
    for p in (-.1, 1., float('nan')):
        with pytest.raises(ValueError):
            build_model('resnet18', pretrained=False, dropout=p)
