"""Prove cached-prefix training preserves logits, gradients and checkpoint keys."""
import pytest
import torch
from src.models.models import build_model, configure_trainable, training_mode, last_block, CachedTail


@pytest.mark.parametrize('name', ['resnet18', 'densenet121'])
def test_cached_prefix_training_equivalence(name):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model(name, pretrained=False)
    configure_trainable(model, name, 'finetune')
    training_mode(model)
    image = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    captured = []
    hook = last_block(model, name).register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    logits = model(image)
    hook.remove()
    parameters = [p for p in model.parameters() if p.requires_grad]
    direct_gradients = torch.autograd.grad(torch.nn.functional.cross_entropy(logits, labels), parameters)
    tail = CachedTail(model, name)
    cached_logits = tail(captured[0])
    cached_gradients = torch.autograd.grad(torch.nn.functional.cross_entropy(cached_logits, labels), parameters)
    torch.testing.assert_close(logits, cached_logits, rtol=1e-5, atol=1e-5)
    for direct, cached in zip(direct_gradients, cached_gradients):
        torch.testing.assert_close(direct, cached, rtol=1e-5, atol=1e-5)
    assert set(tail.state_dict()) == set(model.state_dict())
