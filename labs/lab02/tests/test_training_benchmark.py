"""Training benchmark contracts; synthetic inputs never stand in for published results."""
import pytest
import torch
from torch import nn

from src.utils.training_benchmark import benchmark_steps, prepare_training, validate_settings


def test_counts_fail_before_loading_inputs():
    for change in ({'batch_size': 2}, {'repeats': 0}, {'warmup': -1},
                   {'threads': True}, {'sample_interval': float('nan')}):
        with pytest.raises(ValueError):
            validate_settings(dict(batch_size=1, repeats=2, warmup=0, threads=2,
                                   sample_interval=.001, device='cpu') | change)


def test_training_step_changes_only_open_weights_and_keeps_batchnorm():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(.2), nn.Linear(4, 10))
    for p in model[0].parameters():
        p.requires_grad = False
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    result = benchmark_steps(model, optimizer, torch.ones(1, 4), torch.tensor([1]),
                             dict(batch_size=1, repeats=2, warmup=0, threads=2,
                                  sample_interval=.001, device='cpu'))
    assert result['timing_ms']['count'] == 2
    assert result['timing_ms']['median'] > 0
    assert result['memory']['peak_allocated_bytes'] is None
    assert torch.equal(before['0.weight'], model[0].weight)
    assert not torch.equal(before['3.weight'], model[3].weight)
    assert model[0].weight.grad is None
    assert torch.equal(before['1.running_mean'], model[1].running_mean)
    assert model[2].training and not model[1].training


def test_checkpoint_is_read_only_and_policy_comes_from_metadata(tmp_path):
    from src.models import models
    model = models.build_model('resnet18', pretrained=False, dropout=.2)
    metadata = dict(dataset='CIFAR10', model='resnet18', stage='finetune', dropout=.2,
                    unfreeze='scheduled_tail', backbone_lr=.0001, head_lr=.001,
                    optimizer='adamw', image_size=64)
    checkpoint = tmp_path/'best.pt'
    torch.save(dict(state_dict=model.state_dict(), metadata=metadata), checkpoint)
    original = checkpoint.read_bytes()
    loaded, optimizer, meta = prepare_training(checkpoint, {'weight_decay':.0001,
        'max_trainable_fraction':.5}, 'resnet18', torch.device('cpu'))
    assert checkpoint.read_bytes() == original
    assert meta == metadata
    assert models.parameter_counts(loaded)['trainable'] == 4725770
    assert [g['lr'] for g in optimizer.param_groups] == [.0001, .001]
    assert not loaded.bn1.training and loaded.fc[0].training
    with pytest.raises(ValueError, match='model'):
        prepare_training(checkpoint, {}, 'densenet121', torch.device('cpu'))


def test_evidence_rejects_corruption_and_wrong_run(tmp_path, monkeypatch):
    import json
    import shutil
    from pathlib import Path
    from src.utils import notebook_views as view
    root = Path(__file__).resolve().parents[1]
    evidence = tmp_path/'evidence'
    shutil.copytree(root/'outputs/published/training_benchmark/scheduled64_cpu_batch1', evidence)
    run_name = json.loads((evidence/'provenance.json').read_text())['training_run']
    monkeypatch.setattr(view, 'display', lambda value: None)
    monkeypatch.setattr(view, '_save', lambda fig, *args: view.plt.close(fig))
    view.training_benchmark(tmp_path/run_name, evidence, tmp_path)
    with pytest.raises(ValueError, match='another training run'):
        view.training_benchmark(tmp_path/'other', evidence, tmp_path)
    measurement = evidence/'resnet18/measurement.json'
    measurement.write_text('{}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        view.training_benchmark(tmp_path/run_name, evidence, tmp_path)


def test_cli_refuses_existing_output_without_mutation():
    import importlib.util
    from pathlib import Path
    from src.utils.delivery import file_hash
    root = Path(__file__).resolve().parents[1]
    path = root/'outputs/published/training_benchmark/scheduled64_cpu_batch1/manifest.json'
    before = file_hash(path)
    spec = importlib.util.spec_from_file_location('training_cli', root/'scripts/benchmark_training.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(FileExistsError):
        cli.main([])
    assert file_hash(path) == before
