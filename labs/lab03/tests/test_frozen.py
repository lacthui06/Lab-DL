import pytest
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from labs.lab03.src.training.frozen import configure_frozen_optimizer


def fixture():
    torch.manual_seed(42)
    model = DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=32, dim=16, hidden_dim=24, n_heads=2, n_layers=6,
        dropout=0.2, attention_dropout=0.2, seq_classif_dropout=0.4))
    config = dict(freeze_layers=4, learning_rate=2e-6, head_learning_rate=1e-5, weight_decay=.1)
    return model, config


def test_frozen_parameters_stay_identical_and_trainable_receive_gradients():
    model, config = fixture()
    optimizer, evidence = configure_frozen_optimizer(model, config)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model(input_ids=torch.randint(0, 32, (4, 8)), labels=torch.tensor([0, 1, 0, 1])).loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.distilbert.transformer.layer[4].parameters())
    optimizer.step()
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if not p.requires_grad)
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids) == len(set(ids)) == sum(p.requires_grad for p in model.parameters())
    assert {g['lr'] for g in optimizer.param_groups} == {2e-6, 1e-5}
    assert all(g['weight_decay'] == 0 for g in optimizer.param_groups if all(n.endswith('bias') or 'LayerNorm' in n for n in g['parameter_names']))
    assert evidence['trainable_parameters'] < evidence['total_parameters']


@pytest.mark.parametrize('count', [-1, 6, True])
def test_reject_invalid_freezing(count):
    model, config = fixture()
    config['freeze_layers'] = count
    with pytest.raises(ValueError):
        configure_frozen_optimizer(model, config)


def test_discriminative_optimizer_trainer_ten_tiny_epochs(tiny, tmp_path):
    import json
    from datasets import Dataset, DatasetDict
    from labs.lab03.src.training.trainer import make_trainer
    model, tokenizer = tiny
    cfg = dict(freeze_layers=0, learning_rate=2e-6, head_learning_rate=1e-5,
        weight_decay=.1, num_train_epochs=10, batch_size=8, eval_batch_size=8,
        warmup_ratio=.1, fp16=False, seed=42, eval_strategy='epoch',
        metric_for_best_model='loss', epoch_console_summary=True, save_only_model=True)
    optimizer, _ = configure_frozen_optimizer(model, cfg)
    data = Dataset.from_dict({**tokenizer(['excellent movie', 'terrible film'] * 2), 'label': [1, 0] * 2})
    ds = DatasetDict(train=data, validation=data, test=data)
    trainer = make_trainer(model, tokenizer, ds, cfg, tmp_path)
    trainer.optimizer = optimizer
    trainer.train()
    rows = json.loads((tmp_path/'epoch_history.json').read_text())
    assert [r['epoch'] for r in rows] == list(range(1, 11))
    best = min(rows, key=lambda r: r['validation_loss'])
    assert trainer.state.best_model_checkpoint.endswith('checkpoint-' + str(best['step']))
    assert not trainer.args.greater_is_better
    groups = trainer.optimizer.param_groups
    assert groups[0]['initial_lr'] == 2e-6
    assert groups[-1]['initial_lr'] == 1e-5


def test_frozen_trial_artifacts_and_review_zip(tiny, tmp_path, monkeypatch):
    import json
    import zipfile
    import hashlib
    from datasets import Dataset, DatasetDict
    from labs.lab03.scripts import run_frozen_trial as script
    model, tokenizer = tiny
    raw = Dataset.from_dict({'text': ['excellent movie', 'terrible film'] * 2, 'label': [1, 0] * 2})
    raw = DatasetDict(train=raw, validation=raw, test=raw)
    monkeypatch.setattr(script, 'load_data', lambda cfg: raw)
    monkeypatch.setattr(script, 'load_model', lambda *a, **kw: (model, tokenizer))
    cfg = json.loads((script.ROOT/'config/frozen_trial.json').read_text())
    cfg['exercise2'].update(freeze_layers=0, batch_size=8, eval_batch_size=8,
        fp16=False, augmentation={'enabled': False})
    out = script.run(cfg, root=tmp_path, allow_cpu_fixture=True)
    selection = json.loads((out/'selection_summary.json').read_text())
    assert selection['completed_epochs'] == 10 and not selection['test_used_for_selection']
    assert (out/'learning_curves.png').is_file()
    assert (out/'validation_threshold_curves.png').is_file()
    assert (out/'sample_predictions.png').is_file()
    assert json.loads((out/'sample_predictions.json').read_text())
    assert (out/'best_model/decision_threshold.json').is_file()
    bundle = tmp_path/'outputs'/f'lab03_review_{out.name}.zip'
    with zipfile.ZipFile(bundle) as z:
        manifest = json.loads(z.read('bundle_manifest.json'))
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n,h in manifest.items())
        assert not any('/best_model/' in n or '/checkpoints/' in n for n in z.namelist())
    checkpoint = tmp_path/'outputs'/f'lab03_checkpoint_{out.name}.zip'
    with zipfile.ZipFile(checkpoint) as z:
        assert 'best_model/model.safetensors' in z.namelist()
        assert 'best_model/decision_threshold.json' in z.namelist()


def test_fixture_zip_stays_in_fixture_even_when_kaggle_directory_exists(tmp_path):
    from labs.lab03.scripts.run_frozen_trial import archive_destination
    kaggle = tmp_path/'kaggle/working'
    kaggle.mkdir(parents=True)
    fixture_root = tmp_path/'pytest-fixture'
    fixture_root.mkdir()
    assert archive_destination(fixture_root, True, kaggle) == fixture_root/'outputs'
    assert archive_destination(fixture_root, False, kaggle) == kaggle
