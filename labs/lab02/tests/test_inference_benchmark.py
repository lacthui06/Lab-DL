"""Single-image prediction equivalence and benchmark timing/resource boundaries."""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import importlib.util
from types import SimpleNamespace
import zipfile

import pytest
import torch
from PIL import Image

from src.inference import LoadedModel, load_model, predict_image
from src.utils import inference_benchmark as bench

ROOT = Path(__file__).resolve().parents[1]


def _cli():
    spec = importlib.util.spec_from_file_location('benchmark_cli', ROOT/'scripts/benchmark_inference.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


@pytest.fixture
def loaded(tmp_path):
    from src.datasets.data import image_transform
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 10)).eval()
    image = tmp_path / 'sample.png'
    Image.new('RGB', (32, 32), (20, 100, 180)).save(image)
    return LoadedModel(model, 'resnet18', 64, torch.device('cpu'), image_transform('resnet18', 64)), image


def test_single_image_matches_existing_transform_and_output(loaded):
    session, path = loaded
    stages = []

    @contextmanager
    def record(name):
        stages.append(name)
        yield

    result = predict_image(session, path, actual=3, stages=record)
    with Image.open(path) as image, torch.inference_mode():
        probabilities = session.model(session.transform(image.convert('RGB')).unsqueeze(0)).softmax(1)[0]
    assert result['model'] == 'resnet18' and result['actual'] == 'cat'
    assert result['probability'] == probabilities.max().item()
    assert list(result['probabilities'].values()) == probabilities.tolist()
    assert stages == ['read', 'preprocess', 'transfer', 'forward', 'postprocess']
    assert predict_image(session, path)['actual'] is None


def test_load_model_is_outside_request_and_uses_checkpoint_metadata(tmp_path, monkeypatch):
    import src.models.models as models
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 10))
    checkpoint = tmp_path / 'best.pt'
    torch.save({'state_dict': model.state_dict(), 'metadata': {
        'dataset': 'CIFAR10', 'model': 'resnet18', 'image_size': 64, 'dropout': .2}}, checkpoint)
    calls = []

    def build(name, **kwargs):
        calls.append((name, kwargs))
        return model

    monkeypatch.setattr(models, 'build_model', build)
    session = load_model(checkpoint, device='cpu', threads=2)
    assert calls == [('resnet18', {'pretrained': False, 'dropout': .2})]
    assert session.image_size == 64 and not session.model.training
    image = tmp_path / 'sample.png'; Image.new('RGB', (32, 32)).save(image)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Checkpoint loaded inside request'))
    predict_image(session, image)


def test_loaded_model_benchmark_has_separate_passes_and_cpu_memory_is_not_vram(loaded, monkeypatch):
    session, image = loaded
    original = bench.predict_image
    calls = []

    def predict(current, path, **kwargs):
        assert current is session and path == image
        calls.append(kwargs.get('stages') is not None)
        return original(current, path, **kwargs)

    monkeypatch.setattr(bench, 'predict_image', predict)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Model reloaded during benchmark'))
    result = bench.benchmark_loaded_model(session, image, warmup=1, repeats=3,
                                          stage_repeats=2, memory_repeats=2)
    assert len(calls) == 1 + 1 + 3 + 2 + 2
    assert sum(calls) == 2
    assert len(result['latency_samples_ms']) == 3
    assert set(result['stages_ms']) == {'read', 'preprocess', 'transfer', 'forward', 'postprocess', 'serialize'}
    assert result['latency_ms']['median'] >= 0
    assert result['memory']['peak_rss_bytes'] >= result['memory']['before_pass']['rss_bytes']
    assert result['memory']['peak_allocated_bytes'] is None
    assert result['memory']['peak_reserved_bytes'] is None
    assert result['prediction']['predicted'] in result['prediction']['probabilities']


@pytest.mark.parametrize('kwargs', [{'repeats': 0}, {'warmup': -1}, {'stage_repeats': 0},
                                   {'memory_repeats': 0}, {'sample_interval': 0}, {'sample_interval': float('nan')}])
def test_invalid_measurement_settings_fail_before_prediction(loaded, monkeypatch, kwargs):
    monkeypatch.setattr(bench, 'predict_image', lambda *a, **k: pytest.fail('Invalid benchmark ran'))
    with pytest.raises(ValueError):
        bench.benchmark_loaded_model(*loaded, **kwargs)


def test_timing_statistics_have_explicit_percentile():
    assert bench.timing_summary([1., 2., 3.]) == {'count': 3, 'mean': 2., 'median': 2., 'p95': 2.9}
    assert bench.timing_summary([4.])['p95'] == 4.


def test_missing_cuda_is_an_error_not_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        load_model(tmp_path / 'missing.pt', device='cuda')


def test_request_errors_remain_visible(loaded, tmp_path):
    session, _ = loaded
    with pytest.raises(FileNotFoundError):
        predict_image(session, tmp_path / 'missing.png')
    corrupt = tmp_path / 'broken.png'; corrupt.write_bytes(b'not an image')
    with pytest.raises(OSError):
        predict_image(session, corrupt)


def test_rss_sampler_cleans_up_on_request_failure():
    sampler = bench._RSSSampler(.001)
    with pytest.raises(RuntimeError, match='request failed'):
        with sampler:
            raise RuntimeError('request failed')
    assert not sampler.thread.is_alive()


def test_rss_sampler_failure_is_not_reported_as_success(monkeypatch):
    sampler = bench._RSSSampler(.001)
    failed = threading.Event()
    calls = []

    def read():
        calls.append(1)
        if len(calls) > 1:
            failed.set()
            raise OSError('fixture memory sampler failed')
        return 100

    monkeypatch.setattr(sampler, '_read', read)
    with pytest.raises(RuntimeError, match='RAM sampling failed'):
        with sampler:
            assert failed.wait(2)
    assert not sampler.thread.is_alive()


def test_cuda_stage_timer_waits_before_and_after_work(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: events.append('sync'))
    recorder = bench._Stages(SimpleNamespace(device=torch.device('cuda')))
    with recorder('forward'):
        events.append('forward')
    assert events == ['sync', 'forward', 'sync']
    assert recorder.samples['forward'][0] >= 0


def test_cuda_peaks_are_reset_only_for_separate_memory_pass(loaded, monkeypatch):
    session, image = loaded
    session = LoadedModel(session.model, session.name, session.image_size, torch.device('cuda'), session.transform)
    events = []
    monkeypatch.setattr(bench, 'predict_image', lambda *a, **k: events.append('predict') or {'predicted':'cat'})
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 1000)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 2000)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 1500)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 2500)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda device: events.append('reset'))
    result = bench.benchmark_loaded_model(session, image, warmup=1, repeats=1, stage_repeats=1, memory_repeats=2)
    assert events == ['predict'] * 4 + ['reset'] + ['predict'] * 2
    assert result['memory']['peak_allocated_bytes'] == 1500
    assert result['memory']['allocated_increase_bytes'] == 500
    assert result['memory']['reserved_increase_bytes'] == 500


def test_prepare_image_and_package_are_explicit_and_exclude_unrelated_files(tmp_path, monkeypatch):
    from torchvision import datasets
    class Dataset:
        def __init__(self, *a, **kwargs):
            assert kwargs == {'train':False, 'download':False}
        def __len__(self): return 1
        def __getitem__(self, index): return Image.new('RGB', (32,32)), 3
    monkeypatch.setattr(datasets, 'CIFAR10', Dataset)
    cli = _cli()
    image = cli.main(['prepare-image','--data-dir',str(tmp_path),'--output',str(tmp_path/'image.png')])
    assert json.loads(image.with_suffix('.json').read_text())['actual'] == 'cat'
    with pytest.raises(FileExistsError):
        cli.main(['prepare-image','--data-dir',str(tmp_path),'--output',str(image)])
    source = tmp_path/'source'; (source/'src').mkdir(parents=True); (source/'scripts').mkdir()
    (source/'src/__init__.py').write_text('')
    (source/'.env').write_text('fixture excluded')
    for name in ('benchmark_inference.py','infer.py'): (source/'scripts'/name).write_text('# fixture')
    run = tmp_path/'training_run'
    for name in ('resnet18','densenet121'):
        path = run/name/'finetune/best.pt'; path.parent.mkdir(parents=True); path.write_bytes(b'fixture checkpoint')
    monkeypatch.setattr(cli,'ROOT',source)
    output = tmp_path/'package.zip'
    cli.main(['package','--run',str(run),'--image',str(image),'--output',str(output)])
    with zipfile.ZipFile(output) as archive:
        assert not any('.env' in name for name in archive.namelist())
        manifest = json.loads(archive.read('lab02/package_manifest.json'))
        assert 'lab02/checkpoints/resnet18.pt' in manifest['files']
        import hashlib
        for name, item in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == item['sha256']
    with pytest.raises(ValueError, match='read-only'):
        cli.main(['prepare-image','--data-dir',str(tmp_path),'--output','/kaggle/input/image.png'])
