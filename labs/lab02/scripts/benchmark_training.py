"""Run explicit batch-one training benchmarks; each checkpoint gets a fresh process."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/training_benchmark.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--worker', choices=('resnet18', 'densenet121'), help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from src.utils.delivery import file_hash, iter_source_files, write_json
    from src.utils.training_benchmark import validate_settings
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    validate_settings(settings)
    run = (ROOT/settings['training_run']).resolve()
    if args.worker:
        from src.utils.training_benchmark import measure_checkpoint
        write_json(args.output, measure_checkpoint(ROOT, run, args.worker, settings, args.data_dir))
        return
    if settings['models'] != ['resnet18', 'densenet121']:
        raise ValueError('Select resnet18 and densenet121 exactly once, in that order')
    if type(settings['seed']) is not int or not 0 <= settings['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if type(settings['train_position']) is not int or settings['train_position'] < 0:
        raise ValueError('train_position must be a nonnegative integer')
    required = [run/'config.json', run/'selected_indices.npz',
                *(run/name/'finetune/best.pt' for name in settings['models'])]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (args.data_dir/'cifar-10-batches-py').is_dir():
        raise FileNotFoundError('Local CIFAR10 data missing; benchmark never downloads data')
    output = (ROOT/settings['output']).resolve()
    if not output.is_relative_to(ROOT/'outputs/published/training_benchmark'):
        raise ValueError('Output must be inside outputs/published/training_benchmark')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'status.json', {'status':'running'})
    write_json(output/'config.json', settings)
    write_json(output/'methodology.json', dict(
        scope='forward + CrossEntropyLoss + backward + optimizer.step; batch already on device',
        execution='full_image_forward; checkpoint freeze/dropout/BatchNorm policy; float32',
        initialization='saved best validation weights; NEW optimizer at recipe INITIAL LRs; no scheduler',
        input='same selected train index; deterministic preprocessing; no augmentation; tensor hash recorded',
        memory='separate FIRST step including optimizer allocation; RSS sampled at 1ms; CUDA allocator peaks',
        timing='median/mean/P95 of warmed steps; sampler disabled; CUDA synchronized at boundaries',
        exclusions='checkpoint/model loading, IO, preprocessing, transfer, validation, artifact writing',
        limitations='one sample repeated, one process/model; RSS includes runtime; not convergence or full-fit time; CPU VRAM null'))
    try:
        rows = []
        for name in settings['models']:
            folder = output/name
            folder.mkdir()
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--config',
                str(args.config.resolve()), '--data-dir', str(args.data_dir.resolve()),
                '--worker', name, '--output', str(folder/'measurement.json')],
                cwd=ROOT, capture_output=True, text=True, timeout=300)
            if result.returncode:
                raise RuntimeError(f'{name} benchmark failed: {result.stderr}')
            value = json.loads((folder/'measurement.json').read_text())
            rows.append(dict(model=name, batch_size=1, median_ms=value['timing_ms']['median'],
                mean_ms=value['timing_ms']['mean'], p95_ms=value['timing_ms']['p95'],
                peak_rss_mib=value['memory']['peak_rss_bytes']/2**20,
                rss_increase_mib=value['memory']['rss_increase_bytes']/2**20,
                peak_allocated_mib=(None if value['memory']['peak_allocated_bytes'] is None else
                                    value['memory']['peak_allocated_bytes']/2**20)))
        values = [json.loads((output/name/'measurement.json').read_text()) for name in settings['models']]
        if len({(v['train_index'], v['input_tensor_sha256']) for v in values}) != 1:
            raise ValueError('Models did not receive the same preprocessed training sample')
        with (output/'training_performance.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(output/'provenance.json', dict(training_run=run.name,
            created_utc=datetime.now(timezone.utc).isoformat(),
            checkpoints={v['model']:v['checkpoint_sha256'] for v in values},
            source_hashes={p.relative_to(ROOT).as_posix():file_hash(p) for p in iter_source_files(ROOT)},
            independent_process_per_model=True, original_checkpoints_unchanged=True))
        from src.utils.notebook_views import training_benchmark
        training_benchmark(run, output, output, verify=False)
        write_json(output/'status.json', {'status':'completed'})
        write_json(output/'manifest.json', {p.relative_to(output).as_posix():file_hash(p)
                   for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    except Exception as exc:
        write_json(output/'status.json', {'status':'failed', 'error_type':type(exc).__name__})
        raise
    print(output)


if __name__ == '__main__':
    main()
