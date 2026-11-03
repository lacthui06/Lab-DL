"""Prepare a test PNG, package resident-model benchmarks, or measure one checkpoint."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _writable(path):
    path = Path(path).expanduser().resolve()
    if path == Path('/kaggle/input') or Path('/kaggle/input') in path.parents:
        raise ValueError('Kaggle input is read-only; write artifacts under /kaggle/working')
    return path


def _export_image(args):
    from torchvision.datasets import CIFAR10
    from src.utils.constants import CLASSES
    from src.utils.delivery import file_hash, write_json
    output = _writable(args.output)
    if output.suffix.lower() != '.png':
        raise ValueError('Export a PNG so the input format is explicit')
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    dataset = CIFAR10(str(args.data_dir), train=False, download=False)
    if not 0 <= args.index < len(dataset):
        raise ValueError('Test image index is out of range')
    image, label = dataset[args.index]
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert('RGB').save(output, format='PNG')
    write_json(output.with_suffix('.json'), dict(dataset='CIFAR10', split='test', index=args.index,
               actual=CLASSES[label], original_size=list(image.size), png_sha256=file_hash(output),
               input_policy='PNG exported in a separate process before loading benchmark models'))
    print(output)
    return output


def _package(args):
    from src.utils.delivery import file_hash
    output = _writable(args.output)
    if output.exists():
        raise FileExistsError(output)
    run, image = args.run.resolve(), args.image.resolve()
    files = {f'lab02/{p.relative_to(ROOT).as_posix()}': p for p in (ROOT/'src').rglob('*.py')}
    for name in ('benchmark_inference.py', 'infer.py'):
        files[f'lab02/scripts/{name}'] = ROOT/'scripts'/name
    files['lab02/input.png'] = image
    if image.with_suffix('.json').is_file():
        files['lab02/input.json'] = image.with_suffix('.json')
    for model in ('resnet18', 'densenet121'):
        files[f'lab02/checkpoints/{model}.pt'] = run/model/'finetune/best.pt'
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = dict(training_run=run.name, checkpoints='validation-selected finetune/best.pt',
                    files={name: dict(sha256=file_hash(path), bytes=path.stat().st_size)
                           for name, path in files.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr('lab02/package_manifest.json', json.dumps(manifest, indent=2))
    print(json.dumps(dict(package=str(output), sha256=file_hash(output), bytes=output.stat().st_size), indent=2))
    return output


def _run(args):
    from src.inference import load_model
    from src.utils.inference_benchmark import benchmark_loaded_model
    from src.utils.delivery import file_hash, iter_source_files, write_json
    output = _writable(args.output_dir)
    # Validate settings before loading weights or creating a run directory.
    if args.warmup < 0 or min(args.repeats, args.stage_repeats, args.memory_repeats, args.threads) < 1:
        raise ValueError('Counts/threads must be positive; warmup may be zero')
    import math
    if not math.isfinite(args.sample_interval) or args.sample_interval <= 0:
        raise ValueError('Sampling interval must be positive and finite')
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    import torch
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    session = load_model(args.checkpoint, args.device, args.threads)
    gc.collect()
    run = output/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+
                  '_single_image_'+session.name+'_'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    write_json(run/'status.json', dict(status='running', kind='single_image_benchmark'))
    try:
        environment = dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                           torchvision=importlib.metadata.version('torchvision'),
                           psutil=importlib.metadata.version('psutil'), cuda=torch.version.cuda,
                           cudnn=torch.backends.cudnn.version(), device=str(session.device),
                           device_name=torch.cuda.get_device_name(session.device) if session.device.type=='cuda' else platform.processor(),
                           threads=torch.get_num_threads(), tf32=False, cudnn_benchmark=False)
        write_json(run/'environment_snapshot.json', environment)
        write_json(run/'config.json', dict(kind='single_image_benchmark', model=session.name,
                   image_size=session.image_size, warmup=args.warmup, repeats=args.repeats,
                   stage_repeats=args.stage_repeats, memory_repeats=args.memory_repeats,
                   sample_interval_seconds=args.sample_interval, seed=42, checkpoint_sha256=file_hash(args.checkpoint),
                   image_sha256=file_hash(args.image), batch_size=1, precision='float32',
                   model_load_in_timing=False, independent_process_per_model=True))
        write_json(run/'source_hashes.json', {p.relative_to(ROOT).as_posix():file_hash(p) for p in iter_source_files(ROOT)})
        result = benchmark_loaded_model(session, args.image, args.warmup, args.repeats,
                                        args.stage_repeats, args.memory_repeats, args.sample_interval)
        with (run/'latency_samples.csv').open('w', newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['repeat','total_ms'])
            writer.writerows(enumerate(result['latency_samples_ms'],1))
        with (run/'stage_samples.csv').open('w', newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['stage','repeat','milliseconds'])
            for stage, values in result['stage_samples_ms'].items():
                writer.writerows((stage,index,value) for index,value in enumerate(values,1))
        summary={key:value for key,value in result.items() if key not in ('latency_samples_ms','stage_samples_ms')}
        write_json(run/'summary.json', summary)
        write_json(run/'prediction.json', result['prediction'])
        write_json(run/'status.json', dict(status='completed', kind='single_image_benchmark'))
        write_json(run/'manifest.json', {p.name:file_hash(p) for p in run.iterdir() if p.is_file() and p.name!='manifest.json'})
        print(json.dumps(dict(run=str(run), model=session.name, first_request_ms=result['first_request_ms'],
                              warmed_latency_ms=result['latency_ms'], memory=result['memory']), indent=2))
        return run
    except BaseException as exc:
        write_json(run/'status.json', dict(status='failed', kind='single_image_benchmark', error=type(exc).__name__, message=str(exc)))
        raise


def _publish(args):
    """Export explicit, verified benchmark evidence; never copy logs or checkpoints."""
    import shutil
    from src.utils.delivery import file_hash, write_json
    output = _writable(args.output)
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(args.package) as archive:
        package = json.loads(archive.read('lab02/package_manifest.json'))
    image_hash = file_hash(args.image)
    if image_hash != package['files']['lab02/input.png']['sha256']:
        raise ValueError('Published image differs from the execution package')
    selected = {}
    filenames = ('summary.json','config.json','environment_snapshot.json','source_hashes.json',
                 'prediction.json','latency_samples.csv','stage_samples.csv','status.json','manifest.json')
    for run in args.runs:
        status = json.loads((run/'status.json').read_text())
        if status.get('status') != 'completed' or status.get('kind') != 'single_image_benchmark':
            raise ValueError('Only completed single-image benchmarks can be published')
        manifest = json.loads((run/'manifest.json').read_text())
        for name in filenames[:-1]:
            if file_hash(run/name) != manifest[name]:
                raise ValueError('Benchmark evidence checksum mismatch: '+name)
        cfg = json.loads((run/'config.json').read_text())
        name = cfg['model']
        if name in selected or name not in ('resnet18','densenet121'):
            raise ValueError('Select each supported model at most once')
        if cfg['image_sha256'] != image_hash or cfg['checkpoint_sha256'] != package['files'][f'lab02/checkpoints/{name}.pt']['sha256']:
            raise ValueError('Benchmark inputs differ from the execution package')
        selected[name] = run
    output.mkdir(parents=True, exist_ok=False)
    for name, run in selected.items():
        (output/name).mkdir()
        for filename in filenames:
            shutil.copy2(run/filename, output/name/filename)
    shutil.copy2(args.image, output/'input.png')
    if args.image.with_suffix('.json').is_file():
        shutil.copy2(args.image.with_suffix('.json'), output/'input.json')
    write_json(output/'provenance.json', dict(training_run=package['training_run'],
               benchmark_runs={name:run.name for name,run in selected.items()},
               execution_package_sha256=file_hash(args.package), input_sha256=image_hash,
               dataset_url=args.dataset_url, notebook_url=args.notebook_url,
               scope='Exact selected artifacts; no model weights, raw logs or training metric replacement'))
    print(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare-image', help='Export one official test image in a separate process')
    prepare.add_argument('--data-dir', type=Path, required=True)
    prepare.add_argument('--index', type=int, default=0)
    prepare.add_argument('--output', type=Path, required=True)
    package = commands.add_parser('package', help='Build a private-upload ZIP with source, image and two fine-tune checkpoints')
    package.add_argument('--run', type=Path, required=True)
    package.add_argument('--image', type=Path, required=True)
    package.add_argument('--output', type=Path, required=True)
    publish = commands.add_parser('publish', help='Verify and export explicit completed benchmark evidence')
    publish.add_argument('--runs', type=Path, nargs='+', required=True)
    publish.add_argument('--image', type=Path, required=True)
    publish.add_argument('--package', type=Path, required=True)
    publish.add_argument('--output', type=Path, required=True)
    publish.add_argument('--dataset-url')
    publish.add_argument('--notebook-url')
    run = commands.add_parser('run', help='Benchmark a preloaded model; one fresh process per checkpoint')
    run.add_argument('--checkpoint', type=Path, required=True)
    run.add_argument('--image', type=Path, required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    run.add_argument('--threads', type=int, default=4)
    run.add_argument('--warmup', type=int, default=20)
    run.add_argument('--repeats', type=int, default=100)
    run.add_argument('--stage-repeats', type=int, default=20)
    run.add_argument('--memory-repeats', type=int, default=20)
    run.add_argument('--sample-interval', type=float, default=.001)
    args = parser.parse_args(argv)
    return {'prepare-image':_export_image, 'package':_package, 'run':_run, 'publish':_publish}[args.command](args)


if __name__ == '__main__':
    main()
