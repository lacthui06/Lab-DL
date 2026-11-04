"""Đo hiệu năng theo phạm vi rõ ràng và xuất bằng chứng không cần train lại."""
from pathlib import Path
import threading
import time
import psutil
import torch
from transformers import TrainerCallback
from .evidence import read_json, verify_manifest
from .utils.runs import ROOT, save_json, hash_file, run_context

MIB = 1024 ** 2


class ResourceMonitor:
    """RSS sampled của process hiện tại; CUDA peak của allocator trên một device."""
    def __init__(self, device, sample_count, batch_size, device_batch_size=None, interval=.02):
        if sample_count <= 0 or batch_size <= 0:
            raise ValueError('Positive sample count and batch size required')
        self.device = torch.device(device)
        self.sample_count, self.batch_size = sample_count, batch_size
        self.device_batch_size = device_batch_size or batch_size
        self.interval = interval
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.result = None

    def _sample(self):
        self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)

    def _poll(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def start(self):
        self.rss_peak = self.process.memory_info().rss
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        if self.result is not None:return self.result
        if self.device.type == 'cuda':torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self.started
        self.stop_event.set();self.thread.join();self._sample()
        allocated = torch.cuda.max_memory_allocated(self.device)/MIB if self.device.type=='cuda' else None
        reserved = torch.cuda.max_memory_reserved(self.device)/MIB if self.device.type=='cuda' else None
        self.result = {'seconds':elapsed, 'sample_count':self.sample_count,
            'ms_per_sample':elapsed*1000/self.sample_count,
            'samples_per_second':self.sample_count/elapsed,
            'batch_size':self.batch_size, 'device_batch_size':self.device_batch_size,
            'ram_peak_rss_mib':self.rss_peak/MIB,
            'ram_mib_per_batch_slot':self.rss_peak/MIB/self.batch_size,
            'vram_peak_allocated_mib':allocated, 'vram_peak_reserved_mib':reserved,
            'vram_mib_per_batch_slot':allocated/self.device_batch_size if allocated is not None else None,
            'device':str(self.device), 'rss_poll_interval_seconds':self.interval,
            'memory_scope':'parent process RSS; CUDA allocator on selected device; includes shared model/optimizer',
            'normalization':'peak/batch slots is amortized shared memory, not independent per-example allocation',
            'vram_unavailable_reason':None if allocated is not None else 'CPU execution; CUDA memory not applicable'}
        return self.result


class EpochResources(TrainerCallback):
    """Phải đăng ký trước callback evaluate(train), để loại evaluate/save khỏi timer train."""
    def __init__(self, trainer, dataset_size, output):
        self.trainer, self.size, self.output = trainer, dataset_size, Path(output)
        self.monitor = None;self.rows = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        if args.world_size != 1:
            raise ValueError('Resource telemetry currently requires one training process')
        self.monitor = ResourceMonitor(args.device, self.size, args.train_batch_size,
                                       args.per_device_train_batch_size).start()

    def on_epoch_end(self, args, state, control, **kwargs):
        row = {'epoch':int(round(state.epoch)), 'scope':'train batches including data loading/logging; excludes epoch evaluation and checkpoint save',
               **self.monitor.finish()}
        self.rows.append(row);self.monitor = None
        save_json(self.output/'resource_history.json',self.rows)

    def close(self):
        if self.monitor is not None:self.monitor.finish();self.monitor = None

    def on_train_end(self, args, state, control, **kwargs):self.close()


def architecture_spec(run):
    run = Path(run)
    if (run/'model_architecture.json').is_file():return read_json(run/'model_architecture.json')
    cfg = read_json(run/'config.json')['exercise2']
    if cfg['model_name'] != 'distilbert/distilbert-base-uncased':
        raise ValueError('Historical architecture needs an explicit saved model config')
    return {'architecture':'DistilBertForSequenceClassification','model_name':cfg['model_name'],
        'revision':cfg['model_revision'], 'layers':6, 'hidden_size':768,'attention_heads':12,
        'intermediate_size':3072,'num_labels':2,'freeze_layers':cfg['freeze_layers'],
        'dropout':cfg['model_overrides'],
        'source':'declared pinned DistilBERT base architecture, not locally reloaded checkpoint',
        'reference_url':f"https://huggingface.co/{cfg['model_name']}/blob/{cfg['model_revision']}/config.json"}


def performance_data(run):
    run = Path(run);cfg = read_json(run/'config.json')['exercise2']
    epochs = read_json(run/'epoch_history.json')
    stats = read_json(run/'data_statistics.json')
    train_size = stats['effective_sizes']['train'];test_size = stats['official_sizes']['test']
    measured = read_json(run/'resource_history.json') if (run/'resource_history.json').is_file() else []
    lookup = {r['epoch']:r for r in measured};rows = []
    for point in epochs:
        resource = lookup.get(point['epoch'],{})
        rows.append({'epoch':point['epoch'],'train_accuracy':point['train_accuracy'],
            'validation_accuracy':point['validation_accuracy'],'train_loss':point['train_loss_eval'],
            'validation_loss':point['validation_loss'],
            'train_ms_per_sample':resource.get('ms_per_sample'),
            'epoch_pipeline_ms_per_train_sample':point['epoch_runtime_seconds']*1000/train_size if point.get('epoch_runtime_seconds') is not None else None,
            **{k:resource.get(k) for k in ['ram_peak_rss_mib','ram_mib_per_batch_slot','vram_peak_allocated_mib','vram_peak_reserved_mib','vram_mib_per_batch_slot']}})
    runtime = read_json(run/'test_runtime_metrics.json')
    summary = {'source_run':run.name,'train_samples_per_epoch':train_size,'test_samples':test_size,
        'train_batch_size_per_device':cfg['batch_size'],'eval_batch_size_per_device':cfg['eval_batch_size'],
        'pure_train_resources_available':bool(measured),
        'test_eval_ms_per_sample':runtime['test_runtime']*1000/test_size,
        'test_eval_timing_scope':'Trainer.predict full test, includes batching/transfer/evaluation overhead; not single-request latency',
        'epoch_pipeline_timing_scope':'epoch start through validation callback; includes train, full train-eval and validation; excludes later checkpoint save',
        'memory_status':'measured' if measured else 'not logged; cannot reconstruct training peak from parameter count or saved weights',
        'memory_normalization':'MiB per batch slot = process/device peak divided by nominal batch slots; shared model memory included; not MiB divided by dataset size',
        'cuda_device_available_in_source':read_json(run/'environment_snapshot.json')['cuda_available'],
        'source_device':read_json(run/'environment_snapshot.json').get('device'),
        'execution_device':measured[0]['device'] if measured else cfg.get('device','not recorded; consult source environment'),
        'training_ram_complete':bool(measured) and all(r['ram_peak_rss_mib'] is not None for r in rows),
        'training_vram_complete':bool(measured) and all(r['vram_peak_allocated_mib'] is not None for r in rows)}
    return rows,summary


def write_performance(run, output):
    import matplotlib.pyplot as plt
    from .utils.plots import finish
    output = Path(output);rows,summary = performance_data(run)
    save_json(output/'performance_history.json',rows);save_json(output/'performance_summary.json',summary)
    spec = architecture_spec(run);save_json(output/'model_architecture.json',spec)
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    x = [r['epoch'] for r in rows]
    axes[0].plot(x,[r['epoch_pipeline_ms_per_train_sample'] for r in rows],marker='o',label='Epoch pipeline incl. evaluation')
    if summary['pure_train_resources_available']:
        axes[0].plot(x,[r['train_ms_per_sample'] for r in rows],marker='o',label='Train phase only')
    axes[0].set(xlabel='Epoch',ylabel='ms / train sample',title='Timing scopes kept separate');axes[0].legend(fontsize=7)
    for ax,key,title in [(axes[1],'ram_mib_per_batch_slot','RAM RSS / batch slot'),(axes[2],'vram_mib_per_batch_slot','CUDA allocated / batch slot')]:
        points=[(r['epoch'],r[key]) for r in rows if r[key] is not None]
        if points:
            ax.plot(*zip(*points),marker='o');ax.set(xlabel='Epoch',ylabel='MiB / nominal batch slot')
        else:
            unavailable = 'N/A\nCPU execution' if summary['pure_train_resources_available'] and key.startswith('vram') else 'NOT LOGGED\nNo memory measurement available'
            ax.text(.5,.5,unavailable,ha='center',va='center',transform=ax.transAxes)
            ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(title)
    for ax in axes:ax.grid(alpha=.2)
    finish(fig,output/'performance.png')
    return summary


def ensure_performance_report(run, root=ROOT):
    run,root = Path(run),Path(root);verify_manifest(run)
    key = {'source_run':run.name,'source_manifest_sha256':hash_file(run/'manifest.json'),
           'analysis_sha256':hash_file(Path(__file__))}
    registry = read_json(root/'config/workflow.json')
    for entry in reversed(registry['runs']):
        if entry['kind']=='performance_report' and entry['status']=='completed':
            folder=root/'outputs/runs'/entry['id']
            if read_json(folder/'performance_source.json')==key:
                verify_manifest(folder);return folder
    with run_context('performance_report',key,root=root) as output:
        save_json(output/'performance_source.json',key)
        summary=write_performance(run,output);print(summary)
    return output
