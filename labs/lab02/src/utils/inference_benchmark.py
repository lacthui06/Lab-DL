"""Resident-model single-image latency and a separate sampled memory pass.

No checkpoint is loaded here. CUDA allocator peaks are not total device memory;
sampled RSS is an observed peak and can miss allocations shorter than the interval.
"""
from contextlib import contextmanager, nullcontext
import math
from pathlib import Path
import statistics
import threading
import time
import json

import psutil

from ..inference import LoadedModel, predict_image


def timing_summary(values: list[float]) -> dict:
    """Milliseconds; p95 uses inclusive linear interpolation (or the sole sample)."""
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Timing samples must be nonempty, finite and nonnegative')
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=statistics.quantiles(values, n=100, method='inclusive')[94] if len(values) > 1 else values[0])


def _sync(session):
    if session.device.type == 'cuda':
        import torch
        torch.cuda.synchronize(session.device)


def _memory(session):
    import torch
    cuda = session.device.type == 'cuda'
    return dict(rss_bytes=psutil.Process().memory_info().rss,
                allocated_bytes=torch.cuda.memory_allocated(session.device) if cuda else None,
                reserved_bytes=torch.cuda.memory_reserved(session.device) if cuda else None)


class _RSSSampler:
    """Bounded-memory RSS sampler; always stop the worker and surface read errors."""
    def __init__(self, interval):
        self.interval = interval
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.peak, self.samples, self.error = 0, 0, None

    def _read(self):
        return self.process.memory_info().rss

    def _sample(self):
        self.peak = max(self.peak, self._read())
        self.samples += 1

    def _watch(self):
        try:
            while not self.stop.wait(self.interval):
                self._sample()
        except Exception as exc:
            self.error = exc
            self.stop.set()

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join(timeout=5)
        if kind is None:
            if self.thread.is_alive():
                raise RuntimeError('RAM sampling worker did not stop')
            if self.error is not None:
                raise RuntimeError('RAM sampling failed') from self.error
            self._sample()
        return False


class _Stages:
    def __init__(self, session):
        self.session, self.samples = session, {}

    @contextmanager
    def __call__(self, name):
        _sync(self.session)
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            _sync(self.session)
            self.samples.setdefault(name, []).append((time.perf_counter_ns() - start) / 1e6)


def _request(session, image, stages=None):
    result = predict_image(session, image, stages=stages)
    with stages('serialize') if stages is not None else nullcontext():
        serialized = json.dumps(result, indent=2, allow_nan=False)
    return result, serialized


def _timed_request(session, image):
    _sync(session)
    start = time.perf_counter_ns()
    result, serialized = _request(session, image)
    _sync(session)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return elapsed, result, serialized


def benchmark_loaded_model(session: LoadedModel, image: Path, warmup: int = 20, repeats: int = 100,
                           stage_repeats: int = 20, memory_repeats: int = 20,
                           sample_interval: float = .001) -> dict:
    """Measure first request, warmed total latency, diagnostic stages and memory separately."""
    for name, value, minimum in [('warmup', warmup, 0), ('repeats', repeats, 1),
                                 ('stage_repeats', stage_repeats, 1), ('memory_repeats', memory_repeats, 1)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not math.isfinite(sample_interval) or sample_interval <= 0:
        raise ValueError('sample_interval must be positive and finite')
    if session.model.training:
        raise ValueError('Benchmark requires a preloaded eval model')
    image = Path(image)
    if not image.is_file():
        raise FileNotFoundError(image)
    _sync(session)
    loaded_memory = _memory(session)
    first, prediction, _ = _timed_request(session, image)
    for _ in range(warmup):
        _request(session, image)
    _sync(session)
    warm_memory = _memory(session)
    totals = []
    for _ in range(repeats):
        elapsed, prediction, _ = _timed_request(session, image)
        totals.append(elapsed)
    stages = _Stages(session)
    for _ in range(stage_repeats):
        _request(session, image, stages)
    _sync(session)
    before_memory = _memory(session)
    cuda = session.device.type == 'cuda'
    if cuda:
        import torch
        torch.cuda.reset_peak_memory_stats(session.device)
    with _RSSSampler(sample_interval) as sampler:
        for _ in range(memory_repeats):
            _request(session, image)
            _sync(session)
    after_memory = _memory(session)
    allocated = torch.cuda.max_memory_allocated(session.device) if cuda else None
    reserved = torch.cuda.max_memory_reserved(session.device) if cuda else None
    peak_rss = max(sampler.peak, before_memory['rss_bytes'], after_memory['rss_bytes'])
    return dict(model=session.name, device=str(session.device), image_size=session.image_size,
                batch_size=1, precision='float32', warmup=warmup, repeats=repeats,
                first_request_ms=first, latency_ms=timing_summary(totals), latency_samples_ms=totals,
                stage_repeats=stage_repeats, stages_ms={name: timing_summary(values) for name, values in stages.samples.items()},
                stage_samples_ms=stages.samples, prediction=prediction,
                memory=dict(after_load=loaded_memory, after_warmup=warm_memory,
                            before_pass=before_memory, after_pass=after_memory,
                            peak_rss_bytes=peak_rss,
                            rss_increase_bytes=max(0, peak_rss - before_memory['rss_bytes']),
                            peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                            allocated_increase_bytes=allocated-before_memory['allocated_bytes'] if cuda else None,
                            reserved_increase_bytes=reserved-before_memory['reserved_bytes'] if cuda else None,
                            repeats=memory_repeats, sample_interval_seconds=sample_interval,
                            rss_sample_count=sampler.samples, scope='separate warmed request pass'),
                measurement_policy=dict(
                    total='read/decode PNG -> preprocess -> transfer -> forward -> CPU probabilities -> JSON string; excludes printing/writing',
                    first='first image request after model load; library imports and checkpoint load completed beforehand',
                    stages='separate diagnostic pass with CUDA synchronization at each boundary; do not sum to infer total',
                    memory='process RSS sampled in separate pass; CUDA allocated/reserved use PyTorch allocator peaks, not total device usage',
                    filesystem='same image read repeatedly; OS file cache can contribute; no forced cache flush',
                    limitations='one image, warmed model, no claim of minimum required RAM/VRAM or performance on all inputs'))
