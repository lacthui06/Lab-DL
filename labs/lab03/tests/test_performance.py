import json
import pytest
from labs.lab03.src.utils.runs import ROOT, hash_file
from labs.lab03.src.performance import ResourceMonitor, performance_data, write_performance

SOURCE=ROOT/'outputs/runs/20261103T110437Z_frozen_trial_b021d6eb'


def test_cpu_resource_units_and_vram_unavailable_are_explicit():
    monitor=ResourceMonitor('cpu',sample_count=8,batch_size=4).start()
    result=monitor.finish()
    assert result['seconds']>0
    assert result['ms_per_sample']==pytest.approx(result['seconds']*1000/8)
    assert result['ram_mib_per_batch_slot']==pytest.approx(result['ram_peak_rss_mib']/4)
    assert result['vram_peak_allocated_mib'] is None
    assert result['vram_mib_per_batch_slot'] is None
    assert 'CPU' in result['vram_unavailable_reason']
    assert monitor.finish() is result


def test_saved_t4_performance_is_measured_and_sources_are_unchanged(tmp_path):
    before = {str(p): hash_file(p) for p in SOURCE.rglob('*') if p.is_file()}
    rows, summary = performance_data(SOURCE)
    assert len(rows) == 5 and summary['pure_train_resources_available']
    assert summary['training_ram_complete'] and summary['training_vram_complete']
    assert all(r['ram_peak_rss_mib'] > 0 and r['vram_peak_allocated_mib'] > 0 for r in rows)
    assert summary['test_eval_ms_per_sample'] == pytest.approx(.4263602251407129)
    write_performance(SOURCE, tmp_path)
    assert (tmp_path/'model_architecture.png').is_file() and (tmp_path/'performance.png').is_file()
    assert {str(p): hash_file(p) for p in SOURCE.rglob('*') if p.is_file()} == before
