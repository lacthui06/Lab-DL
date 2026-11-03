import json
import pytest
from labs.lab03.src.utils.runs import ROOT, hash_file
from labs.lab03.src.performance import ResourceMonitor, performance_data, write_performance

SOURCE=ROOT/'outputs/runs/20261027T095655Z_frozen_trial_d8d09116'


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


def test_historical_performance_never_invents_memory_or_pure_train_timer(tmp_path):
    before={str(p):hash_file(p) for p in SOURCE.rglob('*') if p.is_file()}
    rows,summary=performance_data(SOURCE)
    assert len(rows)==10 and not summary['pure_train_resources_available']
    assert all(r['ram_peak_rss_mib'] is None and r['train_ms_per_sample'] is None for r in rows)
    assert all(r['vram_peak_allocated_mib'] is None for r in rows)
    assert summary['test_eval_ms_per_sample']==pytest.approx(.4976*1000/1066)
    assert rows[0]['epoch_pipeline_ms_per_train_sample']==pytest.approx(18.811370074999957*1000/8804)
    write_performance(SOURCE,tmp_path)
    assert (tmp_path/'model_architecture.png').is_file()
    assert (tmp_path/'performance.png').is_file()
    assert json.loads((tmp_path/'performance_summary.json').read_text())['memory_status'].startswith('not logged')
    assert {str(p):hash_file(p) for p in SOURCE.rglob('*') if p.is_file()}==before
