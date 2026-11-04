import pytest
from IPython.display import Image
from labs.lab03.src.notebook_runtime import NotebookSession
from labs.lab03.src.utils.runs import ROOT, hash_file


def test_view_notebooks_show_saved_images_without_training_or_mutation(monkeypatch):
    session = NotebookSession(ROOT.parents[1], mode='view')
    monkeypatch.setattr(session, 'run', lambda *a, **kw: pytest.fail('View must not launch training/preparation'))
    session.prepare();session.train();session.export()
    run = session.paths['data']
    before = {str(p):hash_file(p) for p in run.rglob('*') if p.is_file()}
    shown = []
    monkeypatch.setattr('IPython.display.display', shown.append)
    paths = session.show('data', ['class_distribution.png','token_length_distribution.png'])
    assert len(paths) == sum(isinstance(item,Image) for item in shown) == 2
    assert session.table('data','data_preview.json','examples').shape[0] == 4
    assert {str(p):hash_file(p) for p in run.rglob('*') if p.is_file()} == before


def test_device_selection_is_explicit_and_cuda_unavailability_is_reported(monkeypatch):
    from labs.lab03.scripts.run_frozen_trial import resolve_device
    monkeypatch.setattr('torch.cuda.is_available',lambda:False)
    assert resolve_device('auto') == resolve_device('cpu') == 'cpu'
    with pytest.raises(RuntimeError,match='CUDA requested'):
        resolve_device('cuda')
    monkeypatch.setattr('torch.cuda.is_available',lambda:True)
    assert resolve_device('auto') == 'cuda' and resolve_device('cpu') == 'cpu'
    with pytest.raises(ValueError):resolve_device('invalid')
