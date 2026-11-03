import pytest
from IPython.display import Image
from labs.lab03.src.notebook_runtime import NotebookSession
from labs.lab03.src.utils.runs import ROOT, hash_file


def test_view_notebooks_show_saved_images_without_training_or_mutation(monkeypatch,tmp_path):
    import matplotlib.pyplot as plt
    from labs.lab03.src.utils.runs import save_json
    folder=tmp_path/'labs/lab03/outputs/runs/dataset_preview_fixture';folder.mkdir(parents=True)
    save_json(folder.parents[1]/'latest_preparation.json',{'runs':{'data':folder.name}})
    save_json(folder/'data_preview.json',{'examples':[{'index':i,'text':'unit fixture','label':i%2} for i in range(4)]})
    fig,ax=plt.subplots();ax.plot([0,1])
    for name in ('class_distribution.png','token_length_distribution.png'):fig.savefig(folder/name)
    plt.close(fig)
    session = NotebookSession(tmp_path, mode='view')
    monkeypatch.setattr(session, 'run', lambda *a, **kw: pytest.fail('View must not launch training/preparation'))
    session.prepare();session.train();session.export()
    before = {str(p):hash_file(p) for p in folder.rglob('*') if p.is_file()}
    shown = []
    monkeypatch.setattr('IPython.display.display', shown.append)
    paths = session.show('data', ['class_distribution.png','token_length_distribution.png'])
    assert len(paths) == sum(isinstance(item,Image) for item in shown) == 2
    assert session.table('data','data_preview.json','examples').shape[0] == 4
    assert {str(p):hash_file(p) for p in folder.rglob('*') if p.is_file()} == before


def test_device_selection_is_explicit_and_cuda_unavailability_is_reported(monkeypatch):
    from labs.lab03.scripts.run_frozen_trial import resolve_device
    monkeypatch.setattr('torch.cuda.is_available',lambda:False)
    assert resolve_device('auto') == resolve_device('cpu') == 'cpu'
    with pytest.raises(RuntimeError,match='CUDA requested'):
        resolve_device('cuda')
    monkeypatch.setattr('torch.cuda.is_available',lambda:True)
    assert resolve_device('auto') == 'cuda' and resolve_device('cpu') == 'cpu'
    with pytest.raises(ValueError):resolve_device('invalid')


def test_train_session_starts_without_loading_report_or_previous_selection(tmp_path):
    from labs.lab03.src.utils.runs import save_json
    lab=tmp_path/'labs/lab03'
    save_json(lab/'config/report.json',{'runs':{'frozen_trial':'unrelated'}})
    save_json(lab/'outputs/latest_report.json',{'runs':{'frozen_trial':'another-run'}})
    session=NotebookSession(tmp_path,mode='train')
    assert session.paths=={}
    assert session.env['LAB03_SESSION_ID']
    assert session.env['LAB03_SESSION_ID'] != NotebookSession(tmp_path,mode='train').env['LAB03_SESSION_ID']


def test_preflight_isolates_session_environment_without_changing_training_env(tmp_path,monkeypatch):
    monkeypatch.setenv('LAB03_SESSION_ID','parent-notebook-session')
    session=NotebookSession(tmp_path,mode='train')
    training_session=session.env['LAB03_SESSION_ID']
    captured={}
    class Process:
        stdout=()
        def wait(self):return 0
    def launch(*args,**kwargs):
        captured.update(kwargs)
        return Process()
    monkeypatch.setattr('labs.lab03.src.notebook_runtime.subprocess.Popen',launch)
    session.preflight()
    assert captured['env'].get('LAB03_SESSION_ID') is None
    assert captured['env']['CUDA_VISIBLE_DEVICES']==''
    assert session.env['LAB03_SESSION_ID']==training_session
    import os
    assert os.environ['LAB03_SESSION_ID']=='parent-notebook-session'
