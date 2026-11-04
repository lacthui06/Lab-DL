import pytest
from IPython.display import Image
from src.notebook_runtime import NotebookSession
from src.utils.runs import ROOT, hash_file


def test_view_architecture_reads_verified_text_without_instantiating_models(tmp_path, monkeypatch):
    from src.utils.runs import save_json
    folder=tmp_path/'labs/new-lab03/outputs/architecture';folder.mkdir(parents=True)
    (folder/'model_architectures.txt').write_text('Verified module tree fixture')
    save_json(folder/'summary.json',{'models':[{'role':'exercise1'},{'role':'exercise2'}]})
    save_json(folder/'manifest.json',{name:hash_file(folder/name) for name in ('model_architectures.txt','summary.json')})
    save_json(folder.parent/'latest_report.json',{'runs':{'frozen_trial':'selected-fixture'}})
    session=NotebookSession(tmp_path,mode='view')
    monkeypatch.setattr(session,'run',lambda *a,**kw:pytest.fail('View must not instantiate models'))
    monkeypatch.setattr('IPython.display.display',lambda *a:None)
    before={p.name:hash_file(p) for p in folder.iterdir()}
    assert session.architectures()==folder/'model_architectures.txt'
    assert before=={p.name:hash_file(p) for p in folder.iterdir()}
    (folder/'model_architectures.txt').write_text('tampered')
    with pytest.raises(ValueError,match='SHA256'):session.architectures()


@pytest.mark.parametrize('incomplete_dedicated_clone', [False, True])
def test_notebook_bootstrap_preserves_existing_clone_and_resolves_own_profile(tmp_path, monkeypatch, incomplete_dedicated_clone):
    import json
    working=tmp_path/'kaggle/working';working.mkdir(parents=True)
    original=working/'Lab-DL';original.mkdir()
    (original/'keep.txt').write_text('existing results')
    dedicated=working/'Lab-DL-new-lab03'
    if incomplete_dedicated_clone:
        dedicated.mkdir();(dedicated/'keep.txt').write_text('preserve incomplete clone')
    monkeypatch.chdir(working)
    calls=[]
    def clone(command, check):
        calls.append(command)
        config=__import__('pathlib').Path(command[-1])/'labs/new-lab03/config/frozen_trial.json'
        config.parent.mkdir(parents=True)
        config.write_text('{}')
    monkeypatch.setattr('subprocess.run',clone)
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        notebook=json.loads(path.read_text())
        source=next(''.join(c['source']) for c in notebook['cells'] if 'REPO_REF = "new-lab03"' in ''.join(c['source']))
        prefix=source.split('profile = json.loads',1)[0]
        prefix=prefix.replace('Path("/kaggle/working")',f'Path({str(working)!r})')
        namespace={};exec(prefix,namespace)
        resolved=namespace['REPO']
        assert resolved != original and (resolved/namespace['PROFILE']).is_file()
        assert (original/'keep.txt').read_text()=='existing results'
    assert all(command[command.index('--branch')+1]=='new-lab03' for command in calls)
    if incomplete_dedicated_clone:
        assert (dedicated/'keep.txt').read_text()=='preserve incomplete clone'


def test_view_notebooks_show_saved_images_without_training_or_mutation(monkeypatch,tmp_path):
    import matplotlib.pyplot as plt
    from src.utils.runs import save_json
    folder=tmp_path/'labs/new-lab03/outputs/runs/dataset_preview_fixture';folder.mkdir(parents=True)
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
    from scripts.run_frozen_trial import resolve_device
    monkeypatch.setattr('torch.cuda.is_available',lambda:False)
    assert resolve_device('auto') == resolve_device('cpu') == 'cpu'
    with pytest.raises(RuntimeError,match='CUDA requested'):
        resolve_device('cuda')
    monkeypatch.setattr('torch.cuda.is_available',lambda:True)
    assert resolve_device('auto') == 'cuda' and resolve_device('cpu') == 'cpu'
    with pytest.raises(ValueError):resolve_device('invalid')


def test_preparation_export_is_self_contained_and_detects_corruption(tmp_path, monkeypatch):
    import json
    import zipfile
    import hashlib
    from src.utils.runs import save_json
    session = NotebookSession(tmp_path, mode='prepare')
    session.kaggle = False
    for role in ('data', 'augmentation'):
        folder = session.lab/'outputs/runs'/role
        folder.mkdir(parents=True)
        (folder/'console.log').write_text('actual preparation log')
        (folder/'diagnostic.png').write_bytes(b'fixture-image')
        save_json(folder/'config.json', {'session_id': session.env['LAB03_SESSION_ID']})
        save_json(folder/'status.json', {'status': 'completed'})
        save_json(folder/'manifest.json', {p.name: hash_file(p) for p in folder.iterdir()})
        session.paths[role] = folder
    save_json(session.lab/'outputs/latest_preparation.json', {'runs': {k: p.name for k,p in session.paths.items()}})
    monkeypatch.setattr('IPython.display.display', lambda *a: None)
    archive = session.export_preparation()
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('bundle_manifest.json'))
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n,h in manifest.items())
        assert 'outputs/latest_preparation.json' in manifest
        assert 'outputs/preparation/index.md' in manifest
        assert len([n for n in manifest if n.endswith('.png')]) == 4
        assert 'diagnostic.png' in z.read('outputs/preparation/index.md').decode()
    (session.paths['data']/'console.log').write_text('tampered')
    with pytest.raises(ValueError): session.export_preparation()


def test_train_session_starts_without_loading_report_or_previous_selection(tmp_path):
    from src.utils.runs import save_json
    lab=tmp_path/'labs/new-lab03'
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
    monkeypatch.setattr('src.notebook_runtime.subprocess.Popen',launch)
    session.preflight()
    assert captured['env'].get('LAB03_SESSION_ID') is None
    assert captured['env']['CUDA_VISIBLE_DEVICES']==''
    assert session.env['LAB03_SESSION_ID']==training_session
    import os
    assert os.environ['LAB03_SESSION_ID']=='parent-notebook-session'


@pytest.mark.parametrize('preflight_fails',[False,True])
def test_modeling_preflight_cell_restores_training_session_and_propagates_errors(preflight_fails):
    import nbformat
    source=nbformat.read(ROOT/'notebooks/02_modeling.ipynb',as_version=4).cells[3].source
    class Session:
        def __init__(self):self.env={'LAB03_SESSION_ID':'training-session'}
        def preflight(self):
            assert self.env.get('LAB03_SESSION_ID') is None
            if preflight_fails:raise RuntimeError('fixture preflight failure')
    session=Session()
    if preflight_fails:
        with pytest.raises(RuntimeError,match='fixture preflight failure'):
            exec(source,{'session':session})
    else:exec(source,{'session':session})
    assert session.env.get('LAB03_SESSION_ID')=='training-session'
