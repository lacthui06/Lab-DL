"""Execute a notebook in-place; used by README and delivery.complete_delivery."""
from pathlib import Path
import argparse
import json
import hashlib
import os
import sys

ROOT=Path(__file__).resolve().parents[1]


def scheduled_notebook(nbformat):
    """Dùng notebook chính làm nguồn duy nhất, giữ roadmap và chế độ đã cấu hình."""
    notebook = nbformat.read(ROOT / 'notebooks/02_lab02_modeling.ipynb', as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    return notebook


def execute(notebook_name):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    path=ROOT/'notebooks'/notebook_name
    selected = Path(os.environ.get('LAB02_RUN') or
                    json.loads((ROOT/'outputs/selected_run.json').read_text())['run'])
    selected = selected if selected.is_absolute() else ROOT/selected
    content = path.read_bytes()
    # Keep every overwritten notebook version; content hash makes snapshots immutable.
    backup = selected/'notebook_snapshots'/hashlib.sha256(content).hexdigest()/path.name
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():
        backup.write_bytes(content)
    notebook=nbformat.read(path,as_version=4)
    config=json.loads((selected/'config.json').read_text())
    if notebook_name == '02_lab02_modeling.ipynb' and config.get('profile')=='cifar10_scheduled64':
        notebook=scheduled_notebook(nbformat)
    kernels=ROOT/'.jupyter/kernels'
    spec=kernels/'lab02'
    spec.mkdir(parents=True,exist_ok=True)
    (spec/'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Lab 02',language='python')),encoding='utf-8')
    manager=KernelManager(kernel_name='lab02',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    client=NotebookClient(notebook,timeout=600,resources={'metadata':{'path':str(ROOT)}},kernel_name='lab02',km=manager)
    client.extra_arguments=['--HistoryManager.enabled=False']
    client.execute()
    nbformat.write(notebook,path)
    completed=path.read_bytes()
    archived=selected/'notebook_snapshots'/hashlib.sha256(completed).hexdigest()/path.name
    archived.parent.mkdir(parents=True,exist_ok=True)
    if not archived.exists():
        archived.write_bytes(completed)
    print('Executed successfully:',path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook',nargs='?',default='02_lab02_modeling.ipynb')
    execute(parser.parse_args(argv).notebook)


if __name__=='__main__':
    main()
