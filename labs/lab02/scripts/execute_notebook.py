"""Execute a notebook in-place; used by README and delivery.complete_delivery."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]


def execute(notebook_name):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    path=ROOT/'notebooks'/notebook_name
    notebook=nbformat.read(path,as_version=4)
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
    print('Executed successfully:',path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook',nargs='?',default='02_lab02_modeling.ipynb')
    execute(parser.parse_args(argv).notebook)


if __name__=='__main__':
    main()
