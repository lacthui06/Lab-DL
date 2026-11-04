"""Render notebook từ artifact được chọn, không khởi động training hoặc tải dataset."""
from pathlib import Path
import argparse
import re
import sys
import nbformat
from nbclient import NotebookClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.runs import ROOT,run_context,save_json,hash_file

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kernel',default='python3')
    args=parser.parse_args()
    if not (ROOT/'outputs/latest_report.json').is_file():
        raise FileNotFoundError('Chạy notebook modeling trước để tạo outputs/latest_report.json.')
    paths=sorted((ROOT/'notebooks').glob('*.ipynb'))
    with run_context('notebook_render',{'mode':'view','kernel':args.kernel,'training_executed':False}) as out:
        results=[]
        for path in paths:
            notebook=nbformat.read(path,4);sources={}
            for index,cell in enumerate(notebook.cells):
                if cell.cell_type!='code':continue
                if re.search(r'^MODE = "(?:prepare|train|view)"',cell.source,flags=re.M):
                    sources[index]=cell.source
                    cell.source=re.sub(r'^MODE = "(?:prepare|train|view)"','MODE = "view"',cell.source,flags=re.M)
            if len(sources)!=1:raise ValueError('Notebook cần đúng một khai báo MODE')
            NotebookClient(notebook,timeout=180,kernel_name=args.kernel,
                resources={'metadata':{'path':str(ROOT/'notebooks')}}).execute()
            for index,source in sources.items():notebook.cells[index].source=source
            nbformat.validate(notebook);nbformat.write(notebook,path)
            row={'notebook':path.name,'mode':'view','training_executed':False,
                'notebook_sha256':hash_file(path),'errors':0,
                'executed_code_cells':sum(c.cell_type=='code' for c in notebook.cells),
                'inline_pngs':sum('image/png' in o.get('data',{}) for c in notebook.cells if c.cell_type=='code' for o in c.outputs)}
            results.append(row);print('Rendered:',path.name)
        save_json(out/'execution_summary.json',results)
        save_json(ROOT/'validation/notebook_execution.json',{'artifact_run':out.name,'notebooks':results})
