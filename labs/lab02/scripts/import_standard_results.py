"""Validate downloaded standard artifacts and generate offline report evidence."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]


def main():
    archive=ROOT/'outputs/kaggle/standard/lab02_standard_outputs.zip'
    destination=archive.parent/'extracted'
    with ZipFile(archive) as z:
        for member in z.infolist():
            if not (destination/member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Archive path escapes destination')
        bad=z.testzip()
        if bad:raise ValueError(f'CRC mismatch: {bad}')
        if not destination.exists():
            z.extractall(destination)
    registry=json.loads((destination/'outputs/selected_standard.json').read_text())
    run=destination/registry['run']
    assert json.loads((run/'status.json').read_text())['status']=='completed'
    manifest=json.loads((run/'manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest,name
    expected={(m,s) for m in ('resnet18','densenet121') for s in ('head','finetune')}
    comparison=pd.read_csv(run/'comparison.csv')
    assert set(zip(comparison.model,comparison.stage))==expected
    for row in comparison.itertuples():
        folder=run/row.model/'evaluation'/row.stage
        with np.load(folder/'test_predictions.npz',allow_pickle=False) as data:
            p,y,ids=data['probabilities'],data['labels'],data['indices']
            assert p.shape==(10000,10) and len(np.unique(ids))==10000
            assert np.array_equal(np.sort(ids),np.arange(10000))
            assert np.isfinite(p).all()
            np.testing.assert_allclose(p.sum(1),1,atol=1e-5)
            for key,value in {'accuracy':accuracy_score(y,p.argmax(1)),
                'macro_f1':f1_score(y,p.argmax(1),average='macro'),
                'loss':log_loss(y,p,labels=np.arange(10))}.items():
                assert abs(value-getattr(row,key))<1e-7,(row.model,row.stage,key)
            counts=pd.read_csv(folder/'confusion_counts.csv',index_col=0).to_numpy()
            np.testing.assert_array_equal(counts,confusion_matrix(y,p.argmax(1),labels=np.arange(10)))
            with np.load(folder/'loaded/test_predictions.npz',allow_pickle=False) as restored:
                np.testing.assert_array_equal(restored['labels'],y)
                np.testing.assert_allclose(restored['probabilities'],p,rtol=1e-5,atol=1e-6)
        if row.stage=='finetune':
            select=json.loads((run/row.model/'selection.json').read_text())
            trials=pd.read_csv(run/row.model/'trials.csv')
            assert len(trials)==3 and (trials.state=='COMPLETE').all()
            assert trials.loc[trials.value.idxmin(),'number']==select['trial']
    review=ROOT/'outputs/review'/run.name
    review.mkdir(parents=True,exist_ok=True)
    scopes=[]
    for model,stage in sorted(expected):
        frame=pd.read_csv(destination/'outputs/setup'/f'{model}_{stage}_parameter_scope.csv')
        total=int(frame.parameters.sum())
        trainable=int(frame.loc[frame.trainable.astype(str).str.lower()=='true','parameters'].sum())
        scopes.append({'model':model,'stage':stage,'total':total,'trainable':trainable,
            'frozen':total-trainable,'unfrozen_pct':100*trainable/total,
            'unfrozen_parts':'classifier' if stage=='head' else 'layer4 + fc' if model=='resnet18' else 'denseblock4 + norm5 + classifier'})
    scopes=pd.DataFrame(scopes);scopes.to_csv(review/'parameter_percentages.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    labels=scopes.model+' '+scopes.stage
    ax.barh(labels,100-scopes.unfrozen_pct,label='Frozen (%)',color='#cbd5e1')
    bars=ax.barh(labels,scopes.unfrozen_pct,left=100-scopes.unfrozen_pct,label='Trainable (%)',color='#2563eb')
    for i,r in enumerate(scopes.itertuples()):ax.text(101,i,f'{r.unfrozen_pct:.3f}%',va='center')
    ax.set(xlim=(0,113),xlabel='Percentage of parameters',title='Trainable parameters; not percentage of layers')
    ax.legend();fig.tight_layout();fig.savefig(review/'parameter_percentages.png',dpi=140);plt.close(fig)
    result={'status':'passed','zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'manifest_files_verified':len(manifest),'conditions_verified':4,'prediction_metrics_recomputed':True,
        'loaded_full_test_matches':True,'source_run':run.relative_to(ROOT).as_posix()}
    (review/'import_verification.json').write_text(json.dumps(result,indent=2))
    (ROOT/'outputs/selected_standard.json').write_text(json.dumps({'run':run.relative_to(ROOT).as_posix()},indent=2))
    # Preserve the user's downloaded notebook and all Kaggle output cells.
    import shutil
    import base64
    import nbformat
    notebook_path=ROOT/'notebooks/04_lab02_standard_kaggle.ipynb'
    original=archive.parent/'04_lab02_standard_kaggle_executed_original.ipynb'
    if not original.exists():shutil.copy2(notebook_path,original)
    notebook=nbformat.read(notebook_path,as_version=4)
    notebook.cells=[c for c in notebook.cells if c.get('metadata',{}).get('standard_scope_postprocessing') is not True]
    note=nbformat.v4.new_markdown_cell('## Bổ sung sau training: tỷ lệ tham số unfrozen\n\nTính từ parameter_scope CSV đã lưu trong phiên Kaggle, không train lại. Unfrozen (%) = trainable / total × 100; đây là tỷ lệ tham số, không phải số layer. Bản notebook tải về nguyên gốc được giữ riêng trong outputs/kaggle/standard/.')
    note.metadata['standard_scope_postprocessing']=True
    source="""from pathlib import Path
import sys
candidates = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in candidates if (p/'config/standard.json').is_file()), None) or next(p/'labs/lab02' for p in candidates if (p/'labs/lab02/config/standard.json').is_file())
sys.path.insert(0, str(ROOT))
from src.utils.notebook_views import standard_parameter_summary
scope_table = standard_parameter_summary(ROOT)"""
    cell=nbformat.v4.new_code_cell(source)
    cell.metadata.update(standard_scope_postprocessing=True, evidence_run=run.name,
        execution_origin='Local artifact postprocessing; no training')
    cell.execution_count=max([c.get('execution_count') or 0 for c in notebook.cells]+[0])+1
    cell.outputs=[nbformat.v4.new_output('display_data',data={'text/html':scopes.round({'unfrozen_pct':4}).to_html(index=False),'text/plain':scopes.to_string(index=False)}),
        nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((review/'parameter_percentages.png').read_bytes()).decode()})]
    notebook.cells.extend([note,cell])
    nbformat.validate(notebook)
    nbformat.write(notebook,notebook_path)
    print(comparison.to_string(index=False));print(scopes.to_string(index=False));print(result)


if __name__=='__main__':main()
