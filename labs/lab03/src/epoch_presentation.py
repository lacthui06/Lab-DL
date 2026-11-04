"""Trình bày mỗi epoch một điểm từ history thật; không đổi weights hoặc raw run."""
from pathlib import Path
from .utils.runs import ROOT, hash_file, run_context, save_json
from .utils.plots import plot_epoch_learning
from .evidence import read_json, verify_manifest


def ensure_epoch_presentation(study, root=ROOT):
    root=Path(root).resolve();study=Path(study).resolve();runs=root/'outputs/runs'
    if study.parent!=runs:raise ValueError('Study must be a direct child of outputs/runs')
    verify_manifest(study)
    candidates=read_json(study/'candidate_comparison.json')
    sources=[]
    for row in candidates:
        path=(runs/row['run_id']).resolve()
        if path.parent!=runs:raise ValueError('Invalid candidate path')
        verify_manifest(path);sources.append(path)
    key={'source_study':study.name,'source_manifests':{p.name:hash_file(p/'manifest.json') for p in [study,*sources]},
         'presentation_code':hash_file(Path(__file__)), 'plots_code':hash_file(Path(__file__).parent/'utils/plots.py')}
    state=read_json(root/'config/workflow.json')
    for row in reversed(state['runs']):
        if row['kind']!='epoch_presentation' or row['status']!='completed':continue
        out=runs/row['id']
        if read_json(out/'presentation_source.json')==key:
            verify_manifest(out);return out
    with run_context('epoch_presentation',key,root=root) as out:
        save_json(out/'presentation_source.json',key)
        summary=[]
        for source in sources:
            target=out/source.name;target.mkdir()
            rows=plot_epoch_learning(read_json(source/'trainer_history.json'),target/'learning_curves.png')
            requested=read_json(source/'config.json')['exercise2']['num_train_epochs']
            if len(rows)!=requested:raise ValueError('Epoch plot must contain every requested epoch')
            save_json(target/'epoch_metrics.json',rows)
            summary.append({'run_id':source.name,'epochs':[r['epoch'] for r in rows],'points_per_curve':len(rows),
                'plot':f'{source.name}/learning_curves.png','raw_run_modified':False})
        save_json(out/'summary.json',{'study_run':study.name,'candidates':summary,'model_retrained':False,
            'purpose':'integer end-of-epoch display only; does not reduce model overfitting'})
    return out
