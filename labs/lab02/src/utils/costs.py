"""Compute all-search cost and selected-recipe cost without double counting."""
import json
from pathlib import Path
import pandas as pd


def cost_table(run: Path, comparison: pd.DataFrame) -> pd.DataFrame:
    """Read historical/new evidence; missing fine search means one legacy trial."""
    records = []
    for row in comparison.to_dict('records'):
        name, stage = row['model'], row['stage']
        head = json.loads((run/name/'head_selected/training_summary.json').read_text(encoding='utf-8'))
        extraction = head['feature_extraction_train_val_seconds']
        head_search = head['all_search_training_seconds']
        selected_head = head['train_seconds']
        fine_search = row.get('all_finetune_search_seconds', row['train_seconds']) if stage=='finetune' else 0.
        execution = row.get('execution', 'pooled_feature_cache' if stage=='head' else
            ('cached_frozen_prefix' if (run/name/'prefix_equivalence.json').exists() else 'full_image_forward'))
        records.append(dict(model=name, stage=stage, execution=execution,
            feature_extraction_seconds=extraction, all_head_trials_fit_seconds=head_search,
            all_finetune_trials_fit_seconds=fine_search, selected_stage_fit_seconds=row['train_seconds'],
            pipeline_training_seconds=extraction+head_search+fine_search,
            model_selection_wall_seconds=row.get('model_selection_wall_seconds'),
            selected_recipe_training_seconds=extraction+selected_head+(row['train_seconds'] if stage=='finetune' else 0),
            test_seconds=row['test_seconds'], trainable_parameters=int(row['trainable'])))
    return pd.DataFrame(records)


def validation_recommendation(run: Path, comparison: pd.DataFrame, cfg: dict) -> dict:
    """Use validation only; avoid time ranking if execution policies differ."""
    from ..training.tuning import select_trial
    costs = cost_table(run,comparison)
    records = []
    for row in comparison.to_dict('records'):
        if row['stage'] != 'finetune':
            continue
        loss = row.get('best_val_loss')
        if loss is None:
            history = pd.read_csv(run/row['model']/'finetune/history.csv')
            loss = float(history.loc[history.val_accuracy.idxmax(),'val_loss'])
        cost = costs[(costs.model==row['model']) & (costs.stage=='finetune')].iloc[0]
        records.append(dict(model=row['model'],stage='finetune',best_val_accuracy=row['best_val_accuracy'],
            best_val_loss=loss,train_seconds=cost.selected_recipe_training_seconds, execution=cost.execution))
    policy = cfg.get('selection_policy', {})
    comparable = len({r['execution'] for r in records}) == 1
    if not comparable:
        winner = max(records, key=lambda r:(r['best_val_accuracy'],-r['best_val_loss']))
    else:
        winner = select_trial(records,policy)
    return {**winner, 'validation_accuracy':winner['best_val_accuracy'],
        'time_comparable':comparable, 'selection_policy':policy,
        'criterion':'validation accuracy/loss, selected-recipe cost only with matching execution policies',
        'time_scope':'feature extraction + selected head + selected fine-tune; full search cost reported separately'}
