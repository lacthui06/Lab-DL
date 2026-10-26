"""Validated fine-tuning candidates and validation-only quality/cost selection."""
import math

DEPTHS = ('last_block', 'last_two_blocks')
OPTIMIZERS = ('adamw', 'sgd')


def fine_candidates(cfg: dict) -> list[dict]:
    """Keep legacy configs runnable; explicit candidates replace the fixed fine stage."""
    candidates = cfg.get('finetune_candidates')
    if candidates is None:
        candidates = [dict(id='default', optimizer=cfg.get('optimizer', 'adamw'),
            backbone_lr=cfg['finetune_backbone_lr'], head_lr=cfg['finetune_head_lr'],
            unfreeze=cfg.get('unfreeze', 'last_block'), scheduler=cfg.get('scheduler', 'none'))]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('finetune_candidates must be a nonempty list')
    result, ids = [], set()
    for number, candidate in enumerate(candidates):
        c = dict(candidate)
        c.setdefault('id', f'candidate_{number:02d}')
        c.setdefault('optimizer', 'adamw')
        c.setdefault('unfreeze', 'last_block')
        c.setdefault('scheduler', 'none')
        if not isinstance(c['id'], str) or not c['id'] or c['id'] in ids:
            raise ValueError('Candidate IDs must be unique nonempty strings')
        ids.add(c['id'])
        if c['optimizer'] not in OPTIMIZERS or c['unfreeze'] not in DEPTHS:
            raise ValueError('Unsupported optimizer or unfreeze depth')
        if c['scheduler'] not in ('none', 'cosine'):
            raise ValueError('scheduler must be none or cosine')
        for key in ('backbone_lr', 'head_lr'):
            if key not in c or not math.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if cfg.get('cache_frozen_prefix') and c['unfreeze'] != 'last_block':
            raise ValueError('Opening earlier blocks requires cache_frozen_prefix=false')
        result.append(c)
    return result


def select_trial(records: list[dict], policy: dict | None = None) -> dict:
    """Accuracy band -> loss band -> lower fit time, never inspect test metrics.

    Bands are absolute units: accuracy 0.001 means 0.1 percentage point.
    Defaults preserve strict accuracy selection, with validation loss on ties.
    """
    policy = policy or {}
    acc_tol = policy.get('accuracy_tolerance', 0.)
    loss_tol = policy.get('loss_tolerance', 0.)
    if not records or not all(math.isfinite(x) and x >= 0 for x in (acc_tol, loss_tol)):
        raise ValueError('Nonempty trials and finite nonnegative tolerances required')
    for r in records:
        if not all(math.isfinite(r[k]) for k in ('best_val_accuracy', 'best_val_loss', 'train_seconds')):
            raise ValueError('Validation quality and time must be finite')
        if not 0 <= r['best_val_accuracy'] <= 1 or r['best_val_loss'] < 0 or r['train_seconds'] < 0:
            raise ValueError('Invalid validation quality or time')
    best_acc = max(r['best_val_accuracy'] for r in records)
    eligible = [r for r in records if r['best_val_accuracy'] >= best_acc-acc_tol-1e-12]
    best_loss = min(r['best_val_loss'] for r in eligible)
    eligible = [r for r in eligible if r['best_val_loss'] <= best_loss+loss_tol+1e-12]
    return min(eligible, key=lambda r: r['train_seconds'])


def make_optimizer(groups, name: str, weight_decay: float, momentum: float = .9):
    """Use optimizer-specific LRs in parameter groups. PyTorch optim docs apply.

    https://docs.pytorch.org/docs/stable/optim.html
    """
    import torch
    if name == 'adamw':
        return torch.optim.AdamW(groups, weight_decay=weight_decay)
    if name == 'sgd':
        return torch.optim.SGD(groups, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {name}')


def make_scheduler(optimizer, name: str, epochs: int):
    """Step cosine after each training epoch; log the LR used before stepping.

    https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    """
    if name == 'none':
        return None
    if name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f'Unsupported scheduler: {name}')
