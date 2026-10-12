"""Training and evaluation engine for PyTorch models.
Implements:
1. train_one_epoch: standard forward -> loss -> backward -> optimize loop with zero_grad(set_to_none=True).
2. evaluate: inference evaluation under torch.inference_mode().
3. train_model: full training lifecycle with LR scheduling, checkpointing, and metric logging.
4. sanity_overfit_check: TDD sanity check on 1 batch to verify model learnability before full training.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import build_fold_loaders
from .utils import set_seed
from .config import NUM_CLASSES


def build_optimizer(
    optimizer_type: str,
    model: nn.Module,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Factory function that builds an optimizer by name.

    Used by the optimizer-comparison experiment so that the only varying
    factor between arms is the optimizer itself.

    Args:
        optimizer_type: 'sgd', 'rmsprop', 'adam', or 'adamw'.
        model: The model whose parameters will be optimized.
        learning_rate: Step size shared across all arms.
        momentum: Momentum factor (only applied to SGD).
        weight_decay: L2 penalty; disabled (0.0) for the comparison experiment.

    Returns:
        An instantiated torch.optim.Optimizer.
    """
    name = optimizer_type.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer_type '{optimizer_type}'. "
            f"Choose from 'sgd', 'rmsprop', 'adam', 'adamw'."
        )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs one training epoch over the given dataloader.
    
    Returns:
        (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 1. Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # 2. Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # 3. Backward pass
        loss.backward()

        # 4. Gradient clipping (prevents exploding gradients)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # 5. Optimizer step
        optimizer.step()

        # 6. Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluates the model on validation or test dataloader.
    
    Returns:
        (eval_loss, eval_accuracy, all_predictions, all_targets)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    eval_loss = running_loss / total
    eval_acc = (correct / total) * 100.0
    return eval_loss, eval_acc, all_preds, all_targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    num_epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, List[float]], float]:
    """Executes the full training lifecycle with validation and checkpointing.
    
    Returns:
        (history_dict, best_val_accuracy)
    """
    model.to(device)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train 1 epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Learning rate step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_elapsed = time.time() - epoch_start

        # Checkpointing: Save model with best validation accuracy
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, str(save_path))

        if verbose:
            best_tag = " [BEST CHECKPOINT SAVED]" if (is_best and save_path) else ""
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f}{best_tag}"
            )

    total_time = time.time() - start_time
    if verbose:
        print(f"Training completed in {total_time:.1f}s. Best Validation Accuracy: {best_val_acc:.2f}%\n")

    return history, best_val_acc


def sanity_overfit_check(
    model: nn.Module,
    sample_images: torch.Tensor,
    sample_targets: torch.Tensor,
    device: torch.device,
    num_epochs: int = 40,
    lr: float = 1e-2,
) -> Tuple[bool, float, float]:
    """Sanity Check: Overfit on a tiny mini-batch (8 samples) to verify learnability.
    Mandatory TDD check before full-scale training.
    
    Returns:
        (passed, final_loss, final_accuracy)
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    images = sample_images.to(device)
    targets = sample_targets.to(device)

    final_loss = 1.0
    final_acc = 0.0

    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        final_acc = (preds == targets).float().mean().item() * 100.0
        final_loss = loss.item()

    passed = (final_acc >= 99.0) and (final_loss < 0.1)
    return passed, final_loss, final_acc


def compute_tpr_fpr(
    cm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derives per-class True Positive Rate and False Positive Rate from a confusion matrix.

    Args:
        cm: Square confusion matrix of shape [num_classes, num_classes], rows = true, cols = predicted.

    Returns:
        (tpr, fpr), each a 1D array of length num_classes.
        - TPR (recall / sensitivity) for class i = TP / (TP + FN).
        - FPR (fall-out) for class i = FP / (FP + TN).
    """
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp  # row sum minus diagonal
    fp = cm.sum(axis=0) - tp  # column sum minus diagonal
    tn = total - tp - fn - fp

    tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    fpr = np.divide(fp, fp + tn, out=np.zeros_like(fp), where=(fp + tn) > 0)
    return tpr, fpr


def build_per_class_metrics(
    targets,
    preds,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Builds the full per-class metric set for a classification run.

    Combines scikit-learn's per-class precision, recall, and F1 with the
    per-class True Positive Rate and False Positive Rate derived from the
    confusion matrix. FPR is not part of the scikit-learn classification report,
    so it must be computed here from the confusion matrix; recall and TPR are
    the same quantity and are reported under both names for clarity.

    Args:
        targets: Ground-truth integer labels.
        preds: Predicted integer labels.
        class_names: Ordered class names, index i naming class i.

    Returns:
        Mapping of class name to a dictionary with `precision`, `recall_tpr`,
        `f1`, and `fpr`, each rounded to four decimals.
    """
    report = classification_report(
        targets, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    tpr, fpr = compute_tpr_fpr(cm)

    per_class: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(report[name]["precision"]), 4),
            "recall_tpr": round(float(tpr[i]), 4),
            "f1": round(float(report[name]["f1-score"]), 4),
            "fpr": round(float(fpr[i]), 4),
        }
    return per_class


def build_convergence_metrics(
    histories: List[Dict[str, List[float]]],
    checkpoint_epochs: Optional[List[int]] = None,
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """Derives convergence-speed metrics from the per-fold validation history.

    The final validation accuracy saturates for every optimizer arm on this
    task, so it cannot separate the update rules. The per-epoch history that
    `run_kfold_cv` already records is used instead to measure how quickly each
    arm reaches the common ceiling: the mean validation accuracy and loss at
    fixed epoch checkpoints, the first epoch at which the mean validation
    accuracy crosses a threshold, and the epoch-averaged accuracy/loss (an
    area-under-the-learning-curve summary).

    Args:
        histories: One history dict per fold, each with `val_acc` and `val_loss`
            lists indexed by epoch.
        checkpoint_epochs: Epoch numbers (1-indexed) at which to read the mean
            validation accuracy and loss. Defaults to [3, 5, 10, 15].
        thresholds: Accuracy thresholds (in percent) for which to find the first
            reaching epoch. Defaults to [88.0, 90.0].

    Returns:
        A dictionary with `num_epochs`, `checkpoints`, `first_epoch_to_threshold`,
        `epoch_avg_val_acc`, and `epoch_avg_val_loss`.
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [3, 5, 10, 15]
    if thresholds is None:
        thresholds = [88.0, 90.0]

    val_acc = np.asarray([h["val_acc"] for h in histories], dtype=np.float64)
    val_loss = np.asarray([h["val_loss"] for h in histories], dtype=np.float64)
    num_epochs = int(val_acc.shape[1])

    mean_acc = val_acc.mean(axis=0)
    mean_loss = val_loss.mean(axis=0)

    checkpoints: Dict[str, Dict[str, float]] = {}
    for epoch in checkpoint_epochs:
        if 1 <= epoch <= num_epochs:
            checkpoints[str(epoch)] = {
                "mean_val_acc": round(float(mean_acc[epoch - 1]), 4),
                "mean_val_loss": round(float(mean_loss[epoch - 1]), 4),
            }

    first_epoch_to_threshold: Dict[str, Optional[int]] = {}
    for thr in thresholds:
        reached = np.where(mean_acc >= thr)[0]
        first_epoch_to_threshold[f"acc_{thr:g}"] = int(reached[0]) + 1 if reached.size > 0 else None

    return {
        "num_epochs": num_epochs,
        "checkpoints": checkpoints,
        "first_epoch_to_threshold": first_epoch_to_threshold,
        "epoch_avg_val_acc": round(float(mean_acc.mean()), 4),
        "epoch_avg_val_loss": round(float(mean_loss.mean()), 4),
    }


def run_kfold_cv(
    pool_train: Dataset,
    pool_eval: Dataset,
    targets: torch.Tensor,
    optimizer_type: str,
    model_type: str = "tuned_mlp",
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
    num_epochs: int = 15,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    n_splits: int = 5,
    seed: int = 42,
    num_workers: int = 0,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_suffix: str = "",
    verbose: bool = True,
) -> Dict:
    """Runs Stratified K-Fold cross-validation for a single optimizer arm.

    The full training pool is split into `n_splits` stratified folds. For each
    fold the model is re-initialized with a deterministic per-fold seed, trained
    on the remaining folds, and evaluated on the held-out fold. Per-fold scores
    are aggregated into mean and standard deviation, and the out-of-fold
    predictions of every sample are concatenated into one global confusion
    matrix from which per-class TPR/FPR are derived.

    `checkpoint_suffix` is inserted before the fold index in each checkpoint
    filename. Optimizer scenario 2a and 2b reuse the same arm names (SGD /
    RMSprop / Adam), so without a suffix the second scenario would silently
    overwrite the first scenario's per-fold checkpoints.

    Returns:
        A dictionary with per-fold accuracies/losses, their mean and standard
        deviation, the aggregated confusion matrix, and per-class TPR/FPR.
    """
    from .model import build_model  # local import avoids a module import cycle

    y = np.asarray(targets)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    criterion = nn.CrossEntropyLoss()

    fold_accuracies: List[float] = []
    fold_losses: List[float] = []
    histories: List[Dict[str, List[float]]] = []
    oof_preds: List[int] = []
    oof_targets: List[int] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        set_seed(seed + fold)
        model = build_model(model_type).to(device)
        optimizer = build_optimizer(
            optimizer_type=optimizer_type,
            model=model,
            learning_rate=learning_rate,
            momentum=momentum,
        )

        fold_train_loader, fold_val_loader = build_fold_loaders(
            pool_train=pool_train,
            pool_eval=pool_eval,
            train_indices=train_idx,
            val_indices=val_idx,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        if verbose:
            print(f"  Fold {fold}/{n_splits} | train={len(train_idx)} val={len(val_idx)}")

        save_path = None
        if checkpoint_dir is not None:
            save_path = Path(checkpoint_dir) / f"cv_{optimizer_type}{checkpoint_suffix}_fold{fold}.pt"

        history, _ = train_model(
            model=model,
            train_loader=fold_train_loader,
            val_loader=fold_val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            num_epochs=num_epochs,
            device=device,
            save_path=save_path,
            verbose=verbose,
        )

        # Restore the best-validation checkpoint before scoring the fold
        if save_path is not None and Path(save_path).exists():
            ckpt = torch.load(str(save_path), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        val_loss, val_acc, val_preds, val_targets = evaluate(
            model=model,
            dataloader=fold_val_loader,
            criterion=criterion,
            device=device,
        )

        fold_accuracies.append(val_acc)
        fold_losses.append(val_loss)
        histories.append(history)
        oof_preds.extend(val_preds)
        oof_targets.extend(val_targets)

    cm = confusion_matrix(oof_targets, oof_preds, labels=list(range(NUM_CLASSES)))
    tpr, fpr = compute_tpr_fpr(cm)

    return {
        "optimizer_type": optimizer_type,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "n_splits": n_splits,
        "fold_accuracies": fold_accuracies,
        "fold_losses": fold_losses,
        "mean_acc": float(np.mean(fold_accuracies)),
        "std_acc": float(np.std(fold_accuracies)),
        "mean_loss": float(np.mean(fold_losses)),
        "std_loss": float(np.std(fold_losses)),
        "confusion_matrix": cm,
        "tpr_per_class": tpr,
        "fpr_per_class": fpr,
        "histories": histories,
        "convergence": build_convergence_metrics(histories),
        "oof_preds": oof_preds,
        "oof_targets": oof_targets,
    }


def run_optimizer_scenario(
    scenario,
    pool_train: Dataset,
    pool_eval: Dataset,
    targets: torch.Tensor,
    model_type: str = "tuned_mlp",
    num_epochs: int = 15,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
    n_splits: int = 5,
    seed: int = 42,
    num_workers: int = 0,
    checkpoint_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Runs every optimizer arm of a single learning-rate scenario.

    Args:
        scenario: An `OptimizerScenario` (from `src.config`) exposing `.name`
            and `.arms`; each arm carries its own learning rate and momentum.

    Returns:
        Mapping of arm name to that arm's `run_kfold_cv` result dictionary.
    """
    scenario_results: Dict[str, Dict] = {}
    for arm in scenario.arms:
        if verbose:
            print(
                f"\n>>> Scenario {scenario.name} | arm {arm.name} "
                f"(type={arm.optimizer_type}, lr={arm.learning_rate})"
            )
        scenario_results[arm.name] = run_kfold_cv(
            pool_train=pool_train,
            pool_eval=pool_eval,
            targets=targets,
            optimizer_type=arm.optimizer_type,
            model_type=model_type,
            learning_rate=arm.learning_rate,
            momentum=arm.momentum,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            n_splits=n_splits,
            seed=seed,
            num_workers=num_workers,
            checkpoint_dir=checkpoint_dir,
            checkpoint_suffix=scenario.artifact_suffix,
            verbose=verbose,
        )
    return scenario_results
