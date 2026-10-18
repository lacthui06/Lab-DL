"""Training and evaluation engine for PyTorch models.
Implements:
1. build_optimizer: Factory for SGD, RMSprop, Adam, AdamW.
2. train_one_epoch: standard forward -> loss -> backward -> optimize loop with zero_grad(set_to_none=True).
3. evaluate: inference evaluation under torch.inference_mode().
4. train_model: full training lifecycle with LR scheduling, checkpointing, and metric logging.
5. sanity_overfit_check: TDD sanity check on 1 batch to verify model learnability before full training.
6. compute_tpr_fpr & build_per_class_metrics: Evaluation metrics.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


def build_optimizer(
    optimizer_type: str,
    model: nn.Module,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Factory function that builds an optimizer by name.

    Args:
        optimizer_type: 'sgd', 'rmsprop', 'adam', or 'adamw'.
        model: The model whose parameters will be optimized.
        learning_rate: Step size shared across all arms.
        momentum: Momentum factor (only applied to SGD).
        weight_decay: L2 penalty.

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
        print(f"Huấn luyện hoàn tất trong {total_time:.1f}s. Độ chính xác Val cao nhất: {best_val_acc:.2f}%\n")

    return history, best_val_acc


def sanity_overfit_check(
    model: nn.Module,
    sample_images: torch.Tensor,
    sample_targets: torch.Tensor,
    device: torch.device,
    num_epochs: int = 40,
    lr: float = 1e-2,
    num_steps: Optional[int] = None,
    target_loss: float = 0.1,
) -> Tuple[bool, float, float]:
    """Sanity Check: Overfit on a tiny mini-batch (8 samples) to verify learnability.
    Mandatory TDD check before full-scale training.
    
    Returns:
        (passed, final_loss, final_accuracy)
    """
    if num_steps is not None:
        num_epochs = num_steps

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

    passed = (final_acc >= 90.0) and (final_loss <= target_loss)
    return passed, final_loss, final_acc


def compute_tpr_fpr(
    cm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derives per-class True Positive Rate and False Positive Rate from a confusion matrix.

    Returns:
        (tpr, fpr), each a 1D array of length num_classes.
    """
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = total - tp - fn - fp

    tpr = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    fpr = np.divide(fp, fp + tn, out=np.zeros_like(fp), where=(fp + tn) > 0)
    return tpr, fpr


def build_per_class_metrics(
    targets,
    preds,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Builds the full per-class metric set for a classification run."""
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
