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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
