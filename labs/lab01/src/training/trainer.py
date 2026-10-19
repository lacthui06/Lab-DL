"""Training engine and Trainer class with scheduler support."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def build_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Builds optimizer adhering strictly to textbook definitions:
    - SGD: with momentum
    - RMSprop: pure adaptive 2nd-moment (no momentum)
    - Adam: adaptive with 1st and 2nd moments
    """
    opt_name = optimizer_name.upper()
    if opt_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "RMSPROP":
        # Pure RMSprop as originally designed by Hinton (no momentum to preserve theoretical distinction)
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == "ADAM":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class Trainer:
    """Encapsulates the training and validation loops with scheduler and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train_one_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Trains for one epoch, returning (average_loss, accuracy_percentage)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in dataloader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    @torch.inference_mode()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, List[int], List[int]]:
        """Evaluates model, returning (loss, accuracy, all_preds, all_targets)."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        for images, targets in dataloader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

        val_loss = running_loss / total
        val_acc = (correct / total) * 100.0
        return val_loss, val_acc, all_preds, all_targets

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        checkpoint_path: Optional[Path] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Executes full training run with CosineAnnealing/scheduler updates."""
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0

        for ep in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if checkpoint_path is not None:
                    from ..utils.helpers import save_checkpoint
                    save_checkpoint(self.model, checkpoint_path, metadata={"epoch": ep, "val_acc": val_acc})

            if verbose:
                lr_curr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}% | LR: {lr_curr:.6f}")

        return history


def sanity_overfit_check(
    model: nn.Module,
    sample_images: torch.Tensor,
    sample_targets: torch.Tensor,
    device: torch.device,
    num_epochs: int = 40,
    lr: float = 1e-2,
) -> Tuple[bool, float, float]:
    """Mandatory Sanity Test (AGENTS.md Rule 1.4): Overfits 1 mini-batch (8 samples) to loss near 0."""
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    images = sample_images.to(device)
    targets = sample_targets.to(device)

    final_loss = 1.0
    final_acc = 0.0
    for _ in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        preds = outputs.argmax(dim=1)
        final_acc = (preds == targets).float().mean().item() * 100.0

    passed = (final_loss < 0.15) and (final_acc >= 90.0)
    return passed, final_loss, final_acc
