"""
Training pipeline for brain tumor MRI segmentation using BRISC 2025 dataset.
Mirrors the structure of train.py (classification) for consistency.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import copy
from datetime import datetime

from segmentation_model import create_segmentation_model
from segmentation_dataset import (
    BRISCSegmentationDataset,
    get_segmentation_transforms,
    create_segmentation_kfold_splits
)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def dice_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Compute Dice Score: 2*|A∩B| / (|A|+|B|)"""
    preds = (torch.sigmoid(predictions) > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1).float()
    intersection = (preds * targets).sum()
    return ((2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)).item()


def iou_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Compute IoU (Intersection over Union / Jaccard Index)"""
    preds = (torch.sigmoid(predictions) > threshold).float()
    preds = preds.view(-1)
    targets = targets.view(-1).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


# ─────────────────────────────────────────────
# Early Stopping (mirrors train.py)
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_dice: float, model: nn.Module) -> bool:
        if self.best_score is None or val_dice > self.best_score + self.min_delta:
            self.best_score = val_dice
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class SegmentationTrainer:
    """Training loop for brain tumor segmentation (mirrors ModelTrainer in train.py)"""

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler=None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: str = 'models',
                 use_amp: bool = True):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, loader: DataLoader, fast_dev_run: bool = False) -> Dict[str, float]:
        self.model.train()
        total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

        for batch_idx, batch in enumerate(tqdm(loader, desc="Train")):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                with torch.cuda.amp.autocast():
                    preds = self.model(images)
                    loss = self.criterion(preds, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(images)
                loss = self.criterion(preds, masks)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score(preds.detach(), masks)
            total_iou += iou_score(preds.detach(), masks)

            if fast_dev_run and batch_idx >= 1:
                break

        n = len(loader)
        return {'loss': total_loss / n, 'dice': total_dice / n, 'iou': total_iou / n}

    def val_epoch(self, loader: DataLoader, fast_dev_run: bool = False) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Val")):
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)

                preds = self.model(images)
                loss = self.criterion(preds, masks)

                total_loss += loss.item()
                total_dice += dice_score(preds, masks)
                total_iou += iou_score(preds, masks)

                if fast_dev_run and batch_idx >= 1:
                    break

        n = len(loader)
        return {'loss': total_loss / n, 'dice': total_dice / n, 'iou': total_iou / n}

    def train_fold(self,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   fold: int,
                   epochs: int = 50,
                   patience: int = 10,
                   fast_dev_run: bool = False) -> Dict:
        early_stopping = EarlyStopping(patience=patience)
        best_dice, best_iou = 0.0, 0.0
        history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': [], 'val_iou': []}

        for epoch in range(epochs):
            train_m = self.train_epoch(train_loader, fast_dev_run=fast_dev_run)
            val_m = self.val_epoch(val_loader, fast_dev_run=fast_dev_run)

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_m['dice'])
                else:
                    self.scheduler.step()

            history['train_loss'].append(train_m['loss'])
            history['train_dice'].append(train_m['dice'])
            history['val_loss'].append(val_m['loss'])
            history['val_dice'].append(val_m['dice'])
            history['val_iou'].append(val_m['iou'])

            if val_m['dice'] > best_dice:
                best_dice = val_m['dice']
                self.save_model(f'best_dice_fold_{fold}.pth')

            if val_m['iou'] > best_iou:
                best_iou = val_m['iou']

            print(f"Epoch {epoch+1}/{epochs} — "
                  f"Train Loss: {train_m['loss']:.4f}, Train Dice: {train_m['dice']:.4f} | "
                  f"Val Loss: {val_m['loss']:.4f}, Val Dice: {val_m['dice']:.4f}, Val IoU: {val_m['iou']:.4f}")

            if early_stopping(val_m['dice'], self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break

            if fast_dev_run and epoch >= 1:
                break

        print(f"Fold {fold} done. Best Dice: {best_dice:.4f}, Best IoU: {best_iou:.4f}")
        return history

    def save_model(self, filename: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.save_dir, filename))


# ─────────────────────────────────────────────
# Cross-Validation Entry Point
# ─────────────────────────────────────────────

def train_segmentation_with_cross_validation(
        image_dir: str,
        mask_dir: str,
        n_splits: int = 5,
        batch_size: int = 8,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 10,
        num_workers: int = 4,
        fast_dev_run: bool = False,
        target_size: Tuple[int, int] = (256, 256),
        scheduler_type: str = 'cosine',
        classifier_weights_path: Optional[str] = None) -> Dict:
    """
    Train segmentation model with k-fold cross-validation.
    Mirrors train_with_cross_validation() in train.py.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_transform, val_transform = get_segmentation_transforms(target_size)

    full_dataset = BRISCSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=None,   # transforms applied per-fold below
        target_size=target_size,
        is_training=True
    )

    folds = create_segmentation_kfold_splits(image_dir, n_splits=n_splits)

    cv_results = {'fold_results': [], 'mean_metrics': {}, 'std_metrics': {}}
    all_dice, all_iou = [], []

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*50}\nFOLD {fold+1}/{n_splits}\n{'='*50}")

        train_ds = BRISCSegmentationDataset(image_dir, mask_dir, transform=train_transform,
                                            target_size=target_size, is_training=True)
        val_ds = BRISCSegmentationDataset(image_dir, mask_dir, transform=val_transform,
                                          target_size=target_size, is_training=False)

        # Manually set samples for this fold
        train_ds.samples = [full_dataset.samples[i] for i in train_idx]
        val_ds.samples = [full_dataset.samples[i] for i in val_idx]

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        model, criterion = create_segmentation_model(
            num_classes=1,
            pretrained=True,
            classifier_weights_path=classifier_weights_path
        )

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                              eta_min=learning_rate * 0.01)

        trainer = SegmentationTrainer(
            model=model, criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, device=device, use_amp=True
        )

        fold_history = trainer.train_fold(
            train_loader=train_loader, val_loader=val_loader,
            fold=fold, epochs=epochs, patience=patience, fast_dev_run=fast_dev_run
        )

        final_dice = fold_history['val_dice'][-1]
        final_iou = fold_history['val_iou'][-1]
        all_dice.append(final_dice)
        all_iou.append(final_iou)

        cv_results['fold_results'].append(fold_history)
        print(f"Fold {fold+1} Results — Dice: {final_dice:.4f}, IoU: {final_iou:.4f}")

    cv_results['mean_metrics'] = {'dice': np.mean(all_dice), 'iou': np.mean(all_iou)}
    cv_results['std_metrics'] = {'dice': np.std(all_dice), 'iou': np.std(all_iou)}

    print(f"\n{'='*50}")
    print("CROSS-VALIDATION RESULTS — BRAIN TUMOR SEGMENTATION")
    print(f"{'='*50}")
    print(f"Mean Dice: {cv_results['mean_metrics']['dice']:.4f} ± {cv_results['std_metrics']['dice']:.4f}")
    print(f"Mean IoU:  {cv_results['mean_metrics']['iou']:.4f} ± {cv_results['std_metrics']['iou']:.4f}")

    results_file = f"seg_cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"Results saved to {results_file}")

    return cv_results


if __name__ == "__main__":
    # Quick fast_dev_run test — point to your BRISC 2025 dataset
    results = train_segmentation_with_cross_validation(
        image_dir='dataset/segmentation/train/images',
        mask_dir='dataset/segmentation/train/masks',
        n_splits=2,
        batch_size=2,
        epochs=2,
        learning_rate=1e-4,
        num_workers=0,
        fast_dev_run=True,
        target_size=(256, 256),
        # Optional: transfer encoder weights from your trained classifier
        # classifier_weights_path='models/best_qwk_fold_0.pth'
    )
