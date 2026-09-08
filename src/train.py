"""
Training pipeline for brain tumor MRI classification (GPU optimized)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import copy
from datetime import datetime

from model import create_model
from dataset import BrainTumorDataset, create_stratified_kfold_splits, get_transforms
from evaluate import MedicalEvaluator


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop"""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)  # type: ignore
            return True
        return False

    def save_checkpoint(self, model: nn.Module):
        """Save model weights with deep copy"""
        self.best_weights = copy.deepcopy(model.state_dict())


class ModelTrainer:
    """Comprehensive trainer for brain tumor MRI classification"""

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: str = 'models',
                 val_interval: int = 1,
                 use_amp: bool = True):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.val_interval = val_interval
        self.use_amp = use_amp and torch.cuda.is_available()

        os.makedirs(save_dir, exist_ok=True)
        self.evaluator = MedicalEvaluator()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_qwk': [],
            'val_f1': []
        }

        self.best_qwk = 0.0
        self.best_f1 = 0.0

    def train_epoch(self, train_loader: DataLoader, fast_dev_run: bool = False) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })

            if fast_dev_run and batch_idx >= 1:
                break

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate_epoch(self, val_loader: DataLoader, fast_dev_run: bool = False) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                if fast_dev_run and batch_idx >= 1:
                    break

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * (np.array(all_predictions) == np.array(all_labels)).mean()

        key_metrics = self.evaluator.get_key_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'qwk': key_metrics['quadratic_weighted_kappa'],
            'f1_weighted': key_metrics['f1_weighted'],
            'severe_sensitivity': key_metrics['severe_case_sensitivity']
        }

    def train_fold(self,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   fold: int,
                   epochs: int = 50,
                   patience: int = 10,
                   fast_dev_run: bool = False) -> Dict:
        """Train model for one fold"""
        early_stopping = EarlyStopping(patience=patience)

        fold_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_qwk': [],
            'val_f1': []
        }

        best_val_qwk = 0.0
        best_val_f1 = 0.0

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, fast_dev_run=fast_dev_run)

            if (epoch + 1) % self.val_interval == 0 or epoch == epochs - 1:
                val_metrics = self.validate_epoch(val_loader, fast_dev_run=fast_dev_run)

                if self.scheduler and hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['qwk'])
                    else:
                        self.scheduler.step()

                fold_history['val_loss'].append(val_metrics['loss'])
                fold_history['val_accuracy'].append(val_metrics['accuracy'])
                fold_history['val_qwk'].append(val_metrics['qwk'])
                fold_history['val_f1'].append(val_metrics['f1_weighted'])

                if val_metrics['qwk'] > best_val_qwk:
                    best_val_qwk = val_metrics['qwk']
                    self.save_model(f'best_qwk_fold_{fold}.pth')

                if val_metrics['f1_weighted'] > best_val_f1:
                    best_val_f1 = val_metrics['f1_weighted']
                    self.save_model(f'best_f1_fold_{fold}.pth')

                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val QWK: {val_metrics['qwk']:.4f}, "
                      f"Val F1: {val_metrics['f1_weighted']:.4f}")

                if early_stopping(val_metrics['loss'], self.model):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.2f}%")

            fold_history['train_loss'].append(train_metrics['loss'])
            fold_history['train_accuracy'].append(train_metrics['accuracy'])

            if fast_dev_run and epoch >= 1:
                break

        print(f"Fold {fold} completed. Best Val QWK: {best_val_qwk:.4f}, Best Val F1: {best_val_f1:.4f}")
        return fold_history

    def save_model(self, filename: str):
        """Save model state"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, filepath)

    def load_model(self, filename: str):
        """Load model state"""
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict') is not None:
            scheduler_state = checkpoint['scheduler_state_dict']
            if isinstance(scheduler_state, dict):
                self.scheduler.load_state_dict(scheduler_state)  # type: ignore
            else:
                print("Warning: Scheduler state dict is not a dictionary, skipping...")
        elif self.scheduler:
            print("Warning: No scheduler state dict found in checkpoint, skipping...")


def train_with_cross_validation(csv_file: str,
                                image_dir: str,
                                n_splits: int = 5,
                                batch_size: int = 16,
                                epochs: int = 50,
                                learning_rate: float = 1e-4,
                                weight_decay: float = 1e-4,
                                patience: int = 10,
                                num_workers: int = 4,
                                fast_dev_run: bool = False,
                                val_interval: int = 1,
                                use_attention: bool = False,
                                scheduler_type: str = 'cosine') -> Dict:
    """Train model with stratified k-fold cross-validation"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    folds = create_stratified_kfold_splits(csv_file, n_splits=n_splits)
    df = pd.read_csv(csv_file)
    train_transform, val_transform = get_transforms()

    cv_results = {
        'fold_results': [],
        'mean_metrics': {},
        'std_metrics': {}
    }

    all_val_qwk = []
    all_val_f1 = []
    all_val_accuracy = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"{'='*50}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = BrainTumorDataset(
            csv_file=None,
            image_dir=image_dir,
            transform=train_transform,
            is_training=True
        )
        train_dataset.data = train_df

        val_dataset = BrainTumorDataset(
            csv_file=None,
            image_dir=image_dir,
            transform=val_transform,
            is_training=False
        )
        val_dataset.data = val_df

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        class_counts = train_df['label'].value_counts().sort_index()
        class_weights = 1.0 / class_counts.values
        class_weights = torch.FloatTensor(class_weights).to(device)

        model, criterion = create_model(
            num_classes=4,
            dropout_rate=0.3,
            pretrained=True,
            use_attention=use_attention,
            use_focal_loss=True,
            class_weights=class_weights
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        if scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=learning_rate * 0.01
            )

        trainer = ModelTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,  # type: ignore
            device=device,
            val_interval=val_interval,
            use_amp=True
        )

        fold_history = trainer.train_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            fold=fold,
            epochs=epochs,
            patience=patience,
            fast_dev_run=fast_dev_run
        )

        cv_results['fold_results'].append(fold_history)

        final_val_qwk = fold_history['val_qwk'][-1]
        final_val_f1 = fold_history['val_f1'][-1]
        final_val_accuracy = fold_history['val_accuracy'][-1]

        all_val_qwk.append(final_val_qwk)
        all_val_f1.append(final_val_f1)
        all_val_accuracy.append(final_val_accuracy)

        print(f"Fold {fold + 1} Results:")
        print(f"  Val QWK: {final_val_qwk:.4f}")
        print(f"  Val F1: {final_val_f1:.4f}")
        print(f"  Val Accuracy: {final_val_accuracy:.4f}")

    cv_results['mean_metrics'] = {
        'val_qwk': np.mean(all_val_qwk),
        'val_f1': np.mean(all_val_f1),
        'val_accuracy': np.mean(all_val_accuracy)
    }

    cv_results['std_metrics'] = {
        'val_qwk': np.std(all_val_qwk),
        'val_f1': np.std(all_val_f1),
        'val_accuracy': np.std(all_val_accuracy)
    }

    print(f"\n{'='*50}")
    print("CROSS-VALIDATION RESULTS — BRAIN TUMOR MRI")
    print(f"{'='*50}")
    print(f"Mean Val QWK: {cv_results['mean_metrics']['val_qwk']:.4f} ± {cv_results['std_metrics']['val_qwk']:.4f}")
    print(f"Mean Val F1: {cv_results['mean_metrics']['val_f1']:.4f} ± {cv_results['std_metrics']['val_f1']:.4f}")
    print(f"Mean Val Accuracy: {cv_results['mean_metrics']['val_accuracy']:.4f} ± {cv_results['std_metrics']['val_accuracy']:.4f}")

    results_file = f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"Results saved to {results_file}")

    return cv_results


if __name__ == "__main__":
    results = train_with_cross_validation(
        csv_file='dataset/train.csv',
        image_dir='dataset/Training',
        n_splits=2,
        batch_size=4,
        epochs=2,
        learning_rate=1e-3,
        num_workers=0,
        fast_dev_run=True,
        val_interval=1,
        use_attention=False,
        scheduler_type='cosine'
    )
