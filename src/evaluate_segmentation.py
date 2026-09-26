"""
Evaluation utilities for brain tumor MRI segmentation.
Computes Dice Score, IoU, Precision, Recall and generates visualizations.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# Pixel-Level Metrics
# ─────────────────────────────────────────────

def compute_pixel_metrics(
        pred_mask: np.ndarray,
        true_mask: np.ndarray,
        smooth: float = 1e-6) -> Dict[str, float]:
    """
    Compute all segmentation metrics for a single prediction-mask pair.

    Args:
        pred_mask: Binary predicted mask (H, W) — values 0 or 1
        true_mask: Binary ground truth mask (H, W) — values 0 or 1

    Returns:
        Dict with dice, iou, precision, recall, f1, accuracy
    """
    pred = pred_mask.flatten().astype(float)
    true = true_mask.flatten().astype(float)

    tp = (pred * true).sum()
    fp = (pred * (1 - true)).sum()
    fn = ((1 - pred) * true).sum()
    tn = ((1 - pred) * (1 - true)).sum()

    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)

    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'tp': float(tp),
        'fp': float(fp),
        'fn': float(fn),
        'tn': float(tn)
    }


# ─────────────────────────────────────────────
# Dataset-Level Evaluator
# ─────────────────────────────────────────────

class SegmentationEvaluator:
    """
    Evaluates a segmentation model on a full dataset.
    Mirrors MedicalEvaluator from evaluate.py.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate_model(self,
                       model: torch.nn.Module,
                       loader: DataLoader,
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
        """
        Run full evaluation on a DataLoader.
        Returns per-sample metrics and aggregated statistics.
        """
        model.eval()
        all_metrics = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                images = batch['image'].to(device)
                masks = batch['mask'].cpu().numpy()

                preds = model(images)
                preds = torch.sigmoid(preds).cpu().numpy()
                pred_masks = (preds > self.threshold).astype(np.uint8)

                for i in range(len(images)):
                    metrics = compute_pixel_metrics(pred_masks[i, 0], masks[i, 0])
                    all_metrics.append(metrics)

        # Aggregate
        keys = ['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy']
        aggregated = {
            k: {
                'mean': float(np.mean([m[k] for m in all_metrics])),
                'std': float(np.std([m[k] for m in all_metrics])),
                'min': float(np.min([m[k] for m in all_metrics])),
                'max': float(np.max([m[k] for m in all_metrics]))
            }
            for k in keys
        }

        return {
            'per_sample': all_metrics,
            'aggregated': aggregated,
            'n_samples': len(all_metrics)
        }

    def print_evaluation_report(self, results: Dict):
        """Print a clean tabular segmentation evaluation report."""
        print("\n" + "="*60)
        print("SEGMENTATION EVALUATION REPORT — BRAIN TUMOR MRI")
        print("="*60)
        print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-"*60)

        metric_labels = {
            'dice': 'Dice Score',
            'iou': 'IoU (Jaccard)',
            'precision': 'Precision',
            'recall': 'Recall (Sensitivity)',
            'f1': 'F1 Score',
            'accuracy': 'Pixel Accuracy'
        }

        for key, label in metric_labels.items():
            m = results['aggregated'][key]
            print(f"{label:<20} {m['mean']:>10.4f} {m['std']:>10.4f} {m['min']:>10.4f} {m['max']:>10.4f}")

        print("="*60)
        print(f"Total samples evaluated: {results['n_samples']}")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def visualize_predictions(
        model: torch.nn.Module,
        dataset,
        indices: Optional[List[int]] = None,
        n_samples: int = 6,
        threshold: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_path: Optional[str] = None):
    """
    Visualize segmentation predictions as a grid:
    [Original MRI | Ground Truth Mask | Predicted Mask | Overlay]

    Args:
        model: Trained segmentation model
        dataset: BRISCSegmentationDataset instance
        indices: Specific indices to visualize (random if None)
        n_samples: Number of samples to show
        threshold: Binary threshold for prediction
        save_path: If provided, save figure to this path
    """
    model.eval()
    if indices is None:
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False).tolist()

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original MRI', 'Ground Truth Mask', 'Predicted Mask', 'Overlay']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        image_tensor = sample['image'].unsqueeze(0).to(device)
        true_mask = sample['mask'].squeeze().numpy()

        with torch.no_grad():
            pred_logit = model(image_tensor)
            pred_prob = torch.sigmoid(pred_logit).squeeze().cpu().numpy()
        pred_mask = (pred_prob > threshold).astype(np.uint8)

        # Denormalize image for display
        image_np = sample['image'].permute(1, 2, 0).numpy()
        image_np = (image_np * 0.5 + 0.5).clip(0, 1)
        image_gray = image_np[:, :, 0]  # Show single channel for MRI

        # Overlay: tumor region highlighted in red
        overlay = np.stack([image_gray, image_gray, image_gray], axis=-1)
        overlay[pred_mask == 1] = [1.0, 0.2, 0.2]   # Red = predicted tumor

        # Dice for this sample
        m = compute_pixel_metrics(pred_mask, true_mask)

        axes[row, 0].imshow(image_gray, cmap='gray')
        axes[row, 0].set_ylabel(f"Sample {idx}\nDice: {m['dice']:.3f}", fontsize=9)

        axes[row, 1].imshow(true_mask, cmap='Reds', vmin=0, vmax=1)

        axes[row, 2].imshow(pred_mask, cmap='Blues', vmin=0, vmax=1)

        axes[row, 3].imshow(overlay)

        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle('Brain Tumor Segmentation — BRISC 2025', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to: {save_path}")

    plt.show()
    return fig


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot segmentation training curves (Dice, IoU, Loss)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train Loss', color='steelblue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='tomato')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_dice'], label='Train Dice', color='steelblue')
    axes[1].plot(history['val_dice'], label='Val Dice', color='tomato')
    axes[1].set_title('Dice Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['val_iou'], label='Val IoU', color='seagreen')
    axes[2].set_title('IoU Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Segmentation Training History', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Training history saved to: {save_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    print("Testing SegmentationEvaluator with dummy data...")
    pred = np.random.randint(0, 2, (256, 256)).astype(np.uint8)
    true = np.random.randint(0, 2, (256, 256)).astype(np.uint8)
    metrics = compute_pixel_metrics(pred, true)
    print("Metrics:", {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, float)})
    print("✅ evaluate_segmentation.py test passed!")
