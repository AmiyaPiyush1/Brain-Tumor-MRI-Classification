"""
Medical evaluation metrics for brain tumor MRI classification
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    cohen_kappa_score
)
from typing import Dict, List, Tuple, Optional
import seaborn as sns


class MedicalEvaluator:
    """Comprehensive medical evaluation for brain tumor MRI classification"""

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or [
            'No Tumor', 'Glioma', 'Meningioma', 'Pituitary'
        ]

    def quadratic_weighted_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Quadratic Weighted Kappa — standard metric for ordinal classification"""
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')

    def calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate specificity for each class"""
        cm = confusion_matrix(y_true, y_pred)
        specificity = []

        num_classes = len(self.class_names) if self.class_names is not None else cm.shape[0]
        for i in range(num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity.append(spec)

        return np.array(specificity)

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict:
        """Comprehensive model evaluation with medical metrics"""
        qwk = self.quadratic_weighted_kappa(y_true, y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division='warn')
        recall = recall_score(y_true, y_pred, average='weighted', zero_division='warn')
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division='warn')

        precision_per_class = precision_score(y_true, y_pred, average='macro', zero_division='warn')  # type: ignore
        recall_per_class = recall_score(y_true, y_pred, average='macro', zero_division='warn')  # type: ignore
        f1_per_class = f1_score(y_true, y_pred, average='macro', zero_division='warn')  # type: ignore

        sensitivity = recall_per_class  # type: ignore
        specificity = self.calculate_specificity(y_true, y_pred)

        auc_scores = None
        if y_proba is not None:
            try:
                auc_scores = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            except Exception:
                auc_scores = None

        cm = confusion_matrix(y_true, y_pred)

        # Tumor detection sensitivity: any tumor class (labels 1, 2, 3)
        tumor_mask = (y_true >= 1)
        if tumor_mask.sum() > 0:
            tumor_sensitivity = recall_score(y_true[tumor_mask], y_pred[tumor_mask], average='macro', zero_division='warn')
        else:
            tumor_sensitivity = 0.0

        results = {
            'quadratic_weighted_kappa': qwk,
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'severe_case_sensitivity': tumor_sensitivity,  # kept for train.py compatibility
            'confusion_matrix': cm,
            'auc_scores': auc_scores
        }

        return results

    def get_key_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict:
        """Get only the essential metrics for practical use"""
        qwk = self.quadratic_weighted_kappa(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division='warn')

        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division='warn')  # type: ignore

        # Tumor detection sensitivity: any tumor class (labels 1, 2, 3)
        tumor_mask = (y_true >= 1)
        if tumor_mask.sum() > 0:
            tumor_sensitivity = recall_score(y_true[tumor_mask], y_pred[tumor_mask], average='macro', zero_division='warn')
        else:
            tumor_sensitivity = 0.0

        return {
            'quadratic_weighted_kappa': qwk,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'sensitivity_per_class': sensitivity,
            'severe_case_sensitivity': tumor_sensitivity  # kept for train.py compatibility
        }

    def print_evaluation_report(self, results: Dict):
        """Print evaluation report with key metrics"""
        print("=" * 50)
        print("MEDICAL EVALUATION REPORT — BRAIN TUMOR MRI")
        print("=" * 50)

        print(f"Quadratic Weighted Kappa (QWK): {results['quadratic_weighted_kappa']:.4f}")
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"Weighted F1-Score: {results['f1_weighted']:.4f}")
        print(f"Tumor Detection Sensitivity: {results['severe_case_sensitivity']:.4f}")

        print("\nPer-Class Sensitivity:")
        if self.class_names is not None and 'recall_per_class' in results and results['recall_per_class'] is not None:
            for i, class_name in enumerate(self.class_names):
                print(f"  {class_name}: {results['recall_per_class'][i]:.4f}")  # type: ignore
        else:
            print("  Per-class sensitivity data not available")

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None, show: bool = True):
        """Plot confusion matrix with medical styling"""
        plt.figure(figsize=(10, 8))

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count'}
        )

        plt.title('Confusion Matrix — Brain Tumor MRI Classification', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_metrics_comparison(self, results: Dict, save_path: Optional[str] = None, show: bool = True):
        """Plot per-class metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        if self.class_names is not None and all(key in results for key in ['precision_per_class', 'recall_per_class', 'specificity', 'f1_per_class']):
            axes[0, 0].bar(self.class_names, results['precision_per_class'])  # type: ignore
            axes[0, 0].set_title('Precision per Class')
            axes[0, 0].set_ylabel('Precision')
            axes[0, 0].tick_params(axis='x', rotation=45)

            axes[0, 1].bar(self.class_names, results['recall_per_class'])  # type: ignore
            axes[0, 1].set_title('Recall (Sensitivity) per Class')
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].tick_params(axis='x', rotation=45)

            axes[1, 0].bar(self.class_names, results['specificity'])  # type: ignore
            axes[1, 0].set_title('Specificity per Class')
            axes[1, 0].set_ylabel('Specificity')
            axes[1, 0].tick_params(axis='x', rotation=45)

            axes[1, 1].bar(self.class_names, results['f1_per_class'])  # type: ignore
            axes[1, 1].set_title('F1-Score per Class')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            print("Warning: Per-class metrics data not available for plotting")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()


class GradCAM:
    """Grad-CAM implementation for model interpretability"""

    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM visualization"""
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0].cpu().data.numpy()  # type: ignore
        activations = self.activations[0].cpu().data.numpy()  # type: ignore

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam

        return cam

    def visualize_gradcam(self, input_tensor: torch.Tensor, class_idx: int,
                          original_image: Optional[np.ndarray] = None, save_path: Optional[str] = None):
        """Visualize Grad-CAM with original MRI image"""
        cam = self.generate_cam(input_tensor, class_idx)

        if original_image is not None:
            cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if original_image is not None:
            axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title('Original MRI')
            axes[0].axis('off')

        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title(f'Grad-CAM (Class {class_idx})')
        axes[1].axis('off')

        if original_image is not None:
            overlay = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # type: ignore
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            blended = cv2.addWeighted(original_image, 0.6, overlay, 0.4, 0)

            axes[2].imshow(blended)
            axes[2].set_title('Grad-CAM Overlay')
            axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return cam


if __name__ == "__main__":
    y_true = np.random.randint(0, 4, 100)
    y_pred = np.random.randint(0, 4, 100)
    y_proba = np.random.rand(100, 4)

    evaluator = MedicalEvaluator()

    key_results = evaluator.get_key_metrics(y_true, y_pred, y_proba)
    print("Key Metrics Test:")
    print(f"QWK: {key_results['quadratic_weighted_kappa']:.4f}")
    print(f"Accuracy: {key_results['accuracy']:.4f}")
    print(f"F1-Weighted: {key_results['f1_weighted']:.4f}")

    results = evaluator.evaluate_model(y_true, y_pred, y_proba)
    evaluator.print_evaluation_report(results)
