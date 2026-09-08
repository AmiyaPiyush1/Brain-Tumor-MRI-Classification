"""
Grad-CAM visualization utilities for brain tumor MRI classification
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple, List
import os


class GradCAMVisualizer:
    """Grad-CAM visualization utility for brain tumor MRI analysis"""

    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.class_names = ['No_Tumor', 'Glioma', 'Meningioma', 'Pituitary']

    def generate_gradcam_heatmap(self,
                                  image: torch.Tensor,
                                  class_idx: Optional[int] = None,
                                  alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate Grad-CAM heatmap for a single MRI image"""
        self.model.eval()

        # Move to device
        image = image.to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Generate Grad-CAM
        gradcam = self.model.generate_gradcam(image, class_idx)

        # Get prediction
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1)
            confidence = probabilities[0, predicted_class].item()

        # Convert to numpy
        gradcam_np = gradcam.squeeze().cpu().numpy()
        image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)

        # Normalize image to [0, 1]
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        # Resize Grad-CAM to match image size
        gradcam_resized = cv2.resize(gradcam_np, (image_np.shape[1], image_np.shape[0]))

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Overlay heatmap on image
        overlay = alpha * heatmap + (1 - alpha) * image_np
        overlay = np.clip(overlay, 0, 1)

        return torch.tensor(overlay.transpose(2, 0, 1)), gradcam, predicted_class.item()

    def visualize_gradcam(self,
                           image: torch.Tensor,
                           class_idx: Optional[int] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (15, 5)) -> None:
        """Visualize Grad-CAM for a single MRI image"""
        overlay, gradcam, predicted_class = self.generate_gradcam_heatmap(image, class_idx)

        # Convert to numpy for visualization
        image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        gradcam_np = gradcam.squeeze().cpu().numpy()
        overlay_np = overlay.permute(1, 2, 0).numpy()

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original MRI
        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title('Original MRI Scan')
        axes[0].axis('off')

        # Grad-CAM heatmap
        im1 = axes[1].imshow(gradcam_np, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(overlay_np)
        axes[2].set_title(f'Overlay (Predicted: {self.class_names[predicted_class]})')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")

        plt.show()

    def visualize_multiple_classes(self,
                                    image: torch.Tensor,
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (20, 10)) -> None:
        """Visualize Grad-CAM for all tumor classes"""
        self.model.eval()

        # Move to device
        image = image.to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Get predictions
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()

        # Convert image to numpy
        image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        # Create visualization — 1 original + 4 class Grad-CAMs
        fig, axes = plt.subplots(1, 5, figsize=figsize)

        # Original image
        axes[0].imshow(image_np, cmap='gray')
        axes[0].set_title('Original MRI')
        axes[0].axis('off')

        # Grad-CAM for each class
        for i in range(4):
            gradcam = self.model.generate_gradcam(image, torch.tensor([i]))
            gradcam_np = gradcam.squeeze().cpu().numpy()

            im = axes[i + 1].imshow(gradcam_np, cmap='jet')
            axes[i + 1].set_title(f'{self.class_names[i]}\n(Prob: {probabilities[0, i]:.3f})')
            axes[i + 1].axis('off')
            plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-class Grad-CAM visualization saved to {save_path}")

        plt.show()

    def batch_visualize(self,
                         data_loader,
                         num_samples: int = 4,
                         save_dir: str = 'gradcam_results') -> None:
        """Visualize Grad-CAM for a batch of MRI images"""
        os.makedirs(save_dir, exist_ok=True)

        self.model.eval()
        sample_count = 0

        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= num_samples:
                    break

                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                for i in range(images.shape[0]):
                    if sample_count >= num_samples:
                        break

                    image = images[i:i + 1]
                    label = labels[i].item()

                    # Get prediction
                    output = self.model(image)
                    predicted = torch.argmax(output, dim=1).item()
                    confidence = F.softmax(output, dim=1)[0, predicted].item()

                    # Create filename
                    filename = f"sample_{sample_count}_true_{self.class_names[label]}_pred_{self.class_names[predicted]}_conf_{confidence:.3f}.png"
                    save_path = os.path.join(save_dir, filename)

                    # Visualize
                    self.visualize_gradcam(image, save_path=save_path)

                    sample_count += 1

        print(f"Generated {sample_count} Grad-CAM visualizations in {save_dir}")


def create_gradcam_visualizer(model, device: str = 'cuda') -> GradCAMVisualizer:
    """Create Grad-CAM visualizer instance"""
    return GradCAMVisualizer(model, device)
