"""
EfficientNet-B4 model for brain tumor MRI classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from typing import Optional, Tuple


class BrainTumorClassifier(nn.Module):
    """EfficientNet-B4 based classifier for brain tumor detection"""

    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3, pretrained: bool = True, use_attention: bool = False):
        super(BrainTumorClassifier, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention

        if pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b4')

        self.feature_dim = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()  # type: ignore

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim // 4, self.feature_dim),
                nn.Sigmoid()
            )
        else:
            self.attention = None

        if use_attention:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate * 0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.feature_dim, num_classes)
            )

        # Grad-CAM hooks will be registered when needed
        self.gradcam_activations = None
        self.gradcam_gradients = None
        self.gradcam_layer = None

    def _register_gradcam_hook(self):
        """Register Grad-CAM hook for interpretability"""
        self.gradcam_activations = None
        self.gradcam_gradients = None
        self.gradcam_layer = None

        def forward_hook(module, input, output):
            self.gradcam_activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradcam_gradients = grad_output[0]

        # Find the last convolutional layer for Grad-CAM
        last_conv_layer = None
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module

        if last_conv_layer is not None:
            self.gradcam_layer = last_conv_layer
            last_conv_layer.register_forward_hook(forward_hook)
            last_conv_layer.register_full_backward_hook(backward_hook)

    def _count_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)

        if self.use_attention and self.attention is not None:
            attention_weights = self.attention(features)
            features = features * attention_weights

        output = self.classifier(features)
        return output

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get attention weights for visualization"""
        if not self.use_attention or self.attention is None:
            return None

        with torch.no_grad():
            features = self.backbone(x)
            attention_weights = self.attention(features)
            return attention_weights

    def get_feature_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get intermediate feature maps for Grad-CAM"""
        if self.gradcam_layer is None:
            self._register_gradcam_hook()

        with torch.no_grad():
            _ = self.backbone(x)
        return self.gradcam_activations

    def generate_gradcam(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        """Generate Grad-CAM heatmap for input MRI image"""
        if self.gradcam_layer is None:
            self._register_gradcam_hook()

        if self.gradcam_layer is None:
            raise ValueError("Grad-CAM layer not found. Make sure model is properly initialized.")

        # Forward pass
        output = self.forward(x)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)

        # Zero gradients
        self.zero_grad()

        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, class_idx.unsqueeze(1), 1.0)
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradcam_gradients
        activations = self.gradcam_activations

        if gradients is None or activations is None:
            raise ValueError("Grad-CAM hooks not properly registered")

        # Compute Grad-CAM
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
        gradcam = F.relu(gradcam)

        # Normalize to [0, 1]
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)

        return gradcam


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in medical data"""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_model(num_classes: int = 4,
                 dropout_rate: float = 0.3,
                 pretrained: bool = True,
                 use_attention: bool = False,
                 use_focal_loss: bool = True,
                 class_weights: Optional[torch.Tensor] = None) -> Tuple[nn.Module, nn.Module]:
    """Create model and loss function"""
    model = BrainTumorClassifier(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        use_attention=use_attention
    )

    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    return model, criterion


if __name__ == "__main__":
    input_tensor = torch.randn(2, 3, 512, 512)
    targets = torch.randint(0, 4, (2,))

    model, criterion = create_model(pretrained=False, use_attention=False)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        loss = criterion(output, targets)

        print(f"Output shape: {output.shape}")
        print(f"Loss: {loss.item():.4f}")
        print(f"Parameters: {model._count_parameters():,}")  # type: ignore
