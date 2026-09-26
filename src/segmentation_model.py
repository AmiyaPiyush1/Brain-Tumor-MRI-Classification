"""
U-Net segmentation model with EfficientNet-B4 encoder for brain tumor segmentation.
Reuses the EfficientNet-B4 backbone from the classification model as the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from typing import List, Optional


# ─────────────────────────────────────────────
# U-Net Decoder Blocks
# ─────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample + skip connection + two ConvBNReLU layers"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle size mismatch from skip connection
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ─────────────────────────────────────────────
# Main Segmentation Model
# ─────────────────────────────────────────────

class UNetWithEfficientNet(nn.Module):
    """
    U-Net segmentation model with EfficientNet-B4 encoder.
    The encoder can be initialized with weights from the classification model,
    enabling transfer learning between the two tasks.
    """

    # EfficientNet-B4 intermediate feature channels (at reduction steps)
    ENCODER_CHANNELS = [24, 32, 56, 160, 1792]

    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout_rate: float = 0.3):
        """
        Args:
            num_classes: 1 for binary (tumor vs background), 4 for multi-class
            pretrained: Whether to load ImageNet pretrained EfficientNet-B4 weights
            dropout_rate: Dropout before final prediction head
        """
        super().__init__()
        self.num_classes = num_classes

        # ── Encoder: EfficientNet-B4 backbone ──────────────────────────
        if pretrained:
            self.encoder = EfficientNet.from_pretrained('efficientnet-b4')
        else:
            self.encoder = EfficientNet.from_name('efficientnet-b4')

        # ── Bridge (bottleneck) ─────────────────────────────────────────
        self.bridge = ConvBNReLU(self.ENCODER_CHANNELS[-1], 512)

        # ── Decoder ─────────────────────────────────────────────────────
        self.decoder4 = DecoderBlock(512, self.ENCODER_CHANNELS[3], 256)
        self.decoder3 = DecoderBlock(256, self.ENCODER_CHANNELS[2], 128)
        self.decoder2 = DecoderBlock(128, self.ENCODER_CHANNELS[1], 64)
        self.decoder1 = DecoderBlock(64,  self.ENCODER_CHANNELS[0], 32)

        # ── Final upsample + prediction head ───────────────────────────
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(dropout_rate)
        self.segmentation_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def _extract_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract intermediate feature maps from EfficientNet-B4 encoder.
        Returns skip connection tensors at multiple scales.
        """
        features = []
        # Pass through EfficientNet blocks and collect skip connections
        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(x)))

        for idx, block in enumerate(self.encoder._blocks):
            x = block(x)
            # Collect features at specific reduction points
            if idx in [3, 5, 9, 21]:   # EfficientNet-B4 skip connection indices
                features.append(x)

        x = self.encoder._swish(self.encoder._bn1(self.encoder._conv_head(x)))
        features.append(x)  # bottleneck features

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder forward pass + skip connections
        skips = self._extract_encoder_features(x)
        skip1, skip2, skip3, skip4, bottleneck = skips

        # Bridge
        x = self.bridge(bottleneck)

        # Decoder
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        # Final upsample to input resolution
        x = self.final_upsample(x)
        x = self.dropout(x)
        x = self.segmentation_head(x)

        # Ensure output matches input spatial resolution exactly
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


# ─────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Better than BCE alone for imbalanced masks (small tumor regions).
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class CombinedSegLoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy loss.
    Dice handles spatial overlap, BCE handles pixel-level accuracy.
    """
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        return self.dice_weight * dice + self.bce_weight * bce


# ─────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────

def create_segmentation_model(
    num_classes: int = 1,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
    classifier_weights_path: Optional[str] = None
):
    """
    Create U-Net segmentation model with optional transfer from classifier.

    Args:
        num_classes: 1 for binary tumor mask
        pretrained: Load ImageNet pretrained EfficientNet-B4 encoder
        dropout_rate: Dropout before prediction head
        dice_weight: Weight for Dice component of combined loss
        bce_weight: Weight for BCE component of combined loss
        classifier_weights_path: Optional path to your trained classifier .pth file.
                                  If provided, encoder weights are transferred from it.

    Returns:
        model, criterion
    """
    model = UNetWithEfficientNet(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )

    # Optional: transfer encoder weights from trained classifier
    if classifier_weights_path is not None:
        try:
            checkpoint = torch.load(classifier_weights_path, map_location='cpu')
            classifier_state = checkpoint['model_state_dict']
            # Only load keys that match the encoder (prefix: '_model.')
            encoder_state = {
                k.replace('_model.', ''): v
                for k, v in classifier_state.items()
                if '_model.' in k
            }
            model.encoder.load_state_dict(encoder_state, strict=False)
            print(f"✅ Transferred encoder weights from classifier: {classifier_weights_path}")
        except Exception as e:
            print(f"⚠️  Could not transfer classifier weights: {e}. Using ImageNet pretrained instead.")

    criterion = CombinedSegLoss(dice_weight=dice_weight, bce_weight=bce_weight)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Segmentation Model | Total params: {total_params:,} | Trainable: {trainable_params:,}")

    return model, criterion


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing UNetWithEfficientNet...")
    model, criterion = create_segmentation_model(num_classes=1, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    dummy_mask = torch.randint(0, 2, (2, 1, 256, 256)).float()

    with torch.no_grad():
        output = model(dummy_input)
        loss = criterion(output, dummy_mask)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Loss value:   {loss.item():.4f}")
    print("✅ Segmentation model test passed!")
