"""
U-Net segmentation model with EfficientNet-B4 encoder for brain tumor segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from typing import List, Optional


class ConvBNReLU(nn.Module):
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
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(in_channels // 2 + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UNetWithEfficientNet(nn.Module):
    """
    U-Net segmentation model with EfficientNet-B4 encoder.
    Supports optional weight transfer from a trained classification model.
    """

    ENCODER_CHANNELS = [24, 32, 56, 160, 1792]

    def __init__(self, num_classes: int = 1, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = (EfficientNet.from_pretrained('efficientnet-b4') if pretrained
                        else EfficientNet.from_name('efficientnet-b4'))

        self.bridge  = ConvBNReLU(self.ENCODER_CHANNELS[-1], 512)
        self.decoder4 = DecoderBlock(512, self.ENCODER_CHANNELS[3], 256)
        self.decoder3 = DecoderBlock(256, self.ENCODER_CHANNELS[2], 128)
        self.decoder2 = DecoderBlock(128, self.ENCODER_CHANNELS[1], 64)
        self.decoder1 = DecoderBlock(64,  self.ENCODER_CHANNELS[0], 32)

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(dropout_rate)
        self.segmentation_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def _extract_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(x)))
        for idx, block in enumerate(self.encoder._blocks):
            x = block(x)
            if idx in [1, 5, 9, 21]:
                features.append(x)
        x = self.encoder._swish(self.encoder._bn1(self.encoder._conv_head(x)))
        features.append(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        skip1, skip2, skip3, skip4, bottleneck = self._extract_encoder_features(x)

        x = self.bridge(bottleneck)
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.final_upsample(x)
        x = self.segmentation_head(self.dropout(x))

        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x


class DiceLoss(nn.Module):
    """Dice Loss — better than BCE alone for small, imbalanced tumor regions."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = torch.sigmoid(predictions).view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)


class CombinedSegLoss(nn.Module):
    """Dice + BCE combined loss. Dice handles spatial overlap, BCE handles pixel accuracy."""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        return self.dice_weight * self.dice_loss(predictions, targets) + \
               self.bce_weight * self.bce_loss(predictions, targets)


def create_segmentation_model(
    num_classes: int = 1,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
    classifier_weights_path: Optional[str] = None
):
    """
    Create U-Net segmentation model.

    Args:
        classifier_weights_path: Path to trained classifier .pth — encoder weights are transferred if provided.
    Returns:
        model, criterion
    """
    model = UNetWithEfficientNet(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)

    if classifier_weights_path is not None:
        try:
            checkpoint = torch.load(classifier_weights_path, map_location='cpu')
            encoder_state = {
                k.replace('_model.', ''): v
                for k, v in checkpoint['model_state_dict'].items()
                if '_model.' in k
            }
            model.encoder.load_state_dict(encoder_state, strict=False)
            print(f"✅ Transferred encoder weights from: {classifier_weights_path}")
        except Exception as e:
            print(f"⚠️  Could not transfer classifier weights: {e}. Using ImageNet pretrained instead.")

    criterion = CombinedSegLoss(dice_weight=dice_weight, bce_weight=bce_weight)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Segmentation Model | Total params: {total:,} | Trainable: {trainable:,}")

    return model, criterion


if __name__ == "__main__":
    model, criterion = create_segmentation_model(num_classes=1, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 256, 256)
    dummy_mask  = torch.randint(0, 2, (2, 1, 256, 256)).float()
    with torch.no_grad():
        output = model(dummy_input)
        loss = criterion(output, dummy_mask)
    print(f"Input:  {dummy_input.shape} | Output: {output.shape} | Loss: {loss.item():.4f}")
    print("✅ Segmentation model test passed!")
