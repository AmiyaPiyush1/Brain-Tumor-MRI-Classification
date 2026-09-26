"""
Dataset loader for BRISC 2025 brain tumor segmentation.

Expected folder structure:
    segmentation/
    ├── train/
    │   ├── images/    ← .jpg MRI scans
    │   └── masks/     ← .png binary tumor masks (same filename)
    └── test/
        ├── images/
        └── masks/
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Tuple, Dict
from sklearn.model_selection import KFold


class BRISCSegmentationDataset(Dataset):
    """BRISC 2025 paired image + binary mask dataset."""

    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 transform: Optional[A.Compose] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 is_training: bool = True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.is_training = is_training

        self.samples = []
        for fname in sorted(os.listdir(image_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                stem = os.path.splitext(fname)[0]
                mask_path = os.path.join(mask_dir, stem + '.png')
                image_path = os.path.join(image_dir, fname)
                if os.path.exists(mask_path):
                    self.samples.append({'image_path': image_path, 'mask_path': mask_path})

        print(f"{'Train' if is_training else 'Val/Test'} SegDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        image = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot load image: {sample['image_path']}")
        image = cv2.resize(image, self.target_size)
        image = np.stack([image, image, image], axis=-1)

        mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot load mask: {sample['mask_path']}")
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        mask = mask.clone().detach().long().unsqueeze(0).float()

        return {'image': image, 'mask': mask, 'image_path': sample['image_path']}


def get_segmentation_transforms(target_size: Tuple[int, int] = (256, 256)):
    """Returns (train_transform, val_transform). Spatial transforms are applied to both image and mask."""
    train_transform = A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    return train_transform, val_transform


def create_segmentation_kfold_splits(image_dir: str, n_splits: int = 5):
    """Returns list of (train_indices, val_indices) tuples for k-fold cross-validation."""
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(kf.split(range(len(image_files))))


def generate_segmentation_csv(image_dir: str, mask_dir: str, output_csv: str) -> pd.DataFrame:
    """Generate CSV mapping image paths to mask paths. Columns: image_path, mask_path."""
    records = []
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            stem = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_dir, stem + '.png')
            if os.path.exists(mask_path):
                records.append({'image_path': os.path.join(image_dir, fname), 'mask_path': mask_path})

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Segmentation CSV saved: {output_csv} ({len(df)} pairs)")
    return df


if __name__ == "__main__":
    import tempfile, shutil

    tmp = tempfile.mkdtemp()
    img_dir = os.path.join(tmp, 'images')
    msk_dir = os.path.join(tmp, 'masks')
    os.makedirs(img_dir)
    os.makedirs(msk_dir)

    for i in range(5):
        cv2.imwrite(os.path.join(img_dir, f'sample_{i}.jpg'),
                    np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        cv2.imwrite(os.path.join(msk_dir, f'sample_{i}.png'),
                    np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255)

    train_t, _ = get_segmentation_transforms()
    ds = BRISCSegmentationDataset(img_dir, msk_dir, transform=train_t)
    sample = ds[0]
    print(f"Image: {sample['image'].shape} | Mask: {sample['mask'].shape}")
    print("✅ BRISCSegmentationDataset test passed!")
    shutil.rmtree(tmp)
