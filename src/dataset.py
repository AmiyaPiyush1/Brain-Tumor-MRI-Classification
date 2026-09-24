"""
Dataset handling for brain tumor MRI classification
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional

from .mri_processor import MRIImageProcessor


class BrainTumorDataset(Dataset):
    """PyTorch dataset for brain tumor MRI classification"""

    CLASS_FOLDERS = {
        0: 'notumor',
        1: 'glioma',
        2: 'meningioma',
        3: 'pituitary'
    }

    def __init__(
        self,
        csv_file: Optional[str],
        image_dir: str,
        transform: Optional[A.Compose] = None,
        is_training: bool = True
    ):
        if csv_file is not None:
            self.data = pd.read_csv(csv_file)
        else:
            self.data = None
            
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training
        self.processor = MRIImageProcessor(target_size=(224, 224))

        if self.data is not None:
            self.class_counts = self.data['label'].value_counts().sort_index()
        else:
            self.class_counts = None

    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx: int) -> dict:
        if self.data is None:
            raise ValueError("Dataset data is not loaded")

        row = self.data.iloc[idx]
        label = int(row['label'])

        # Explicitly raise KeyError if an unexpected integer label appears
        if label not in self.CLASS_FOLDERS:
            raise ValueError(f"Invalid label '{label}' at index {idx}. Expected one of {list(self.CLASS_FOLDERS.keys())}")

        class_folder = self.CLASS_FOLDERS[label]
        base_path = os.path.join(self.image_dir, class_folder, f"{row['id_code']}").replace('\\', '/')

        # Find valid extension (.jpg, .png, .jpeg)
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            possible_path = f"{base_path}{ext}"
            if os.path.exists(possible_path):
                image_path = possible_path
                break

        if image_path is None:
            raise FileNotFoundError(f"Image for ID code '{row['id_code']}' not found in '{os.path.join(self.image_dir, class_folder)}'")

        # Load and preprocess image
        image = self.processor.preprocess_mri_image(image_path)

        # MRI preprocessor returns grayscale — stack to 3 channels for EfficientNet
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)

        if self.transform:
            image_uint8 = (image * 255).astype(np.uint8)
            transformed = self.transform(image=image_uint8)
            image = transformed['image']
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': image_path
        }

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Create weighted sampler to handle class imbalance"""
        if self.data is None:
            raise ValueError("Dataset data is not loaded")
        labels = self.data['label'].values
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        sample_weights = [float(class_weights[label]) for label in labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


def get_transforms() -> Tuple[A.Compose, A.Compose]:
    """Get lightweight transforms for 3-channel RGB image inputs"""
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    return train_transform, val_transform
