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

        # Map integer label to folder name
        class_folders = {
            0: 'no_tumor',
            1: 'glioma',
            2: 'meningioma',
            3: 'pituitary'
        }

        label = int(row['label'])
        class_folder = class_folders.get(label, 'no_tumor')
        image_path = os.path.join(self.image_dir, class_folder, f"{row['id_code']}.jpg").replace('\\', '/')

        try:
            if not os.path.exists(image_path):
                alt_paths = [
                    image_path.replace('.jpg', '.png'),
                    image_path.replace('.jpg', '.jpeg')
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
                else:
                    return self.__getitem__((idx + 1) % len(self.data))

            image = self.processor.preprocess_mri_image(image_path)
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.data))

        # MRI preprocessor returns grayscale — stack to 3 channels for EfficientNet
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)

        if self.transform:
            image_uint8 = (image * 255).astype(np.uint8)
            transformed = self.transform(image=image_uint8)
            image = transformed['image']
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0

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
    """Get lightweight transforms for fast training"""
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    return train_transform, val_transform


def create_data_loaders(
    csv_file: str,
    image_dir: str,
    batch_size: int = 2,
    num_workers: int = 2,
    train_split: float = 0.8,
    random_state: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    df = pd.read_csv(csv_file)

    train_df, temp_df = train_test_split(
        df,
        test_size=1 - train_split,
        random_state=random_state,
        stratify=df['label']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_df['label']
    )

    train_transform, val_transform = get_transforms()

    train_dataset = BrainTumorDataset(csv_file=None, image_dir=image_dir, transform=train_transform)
    train_dataset.data = train_df

    val_dataset = BrainTumorDataset(csv_file=None, image_dir=image_dir, transform=val_transform)
    val_dataset.data = val_df

    test_dataset = BrainTumorDataset(csv_file=None, image_dir=image_dir, transform=val_transform)
    test_dataset.data = test_df

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_dataset.get_weighted_sampler(),
                              num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def create_stratified_kfold_splits(csv_file: str, n_splits: int = 5, random_state: int = 42) -> list:
    """Create stratified k-fold splits for cross-validation"""
    df = pd.read_csv(csv_file)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits = []
    for train_idx, val_idx in skf.split(df, df['label']):
        splits.append((train_idx, val_idx))

    return splits
