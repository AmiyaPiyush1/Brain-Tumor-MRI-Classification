"""
MRI Image Preprocessing Pipeline for Brain Tumor Classification
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional


class MRIImageProcessor:
    """Specialized image processor for brain MRI images"""

    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 enhance_edges: bool = True,
                 apply_gaussian_blur: bool = False):
        self.target_size = target_size
        self.enhance_edges = enhance_edges
        self.apply_gaussian_blur = apply_gaussian_blur

    def preprocess_mri_image(self, image_path: str) -> np.ndarray:
        """Complete preprocessing pipeline for brain MRI images"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale — MRI scans are grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply CLAHE for adaptive contrast enhancement (critical for MRI)
        enhanced = self._apply_clahe(gray)

        if self.apply_gaussian_blur:
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Enhance tumor boundaries using edge detection
        if self.enhance_edges:
            enhanced = self._enhance_edges(enhanced)

        # Crop to focus region (removes black background borders)
        enhanced = self._skull_strip_crop(enhanced)

        # Resize and normalize
        resized = cv2.resize(enhanced, self.target_size)
        normalized = resized.astype(np.float32) / 255.0

        return normalized

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance tumor boundaries using morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def _skull_strip_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply circular crop to remove background borders in MRI scans"""
        h, w = image.shape
        center = (w // 2, h // 2)
        radius = min(h, w) // 2 - 10

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        return cv2.bitwise_and(image, image, mask=mask)

    def batch_preprocess(self, image_paths: list, output_dir: str, save_format: str = "npy"):
        """Batch preprocess a list of MRI images"""
        os.makedirs(output_dir, exist_ok=True)

        for i, image_path in enumerate(image_paths):
            try:
                processed = self.preprocess_mri_image(image_path)
                filename = os.path.basename(image_path).split('.')[0]

                if save_format.lower() == "npy":
                    output_path = os.path.join(output_dir, f"{filename}_processed.npy")
                    np.save(output_path, processed)
                elif save_format.lower() == "png":
                    output_path = os.path.join(output_dir, f"{filename}_processed.png")
                    processed_uint8 = (processed * 255).astype(np.uint8)
                    cv2.imwrite(output_path, processed_uint8)
                else:
                    raise ValueError(f"Unsupported save format: {save_format}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
