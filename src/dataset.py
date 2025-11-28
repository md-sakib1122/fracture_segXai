"""
FracAtlas for Fracture Segmentation
Handles:
- Mixed RGB/Grayscale images (converts all to RGB)
- Class imbalance (19.6% fractured vs 80.4% non-fractured)
- Medical image augmentations
"""

import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class FracAtlasDataset(Dataset):
    """
    FracAtlas for FracAtlas fracture segmentation
    Handles mixed RGB/grayscale X-rays and binary masks
    """

    def __init__(self, image_dir, mask_dir, transform=None, return_metadata=False):
        """
        Args:
            image_dir: Path to images folder
            mask_dir: Path to masks folder
            transform: Albumentations transforms
            return_metadata: If True, return image filename with data
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.return_metadata = return_metadata

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Map each image to its corresponding mask
        self.masks = [os.path.splitext(f)[0] + '_mask.png' for f in self.images]
        for mask in self.masks:
            assert os.path.exists(os.path.join(mask_dir, mask)), f"Missing mask: {mask}"

        print(f"FracAtlas initialized: {len(self.images)} image-mask pairs")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)

        # Handle grayscale images - convert to RGB
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # Already RGB or BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Binarize mask (0 or 1)
        mask = (mask > 127).astype(np.uint8)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            # Add channel dimension after transformation
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        else:
            # No transform - manually convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Convert to float
        mask = mask.float()

        if self.return_metadata:
            return image, mask, self.images[idx]
        return image, mask

    def get_class_weights(self):
        """
        Calculate class weights for handling imbalance
        Returns weights for [background, fracture]
        """
        total_pixels = 0
        fracture_pixels = 0

        print("Calculating class weights...")
        for idx in range(len(self)):
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)

            total_pixels += mask.size
            fracture_pixels += mask.sum()

        background_pixels = total_pixels - fracture_pixels

        # Inverse frequency weighting
        weight_background = total_pixels / (2 * background_pixels)
        weight_fracture = total_pixels / (2 * fracture_pixels)

        print(f"Background pixels: {background_pixels:,} ({background_pixels / total_pixels * 100:.2f}%)")
        print(f"Fracture pixels: {fracture_pixels:,} ({fracture_pixels / total_pixels * 100:.2f}%)")
        print(f"Class weights: background={weight_background:.4f}, fracture={weight_fracture:.4f}")

        return torch.tensor([weight_background, weight_fracture])

    def get_sample_weights(self):
        """
        Calculate per-sample weights for WeightedRandomSampler
        Returns higher weights for images with fractures
        """
        weights = []
        print("Calculating sample weights...")

        for idx in range(len(self)):
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)

            # Higher weight for fractured images
            has_fracture = mask.sum() > 0
            weight = 3.0 if has_fracture else 1.0  # 3x weight for fractured
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float)


def get_train_transforms(image_size=512):
    """
    Training augmentations for medical images
    Conservative augmentations to preserve medical validity
    """
    return A.Compose([
        # Resize
        A.Resize(image_size, image_size),

        # Geometric augmentations (mild)
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5
        ),


        # Intensity augmentations (important for X-rays)
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),

        # Normalize to [0, 1]
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),

        ToTensorV2()
    ])


def get_val_transforms(image_size=512):
    """
    Validation/Test transforms (no augmentation)
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def create_dataloaders(
        data_root,
        batch_size=8,
        image_size=512,
        num_workers=4,
        use_weighted_sampling=True
):
    """
    Create train, val, test dataloaders with proper handling of class imbalance

    Args:
        data_root: Root directory containing FracAtlas/
        batch_size: Batch size for training
        image_size: Image size (square)
        num_workers: Number of workers for data loading
        use_weighted_sampling: Use WeightedRandomSampler for training

    Returns:
        train_loader, val_loader, test_loader
    """

    # Paths
    train_img_dir = os.path.join(data_root, 'FracAtlas', 'train', 'images')
    train_mask_dir = os.path.join(data_root, 'FracAtlas', 'train', 'masks')
    val_img_dir = os.path.join(data_root, 'FracAtlas', 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'FracAtlas', 'val', 'masks')
    test_img_dir = os.path.join(data_root, 'FracAtlas', 'test', 'images')
    test_mask_dir = os.path.join(data_root, 'FracAtlas', 'test', 'masks')

    # Datasets
    train_dataset = FracAtlasDataset(
        train_img_dir,
        train_mask_dir,
        transform=get_train_transforms(image_size)
    )

    val_dataset = FracAtlasDataset(
        val_img_dir,
        val_mask_dir,
        transform=get_val_transforms(image_size)
    )

    test_dataset = FracAtlasDataset(
        test_img_dir,
        test_mask_dir,
        transform=get_val_transforms(image_size),
        return_metadata=True  # Return filenames for visualization
    )

    # Weighted sampler for handling class imbalance
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Don't shuffle when using sampler
    else:
        sampler = None
        shuffle = True

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # For inference, process one at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n✓ DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# Test the dataset
if __name__ == "__main__":
    # Test with your data
    data_root = "./.."  # Adjust path

    print("Testing dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=4,
        image_size=512,
        use_weighted_sampling=True
    )

    # Test loading a batch
    images, masks = next(iter(train_loader))
    print(f"\nBatch test:")
    print(f"  Images shape: {images.shape}")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Masks unique values: {masks.unique()}")

    # Check class distribution in batch
    fracture_ratio = masks.sum() / masks.numel()
    print(f"  Fracture pixel ratio in batch: {fracture_ratio:.4f}")