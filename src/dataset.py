"""
"""

import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# SIMPLE PATCH UTILITY (ADD THIS)
# ============================================================================

class PatchPredictor:
    """Extract patches, predict, combine results"""

    def __init__(self, patch_size=512, overlap=128):
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap

    def get_patches(self, image):
        """Extract patches from image with overlap"""
        h, w = image.shape[:2]
        patches = []
        positions = []

        # Extract patches with overlap

        # ===== IMPROVED: Pre-compute y positions deterministically =====
        ys = []
        y = 0
        while y < h:
            ys.append(y)
            if y + self.patch_size >= h:
                break  # Last patch reaches the end
            y += self.stride

        # ===== IMPROVED: Pre-compute x positions deterministically =====
        xs = []
        x = 0
        while x < w:
            xs.append(x)
            if x + self.patch_size >= w:
                break  # Last patch reaches the end
            x += self.stride

        # ===== IMPROVED: Extract patches at all positions =====
        for y in ys:
            for x in xs:
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)

                patch = image[y:y_end, x:x_end]

                # Pad if needed
                if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                    pad_h = self.patch_size - patch.shape[0]
                    pad_w = self.patch_size - patch.shape[1]
                    patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w,
                                               cv2.BORDER_REFLECT_101)

                patches.append(patch)
                positions.append((y, x))

        return patches, positions, (h, w)

    def combine_predictions(self, predictions, positions, original_shape):
        """Combine patch predictions using averaging"""
        h, w = original_shape
        output = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)

        for pred, (y, x) in zip(predictions, positions):
            # Crop prediction to fit original image
            pred_h, pred_w = pred.shape[:2]
            y_end = min(y + pred_h, h)
            x_end = min(x + pred_w, w)

            pred_crop = pred[:y_end - y, :x_end - x]

            # Add to output
            output[y:y_end, x:x_end] += pred_crop
            weights[y:y_end, x:x_end] += 1.0

        # Average
        output = output / (weights + 1e-6)
        return np.clip(output, 0, 1)


# ============================================================================
# ORIGINAL FRACATLAS DATASET (UNCHANGED)
# ============================================================================

class FracAtlasDataset(Dataset):
    """Enhanced FracAtlas dataset with improved sample weighting"""

    def __init__(self, image_dir, mask_dir, transform=None, return_metadata=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.return_metadata = return_metadata

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

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

        # Handle grayscale images
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        mask = mask.float()

        if self.return_metadata:
            return image, mask, self.images[idx]
        return image, mask

    def get_sample_weights_improved(self):
        """Size-based weighting"""
        weights = []
        fracture_counts = []

        print("Calculating improved sample weights...")

        for idx in range(len(self)):
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
            fracture_pixels = mask.sum()
            fracture_counts.append(fracture_pixels)

        fracture_counts = np.array(fracture_counts)

        has_fracture = fracture_counts > 0
        num_with = has_fracture.sum()
        num_without = len(fracture_counts) - num_with

        print(f"  With fractures: {num_with} ({num_with / len(fracture_counts) * 100:.1f}%)")
        print(f"  Without fractures: {num_without}")

        if num_with > 0:
            print(f"  Fracture size range: {fracture_counts[has_fracture].min():.0f} - "
                  f"{fracture_counts[has_fracture].max():.0f} pixels")
            print(f"  Mean fracture size: {fracture_counts[has_fracture].mean():.0f} pixels")

        for count in fracture_counts:
            if count == 0:
                weight = 0.5
            elif count < 500:
                weight = 8.0
            elif count < 2000:
                weight = 5.0
            elif count < 5000:
                weight = 3.0
            else:
                weight = 2.0
            weights.append(weight)

        weights = torch.tensor(weights, dtype=torch.float)

        print(f"\n  Weight distribution:")
        print(f"    No fracture (0.5):  {(weights == 0.5).sum().item():4d} samples")
        print(f"    Large (2.0):        {(weights == 2.0).sum().item():4d} samples")
        print(f"    Medium (3.0):       {(weights == 3.0).sum().item():4d} samples")
        print(f"    Small (5.0):        {(weights == 5.0).sum().item():4d} samples")
        print(f"    Tiny (8.0):         {(weights == 8.0).sum().item():4d} samples (highest priority)")

        return weights

def get_train_transforms(image_size=512):
    """Training augmentations"""
    return A.Compose([
        # ===== FIX: Preserve aspect ratio =====
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                     border_mode=cv2.BORDER_REFLECT_101),
        # ===== END FIX =====
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=0.5
        ),
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])


def get_val_transforms(image_size=512):
    """Validation transforms"""
    return A.Compose([
        # ===== FIX: Preserve aspect ratio =====
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                     border_mode=cv2.BORDER_REFLECT_101),
        # ===== END FIX =====
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])


def create_dataloaders(
        data_root,
        batch_size=8,
        image_size=512,
        num_workers=4,
        use_weighted_sampling=True,
        weighting_method='improved'
):
    """Create dataloaders with improved sample weighting"""

    train_img_dir = os.path.join(data_root, 'FracAtlas', 'train', 'images')
    train_mask_dir = os.path.join(data_root, 'FracAtlas', 'train', 'masks')
    val_img_dir = os.path.join(data_root, 'FracAtlas', 'val', 'images')
    val_mask_dir = os.path.join(data_root, 'FracAtlas', 'val', 'masks')
    test_img_dir = os.path.join(data_root, 'FracAtlas', 'test', 'images')
    test_mask_dir = os.path.join(data_root, 'FracAtlas', 'test', 'masks')

    train_dataset = FracAtlasDataset(
        train_img_dir, train_mask_dir,
        transform=get_train_transforms(image_size)
    )

    val_dataset = FracAtlasDataset(
        val_img_dir, val_mask_dir,
        transform=get_val_transforms(image_size)
    )

    test_dataset = FracAtlasDataset(
        test_img_dir, test_mask_dir,
        transform=get_val_transforms(image_size),
        return_metadata=True
    )

    if use_weighted_sampling:
        print(f"\nUsing weighted sampling with method: {weighting_method}")
        sample_weights = train_dataset.get_sample_weights_improved()

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"\n✓ DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# ============================================================================
# PATCH-BASED INFERENCE (ADD THIS FOR BEST ACCURACY)
# ============================================================================
def predict_with_patches(model, image_path, device, patch_size=512, overlap=128):
    """
    Correct patch-based inference:
    1. Extract patches from ORIGINAL resolution
    2. Preprocess EACH patch individually
    3. Predict and combine
    """
    # Load image at full resolution
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    print(f"Processing {h}×{w} image")

    # ✅ Extract patches at FULL resolution
    predictor = PatchPredictor(patch_size=patch_size, overlap=overlap)
    patches, positions, original_shape = predictor.get_patches(image)

    print(f"Extracted {len(patches)} patches")

    # ✅ Create preprocessing transform (same as validation, but for individual patches)
    preprocess = A.Compose([
        A.LongestMaxSize(max_size=patch_size),  # Applied to EACH patch
        A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                      border_mode=cv2.BORDER_REFLECT_101),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    predictions = []
    model.eval()

    with torch.no_grad():  # Consider changing to torch.inference_mode()
        for idx, patch in enumerate(patches):
            # ✅ Preprocess EACH patch (ensures consistent preprocessing)
            preprocessed_patch = preprocess(image=patch)['image']

            # Convert to tensor
            patch_tensor = torch.from_numpy(preprocessed_patch).permute(2, 0, 1)
            patch_tensor = patch_tensor.unsqueeze(0).to(device)

            # Predict
            output = model(patch_tensor)
            pred = torch.sigmoid(output).cpu().numpy().squeeze()
            predictions.append(pred)

            if (idx + 1) % max(1, len(patches) // 5) == 0:
                print(f"  Progress: {idx + 1}/{len(patches)} patches")

    # Combine predictions
    full_prediction = predictor.combine_predictions(predictions, positions, original_shape)

    print(f"✓ Prediction complete: {full_prediction.shape}")
    return full_prediction

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # 1. Create dataloaders (training - unchanged)
    data_root = "./.."
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=8,
        image_size=512,
        use_weighted_sampling=True,
        weighting_method='improved'
    )

    print("\n" + "=" * 70)
    print("TO USE PATCH-BASED INFERENCE AFTER TRAINING:")
    print("=" * 70)
    print("""
    from dataset import predict_with_patches
    import torch

    # Load your trained model
    model = load_your_model()
    device = torch.device('cuda')

    # Predict on image with patches (full resolution)
    prediction = predict_with_patches(model, 'test_image.jpg', device)

    # Or test with ground truth mask
    from dataset import test_patches_on_image
    pred, iou, f1 = test_patches_on_image(model, 'test_image.jpg', 'test_mask.png', device)
    """)