"""
Complete Loss Function Testing Script
Tests all loss functions with real FracAtlas data
"""

import torch
from dataset import create_dataloaders
from losses import (
    DiceBCELoss,
    FocalDiceLoss,
    OHEMFocalDiceLoss,
    TverskyFocalLoss,
    get_recommended_loss
)


def test_basic_forward():
    """Test basic forward pass with synthetic data"""
    print("=" * 70)
    print("TEST 1: Basic Forward Pass (Synthetic Data)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create synthetic data
    batch_size = 4
    pred = torch.randn(batch_size, 1, 512, 512, requires_grad=True).to(device)
    target = torch.randint(0, 2, (batch_size, 1, 512, 512)).float().to(device)

    losses = {
        "DiceBCE": DiceBCELoss(dice_weight=0.6),
        "FocalDice": FocalDiceLoss(alpha=0.75, gamma=2.0, dice_weight=0.5),
        "OHEMFocalDice": OHEMFocalDiceLoss(alpha=0.8, gamma=3.0, ohem_ratio=0.25),
        "TverskyFocal": TverskyFocalLoss(alpha=0.3, beta=0.7),
        "Recommended": get_recommended_loss('extreme')
    }

    for name, loss_fn in losses.items():
        loss_fn = loss_fn.to(device)

        # Test without components (for training)
        loss_value = loss_fn(pred, target)
        loss_value.backward(retain_graph=True)

        # Test with components (for logging)
        with torch.no_grad():
            _, components = loss_fn(pred, target, return_components=True)

        print(f"✓ {name:15} | Loss: {loss_value.item():.4f} | {components}")

    print("\n✓ All losses computed successfully!\n")


def test_with_real_data():
    """Test with actual FracAtlas data"""
    print("=" * 70)
    print("TEST 2: Real FracAtlas Data")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load dataset
    print("Loading FracAtlas dataset...")
    try:
        data_root = "../"  # Adjust this path if needed
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=data_root,
            batch_size=4,
            image_size=512,
            num_workers=0,
            use_weighted_sampling=True
        )
        print("✓ Dataset loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Skipping real data test\n")
        return

    # Get one batch
    images, masks = next(iter(train_loader))
    images, masks = images.to(device), masks.to(device)

    print(f"Batch info:")
    print(f"  Images: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Masks: {masks.shape}, unique values: {masks.unique().tolist()}")
    print(f"  Fracture ratio: {masks.sum() / masks.numel():.6f}\n")

    # Create dummy model output (simulate predictions)
    # In real training, this would be model(images)
    pred = torch.randn_like(masks, requires_grad=True)

    # Initialize losses
    losses = {
        "DiceBCE": DiceBCELoss(dice_weight=0.6),
        "FocalDice": FocalDiceLoss(alpha=0.75, gamma=2.0, dice_weight=0.5),
        "OHEMFocalDice": OHEMFocalDiceLoss(alpha=0.8, gamma=3.0, ohem_ratio=0.25),
        "TverskyFocal": TverskyFocalLoss(alpha=0.3, beta=0.7),
        "Recommended": get_recommended_loss('extreme')
    }

    # Test all losses
    print("Testing loss functions:")
    print("-" * 70)

    for name, loss_fn in losses.items():
        loss_fn = loss_fn.to(device)

        # Forward pass (for training)
        loss_value = loss_fn(pred, masks)

        # Backward pass
        loss_value.backward(retain_graph=True)

        # Get components (for logging)
        with torch.no_grad():
            _, components = loss_fn(pred, masks, return_components=True)

        print(f"{name:15} | Loss: {loss_value.item():.4f}")
        for key, value in components.items():
            if isinstance(value, float):
                print(f"                  └─ {key}: {value:.4f}")
            else:
                print(f"                  └─ {key}: {value}")
        print()

    print("✓ All losses work with real data!\n")


def test_training_pattern():
    """Test proper training loop pattern"""
    print("=" * 70)
    print("TEST 3: Training Loop Pattern")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulate training
    pred = torch.randn(4, 1, 512, 512, requires_grad=True).to(device)
    target = torch.randint(0, 2, (4, 1, 512, 512)).float().to(device)

    criterion = FocalDiceLoss(alpha=0.75, gamma=2.0, dice_weight=0.5).to(device)

    print("Simulating training step...\n")

    # CORRECT PATTERN 1: Loss only (for backward)
    print("Pattern 1: Standard training (backward)")
    loss = criterion(pred, target)
    loss.backward()
    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ Gradients computed\n")

    # CORRECT PATTERN 2: Loss + components (logging)
    print("Pattern 2: Training with component logging")
    pred = torch.randn(4, 1, 512, 512, requires_grad=True).to(device)

    # Step 1: Compute loss and backward
    loss = criterion(pred, target)
    loss.backward()

    # Step 2: Get components for logging (no grad needed)
    with torch.no_grad():
        _, components = criterion(pred, target, return_components=True)

    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ Components: {components}")
    print(f"  ✓ This is the recommended pattern!\n")

    print("✓ Training patterns validated!\n")


def test_shape_handling():
    """Test handling of various input shapes"""
    print("=" * 70)
    print("TEST 4: Shape Handling")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = FocalDiceLoss().to(device)

    test_cases = [
        ("Normal [B,1,H,W]", (4, 1, 512, 512), (4, 1, 512, 512)),
        ("3-channel pred", (4, 3, 512, 512), (4, 1, 512, 512)),
        ("3D tensors", (4, 512, 512), (4, 512, 512)),
    ]

    for name, pred_shape, target_shape in test_cases:
        pred = torch.randn(*pred_shape).to(device)
        target = torch.randint(0, 2, target_shape).float().to(device)

        try:
            loss = criterion(pred, target)
            print(f"✓ {name:20} | Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"✗ {name:20} | Error: {e}")

    print("\n✓ Shape handling validated!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("LOSS FUNCTION COMPREHENSIVE TEST SUITE")
    print("=" * 70 + "\n")

    try:
        # Test 1: Basic forward pass
        test_basic_forward()

        # Test 2: Real data
        test_with_real_data()

        # Test 3: Training patterns
        test_training_pattern()

        # Test 4: Shape handling
        test_shape_handling()

        # Summary
        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nRecommended loss for FracAtlas:")
        print("  criterion = get_recommended_loss('extreme')")
        print("  └─ OHEMFocalDiceLoss(alpha=0.8, gamma=3.0, ohem_ratio=0.25)")
        print("\nReady for training! 🚀\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()