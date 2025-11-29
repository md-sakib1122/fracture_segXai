"""
Improved Loss Functions for Medical Image Segmentation
Optimized for extreme class imbalance (FracAtlas dataset)
FIXED: Handles shape mismatches and validates inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_inputs(pred, target):
    """
    Validate and fix input shapes for loss calculation

    Args:
        pred: Model predictions - can be [B, C, H, W] or [B, 1, H, W]
        target: Ground truth - should be [B, 1, H, W]

    Returns:
        pred, target with validated shapes
    """
    # Ensure both are 4D tensors
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    # Check batch size matches
    assert pred.shape[0] == target.shape[0], \
        f"Batch size mismatch: pred={pred.shape[0]}, target={target.shape[0]}"

    # Handle multi-channel predictions (take only first channel)
    if pred.shape[1] > 1:
        print(f"⚠ Warning: Predictions have {pred.shape[1]} channels, using first channel only")
        pred = pred[:, 0:1, :, :]

    # Ensure target is single channel
    if target.shape[1] > 1:
        print(f"⚠ Warning: Target has {target.shape[1]} channels, using first channel only")
        target = target[:, 0:1, :, :]

    # Check spatial dimensions match
    if pred.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"Spatial dimension mismatch: pred={pred.shape[2:]}, target={target.shape[2:]}"
        )

    return pred, target


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss for binary segmentation

    Good baseline for moderate imbalance
    Use if fracture pixels > 5% of total
    """

    def __init__(self, dice_weight=0.6, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight
        self.smooth = smooth

    def forward(self, pred, target, return_components=False):
        """
        Args:
            pred: Model predictions (logits) - shape [B, 1, H, W]
            target: Ground truth masks - shape [B, 1, H, W]
            return_components: If True, return loss breakdown dict
        """
        # Validate and fix shapes
        pred, target = validate_inputs(pred, target)

        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        # Dice Loss
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
                pred_flat.sum() + target_flat.sum() + self.smooth
        )

        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)

        # Combined
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        if return_components:
            # Don't call .item() on component losses - return raw tensors
            return total_loss, {
                'dice': dice_loss.detach().item(),
                'bce': bce_loss.detach().item(),
                'total': total_loss.detach().item()
            }
        return total_loss


class FocalDiceLoss(nn.Module):
    """
    Focal + Dice Loss with proper class-specific alpha weighting

    Recommended for fracture segmentation (0.5-5% positive pixels)

    Args:
        alpha: Weight for positive (fracture) class (0.5-0.8)
               Higher = more weight on fractures
        gamma: Focus on hard examples (1.5-3.0)
               Higher = more focus on mistakes
        dice_weight: Balance between Dice and Focal (0.4-0.6)
    """

    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=0.5, smooth=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred, target, return_components=False):
        # Validate and fix shapes
        pred, target = validate_inputs(pred, target)

        # Focal Loss with class-specific alpha
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)

        # Apply alpha based on class
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        focal_loss = focal_loss.mean()

        # Dice Loss
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
                pred_flat.sum() + target_flat.sum() + self.smooth
        )

        # Combined
        total_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * focal_loss

        if return_components:
            return total_loss, {
                'focal': focal_loss.detach().item(),
                'dice': dice_loss.detach().item(),
                'total': total_loss.detach().item()
            }
        return total_loss


class OHEMFocalDiceLoss(nn.Module):
    """
    Online Hard Example Mining + Focal + Dice Loss

    Best for EXTREME imbalance (< 0.5% positive pixels)
    Use if FocalDiceLoss gives validation Dice < 0.60 after 20 epochs

    Args:
        ohem_ratio: Keep top X% hardest pixels (0.2-0.3)
                   Lower = more aggressive mining
    """

    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=0.5,
                 ohem_ratio=0.25, smooth=1e-6):
        super(OHEMFocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.ohem_ratio = ohem_ratio
        self.smooth = smooth

    def forward(self, pred, target, return_components=False):
        # Validate and fix shapes
        pred, target = validate_inputs(pred, target)

        # Calculate per-pixel focal loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_loss_map = alpha_t * (1 - pt) ** self.gamma * bce

        # OHEM: Keep only hardest examples
        num_pixels = focal_loss_map.numel()
        num_hard = max(int(num_pixels * self.ohem_ratio), 1)

        focal_loss_flat = focal_loss_map.view(-1)
        hard_losses, _ = torch.topk(focal_loss_flat, num_hard)
        focal_loss = hard_losses.mean()

        # Dice Loss (computed on all pixels)
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
                pred_flat.sum() + target_flat.sum() + self.smooth
        )

        # Combined
        total_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * focal_loss

        if return_components:
            return total_loss, {
                'focal': focal_loss.detach().item(),
                'dice': dice_loss.detach().item(),
                'total': total_loss.detach().item(),
                'hard_pixels_ratio': num_hard / num_pixels
            }
        return total_loss


class TverskyFocalLoss(nn.Module):
    """
    Tversky + Focal Loss

    Alternative to Dice - better control over FP/FN tradeoff
    Use if you want to prioritize recall (catching all fractures)

    Args:
        alpha: False Positive weight (0.2-0.4)
               Lower = less penalty for FP (more detections)
        beta: False Negative weight (0.6-0.8)
              Higher = more penalty for FN (fewer misses)
    """

    def __init__(self, alpha=0.3, beta=0.7, focal_alpha=0.75,
                 gamma=2.0, tversky_weight=0.5, smooth=1e-6):
        super(TverskyFocalLoss, self).__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta  # FN weight
        self.focal_alpha = focal_alpha
        self.gamma = gamma
        self.tversky_weight = tversky_weight
        self.smooth = smooth

    def forward(self, pred, target, return_components=False):
        # Validate and fix shapes
        pred, target = validate_inputs(pred, target)

        # Focal Loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        focal_loss = focal_loss.mean()

        # Tversky Loss
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        TP = (pred_flat * target_flat).sum()
        FP = (pred_flat * (1 - target_flat)).sum()
        FN = ((1 - pred_flat) * target_flat).sum()

        tversky_index = (TP + self.smooth) / (
                TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        tversky_loss = 1 - tversky_index

        # Combined
        total_loss = (self.tversky_weight * tversky_loss +
                      (1 - self.tversky_weight) * focal_loss)

        if return_components:
            return total_loss, {
                'focal': focal_loss.detach().item(),
                'tversky': tversky_loss.detach().item(),
                'total': total_loss.detach().item()
            }
        return total_loss


def get_criterion(loss_type='focal_dice', **kwargs):
    """
    Factory function to get loss criterion

    Args:
        loss_type: 'dice_bce', 'focal_dice', 'ohem_focal_dice', 'tversky_focal'
        **kwargs: Loss-specific parameters

    Returns:
        Loss function instance

    Examples:
        >>> criterion = get_criterion('focal_dice', alpha=0.75, gamma=2.0)
        >>> criterion = get_criterion('ohem_focal_dice', ohem_ratio=0.25)
    """
    loss_map = {
        'dice_bce': DiceBCELoss,
        'focal_dice': FocalDiceLoss,
        'ohem_focal_dice': OHEMFocalDiceLoss,
        'tversky_focal': TverskyFocalLoss
    }

    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                         f"Choose from {list(loss_map.keys())}")

    return loss_map[loss_type](**kwargs)


# ============================================================================
# RECOMMENDED CONFIGURATIONS FOR FRACATLAS
# ============================================================================

def get_recommended_loss(imbalance_severity='extreme'):
    """
    Get recommended loss based on class imbalance

    Args:
        imbalance_severity: 'mild' (>10%), 'moderate' (1-10%),
                           'severe' (0.1-1%), 'extreme' (<0.1%)

    Returns:
        Configured loss function
    """
    configs = {
        'mild': {
            'type': 'dice_bce',
            'dice_weight': 0.6
        },
        'moderate': {
            'type': 'focal_dice',
            'alpha': 0.7,
            'gamma': 2.0,
            'dice_weight': 0.5
        },
        'severe': {
            'type': 'focal_dice',
            'alpha': 0.75,
            'gamma': 2.5,
            'dice_weight': 0.5
        },
        'extreme': {  # FracAtlas falls here (0.05-0.18%)
            'type': 'ohem_focal_dice',
            'alpha': 0.8,
            'gamma': 3.0,
            'dice_weight': 0.5,
            'ohem_ratio': 0.25
        }
    }

    config = configs[imbalance_severity]
    loss_type = config.pop('type')
    return get_criterion(loss_type, **config)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Loss Functions with Shape Validation...\n")

    # Test Case 1: Normal case - matching shapes
    print("=" * 60)
    print("TEST 1: Normal shapes [B, 1, H, W]")
    print("=" * 60)
    batch_size, height, width = 4, 512, 512
    pred = torch.randn(batch_size, 1, height, width)
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    criterion = FocalDiceLoss(alpha=0.75, gamma=2.0, dice_weight=0.5)
    loss, components = criterion(pred, target, return_components=True)
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"  Components: {components}\n")

    # Test Case 2: Multi-channel predictions (model output issue)
    print("=" * 60)
    print("TEST 2: Multi-channel prediction [B, 3, H, W]")
    print("=" * 60)
    pred = torch.randn(batch_size, 3, height, width)  # 3 channels
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    loss, components = criterion(pred, target, return_components=True)
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"  (Used first channel only)\n")

    # Test Case 3: 3D tensors (missing batch/channel dim)
    print("=" * 60)
    print("TEST 3: 3D tensors [B, H, W]")
    print("=" * 60)
    pred = torch.randn(batch_size, height, width)
    target = torch.randint(0, 2, (batch_size, height, width)).float()

    loss, components = criterion(pred, target, return_components=True)
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"  (Added channel dimension)\n")

    # Test all loss types
    print("=" * 60)
    print("TEST 4: All Loss Types")
    print("=" * 60)

    pred = torch.randn(batch_size, 1, height, width)
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    loss_configs = [
        ('DiceBCE', DiceBCELoss(dice_weight=0.6)),
        ('FocalDice', FocalDiceLoss(alpha=0.75, gamma=2.0, dice_weight=0.5)),
        ('OHEM', OHEMFocalDiceLoss(alpha=0.8, gamma=3.0, ohem_ratio=0.25)),
        ('Tversky', TverskyFocalLoss(alpha=0.3, beta=0.7))
    ]

    for name, criterion in loss_configs:
        loss, components = criterion(pred, target, return_components=True)
        loss.backward(retain_graph=True)
        print(f"✓ {name:12} | Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

    # Show recommended configuration
    print("\nRECOMMENDED FOR FRACATLAS:")
    criterion = get_recommended_loss('extreme')
    print(f"  Loss: {criterion.__class__.__name__}")
    print(f"  Config: alpha={criterion.alpha}, gamma={criterion.gamma}")