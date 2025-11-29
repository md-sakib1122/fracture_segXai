"""
Loss Functions for Medical Image Segmentation
Dice + BCE Loss for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss for binary segmentation

    Dice Loss: Measures overlap (good for imbalanced data)
    BCE Loss: Provides stable gradients

    Args:
        dice_weight: Weight for Dice loss (0-1). BCE weight = 1 - dice_weight
        smooth: Smoothing factor to avoid division by zero
    """

    def __init__(self, dice_weight=0.6, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (logits) - shape [B, 1, H, W]
            target: Ground truth masks - shape [B, 1, H, W]

        Returns:
            Combined loss value
        """
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)

        # Flatten predictions and targets
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        # Dice Loss calculation
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
                pred_flat.sum() + target_flat.sum() + self.smooth
        )

        # BCE Loss calculation
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)

        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        return total_loss


class FocalDiceLoss(nn.Module):
    """
    Advanced: Focal + Dice Loss
    Use this if DiceBCE doesn't give Dice > 0.80

    Focal Loss: Focuses on hard examples
    Dice Loss: Measures overlap
    """

    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.6, smooth=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred, target):
        # Focal Loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
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

        return total_loss


# For backward compatibility with your code
def get_criterion(loss_type='dice_bce', dice_weight=0.6):
    """
    Factory function to get loss criterion

    Args:
        loss_type: 'dice_bce' or 'focal_dice'
        dice_weight: Weight for Dice component (0.5-0.7 recommended)

    Returns:
        Loss function
    """
    if loss_type == 'dice_bce':
        return DiceBCELoss(dice_weight=dice_weight)
    elif loss_type == 'focal_dice':
        return FocalDiceLoss(dice_weight=dice_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test the loss function
    print("Testing Dice + BCE Loss...")

    criterion = DiceBCELoss(dice_weight=0.6)

    # Simulate predictions and targets
    batch_size, height, width = 4, 512, 512
    pred = torch.randn(batch_size, 1, height, width)  # Logits
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    # Calculate loss
    loss = criterion(pred, target)
    print(f"Loss value: {loss.item():.4f}")

    # Test backward pass
    loss.backward()
    print("✓ Backward pass successful")