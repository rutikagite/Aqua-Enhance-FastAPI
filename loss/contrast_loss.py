"""
Contrast Loss Implementation for IACC Dehazing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastLoss(nn.Module):
    """Basic Contrast Loss for image dehazing"""
    
    def __init__(self, reduction='mean'):
        super(ContrastLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        # Simple L1 loss as placeholder
        return F.l1_loss(pred, target, reduction=self.reduction)

class PerceptualContrastLoss(nn.Module):
    """Perceptual Contrast Loss"""
    
    def __init__(self):
        super(PerceptualContrastLoss, self).__init__()
    
    def forward(self, pred, target):
        return F.mse_loss(pred, target)

class ColorLoss(nn.Module):
    """Color consistency loss"""
    
    def __init__(self):
        super(ColorLoss, self).__init__()
    
    def forward(self, pred, target):
        return F.mse_loss(pred, target)

class EdgeLoss(nn.Module):
    """Edge preservation loss"""
    
    def __init__(self):
        super(EdgeLoss, self).__init__()
    
    def forward(self, pred, target):
        return F.l1_loss(pred, target)

class AquaEnhanceLoss(nn.Module):
    """Combined AquaEnhance Loss"""
    
    def __init__(self, l1_weight=1.0):
        super(AquaEnhanceLoss, self).__init__()
        self.l1_weight = l1_weight
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        return l1, {'l1': l1.item(), 'total': l1.item()}
