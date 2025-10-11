"""
Focal Frequency Loss Implementation
This implements the Focal Frequency Loss used in image dehazing and restoration tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss for image dehazing
    
    This loss focuses on frequency domain differences between predicted and target images.
    It's particularly effective for image restoration tasks like dehazing.
    
    Args:
        loss_weight (float): Weight of the focal frequency loss. Default: 1.0
        alpha (float): The focusing parameter. Default: 1.0
        patch_factor (int): The factor to crop the frequency domain. Default: 1
        ave_spectrum (bool): Whether to use average spectrum. Default: False
        log_matrix (bool): Whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): Whether to calculate the spectrum weight matrix in batch mode. Default: False
    """
    
    def __init__(self, 
                 loss_weight=1.0,
                 alpha=1.0, 
                 patch_factor=1, 
                 ave_spectrum=False, 
                 log_matrix=False, 
                 batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        """Convert tensor to frequency domain using FFT"""
        # Perform 2D FFT
        freq = torch.fft.fft2(x, dim=(-2, -1))
        freq = torch.stack([freq.real, freq.imag], dim=-1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        """Calculate the focal frequency loss"""
        # Spectrum weight matrix
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1])

            # Apply logarithm to the matrix
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # Apply focusing parameter
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            f'The values of spectrum weight matrix should be in the range [0, 1], '
            f'but got Min: {weight_matrix.min().item()}, Max: {weight_matrix.max().item()}.')

        # Calculate frequency distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # Apply focusing
        loss = weight_matrix ** self.alpha * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """
        Forward function for focal frequency loss
        
        Args:
            pred (Tensor): Predicted tensor with shape (N, C, H, W)
            target (Tensor): Target tensor with shape (N, C, H, W)  
            matrix (Tensor, optional): Element-wise spectrum weight matrix.
        
        Returns:
            loss (Tensor): Calculated focal frequency loss
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # Crop the frequency domain
        if self.patch_factor > 1:
            _, _, h, w = pred_freq.shape
            crop_h = h // self.patch_factor
            crop_w = w // self.patch_factor
            
            pred_freq = pred_freq[:, :, :crop_h, :crop_w, :]
            target_freq = target_freq[:, :, :crop_h, :crop_w, :]

        # Calculate loss
        if matrix is None:
            loss = self.loss_formulation(pred_freq, target_freq, matrix)
        else:
            loss = self.loss_formulation(pred_freq, target_freq, matrix)

        loss = loss * self.loss_weight
        return loss


class MultiFocalFrequencyLoss(nn.Module):
    """
    Multi-scale Focal Frequency Loss
    Applies focal frequency loss at multiple scales
    """
    
    def __init__(self, 
                 loss_weight=1.0,
                 alpha=1.0,
                 scales=[1, 2, 4]):
        super(MultiFocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.scales = scales
        self.focal_losses = nn.ModuleList([
            FocalFrequencyLoss(loss_weight=1.0, alpha=alpha, patch_factor=scale)
            for scale in scales
        ])
    
    def forward(self, pred, target):
        """Calculate multi-scale focal frequency loss"""
        total_loss = 0
        for focal_loss in self.focal_losses:
            total_loss += focal_loss(pred, target)
        
        return total_loss * self.loss_weight / len(self.scales)


# Additional frequency-based losses
class SpectralLoss(nn.Module):
    """
    Simple spectral loss in frequency domain
    """
    
    def __init__(self, loss_weight=1.0):
        super(SpectralLoss, self).__init__()
        self.loss_weight = loss_weight
    
    def forward(self, pred, target):
        # Convert to frequency domain
        pred_freq = torch.fft.fft2(pred, dim=(-2, -1))
        target_freq = torch.fft.fft2(target, dim=(-2, -1))
        
        # Calculate L2 loss in frequency domain
        loss = F.mse_loss(pred_freq.real, target_freq.real) + F.mse_loss(pred_freq.imag, target_freq.imag)
        
        return loss * self.loss_weight


class AdaptiveFocalFrequencyLoss(nn.Module):
    """
    Adaptive Focal Frequency Loss with learnable parameters
    """
    
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super(AdaptiveFocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, pred, target):
        # Use the basic focal frequency loss with learnable alpha
        focal_loss = FocalFrequencyLoss(alpha=self.alpha)
        return focal_loss(pred, target) * self.loss_weight


# Utility functions
def get_frequency_loss(loss_type='focal', **kwargs):
    """
    Factory function to get frequency-based loss
    
    Args:
        loss_type (str): Type of frequency loss ('focal', 'spectral', 'multi_focal', 'adaptive')
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function instance
    """
    if loss_type == 'focal':
        return FocalFrequencyLoss(**kwargs)
    elif loss_type == 'spectral':
        return SpectralLoss(**kwargs)
    elif loss_type == 'multi_focal':
        return MultiFocalFrequencyLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveFocalFrequencyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown frequency loss type: {loss_type}")


# Export all classes
__all__ = [
    'FocalFrequencyLoss', 
    'MultiFocalFrequencyLoss', 
    'SpectralLoss', 
    'AdaptiveFocalFrequencyLoss',
    'get_frequency_loss'
]