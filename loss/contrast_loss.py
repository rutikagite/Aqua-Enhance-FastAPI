"""
Enhanced Contrast Loss Implementation for Image Dehazing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastLoss(nn.Module):
    """
    Enhanced Contrast Loss for image dehazing
    Encourages better contrast in dehazed images
    """
    
    def __init__(self, reduction='mean', window_size=5):
        super(ContrastLoss, self).__init__()
        self.reduction = reduction
        self.window_size = window_size
    
    def compute_local_contrast(self, x):
        """Compute local contrast using standard deviation in local windows"""
        # Convert to grayscale if RGB
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        # Compute local mean using average pooling
        kernel_size = self.window_size
        padding = kernel_size // 2
        
        # Local mean
        mean = F.avg_pool2d(gray, kernel_size, stride=1, padding=padding)
        
        # Local variance
        squared = gray ** 2
        mean_squared = F.avg_pool2d(squared, kernel_size, stride=1, padding=padding)
        variance = mean_squared - mean ** 2
        variance = torch.clamp(variance, min=1e-8)
        
        # Standard deviation as contrast measure
        std = torch.sqrt(variance)
        
        return std
    
    def forward(self, pred, target):
        """
        Calculate contrast loss
        Args:
            pred: predicted dehazed image [B, C, H, W]
            target: ground truth clear image [B, C, H, W]
        """
        # Compute local contrast for both images
        pred_contrast = self.compute_local_contrast(pred)
        target_contrast = self.compute_local_contrast(target)
        
        # Encourage predicted image to have similar contrast to target
        loss = F.l1_loss(pred_contrast, target_contrast, reduction=self.reduction)
        
        return loss


class EdgeContrastLoss(nn.Module):
    """
    Edge-aware contrast loss using Sobel filters
    """
    
    def __init__(self):
        super(EdgeContrastLoss, self).__init__()
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def compute_edges(self, x):
        """Compute edge magnitude using Sobel filters"""
        if x.shape[1] == 3:
            # Convert to grayscale
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        # Apply Sobel filters
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Compute edge magnitude
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        
        return edge_magnitude
    
    def forward(self, pred, target):
        """Calculate edge contrast loss"""
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        
        return F.l1_loss(pred_edges, target_edges)


class PerceptualContrastLoss(nn.Module):
    """Perceptual Contrast Loss using color channels"""
    
    def __init__(self):
        super(PerceptualContrastLoss, self).__init__()
    
    def forward(self, pred, target):
        """
        Calculate perceptual contrast loss
        Encourages similar color distribution
        """
        # Compute channel-wise mean and std
        pred_mean = torch.mean(pred, dim=(2, 3), keepdim=True)
        target_mean = torch.mean(target, dim=(2, 3), keepdim=True)
        
        pred_std = torch.std(pred, dim=(2, 3), keepdim=True)
        target_std = torch.std(target, dim=(2, 3), keepdim=True)
        
        # Loss on mean and std
        mean_loss = F.mse_loss(pred_mean, target_mean)
        std_loss = F.mse_loss(pred_std, target_std)
        
        return mean_loss + std_loss


class ColorLoss(nn.Module):
    """Color consistency loss"""
    
    def __init__(self):
        super(ColorLoss, self).__init__()
    
    def forward(self, pred, target):
        """Calculate color loss in RGB space"""
        return F.mse_loss(pred, target)


class EdgeLoss(nn.Module):
    """Edge preservation loss using Laplacian"""
    
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Laplacian kernel for edge detection
        self.register_buffer('laplacian', torch.tensor([
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        """Calculate edge preservation loss"""
        if pred.shape[1] == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target
        
        pred_edges = F.conv2d(pred_gray, self.laplacian, padding=1)
        target_edges = F.conv2d(target_gray, self.laplacian, padding=1)
        
        return F.l1_loss(pred_edges, target_edges)


class DetailLoss(nn.Module):
    """
    High-frequency detail preservation loss
    """
    
    def __init__(self):
        super(DetailLoss, self).__init__()
    
    def forward(self, pred, target):
        """
        Extract and compare high-frequency details
        """
        # Apply average pooling to get low-frequency component
        low_pred = F.avg_pool2d(pred, 3, stride=1, padding=1)
        low_target = F.avg_pool2d(target, 3, stride=1, padding=1)
        
        # High-frequency = original - low-frequency
        high_pred = pred - low_pred
        high_target = target - low_target
        
        # Loss on high-frequency components
        return F.l1_loss(high_pred, high_target)


class AquaEnhanceLoss(nn.Module):
    """Combined AquaEnhance Loss with multiple components"""
    
    def __init__(self, l1_weight=1.0, contrast_weight=0.1, edge_weight=0.1):
        super(AquaEnhanceLoss, self).__init__()
        self.l1_weight = l1_weight
        self.contrast_weight = contrast_weight
        self.edge_weight = edge_weight
        
        self.l1_loss = nn.L1Loss()
        self.contrast_loss = ContrastLoss()
        self.edge_loss = EdgeLoss()
    
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        contrast = self.contrast_loss(pred, target)
        edge = self.edge_loss(pred, target)
        
        total = (self.l1_weight * l1 + 
                self.contrast_weight * contrast +
                self.edge_weight * edge)
        
        return total, {
            'l1': l1.item(), 
            'contrast': contrast.item(),
            'edge': edge.item(),
            'total': total.item()
        }