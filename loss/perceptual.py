"""
Perceptual Loss Implementation using VGG Features
This implements perceptual losses commonly used in image dehazing and restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms


class VGGPerceptualLoss(nn.Module):
    """
    VGG Perceptual Loss for image dehazing
    
    Uses pre-trained VGG features to compute perceptual similarity between images.
    This is particularly effective for maintaining visual quality in image restoration.
    
    Args:
        feature_layers (list): List of VGG layer indices to use for feature extraction
        weights (list): Weights for each feature layer loss
        resize (bool): Whether to resize input to 224x224 (VGG input size)
        normalize (bool): Whether to normalize inputs with ImageNet stats
    """
    
    def __init__(self, 
                 feature_layers=[3, 8, 15, 22], 
                 weights=[1.0, 1.0, 1.0, 1.0],
                 resize=False,
                 normalize=True):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.feature_layers = feature_layers
        self.weights = weights
        self.resize = resize
        self.normalize = normalize
        
        # Create feature extractor
        self.feature_extractor = nn.ModuleList()
        self.layer_names = []
        
        layer_idx = 0
        for i, layer in enumerate(vgg):
            self.feature_extractor.append(layer)
            if isinstance(layer, nn.ReLU):
                layer_idx += 1
                self.layer_names.append(f'relu{layer_idx}')
            elif isinstance(layer, nn.MaxPool2d):
                self.layer_names.append(f'pool{i}')
            elif isinstance(layer, nn.Conv2d):
                self.layer_names.append(f'conv{i}')
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        if self.normalize:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def normalize_tensor(self, x):
        """Normalize tensor with ImageNet statistics"""
        if self.normalize:
            if self.mean.device != x.device:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
            return (x - self.mean) / self.std
        return x
    
    def extract_features(self, x):
        """Extract VGG features from input tensor"""
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = self.normalize_tensor(x)
        features = []
        
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features
    
    def forward(self, pred, target):
        """
        Calculate VGG perceptual loss
        
        Args:
            pred (Tensor): Predicted image [B, C, H, W]
            target (Tensor): Target image [B, C, H, W]
        
        Returns:
            loss (Tensor): Perceptual loss value
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            weight = self.weights[i] if i < len(self.weights) else 1.0
            loss += weight * F.mse_loss(pred_feat, target_feat)
        
        return loss


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS)
    A more advanced perceptual loss that's learned rather than fixed
    """
    
    def __init__(self, net='vgg', spatial=False):
        super(LPIPSLoss, self).__init__()
        
        # Use VGG as backbone
        if net == 'vgg':
            self.feature_extractor = models.vgg16(pretrained=True).features[:23]
        else:
            self.feature_extractor = models.vgg16(pretrained=True).features[:23]
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.spatial = spatial
        
        # Learned linear layers
        self.linear_layers = nn.ModuleList([
            nn.Conv2d(64, 1, 1, bias=False),
            nn.Conv2d(128, 1, 1, bias=False),
            nn.Conv2d(256, 1, 1, bias=False),
            nn.Conv2d(512, 1, 1, bias=False),
        ])
    
    def forward(self, pred, target):
        # Placeholder implementation - simplified
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return F.mse_loss(pred_features, target_features)


class StyleLoss(nn.Module):
    """
    Style Loss using Gram matrices
    Captures texture and style information
    """
    
    def __init__(self, feature_layers=[3, 8, 15, 22]):
        super(StyleLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:max(feature_layers)+1])
        self.feature_layers = feature_layers
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def gram_matrix(self, features):
        """Calculate Gram matrix for style representation"""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def extract_features(self, x):
        """Extract features at specified layers"""
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features
    
    def forward(self, pred, target):
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += F.mse_loss(pred_gram, target_gram)
        
        return loss


class CombinedPerceptualLoss(nn.Module):
    """
    Combined perceptual loss with content and style components
    """
    
    def __init__(self, 
                 content_weight=1.0, 
                 style_weight=0.1,
                 content_layers=[15, 22],
                 style_layers=[3, 8, 15, 22]):
        super(CombinedPerceptualLoss, self).__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        self.content_loss = VGGPerceptualLoss(feature_layers=content_layers)
        self.style_loss = StyleLoss(feature_layers=style_layers)
    
    def forward(self, pred, target):
        content_loss = self.content_loss(pred, target)
        style_loss = self.style_loss(pred, target)
        
        total_loss = (self.content_weight * content_loss + 
                     self.style_weight * style_loss)
        
        return total_loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss using multiple VGG layers
    Often used in GANs and image restoration
    """
    
    def __init__(self, layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']):
        super(FeatureMatchingLoss, self).__init__()
        
        # Use pre-trained VGG
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = vgg
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.layers = layers
    
    def forward(self, pred, target):
        # Simple implementation using full VGG features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        return F.l1_loss(pred_features, target_features)


# Factory function
def get_perceptual_loss(loss_type='vgg', **kwargs):
    """
    Factory function to get perceptual loss
    
    Args:
        loss_type (str): Type of perceptual loss
        **kwargs: Additional arguments
    
    Returns:
        Perceptual loss instance
    """
    if loss_type == 'vgg':
        return VGGPerceptualLoss(**kwargs)
    elif loss_type == 'lpips':
        return LPIPSLoss(**kwargs)
    elif loss_type == 'style':
        return StyleLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedPerceptualLoss(**kwargs)
    elif loss_type == 'feature_matching':
        return FeatureMatchingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown perceptual loss type: {loss_type}")


# Export all classes
__all__ = [
    'VGGPerceptualLoss',
    'LPIPSLoss', 
    'StyleLoss',
    'CombinedPerceptualLoss',
    'FeatureMatchingLoss',
    'get_perceptual_loss'
]