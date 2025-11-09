import math

import torch.nn as nn
from kornia.color import rgb_to_hsv
import torch
from ptflops import get_model_complexity_info


class UIA(nn.Module):
    def __init__(self, channels, ks, num_bins=4):
        super(UIA, self).__init__()
        self.num_bins = num_bins

        # Pooling for directional attention
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_v = nn.AdaptiveAvgPool2d((None, 1))

        # Channel-wise brightness adjustment with better initialization
        self.channel_conv = nn.Conv2d(channels, channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Combine attention with reduced smoothing
        self.combine_conv = nn.Conv2d(channels, channels, ks, padding=ks // 2, 
                                     padding_mode='reflect', groups=channels, bias=True)
        self.final_conv = nn.Conv2d(channels, 1, 1)
        self.final_sigmoid = nn.Sigmoid()
        
        # Better initialization to preserve details
        nn.init.xavier_uniform_(self.channel_conv.weight, gain=0.02)
        nn.init.constant_(self.channel_conv.bias, 0.0)
        nn.init.xavier_uniform_(self.combine_conv.weight, gain=0.02)
        nn.init.constant_(self.combine_conv.bias, 0.0)

    def forward(self, x):
        with torch.no_grad():
            if x.shape[1] == 3:
                hsv = rgb_to_hsv(x)
            else:
                hsv = x 
            brightness = hsv[:, 2:3, :, :]

        # Softer clustering to preserve gradients
        bin_width = 1.0 / self.num_bins
        clustered_brightness = torch.floor(brightness / bin_width + 0.5) * bin_width
        clustered_brightness = torch.clamp(clustered_brightness, 0, 1)

        # Directional attention maps
        h_attn = self.pool_h(x)
        v_attn = self.pool_v(x)
        directional_attn = h_attn + v_attn

        # Channel-wise modulation
        c_attn = self.channel_conv(x)
        c_attn = self.sigmoid(c_attn)

        # Fuse with residual connection to preserve input
        combined = directional_attn * c_attn * (clustered_brightness + 0.1)

        # Refine and weight original input
        attn_map = self.combine_conv(combined)
        attn_weight = self.final_sigmoid(self.final_conv(attn_map))
        
        # Ensure attention doesn't suppress too much
        attn_weight = attn_weight * 0.9 + 0.1

        return x * attn_weight


class NormGate(nn.Module):
    def __init__(self, channels, ks, norm=nn.InstanceNorm2d):
        super(NormGate, self).__init__()
        self._norm_branch = nn.Sequential(
            norm(channels, affine=True),
            nn.Conv2d(channels, channels, ks, padding=ks // 2, padding_mode='reflect', bias=True)
        )
        self._sig_branch = nn.Sequential(
            nn.Conv2d(channels, channels, ks, padding=ks // 2, padding_mode='reflect', bias=True),
            nn.Sigmoid()
        )
        
        # Better initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        norm = self._norm_branch(x)
        sig = self._sig_branch(x)
        return norm * sig


class UCB(nn.Module):
    def __init__(self, channels, ks):
        super(UCB, self).__init__()
        self._body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, padding=ks // 2,
                      padding_mode='reflect', bias=True),
            NormGate(channels, ks),
            UIA(channels, ks)
        )
        
        # Initialize conv with small weights
        nn.init.xavier_uniform_(self._body[0].weight, gain=0.02)
        nn.init.constant_(self._body[0].bias, 0.0)

    def forward(self, x):
        y = self._body(x)
        return y + x


class PWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(PWConv, self).__init__()
        self._body = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                               padding=kernel_size // 2, padding_mode='reflect', bias=bias)
        
        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self._body.weight, gain=0.02)
        if bias:
            nn.init.constant_(self._body.bias, 0.0)

    def forward(self, x):
        return self._body(x)


class GlobalColorCompensationNet(nn.Module):
    def __init__(self, channel_scale, kernel_size):
        super(GlobalColorCompensationNet, self).__init__()
        self._body = nn.Sequential(
            PWConv(3, channel_scale, kernel_size),
            UCB(channel_scale, kernel_size),
            UCB(channel_scale, kernel_size),
            UCB(channel_scale, kernel_size),
            PWConv(channel_scale, 3, kernel_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self._body(x)
        return y


class CLCC(nn.Module):
    def __init__(self, channel_scale, main_ks, gcc_ks):
        super(CLCC, self).__init__()
        self._color_branch = GlobalColorCompensationNet(channel_scale, gcc_ks)
        self._in_conv = nn.Sequential(
            PWConv(3, channel_scale, main_ks),
            UIA(channel_scale, main_ks)
        )
        self._group1 = nn.Sequential(
            *[UCB(channel_scale, main_ks) for _ in range(4)]
        )
        self._group2 = nn.Sequential(
            *[UCB(channel_scale, main_ks) for _ in range(4)]
        )
        self._group3 = nn.Sequential(
            *[UCB(channel_scale, main_ks) for _ in range(4)]
        )
        self._group1_adaptation = nn.Sequential(
            PWConv(3, channel_scale, main_ks),
            UCB(channel_scale, main_ks)
        )
        self._group2_adaptation = nn.Sequential(
            PWConv(3, channel_scale, main_ks),
            UCB(channel_scale, main_ks)
        )
        self._group3_adaptation = nn.Sequential(
            PWConv(3, channel_scale, main_ks),
            UCB(channel_scale, main_ks)
        )
        
        # CRITICAL FIX: Changed from Tanh to Sigmoid for [0, 1] output range
        # Also add residual connection to preserve input details
        self._out_conv = nn.Sequential(
            PWConv(channel_scale, 3, main_ks),
            nn.Sigmoid()  # Changed from Tanh to Sigmoid
        )
        
        # Optional: Add a learnable alpha for residual connection
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Initialize all weights with small values to prevent saturation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Store original input for residual connection
        identity = x
        
        # Color compensation with reduced influence
        color_comp = 1 - x
        color_comp_map = self._color_branch(color_comp)
        
        # Reduce color compensation influence to prevent over-correction
        color_comp_map = color_comp_map * 0.5
        
        in_feat = self._in_conv(x)
        group1_out = self._group1(in_feat)
        group1_comp_out = group1_out + 0.3 * self._group1_adaptation(color_comp_map * color_comp)
        
        group2_out = self._group2(group1_comp_out)
        group2_comp_out = group2_out + 0.3 * self._group2_adaptation(color_comp_map * color_comp)
        
        group3_out = self._group3(group2_comp_out)
        group3_comp_out = group3_out + 0.3 * self._group3_adaptation(color_comp_map * color_comp)
        
        out = self._out_conv(group3_comp_out)
        
        # Add residual connection to preserve details from input
        # This prevents the network from producing overly smooth outputs
        alpha = torch.sigmoid(self.alpha)  # Keep alpha in [0, 1]
        out = alpha * out + (1 - alpha) * identity
        
        # Ensure output is in valid range
        out = torch.clamp(out, 0.0, 1.0)
        
        return out


if __name__ == '__main__':
    import torch
    x = torch.randn((2, 3, 256, 256))
    model = CLCC(64, 3, 3)
    macs, params = get_model_complexity_info(model, (3, 256, 256), verbose=False, print_per_layer_stat=False)
    print('MACS: ' + str(macs))
    print('Params: ' + str(params))
    y = model(x)
    print(y.shape)
    print('Output range:', y.min().item(), y.max().item())