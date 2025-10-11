import math

import torch.nn as nn
from kornia.color import rgb_to_hsv
import torch
from ptflops import get_model_complexity_info


class UIA(nn.Module):
    def __init__(self, channels, ks, num_bins=4):
        super(UIA, self).__init__()
        self.num_bins = num_bins  # For grouping brightness levels (clustering-inspired)

        # Pooling for directional attention
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))   # Horizontal
        self.pool_v = nn.AdaptiveAvgPool2d((None, 1))   # Vertical

        # Channel-wise brightness adjustment
        self.channel_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Combine attention
        self.combine_conv = nn.Conv2d(channels, channels, ks, padding=ks // 2, padding_mode='reflect', groups=channels, bias=False)
        self.final_conv = nn.Conv2d(channels, 1, 1)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        with torch.no_grad():
            if x.shape[1] == 3:
                hsv = rgb_to_hsv(x)
            else:
                hsv = x 
            brightness = hsv[:, 2:3, :, :]  # V channel [B,1,H,W]

        # Step 2: Cluster brightness into bins
        bin_width = 1.0 / self.num_bins
        clustered_brightness = torch.floor(brightness / bin_width) * bin_width  # Like clustering

        # Step 3: Directional attention maps
        h_attn = self.pool_h(x)
        v_attn = self.pool_v(x)
        directional_attn = h_attn + v_attn

        # Step 4: Channel-wise modulation
        c_attn = self.channel_conv(x)
        c_attn = self.sigmoid(c_attn)

        # Step 5: Fuse all
        combined = directional_attn * c_attn * clustered_brightness  # shape [B,C,H,W]

        # Step 6: Refine and weight original input
        attn_map = self.combine_conv(combined)
        attn_weight = self.final_sigmoid(self.final_conv(attn_map))

        return x * attn_weight


class NormGate(nn.Module):
    def __init__(self, channels, ks, norm=nn.InstanceNorm2d):
        super(NormGate, self).__init__()
        self._norm_branch = nn.Sequential(
            norm(channels),
            nn.Conv2d(channels, channels, ks, padding=ks // 2, padding_mode='reflect', bias=False)
        )
        self._sig_branch = nn.Sequential(
            nn.Conv2d(channels, channels, ks, padding=ks // 2, padding_mode='reflect', bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        norm = self._norm_branch(x)
        sig = self._sig_branch(x)
        return norm * sig


class UCB(nn.Module):
    def __init__(self, channels, ks):
        super(UCB, self).__init__()
        self._body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, padding=ks // 2,
                      padding_mode='reflect', bias=False),
            NormGate(channels, ks),
            UIA(channels, ks)
        )

    def forward(self, x):
        y = self._body(x)
        return y + x


class PWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(PWConv, self).__init__()
        self._body = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                               padding=kernel_size // 2, padding_mode='reflect', bias=bias)

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
        self._out_conv = nn.Sequential(
            PWConv(channel_scale, 3, main_ks),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            # elif isinstance(m, nn.InstanceNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()


    def forward(self, x):
        color_comp = 1 - x
        color_comp_map = self._color_branch(color_comp)
        in_feat = self._in_conv(x)
        group1_out = self._group1(in_feat)
        group1_comp_out = group1_out + self._group1_adaptation(color_comp_map * color_comp)
        group2_out = self._group2(group1_comp_out)
        group2_comp_out = group2_out + self._group2_adaptation(color_comp_map * color_comp)
        group3_out = self._group3(group2_comp_out)
        group3_comp_out = group3_out + self._group3_adaptation(color_comp_map * color_comp)
        out = self._out_conv(group3_comp_out)
        return out


if __name__ == '__main__':
    import torch
    x = torch.randn((2, 3, 256, 256))
    model = CLCC(64, 3, 3)
    macs, params = get_model_complexity_info(model, (3, 256, 256), verbose=False, print_per_layer_stat=False)
    print('MACS: ' + str(macs))
    print('Params: ' + str(params))
    # model = GlobalColorCompensationNet(64)
    y = model(x)
    print(y.shape)
