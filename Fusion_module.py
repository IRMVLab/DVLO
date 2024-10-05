import torch
import torch.nn as nn
from torch.nn.functional import softmax
from conv_util import (
    Conv1dNormRelu,
    Conv2dNormRelu,
)

class GlobalFuser(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, fusion_fn="gated", norm=None):
        super().__init__()

        self.mlps3d = Conv2dNormRelu(in_channels_2d, in_channels_2d, norm=norm)

        if fusion_fn == "gated":
            self.fuse3d = GatedFusion(
                in_channels_2d, in_channels_3d, in_channels_3d, "nchw", norm
            )
        else:
            raise ValueError

    def forward(self, feat_2d, feat_3d):
        """
        :param feat_2d: features of images: [B, n_channels_2d, H, W]
        :param feat_3d: features of points: [B, n_channels_3d, H, W]
        :return: out3d: fused features of points: [B, n_channels_3d, H, W]
        """
        feat_2d = feat_2d.float().permute(0, 3, 1, 2)
        feat_3d = feat_3d.float().permute(0, 3, 1, 2)

        out3d = self.fuse3d(self.mlps3d(feat_2d.detach().clone()), feat_3d)

        return out3d


class GatedFusion(nn.Module):
    def __init__(
        self, in_channels_2d, in_channels_3d, out_channels, feat_format, norm=None
    ):
        super().__init__()

        if feat_format == "nchw":
            self.align1 = Conv2dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv2dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.mlp1 = Conv2dNormRelu(out_channels, 2, norm=None, activation="sigmoid")
            self.mlp2 = Conv2dNormRelu(out_channels, 2, norm=None, activation="sigmoid")
        elif feat_format == "ncm":
            self.align1 = Conv1dNormRelu(in_channels_2d, out_channels, norm=norm)
            self.align2 = Conv1dNormRelu(in_channels_3d, out_channels, norm=norm)
            self.mlp1 = Conv1dNormRelu(out_channels, 2, norm=None, activation="sigmoid")
            self.mlp2 = Conv1dNormRelu(out_channels, 2, norm=None, activation="sigmoid")
        else:
            raise ValueError

    def forward(self, feat_2d, feat_3d):
        feat_2d = self.align1(feat_2d)  # [N, C_out, H, W]
        feat_3d = self.align2(feat_3d)  # [N, C_out, H, W]
        weight = self.mlp1(feat_2d) + self.mlp2(feat_3d)  # [N, 2, H, W]
        weight = softmax(weight, dim=1)  # [N, 2, H, W]
        return feat_2d * weight[:, 0:1] + feat_3d * weight[:, 1:2]

