import os
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
import torch.nn.functional as F
from conv_util import Feature_Gather

import time

def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        """

        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center

    def forward(self, points, x):  # [b,c,h,w]

        value = self.v(x)
        x = self.f(x)

        x = rearrange(x, "b (e c) h w -> (b e) c h w", e=self.heads)
        value = rearrange(value, "b (e c) h w -> (b e) c h w", e=self.heads)

        _, N, _ = points.shape
        size_range = [1296.0, 384.0]
        # split the big feature maps to small local regions to reduce computations.
        b0, c0, h0, w0 = x.shape
        assert h0 % self.fold_h == 0 and w0 % self.fold_w == 0, \
            f"Ensure the feature map size ({h0}*{w0}) can be divided by fold {self.fold_h}*{self.fold_w}"
        x = rearrange(x, "b c (f1 h) (f2 w) -> (b f1 f2) c h w", f1=self.fold_h, f2=self.fold_w)
        value = rearrange(value, "b c (f1 h) (f2 w) -> (b f1 f2) c h w", f1=self.fold_h, f2=self.fold_w)

        regions_h = size_range[1] / self.fold_h
        regions_w = size_range[0] / self.fold_w
        num_regions = self.fold_h * self.fold_w

        i = torch.arange(self.fold_h).view(-1, 1, 1, 1).to(points.device)
        j = torch.arange(self.fold_w).view(1, -1, 1, 1).to(points.device)
        val_flag_1 = (points[:, :, 1] > regions_h * i) & (points[:, :, 1] <= regions_h * (i + 1))
        val_flag_2 = (points[:, :, 0] > regions_w * j) & (points[:, :, 0] <= regions_w * (j + 1))
        mask_split = val_flag_1 & val_flag_2
        mask_split = rearrange(mask_split, "f1 f2 b n -> (b f1 f2) n", f1=self.fold_h, f2=self.fold_w)
        points_split_origin = points.repeat(num_regions, 1, 1) * mask_split.unsqueeze(-1).repeat(1, 1, 2)
        mask_points_split_origin = torch.any(points_split_origin != 0, dim=-1, keepdim=True).to(torch.bool).cuda()  # [b*blocks, N, 1]
        mask_origin = mask_points_split_origin.squeeze(-1)
        bb, nn, cc = points_split_origin.shape
        points_split = torch.zeros(bb, nn//self.fold_h, cc)           # [b*blocks, n, 2]
        for batch in range(bb):
            points_batch_split = points_split_origin[batch:batch + 1, :][mask_origin[batch:batch + 1, :]][:, :2] # [n, 2]
            points_split[batch, :points_batch_split.shape[0], :] = points_batch_split  # [b*blocks, n, 2]
        mask_points_split = torch.any(points_split != 0, dim=-1, keepdim=False).to(torch.bool).cuda()  # [b*blocks, n]
        b, c, h, w = x.shape
        points_split[:, :, 0] = points_split[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
        points_split[:, :, 1] = points_split[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0
        centers = Feature_Gather(x, points_split)  # points是直接对x的feature gather
        value_centers = Feature_Gather(value, points_split)
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,n,H*W]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,n,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c h w -> b (h w) c')  # [B,H*W,D]
        output = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers.reshape(b, c, -1).permute(0, 2, 1)) / (
                sim.sum(dim=-1, keepdim=True) + 1.0)  # [b*blocks, n, D]
        out = torch.zeros(bb, nn, c).cuda()
        out[mask_origin] = output[mask_points_split]
        mask_points_split_origin = mask_points_split_origin.repeat(1, 1, out.shape[2])
        out = out * mask_points_split_origin
        # out = torch.where(out != 0, out, torch.tensor(float('-inf')).cuda())
        out = rearrange(out, "(b f1) n c -> b f1 n c", f1=num_regions)
        out = out.sum(dim=1)  # [B N C]
        # out, _ = torch.max(out, dim=1)      # [B N C]
        # out = torch.where(out != float('-inf'), out, torch.tensor(0.).float().cuda())
        mask_points_split_origin = torch.any(out != 0, dim=-1, keepdim=True).to(torch.bool).cuda().repeat(1, 1, out.shape[2])
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.proj(out)  # [B, C, 1, N]
        out = out.squeeze(2).permute(0, 2, 1)  # [B  N  C]
        out = out * mask_points_split_origin

        return out.permute(0, 2, 1).unsqueeze(2) # [B C 1 N]
