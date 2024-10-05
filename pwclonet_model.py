# -*- coding:UTF-8 -*-

"""
PointPWC-Net model and losses
Author: Wenxuan Wu
Date: May 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from conv_util import PointNetSaModule, cost_volume, set_upconv_module, FlowPredictor, Conv1d, BasicBlock, Feature_Gather
from pwclonet_model_utils import ProjectPCflagxy2SphericalRing, ProjectPCflag2SphericalRing,ProjectPC2SphericalRing, PreProcess, softmax_valid, quat2mat, inv_q, mul_q_point, mul_point_q
from context_cluster import Cluster
from cross_swin_transformer import Cross_BasicLayer
from Fusion_module import GlobalFuser
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange


scale = 1.0

def get_selected_idx(batch_size, out_H: int, out_W: int, stride_H: int, stride_W: int):
    """According to given stride and output size, return the corresponding selected points

    Args:
        array (tf.Tensor): [any array with shape (B, H, W, 3)]
        stride_H (int): [stride in height]
        stride_W (int): [stride in width]
        out_H (int): [height of output array]
        out_W (int): [width of output array]
    Returns:
        [tf.Tensor]: [shape (B, outh, outw, 3) indices]
    """
    select_h_idx = torch.arange(0, out_H * stride_H, stride_H)
    select_w_idx = torch.arange(0, out_W * stride_W, stride_W)
    height_indices = (torch.reshape(select_h_idx, (1, -1, 1))).expand(batch_size, out_H, out_W)         # b out_H out_W 
    width_indices = (torch.reshape(select_w_idx, (1, 1, -1))).expand(batch_size, out_H, out_W)            # b out_H out_W 
    padding_indices = torch.reshape(torch.arange(batch_size), (-1, 1, 1)).expand(batch_size, out_H, out_W)   # b out_H out_W 

    return padding_indices, height_indices, width_indices


class pwc_model(nn.Module):
    def __init__(self, batch_size, H_input, W_input, is_training, bn_decay=None):
        super(pwc_model, self).__init__()
        self.count = 0
        #####   initialize the parameters (distance  &  stride ) ######
        self.H_input = H_input; self.W_input = W_input

        self.Down_conv_dis = [0.75, 3.0, 6.0, 12.0]
        self.Up_conv_dis = [3.0, 6.0, 9.0]
        self.Cost_volume_dis = [1.0, 2.0, 4.5]

        self.stride_H_list = [4, 2, 2, 1]
        self.stride_W_list = [8, 2, 2, 2]
        self.length = [700, 250, 100, 70]

        self.out_H_list = [math.ceil(self.H_input / self.stride_H_list[0])]
        self.out_W_list = [math.ceil(self.W_input / self.stride_W_list[0])]

        for i in range(1, 4):
            self.out_H_list.append(math.ceil(self.out_H_list[i - 1] / self.stride_H_list[i]))
            self.out_W_list.append(math.ceil(self.out_W_list[i - 1] / self.stride_W_list[i]))  # generate the output shape list


        self.training = is_training
        self.w_x = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.w_q = torch.nn.Parameter(torch.tensor([-2.5]), requires_grad=True)


        self.layer0 = PointNetSaModule(batch_size = batch_size, K_sample = 32, kernel_size = [9, 15], H = self.out_H_list[0], W = self.out_W_list[0], \
                                       stride_H = self.stride_H_list[0], stride_W = self.stride_W_list[0], distance = self.Down_conv_dis[0], in_channels = 3,
                                       mlp = [8, 8, 16], is_training = self.training,
                                       bn_decay = bn_decay)  

        self.layer1 = PointNetSaModule(batch_size = batch_size, K_sample = 32, kernel_size = [7, 11], H = self.out_H_list[1], W = self.out_W_list[1], \
                                       stride_H = self.stride_H_list[1], stride_W = self.stride_W_list[1], distance = self.Down_conv_dis[1],
                                       in_channels = 16,
                                       mlp=[16, 16, 32], is_training=self.training,
                                       bn_decay = bn_decay) 

        self.layer2 = PointNetSaModule(batch_size = batch_size, K_sample = 16, kernel_size = [5, 9], H = self.out_H_list[2], W = self.out_W_list[2], \
                                       stride_H = self.stride_H_list[2], stride_W = self.stride_W_list[2], distance = self.Down_conv_dis[2],
                                       in_channels=32,
                                       mlp=[32, 32, 64], is_training=self.training,
                                       bn_decay=bn_decay)

        self.layer3 = PointNetSaModule(batch_size = batch_size, K_sample = 16, kernel_size = [5, 9], H = self.out_H_list[3], W = self.out_W_list[3], \
                                       stride_H = self.stride_H_list[3], stride_W = self.stride_W_list[3], distance = self.Down_conv_dis[3],
                                       in_channels=64,
                                       mlp=[64, 64, 128], is_training=self.training,
                                       bn_decay=bn_decay)  

        self.laye3_1 = PointNetSaModule(batch_size = batch_size, K_sample = 16, kernel_size = [5, 9], H = self.out_H_list[3], W = self.out_W_list[3], \
                                       stride_H = self.stride_H_list[3], stride_W = self.stride_W_list[3], distance = self.Down_conv_dis[3],
                                       in_channels=64,
                                       mlp=[128, 64, 64], is_training=self.training,
                                       bn_decay=bn_decay)  


        self.cost_volume1 = cost_volume(batch_size = batch_size, kernel_size1 = [3, 5], kernel_size2 = [5, 35] , nsample=4, nsample_q=32, \
                                       H = self.out_H_list[2], W = self.out_W_list[2], \
                                       stride_H = 1, stride_W = 1, distance = self.Cost_volume_dis[2],
                                       in_channels = [64, 64],
                                       mlp1=[128, 64, 64], mlp2=[128, 64], is_training=self.training, bn_decay=bn_decay,
                                       bn=True, pooling='max', knn=True, corr_func='concat')  
                                       
        self.cost_volume2 = cost_volume(batch_size = batch_size, kernel_size1 = [3, 5], kernel_size2 = [5, 15] , nsample=4, nsample_q = 6, \
                                       H = self.out_H_list[2], W = self.out_W_list[2], \
                                       stride_H = 1, stride_W = 1, distance = self.Cost_volume_dis[2],
                                       in_channels = [64, 64],
                                       mlp1=[128, 64, 64], mlp2=[128, 64], is_training=self.training, bn_decay=bn_decay,
                                       bn=True,
                                       pooling='max', knn=True, corr_func='concat')

        self.cost_volume3 = cost_volume(batch_size = batch_size, kernel_size1 = [3, 5], kernel_size2 = [7, 25] , nsample=4, nsample_q = 6, \
                                       H = self.out_H_list[1], W = self.out_W_list[1], \
                                       stride_H = 1, stride_W = 1, distance = self.Cost_volume_dis[1],
                                       in_channels = [32, 32],
                                       mlp1=[128, 64, 64], mlp2=[128, 64], is_training=self.training, bn_decay=bn_decay,
                                       bn=True,
                                       pooling='max', knn=True, corr_func='concat')  


        self.cost_volume4 = cost_volume(batch_size = batch_size, kernel_size1 = [3, 5], kernel_size2 = [11, 41] , nsample=4, nsample_q = 6, \
                                       H = self.out_H_list[0], W = self.out_W_list[0], \
                                       stride_H = 1, stride_W = 1, distance = self.Cost_volume_dis[0],
                                       in_channels = [16, 16],
                                       mlp1=[128, 64, 64], mlp2=[128, 64], is_training=self.training, bn_decay=bn_decay,
                                       bn=True,
                                       pooling='max', knn=True, corr_func='concat') 


        self.flow_predictor0 = FlowPredictor(in_channels=64 * 3, mlp=[128, 64], is_training=self.training,
                                             bn_decay=bn_decay)  
        self.flow_predictor1_predict = FlowPredictor(in_channels=64 * 3, mlp=[128, 64], is_training=self.training,
                                             bn_decay=bn_decay)  
        self.flow_predictor1_w = FlowPredictor(in_channels=64 * 3, mlp=[128, 64], is_training=self.training,
                                             bn_decay=bn_decay)  
        self.flow_predictor2_predict = FlowPredictor(in_channels=64 * 2 + 32, mlp=[128, 64], is_training=self.training,
                                             bn_decay=bn_decay)  
        self.flow_predictor2_w = FlowPredictor(in_channels=64 * 2 + 32, mlp=[128, 64], is_training=self.training,
                                             bn_decay=bn_decay)  
        self.flow_predictor3_predict = FlowPredictor(in_channels=64 * 2 + 16, mlp=[128, 64], is_training=self.training,
                                             bn_decay=bn_decay)  
        self.flow_predictor3_w = FlowPredictor(in_channels=64 * 2 + 16, mlp=[128, 64], is_training=self.training,
                                             bn_decay=bn_decay)  


        self.set_upconv1_w_upsample = set_upconv_module(batch_size = batch_size, kernel_size = [7, 15],
                                            H = self.out_H_list[2], W = self.out_W_list[2],
                                            stride_H = self.stride_H_list[-1], stride_W = self.stride_W_list[-1],
                                            nsample=8, distance = self.Up_conv_dis[2],
                                            in_channels=[64, 64],
                                            mlp=[128, 64], mlp2=[64], is_training=self.training,
                                            bn_decay=bn_decay, knn=True)  

        self.set_upconv1_upsample = set_upconv_module(batch_size = batch_size, kernel_size = [7, 15],
                                            H = self.out_H_list[2], W = self.out_W_list[2],                               
                                            stride_H = self.stride_H_list[-1], stride_W = self.stride_W_list[-1],
                                            nsample=8, distance = self.Up_conv_dis[2],
                                            in_channels=[64, 64],
                                            mlp=[128, 64], mlp2=[64], is_training=self.training,
                                            bn_decay=bn_decay, knn=True)  

        self.set_upconv2_w_upsample = set_upconv_module(batch_size = batch_size, kernel_size = [7, 15], 
                                            H = self.out_H_list[1], W = self.out_W_list[1],
                                            stride_H = self.stride_H_list[-2], stride_W = self.stride_W_list[-2], \
                                            nsample=8, distance = self.Up_conv_dis[1],
                                            in_channels=[32, 64],
                                            mlp=[128, 64], mlp2=[64], is_training=self.training,
                                            bn_decay=bn_decay, knn=True)  

        self.set_upconv2_upsample = set_upconv_module(batch_size = batch_size, kernel_size = [7, 15], 
                                            H = self.out_H_list[1], W = self.out_W_list[1],
                                            stride_H = self.stride_H_list[-2], stride_W = self.stride_W_list[-2], \
                                            nsample=8, distance = self.Up_conv_dis[1],
                                            in_channels=[32, 64],
                                            mlp=[128, 64], mlp2=[64], is_training=self.training,
                                            bn_decay=bn_decay, knn=True)  

        self.set_upconv3_w_upsample = set_upconv_module(batch_size = batch_size, kernel_size = [7, 15], 
                                            H = self.out_H_list[0], W = self.out_W_list[0],
                                            stride_H = self.stride_H_list[-3], stride_W = self.stride_W_list[-3], \
                                            nsample=8, distance = self.Up_conv_dis[0],
                                            in_channels=[16, 64],
                                            mlp=[128, 64], mlp2=[64], is_training=self.training,
                                            bn_decay=bn_decay, knn=True) 

        self.set_upconv3_upsample = set_upconv_module(batch_size = batch_size, kernel_size = [7, 15], 
                                            H = self.out_H_list[0], W = self.out_W_list[0],
                                            stride_H = self.stride_H_list[-3], stride_W = self.stride_W_list[-3], \
                                            nsample=8, distance = self.Up_conv_dis[0],
                                            in_channels=[16, 64],
                                            mlp=[128, 64], mlp2=[64], is_training=self.training,
                                            bn_decay=bn_decay, knn=True)  


        self.conv1_l3 = Conv1d(256, 4, use_activation=False)  
        self.conv1_l2 = Conv1d(256, 4, use_activation=False)  
        self.conv1_l1 = Conv1d(256, 4, use_activation=False)  
        self.conv1_l0 = Conv1d(256, 4, use_activation=False)  
        self.conv2_l3 = Conv1d(256, 3, use_activation=False)  
        self.conv2_l2 = Conv1d(256, 3, use_activation=False)  
        self.conv2_l1 = Conv1d(256, 3, use_activation=False)  
        self.conv2_l0 = Conv1d(256, 3, use_activation=False)  
        self.conv3_l3 = Conv1d(64, 256, use_activation=False)  
        self.conv3_l2 = Conv1d(64, 256, use_activation=False)  
        self.conv3_l1 = Conv1d(64, 256, use_activation=False)  
        self.conv3_l0 = Conv1d(64, 256, use_activation=False)

        # 图片卷积模块
        self.BasicBlock_0 = BasicBlock(3, 16, stride=2)
        self.BasicBlock_1 = BasicBlock(16, 32, stride=1)
        self.BasicBlock_2 = BasicBlock(32, 64, stride=1)
        self.BasicBlock_3 = BasicBlock(64, 128, stride=1)

        self.cluster_l0 = Cluster(dim=16, out_dim=16, fold_w=8, fold_h=8, heads=1, head_dim=16, return_center=True)
        self.cluster_l1 = Cluster(dim=32, out_dim=32, fold_w=4, fold_h=4, heads=1, head_dim=32, return_center=True)
        self.cluster_l2 = Cluster(dim=64, out_dim=64, fold_w=2, fold_h=2, heads=1, head_dim=64, return_center=True)
        self.cluster_l3 = Cluster(dim=128, out_dim=128, fold_w=1, fold_h=1, heads=1, head_dim=128, return_center=True)

        self.feat_fuserl0 = GlobalFuser(in_channels_2d=16, in_channels_3d=16)
        self.feat_fuserl1 = GlobalFuser(in_channels_2d=32, in_channels_3d=32)
        self.feat_fuserl2 = GlobalFuser(in_channels_2d=64, in_channels_3d=64)
        self.feat_fuserl3 = GlobalFuser(in_channels_2d=128, in_channels_3d=128)

    def forward(self, input_xyz_f1, input_xyz_f2, input_img_f1, input_img_f2, input_xy_f1, input_xy_f2, T_gt, T_trans, T_trans_inv, pts_valid_flag_f1, pts_valid_flag_f2):


        start_train = time.time()

        batch_size = input_xyz_f1.shape[0]

        input_points_proj_f1 = torch.zeros(batch_size, self.H_input, self.W_input, 3).cuda().detach()
        input_points_proj_f2 = torch.zeros(batch_size, self.H_input, self.W_input, 3).cuda().detach()


        self.l0_b_idx, self.l0_h_idx, self.l0_w_idx = get_selected_idx( batch_size, self.out_H_list[0], self.out_W_list[0], self.stride_H_list[0], self.stride_W_list[0] )
        self.l1_b_idx, self.l1_h_idx, self.l1_w_idx = get_selected_idx( batch_size, self.out_H_list[1], self.out_W_list[1], self.stride_H_list[1], self.stride_W_list[1] )
        self.l2_b_idx, self.l2_h_idx, self.l2_w_idx = get_selected_idx( batch_size, self.out_H_list[2], self.out_W_list[2], self.stride_H_list[2], self.stride_W_list[2] )
        self.l3_b_idx, self.l3_h_idx, self.l3_w_idx = get_selected_idx( batch_size, self.out_H_list[3], self.out_W_list[3], self.stride_H_list[3], self.stride_W_list[3] )

        torch.cuda.synchronize()
        start_time = time.time()

        aug_frame = np.random.choice([1, 2], size = batch_size, replace = True) # random choose aug frame 1 or 2
        input_xyz_aug_f1, input_xyz_aug_f2, q_gt, t_gt = PreProcess(input_xyz_f1, input_xyz_f2, T_gt, T_trans, T_trans_inv, aug_frame)

        # input_xyz_aug_proj_f1 = ProjectPC2SphericalRing(input_xyz_aug_f1, None, self.H_input, self.W_input)  ## proj func
        # input_xyz_aug_proj_f2 = ProjectPC2SphericalRing(input_xyz_aug_f2, None, self.H_input, self.W_input)

        input_xyz_aug_proj_f1, input_flag_proj_f1, input_xy_proj_f1 = ProjectPCflagxy2SphericalRing(input_xyz_aug_f1, pts_valid_flag_f1, input_xy_f1, None, self.H_input,self.W_input)  ## proj func
        input_xyz_aug_proj_f2, input_flag_proj_f2, input_xy_proj_f2 = ProjectPCflagxy2SphericalRing(input_xyz_aug_f2, pts_valid_flag_f2, input_xy_f2, None, self.H_input, self.W_input)
        
    
        # print('data_pre_process_proj+aug: ', time.time() - start_time)


        ####  the l0 select bn3 xyz

        l0_xyz_proj_f1 = input_xyz_aug_proj_f1[self.l0_b_idx.cuda().long(), self.l0_h_idx.cuda().long(), self.l0_w_idx.cuda().long(), :]
        l0_xyz_proj_f2 = input_xyz_aug_proj_f2[self.l0_b_idx.cuda().long(), self.l0_h_idx.cuda().long(), self.l0_w_idx.cuda().long(), :]
        l0_flag_proj_f1 = input_flag_proj_f1[self.l0_b_idx.cuda().long(), self.l0_h_idx.cuda().long(), self.l0_w_idx.cuda().long(), :]
        l0_flag_proj_f2 = input_flag_proj_f2[self.l0_b_idx.cuda().long(), self.l0_h_idx.cuda().long(), self.l0_w_idx.cuda().long(), :]
        l0_xy_proj_f1 = input_xy_proj_f1[self.l0_b_idx.cuda().long(), self.l0_h_idx.cuda().long(), self.l0_w_idx.cuda().long(), :]
        l0_xy_proj_f2 = input_xy_proj_f2[self.l0_b_idx.cuda().long(), self.l0_h_idx.cuda().long(), self.l0_w_idx.cuda().long(), :]

        ####  the l1 select bn3 xyz

        l1_xyz_proj_f1 = l0_xyz_proj_f1[self.l1_b_idx.cuda().long(), self.l1_h_idx.cuda().long(), self.l1_w_idx.cuda().long(), :]
        l1_xyz_proj_f2 = l0_xyz_proj_f2[self.l1_b_idx.cuda().long(), self.l1_h_idx.cuda().long(), self.l1_w_idx.cuda().long(), :]
        l1_flag_proj_f1 = l0_flag_proj_f1[self.l1_b_idx.cuda().long(), self.l1_h_idx.cuda().long(), self.l1_w_idx.cuda().long(), :]
        l1_flag_proj_f2 = l0_flag_proj_f2[self.l1_b_idx.cuda().long(), self.l1_h_idx.cuda().long(), self.l1_w_idx.cuda().long(), :]
        l1_xy_proj_f1 = l0_xy_proj_f1[self.l1_b_idx.cuda().long(), self.l1_h_idx.cuda().long(), self.l1_w_idx.cuda().long(), :]
        l1_xy_proj_f2 = l0_xy_proj_f2[self.l1_b_idx.cuda().long(), self.l1_h_idx.cuda().long(), self.l1_w_idx.cuda().long(), :]

        ####  the l2 select bn3 xyz
        
        l2_xyz_proj_f1 = l1_xyz_proj_f1[self.l2_b_idx.cuda().long(), self.l2_h_idx.cuda().long(), self.l2_w_idx.cuda().long(), :]
        l2_xyz_proj_f2 = l1_xyz_proj_f2[self.l2_b_idx.cuda().long(), self.l2_h_idx.cuda().long(), self.l2_w_idx.cuda().long(), :]
        l2_flag_proj_f1 = l1_flag_proj_f1[self.l2_b_idx.cuda().long(), self.l2_h_idx.cuda().long(), self.l2_w_idx.cuda().long(), :]
        l2_flag_proj_f2 = l1_flag_proj_f2[self.l2_b_idx.cuda().long(), self.l2_h_idx.cuda().long(), self.l2_w_idx.cuda().long(), :]
        l2_xy_proj_f1 = l1_xy_proj_f1[self.l2_b_idx.cuda().long(), self.l2_h_idx.cuda().long(), self.l2_w_idx.cuda().long(), :]
        l2_xy_proj_f2 = l1_xy_proj_f2[self.l2_b_idx.cuda().long(), self.l2_h_idx.cuda().long(), self.l2_w_idx.cuda().long(), :]

        ####  the l3 select bn3 xyz

        l3_xyz_proj_f1 = l2_xyz_proj_f1[self.l3_b_idx.cuda().long(), self.l3_h_idx.cuda().long(), self.l3_w_idx.cuda().long(), :]
        l3_xyz_proj_f2 = l2_xyz_proj_f2[self.l3_b_idx.cuda().long(), self.l3_h_idx.cuda().long(), self.l3_w_idx.cuda().long(), :]
        l3_flag_proj_f1 = l2_flag_proj_f1[self.l3_b_idx.cuda().long(), self.l3_h_idx.cuda().long(), self.l3_w_idx.cuda().long(), :]
        l3_flag_proj_f2 = l2_flag_proj_f2[self.l3_b_idx.cuda().long(), self.l3_h_idx.cuda().long(), self.l3_w_idx.cuda().long(), :]
        l3_xy_proj_f1 = l2_xy_proj_f1[self.l3_b_idx.cuda().long(), self.l3_h_idx.cuda().long(), self.l3_w_idx.cuda().long(), :]
        l3_xy_proj_f2 = l2_xy_proj_f2[self.l3_b_idx.cuda().long(), self.l3_h_idx.cuda().long(), self.l3_w_idx.cuda().long(), :]

        # print('pre_process: ', time.time() - start_train)

        set_conv_start = time.time()

        l0_mask_f1 = l0_flag_proj_f1.squeeze(-1).to(torch.bool)
        l0_mask_f2 = l0_flag_proj_f2.squeeze(-1).to(torch.bool)
        l1_mask_f1 = l1_flag_proj_f1.squeeze(-1).to(torch.bool)
        l1_mask_f2 = l1_flag_proj_f2.squeeze(-1).to(torch.bool)
        l2_mask_f1 = l2_flag_proj_f1.squeeze(-1).to(torch.bool)
        l2_mask_f2 = l2_flag_proj_f2.squeeze(-1).to(torch.bool)
        l3_mask_f1 = l3_flag_proj_f1.squeeze(-1).to(torch.bool)
        l3_mask_f2 = l3_flag_proj_f2.squeeze(-1).to(torch.bool)

        ## flame 1
        # layer 0
        l0_points_f1, l0_points_proj_f1 = self.layer0(input_xyz_aug_proj_f1, input_points_proj_f1, l0_xyz_proj_f1)
        image0_f1 = self.BasicBlock_0(input_img_f1.to(torch.float32))
        b = l0_points_proj_f1.shape[0]
        h = l0_points_proj_f1.shape[1]
        w = l0_points_proj_f1.shape[2]
        c = l0_points_proj_f1.shape[3]

        l0_xy_cor_f1 = torch.zeros(b, self.length[0], 2)
        l0_proj_cor_f1 = torch.zeros(b, self.length[0], c)
        start_time = time.time()
        for batch in range(batch_size):
            l0_batch_xy_f1 = l0_xy_proj_f1[batch:batch + 1, :, :][l0_mask_f1[batch:batch + 1, :, :]][:, :2]  # [N, 2]
            l0_xy_cor_f1[batch, :l0_batch_xy_f1.shape[0], :] = l0_batch_xy_f1  # [B, N, 2]
            l0_batch_proj_f1 = l0_points_proj_f1[batch:batch + 1, :, :][l0_mask_f1[batch:batch + 1, :, :]][:,
                               :]  # [N, C]
            l0_proj_cor_f1[batch, :l0_batch_proj_f1.shape[0], :] = l0_batch_proj_f1  # [B, N, C]

        mask_valid_xy_f1 = torch.any(l0_xy_cor_f1 != 0, dim=-1, keepdim=True).squeeze(-1).to(torch.bool)
        l0_points_origin_f1 = l0_points_proj_f1.clone().cuda()
        l0_img_gather_feature_full_f1 = self.cluster_l0(l0_xy_cor_f1, image0_f1)  # [B, C, 1, N]
        l0_img_gather_feature_full_f1 = l0_img_gather_feature_full_f1.squeeze(2).permute(0, 2, 1)  # [B, N, C]
        l0_points_proj_f1[l0_mask_f1] = l0_img_gather_feature_full_f1[mask_valid_xy_f1]
        l0_points_proj_f1 = self.feat_fuserl0(l0_points_proj_f1, l0_points_origin_f1).permute(0, 2, 3, 1)  # [B,h,w,c]

        # layer 1
        l1_points_f1, l1_points_proj_f1 = self.layer1(l0_xyz_proj_f1, l0_points_proj_f1, l1_xyz_proj_f1)
        image1_f1 = self.BasicBlock_1(image0_f1.to(torch.float32))
        b = l1_points_proj_f1.shape[0]
        h = l1_points_proj_f1.shape[1]
        w = l1_points_proj_f1.shape[2]
        c = l1_points_proj_f1.shape[3]

        l1_xy_cor_f1 = torch.zeros(b, self.length[1], 2)
        l1_proj_cor_f1 = torch.zeros(b, self.length[1], c)
        for batch in range(batch_size):
            l1_batch_xy_f1 = l1_xy_proj_f1[batch:batch + 1, :, :][l1_mask_f1[batch:batch + 1, :, :]][:, :2]  # [N, 2]
            l1_xy_cor_f1[batch, :l1_batch_xy_f1.shape[0], :] = l1_batch_xy_f1  # [B, N, 2]
            l1_batch_proj_f1 = l1_points_proj_f1[batch:batch + 1, :, :][l1_mask_f1[batch:batch + 1, :, :]][:,
                               :]  # [N, C]
            l1_proj_cor_f1[batch, :l1_batch_proj_f1.shape[0], :] = l1_batch_proj_f1  # [B, N, C]

        mask_valid_xy_f1 = torch.any(l1_xy_cor_f1 != 0, dim=-1, keepdim=True).squeeze(-1).to(torch.bool)

        l1_points_origin_f1 = l1_points_proj_f1.clone().cuda()
        l1_img_gather_feature_full_f1 = self.cluster_l1(l1_xy_cor_f1, image1_f1)
        l1_img_gather_feature_full_f1 = l1_img_gather_feature_full_f1.permute(0, 2, 3, 1).squeeze(1)
        l1_points_proj_f1[l1_mask_f1] = l1_img_gather_feature_full_f1[mask_valid_xy_f1]
        l1_points_proj_f1 = self.feat_fuserl1(l1_points_proj_f1, l1_points_origin_f1).permute(0, 2, 3, 1)  # [B,h,w,c]

        # layer 2
        l2_points_f1, l2_points_proj_f1 = self.layer2(l1_xyz_proj_f1, l1_points_proj_f1, l2_xyz_proj_f1)
        image2_f1 = self.BasicBlock_2(image1_f1.to(torch.float32))
        b = l2_points_proj_f1.shape[0]
        h = l2_points_proj_f1.shape[1]
        w = l2_points_proj_f1.shape[2]
        c = l2_points_proj_f1.shape[3]

        l2_xy_cor_f1 = torch.zeros(b, self.length[2], 2)
        l2_proj_cor_f1 = torch.zeros(b, self.length[2], c)
        for batch in range(batch_size):
            l2_batch_xy_f1 = l2_xy_proj_f1[batch:batch + 1, :, :][l2_mask_f1[batch:batch + 1, :, :]][:, :2]  # [N, 2]
            l2_xy_cor_f1[batch, :l2_batch_xy_f1.shape[0], :] = l2_batch_xy_f1  # [B, N, 2]
            l2_batch_proj_f1 = l2_points_proj_f1[batch:batch + 1, :, :][l2_mask_f1[batch:batch + 1, :, :]][:,
                               :]  # [N, C]
            l2_proj_cor_f1[batch, :l2_batch_proj_f1.shape[0], :] = l2_batch_proj_f1  # [B, N, C]

        mask_valid_xy_f1 = torch.any(l2_xy_cor_f1 != 0, dim=-1, keepdim=True).squeeze(-1).to(torch.bool)
        l2_points_origin_f1 = l2_points_proj_f1.clone().cuda()
        l2_img_gather_feature_full_f1 = self.cluster_l2(l2_xy_cor_f1, image2_f1)
        l2_img_gather_feature_full_f1 = l2_img_gather_feature_full_f1.permute(0, 2, 3, 1).squeeze(1)
        l2_points_proj_f1[l2_mask_f1] = l2_img_gather_feature_full_f1[mask_valid_xy_f1]
        l2_points_proj_f1 = self.feat_fuserl2(l2_points_proj_f1, l2_points_origin_f1).permute(0, 2, 3,
                                                                                              1)  # [B, C, 1, N]

        # layer 3
        l3_points_f1, l3_points_proj_f1 = self.layer3(l2_xyz_proj_f1, l2_points_proj_f1, l3_xyz_proj_f1)
        image3_f1 = self.BasicBlock_3(image2_f1.to(torch.float32))
        b = l3_points_proj_f1.shape[0]
        h = l3_points_proj_f1.shape[1]
        w = l3_points_proj_f1.shape[2]
        c = l3_points_proj_f1.shape[3]

        l3_xy_cor_f1 = torch.zeros(b, self.length[3], 2)
        l3_proj_cor_f1 = torch.zeros(b, self.length[3], c)
        for batch in range(batch_size):
            l3_batch_xy_f1 = l3_xy_proj_f1[batch:batch + 1, :, :][l3_mask_f1[batch:batch + 1, :, :]][:, :2]  # [N, 2]
            l3_xy_cor_f1[batch, :l3_batch_xy_f1.shape[0], :] = l3_batch_xy_f1  # [B, N, 2]
            l3_batch_proj_f1 = l3_points_proj_f1[batch:batch + 1, :, :][l3_mask_f1[batch:batch + 1, :, :]][:,
                               :]  # [N, C]
            l3_proj_cor_f1[batch, :l3_batch_proj_f1.shape[0], :] = l3_batch_proj_f1  # [B, N, C]
        mask_valid_xy_f1 = torch.any(l3_xy_cor_f1 != 0, dim=-1, keepdim=True).squeeze(-1).to(torch.bool)
        l3_points_origin_f1 = l3_points_proj_f1.clone().cuda()
        l3_img_gather_feature_full_f1 = self.cluster_l3(l3_xy_cor_f1, image3_f1)
        l3_img_gather_feature_full_f1 = l3_img_gather_feature_full_f1.permute(0, 2, 3, 1).squeeze(1)
        l3_points_proj_f1[l3_mask_f1] = l3_img_gather_feature_full_f1[mask_valid_xy_f1]
        l3_points_proj_f1 = self.feat_fuserl3(l3_points_proj_f1, l3_points_origin_f1).permute(0, 2, 3, 1)

        ## flame 2
        # layer 0
        l0_points_f2, l0_points_proj_f2 = self.layer0(input_xyz_aug_proj_f2, input_points_proj_f2, l0_xyz_proj_f2)
        image0_f2 = self.BasicBlock_0(input_img_f2.to(torch.float32))
        b = l0_points_proj_f2.shape[0]
        h = l0_points_proj_f2.shape[1]
        w = l0_points_proj_f2.shape[2]
        c = l0_points_proj_f2.shape[3]

        l0_xy_cor_f2 = torch.zeros(b, self.length[0], 2)
        l0_proj_cor_f2 = torch.zeros(b, self.length[0], c)
        for batch in range(batch_size):
            l0_batch_xy_f2 = l0_xy_proj_f2[batch:batch + 1, :, :][l0_mask_f2[batch:batch + 1, :, :]][:, :2]  # [N, 2]
            l0_xy_cor_f2[batch, :l0_batch_xy_f2.shape[0], :] = l0_batch_xy_f2  # [B, N, 2]
            l0_batch_proj_f2 = l0_points_proj_f2[batch:batch + 1, :, :][l0_mask_f2[batch:batch + 1, :, :]][:,
                               :]  # [N, C]
            l0_proj_cor_f2[batch, :l0_batch_proj_f2.shape[0], :] = l0_batch_proj_f2  # [B, N, C]

        mask_valid_xy_f2 = torch.any(l0_xy_cor_f2 != 0, dim=-1, keepdim=True).squeeze(-1).to(torch.bool)

        l0_points_origin_f2 = l0_points_proj_f2.clone().cuda()
        l0_img_gather_feature_full_f2 = self.cluster_l0(l0_xy_cor_f2, image0_f2)
        l0_img_gather_feature_full_f2 = l0_img_gather_feature_full_f2.permute(0, 2, 3, 1).squeeze(1)
        l0_points_proj_f2[l0_mask_f2] = l0_img_gather_feature_full_f2[mask_valid_xy_f2]
        l0_points_proj_f2 = self.feat_fuserl0(l0_points_proj_f2, l0_points_origin_f2).permute(0, 2, 3,
                                                                                              1)  # [B, C, 1, N]

        # layer 1
        l1_points_f2, l1_points_proj_f2 = self.layer1(l0_xyz_proj_f2, l0_points_proj_f2, l1_xyz_proj_f2)
        image1_f2 = self.BasicBlock_1(image0_f2.to(torch.float32))
        b = l1_points_proj_f2.shape[0]
        h = l1_points_proj_f2.shape[1]
        w = l1_points_proj_f2.shape[2]
        c = l1_points_proj_f2.shape[3]

        l1_xy_cor_f2 = torch.zeros(b, self.length[1], 2)
        l1_proj_cor_f2 = torch.zeros(b, self.length[1], c)
        for batch in range(batch_size):
            l1_batch_xy_f2 = l1_xy_proj_f2[batch:batch + 1, :, :][l1_mask_f2[batch:batch + 1, :, :]][:, :2]  # [N, 2]
            l1_xy_cor_f2[batch, :l1_batch_xy_f2.shape[0], :] = l1_batch_xy_f2  # [B, N, 2]
            l1_batch_proj_f2 = l1_points_proj_f2[batch:batch + 1, :, :][l1_mask_f2[batch:batch + 1, :, :]][:,
                               :]  # [N, C]
            l1_proj_cor_f2[batch, :l1_batch_proj_f2.shape[0], :] = l1_batch_proj_f2  # [B, N, C]

        mask_valid_xy_f2 = torch.any(l1_xy_cor_f2 != 0, dim=-1, keepdim=True).squeeze(-1).to(torch.bool)

        l1_points_origin_f2 = l1_points_proj_f2.clone().cuda()
        l1_img_gather_feature_full_f2 = self.cluster_l1(l1_xy_cor_f2, image1_f2)
        l1_img_gather_feature_full_f2 = l1_img_gather_feature_full_f2.permute(0, 2, 3, 1).squeeze(1)
        l1_points_proj_f2[l1_mask_f2] = l1_img_gather_feature_full_f2[mask_valid_xy_f2]
        l1_points_proj_f2 = self.feat_fuserl1(l1_points_proj_f2, l1_points_origin_f2).permute(0, 2, 3,
                                                                                              1)  # [B, C, 1, N]

        # layer 2
        l2_points_f2, l2_points_proj_f2 = self.layer2(l1_xyz_proj_f2, l1_points_proj_f2, l2_xyz_proj_f2)
        image2_f2 = self.BasicBlock_2(image1_f2.to(torch.float32))
        b = l2_points_proj_f2.shape[0]
        h = l2_points_proj_f2.shape[1]
        w = l2_points_proj_f2.shape[2]
        c = l2_points_proj_f2.shape[3]

        l2_xy_cor_f2 = torch.zeros(b, self.length[2], 2)
        l2_proj_cor_f2 = torch.zeros(b, self.length[2], c)
        for batch in range(batch_size):
            l2_batch_xy_f2 = l2_xy_proj_f2[batch:batch + 1, :, :][l2_mask_f2[batch:batch + 1, :, :]][:, :2]  # [N, 2]
            l2_xy_cor_f2[batch, :l2_batch_xy_f2.shape[0], :] = l2_batch_xy_f2  # [B, N, 2]
            l2_batch_proj_f2 = l2_points_proj_f2[batch:batch + 1, :, :][l2_mask_f2[batch:batch + 1, :, :]][:,
                               :]  # [N, C]
            l2_proj_cor_f2[batch, :l2_batch_proj_f2.shape[0], :] = l2_batch_proj_f2  # [B, N, C]

        mask_valid_xy_f2 = torch.any(l2_xy_cor_f2 != 0, dim=-1, keepdim=True).squeeze(-1).to(torch.bool)
        l2_points_origin_f2 = l2_points_proj_f2.clone().cuda()
        l2_img_gather_feature_full_f2 = self.cluster_l2(l2_xy_cor_f2, image2_f2)
        l2_img_gather_feature_full_f2 = l2_img_gather_feature_full_f2.permute(0, 2, 3, 1).squeeze(1)
        l2_points_proj_f2[l2_mask_f2] = l2_img_gather_feature_full_f2[mask_valid_xy_f2]
        l2_points_proj_f2 = self.feat_fuserl2(l2_points_proj_f2, l2_points_origin_f2).permute(0, 2, 3, 1)

        # layer 3
        l3_points_f2, l3_points_proj_f2 = self.layer3(l2_xyz_proj_f2, l2_points_proj_f2, l3_xyz_proj_f2)
        image3_f2 = self.BasicBlock_3(image2_f2.to(torch.float32))
        b = l3_points_proj_f2.shape[0]
        h = l3_points_proj_f2.shape[1]
        w = l3_points_proj_f2.shape[2]
        c = l3_points_proj_f2.shape[3]

        l3_xy_cor_f2 = torch.zeros(b, self.length[3], 2)
        l3_proj_cor_f2 = torch.zeros(b, self.length[3], c)
        for batch in range(batch_size):
            l3_batch_xy_f2 = l3_xy_proj_f2[batch:batch + 1, :, :][l3_mask_f2[batch:batch + 1, :, :]][:, :2]  # [N, 2]
            l3_xy_cor_f2[batch, :l3_batch_xy_f2.shape[0], :] = l3_batch_xy_f2  # [B, N, 2]
            l3_batch_proj_f2 = l3_points_proj_f2[batch:batch + 1, :, :][l3_mask_f2[batch:batch + 1, :, :]][:,
                               :]  # [N, C]
            l3_proj_cor_f2[batch, :l3_batch_proj_f2.shape[0], :] = l3_batch_proj_f2  # [B, N, C]

        mask_valid_xy_f2 = torch.any(l3_xy_cor_f2 != 0, dim=-1, keepdim=True).squeeze(-1).to(torch.bool)

        l3_points_origin_f2 = l3_points_proj_f2.clone().cuda()
        l3_img_gather_feature_full_f2 = self.cluster_l3(l3_xy_cor_f2, image3_f2)
        l3_img_gather_feature_full_f2 = l3_img_gather_feature_full_f2.permute(0, 2, 3, 1).squeeze(1)
        l3_points_proj_f2[l3_mask_f2] = l3_img_gather_feature_full_f2[mask_valid_xy_f2]
        l3_points_proj_f2 = self.feat_fuserl3(l3_points_proj_f2, l3_points_origin_f2).permute(0, 2, 3, 1)

        l2_cost_volume_origin = self.cost_volume1(l2_xyz_proj_f1, l2_xyz_proj_f2, l2_points_proj_f1, l2_points_proj_f2)
        l2_cost_volume_origin_proj = torch.reshape(l2_cost_volume_origin,  [batch_size, self.out_H_list[2], self.out_W_list[2], -1])

        # Layer 3 ##################
        
        l3_cost_volume, l3_cost_volume_proj = self.laye3_1(l2_xyz_proj_f1, l2_cost_volume_origin_proj, l3_xyz_proj_f1)
        l3_cost_volume_w = self.flow_predictor0(l3_points_f1, None, l3_cost_volume)
        l3_cost_volume_w_proj = torch.reshape(l3_cost_volume_w, [batch_size, self.out_H_list[3], self.out_W_list[3], -1])


        l3_xyz_f1 = torch.reshape(l3_xyz_proj_f1, [batch_size, -1, 3])
        mask_l3 = torch.any(l3_xyz_f1 != 0, dim = -1)

        l3_points_f1_new = softmax_valid(feature_bnc = l3_cost_volume, weight_bnc = l3_cost_volume_w, mask_valid = mask_l3)  # B 1 C

        l3_points_f1_new_big = self.conv3_l3(l3_points_f1_new)
        l3_points_f1_new_q = F.dropout(l3_points_f1_new_big, p = 0.5, training = self.training)
        l3_points_f1_new_t = F.dropout(l3_points_f1_new_big, p = 0.5, training = self.training)

        l3_q_coarse = self.conv1_l3(l3_points_f1_new_q)
        l3_q_coarse = l3_q_coarse / (torch.sqrt(torch.sum(l3_q_coarse * l3_q_coarse, dim=-1, keepdim=True) + 1e-10) + 1e-10)

        l3_t_coarse = self.conv2_l3(l3_points_f1_new_t)

        l3_q = torch.squeeze(l3_q_coarse, dim=1)
        l3_t = torch.squeeze(l3_t_coarse, dim=1)

        ################ layer 2 #################

        l2_q_coarse = torch.reshape(l3_q, [batch_size, 1, -1])
        l2_t_coarse = torch.reshape(l3_t, [batch_size, 1, -1])
        l2_q_inv = inv_q(l2_q_coarse, batch_size)

        ### warp layer2 pose

        l2_xyz_f1 = torch.reshape(l2_xyz_proj_f1, [batch_size, -1, 3])
        l2_xyz_bnc_q = torch.cat([torch.zeros([batch_size, self.out_H_list[2] * self.out_W_list[2], 1]).cuda(), l2_xyz_f1], dim=-1)

        l2_flow_warped = mul_q_point(l2_q_coarse, l2_xyz_bnc_q, batch_size)
        l2_flow_warped = torch.index_select(mul_point_q(l2_flow_warped, l2_q_inv, batch_size), 2, torch.LongTensor(range(1, 4)).cuda()) + l2_t_coarse

        l2_mask = torch.any(l2_xyz_f1 !=0, dim = -1, keepdim = True).to(torch.float32)
        l2_flow_warped = l2_flow_warped * l2_mask


        ### re-project 改成新方法

        l2_xyz_warp_proj_f1, l2_points_warp_proj_f1 = ProjectPC2SphericalRing(l2_flow_warped, l2_points_f1, self.out_H_list[2], self.out_W_list[2])  # 
        l2_xyz_warp_f1 = torch.reshape(l2_xyz_warp_proj_f1, [batch_size, -1, 3])
        l2_points_warp_f1 = torch.reshape(l2_points_warp_proj_f1, [batch_size, self.out_H_list[2] * self.out_W_list[2], -1])

        l2_mask_warped = torch.any(l2_xyz_warp_f1 !=0, dim = -1, keepdim = False)


        # get the cost volume of warped layer3 flow and the points of frame2
        l2_cost_volume = self.cost_volume2(l2_xyz_warp_proj_f1, l2_xyz_proj_f2, l2_points_warp_proj_f1, l2_points_proj_f2)

        l2_cost_volume_w_upsample = self.set_upconv1_w_upsample(l2_xyz_warp_proj_f1, l3_xyz_proj_f1, l2_points_warp_proj_f1, l3_cost_volume_w_proj)
        l2_cost_volume_upsample = self.set_upconv1_upsample(l2_xyz_warp_proj_f1, l3_xyz_proj_f1, l2_points_warp_proj_f1, l3_cost_volume_proj)
        
        l2_cost_volume_predict = self.flow_predictor1_predict(l2_points_warp_f1, l2_cost_volume_upsample, l2_cost_volume)
        l2_cost_volume_w = self.flow_predictor1_w(l2_points_warp_f1, l2_cost_volume_w_upsample, l2_cost_volume)

        l2_cost_volume_proj = torch.reshape(l2_cost_volume_predict, [batch_size, self.out_H_list[2], self.out_W_list[2], -1])
        l2_cost_volume_w_proj = torch.reshape(l2_cost_volume_w, [batch_size, self.out_H_list[2], self.out_W_list[2], -1])

        l2_cost_volume_sum = softmax_valid(feature_bnc = l2_cost_volume_predict, weight_bnc = l2_cost_volume_w, mask_valid = l2_mask_warped)  # B 1 C


        l2_points_f1_new_big = self.conv3_l2(l2_cost_volume_sum)
        l2_points_f1_new_q = F.dropout(l2_points_f1_new_big, p = 0.5, training = self.training)
        l2_points_f1_new_t = F.dropout(l2_points_f1_new_big, p = 0.5, training = self.training)

        l2_q_det = self.conv1_l2(l2_points_f1_new_q)
        l2_q_det = l2_q_det / (torch.sqrt(torch.sum(l2_q_det * l2_q_det, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l2_t_det = self.conv2_l2(l2_points_f1_new_t)

        l2_q_det_inv = inv_q(l2_q_det, batch_size)

        l2_t_coarse_trans = torch.cat([torch.zeros([batch_size, 1, 1]).cuda(), l2_t_coarse], dim=-1)
        l2_t_coarse_trans = mul_q_point(l2_q_det, l2_t_coarse_trans, batch_size)
        l2_t_coarse_trans = torch.index_select(mul_point_q(l2_t_coarse_trans, l2_q_det_inv, batch_size), 2,
                                                  torch.LongTensor(range(1, 4)).cuda())

        l2_q = torch.squeeze(mul_point_q(l2_q_det, l2_q_coarse, batch_size), dim=1)
        l2_t = torch.squeeze(l2_t_coarse_trans + l2_t_det, dim=1)



        ############# layer1
        start_l1_refine = time.time()

        l1_q_coarse = torch.reshape(l2_q, [batch_size, 1, -1])
        l1_t_coarse = torch.reshape(l2_t, [batch_size, 1, -1])
        l1_q_inv = inv_q(l1_q_coarse, batch_size)

        ############# warp layer2 pose

        l1_xyz_f1 = torch.reshape(l1_xyz_proj_f1, [batch_size, -1, 3])
        l1_xyz_bnc_q = torch.cat([torch.zeros([batch_size, self.out_H_list[1] * self.out_W_list[1], 1]).cuda(), l1_xyz_f1], dim=-1)

        l1_flow_warped = mul_q_point(l1_q_coarse, l1_xyz_bnc_q, batch_size)
        l1_flow_warped = torch.index_select(mul_point_q(l1_flow_warped, l1_q_inv, batch_size), 2, torch.LongTensor(range(1, 4)).cuda()) + l1_t_coarse

        l1_mask = torch.any(l1_xyz_f1 !=0, dim = -1, keepdim = True).to(torch.float32)
        l1_flow_warped = l1_flow_warped * l1_mask


        ########## re-project

        l1_xyz_warp_proj_f1, l1_points_warp_proj_f1 = ProjectPC2SphericalRing(l1_flow_warped, l1_points_f1, self.out_H_list[1], self.out_W_list[1])  # 
        l1_xyz_warp_f1 = torch.reshape(l1_xyz_warp_proj_f1, [batch_size, -1, 3])
        l1_points_warp_f1 = torch.reshape(l1_points_warp_proj_f1, [batch_size, self.out_H_list[1] * self.out_W_list[1], -1])

        l1_mask_warped = torch.any(l1_xyz_warp_f1 !=0, dim = -1, keepdim = False)


        # get the cost volume of warped layer3 flow and the points of frame2
        l1_cost_volume = self.cost_volume3(l1_xyz_warp_proj_f1, l1_xyz_proj_f2, l1_points_warp_proj_f1, l1_points_proj_f2)

        l1_cost_volume_w_upsample = self.set_upconv2_w_upsample(l1_xyz_warp_proj_f1, l2_xyz_warp_proj_f1, l1_points_warp_proj_f1, l2_cost_volume_w_proj)
        l1_cost_volume_upsample = self.set_upconv2_upsample(l1_xyz_warp_proj_f1, l2_xyz_warp_proj_f1, l1_points_warp_proj_f1, l2_cost_volume_proj)
        
        l1_cost_volume_predict = self.flow_predictor2_predict(l1_points_warp_f1, l1_cost_volume_upsample, l1_cost_volume)
        l1_cost_volume_w = self.flow_predictor2_w(l1_points_warp_f1, l1_cost_volume_w_upsample, l1_cost_volume)

        l1_cost_volume_proj = torch.reshape(l1_cost_volume_predict, [batch_size, self.out_H_list[1], self.out_W_list[1], -1])
        l1_cost_volume_w_proj = torch.reshape(l1_cost_volume_w, [batch_size, self.out_H_list[1], self.out_W_list[1], -1])


        l1_cost_volume_sum = softmax_valid(feature_bnc = l1_cost_volume_predict, weight_bnc = l1_cost_volume_w, mask_valid = l1_mask_warped)  # B 1 C

        l1_points_f1_new_big = self.conv3_l1(l1_cost_volume_sum)
        l1_points_f1_new_q = F.dropout(l1_points_f1_new_big, p = 0.5, training = self.training)
        l1_points_f1_new_t = F.dropout(l1_points_f1_new_big, p = 0.5, training = self.training)

        l1_q_det = self.conv1_l1(l1_points_f1_new_q)
        l1_q_det = l1_q_det / (torch.sqrt(torch.sum(l1_q_det * l1_q_det, dim=-1, keepdim=True) + 1e-10) + 1e-10)

        l1_t_det = self.conv2_l1(l1_points_f1_new_t)

        l1_q_det_inv = inv_q(l1_q_det, batch_size)

        l1_t_coarse_trans = torch.cat([torch.zeros([batch_size, 1, 1]).cuda(), l1_t_coarse], dim=-1)
        l1_t_coarse_trans = mul_q_point(l1_q_det, l1_t_coarse_trans, batch_size)

        l1_t_coarse_trans = torch.index_select(mul_point_q(l1_t_coarse_trans, l1_q_det_inv, batch_size), 2,
                                               torch.LongTensor(range(1, 4)).cuda())

        l1_q = torch.squeeze(mul_point_q(l1_q_det, l1_q_coarse, batch_size), dim=1)
        l1_t = torch.squeeze(l1_t_coarse_trans + l1_t_det, dim=1)


        # print('l1_refine_time--------', time.time() - start_l1_refine)

    ################# layer0

        # start_l0_refine = time.time()

        l0_q_coarse = torch.reshape(l1_q, [batch_size, 1, -1])
        l0_t_coarse = torch.reshape(l1_t, [batch_size, 1, -1])

        l0_q_inv = inv_q(l0_q_coarse, batch_size)

        ############# warp layer2 pose

        l0_xyz_f1 = torch.reshape(l0_xyz_proj_f1, [batch_size, -1, 3])
        l0_xyz_bnc_q = torch.cat([torch.zeros([batch_size, self.out_H_list[0] * self.out_W_list[0], 1]).cuda(), l0_xyz_f1], dim=-1)

        l0_flow_warped = mul_q_point(l0_q_coarse, l0_xyz_bnc_q, batch_size)
        l0_flow_warped = torch.index_select(mul_point_q(l0_flow_warped, l0_q_inv, batch_size), 2, torch.LongTensor(range(1, 4)).cuda()) + l0_t_coarse

        l0_mask = torch.any(l0_xyz_f1 !=0, dim = -1, keepdim = True).to(torch.float32)
        l0_flow_warped = l0_flow_warped * l0_mask

        ########## re-project

        l0_xyz_warp_proj_f1, l0_points_warp_proj_f1 = ProjectPC2SphericalRing(l0_flow_warped, l0_points_f1, self.out_H_list[0], self.out_W_list[0])  # 
        l0_xyz_warp_f1 = torch.reshape(l0_xyz_warp_proj_f1, [batch_size, -1, 3])
        l0_points_warp_f1 = torch.reshape(l0_points_warp_proj_f1, [batch_size, self.out_H_list[0] * self.out_W_list[0], -1])

        l0_mask_warped = torch.any(l0_xyz_warp_f1 !=0, dim = -1, keepdim = False)


        # get the cost volume of warped layer3 flow and the points of frame2
        l0_cost_volume = self.cost_volume4(l0_xyz_warp_proj_f1, l0_xyz_proj_f2, l0_points_warp_proj_f1, l0_points_proj_f2)

        l0_cost_volume_w_upsample = self.set_upconv3_w_upsample(l0_xyz_warp_proj_f1, l1_xyz_warp_proj_f1, l0_points_warp_proj_f1, l1_cost_volume_w_proj)
        l0_cost_volume_upsample = self.set_upconv3_upsample(l0_xyz_warp_proj_f1, l1_xyz_warp_proj_f1, l0_points_warp_proj_f1, l1_cost_volume_proj)
        
        l0_cost_volume_predict = self.flow_predictor3_predict(l0_points_warp_f1, l0_cost_volume_upsample, l0_cost_volume)
        l0_cost_volume_w = self.flow_predictor3_w(l0_points_warp_f1, l0_cost_volume_w_upsample, l0_cost_volume)

        l0_cost_volume_sum = softmax_valid(feature_bnc = l0_cost_volume_predict, weight_bnc = l0_cost_volume_w, mask_valid = l0_mask_warped)  # B 1 C

        l0_points_f1_new_big = self.conv3_l0(l0_cost_volume_sum)

        l0_points_f1_new_q = F.dropout(l0_points_f1_new_big, p = 0.5, training = self.training)
        l0_points_f1_new_t = F.dropout(l0_points_f1_new_big, p = 0.5, training = self.training)

        l0_q_det = self.conv1_l0(l0_points_f1_new_q)
        l0_q_det = l0_q_det / (torch.sqrt(torch.sum(l0_q_det * l0_q_det, dim=-1, keepdim=True) + 1e-10) + 1e-10)

        l0_t_det = self.conv2_l0(l0_points_f1_new_t)

        l0_q_det_inv = inv_q(l0_q_det, batch_size)
        
        l0_t_coarse_trans = torch.cat([torch.zeros([batch_size, 1, 1]).cuda(), l0_t_coarse], dim=-1)
        l0_t_coarse_trans = mul_q_point(l0_q_det, l0_t_coarse_trans, batch_size)
        l0_t_coarse_trans = torch.index_select(mul_point_q(l0_t_coarse_trans, l0_q_det_inv, batch_size), 2,
                                               torch.LongTensor(range(1, 4)).cuda())

        l0_q = torch.squeeze(mul_point_q(l0_q_det, l0_q_coarse, batch_size), dim=1)
        l0_t = torch.squeeze(l0_t_coarse_trans + l0_t_det, dim=1)

        #l0_q = l1_q
        #l0_t = l1_t

        l0_q_norm = l0_q / (torch.sqrt(torch.sum(l0_q * l0_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l1_q_norm = l1_q / (torch.sqrt(torch.sum(l1_q * l1_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l2_q_norm = l2_q / (torch.sqrt(torch.sum(l2_q * l2_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l3_q_norm = l3_q / (torch.sqrt(torch.sum(l3_q * l3_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)

        # print('l0_refime_time: ---------', time.time() - start_l0_refine)

        return l0_q_norm, l0_t, l1_q_norm, l1_t, l2_q_norm, l2_t, l3_q_norm, l3_t, l1_xyz_f1, q_gt, t_gt, self.w_x, self.w_q


def get_loss(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, qq_gt, t_gt, w_x, w_q):

    t_gt = torch.squeeze(t_gt)

    l0_q_norm = l0_q / (torch.sqrt(torch.sum(l0_q * l0_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
    l0_loss_q = torch.mean(torch.sqrt(torch.sum((qq_gt - l0_q_norm) * (qq_gt - l0_q_norm), dim=-1, keepdim=True) + 1e-10))
    l0_loss_x = torch.mean(torch.sqrt((l0_t - t_gt) * (l0_t - t_gt) + 1e-10))
    l0_loss = l0_loss_x * torch.exp(-w_x) + w_x + l0_loss_q * torch.exp(-w_q) + w_q

    l1_q_norm = l1_q / (torch.sqrt(torch.sum(l1_q * l1_q, -1, keepdim=True) + 1e-10) + 1e-10)
    l1_loss_q = torch.mean( torch.sqrt(torch.sum((qq_gt - l1_q_norm) * (qq_gt - l1_q_norm), -1, keepdim=True) + 1e-10))
    l1_loss_x = torch.mean(torch.sqrt((l1_t - t_gt) * (l1_t - t_gt) + 1e-10))
    l1_loss = l1_loss_x * torch.exp(-w_x) + w_x + l1_loss_q * torch.exp(-w_q) + w_q

    l2_q_norm = l2_q / (torch.sqrt(torch.sum(l2_q * l2_q, -1, keepdim=True) + 1e-10) + 1e-10)
    l2_loss_q = torch.mean(torch.sqrt(torch.sum((qq_gt - l2_q_norm) * (qq_gt - l2_q_norm), -1, keepdim=True) + 1e-10))
    l2_loss_x = torch.mean(torch.sqrt((l2_t - t_gt) * (l2_t - t_gt) + 1e-10))
    l2_loss = l2_loss_x * torch.exp(-w_x) + w_x + l2_loss_q * torch.exp(-w_q) + w_q

    l3_q_norm = l3_q / (torch.sqrt(torch.sum(l3_q * l3_q, -1, keepdim=True) + 1e-10) + 1e-10)
    l3_loss_q = torch.mean(torch.sqrt(torch.sum((qq_gt - l3_q_norm) * (qq_gt - l3_q_norm), -1, keepdim=True) + 1e-10))
    l3_loss_x = torch.mean(torch.sqrt((l3_t - t_gt) * (l3_t - t_gt) + 1e-10))
    l3_loss = l3_loss_x * torch.exp(-w_x) + w_x + l3_loss_q * torch.exp(-w_q) + w_q

    loss_sum = 1.6 * l3_loss + 0.8 * l2_loss + 0.4 * l1_loss + 0.2 * l0_loss

    return loss_sum




# if __name__ == "__main__":


