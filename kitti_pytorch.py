# -*- coding:UTF-8 -*-

import os
import yaml
import argparse
import numpy as np
import torch.utils.data as data
from PIL import Image
import lib2.utils.calibration as calibration
from lib2.config import cfg
from tools.euler_tools import euler2quat, mat2euler
from tools.points_process import aug_matrix, limited_points, filter_points

# author:Zhiheng Feng
# contact: fzhsjtu@foxmail.com
# datetime:2021/10/21 20:04
# software: PyCharm

"""
文件说明：数据集读取

"""


class points_dataset(data.Dataset):

    def __init__(self, is_training: int = 1, num_point: int = 150000, data_dir_list: list = [0, 1, 2, 3, 4, 5, 6], config: argparse.Namespace = None):
        """

        :param train: 0训练集，1验证集，2测试集
        :param data_dir_list: 数据集序列
        :param config: 配置参数
        """
        self.count = 0
        data_dir_list.sort()
        self.num_point = num_point
        self.is_training = is_training
        self.args = config
        self.data_list = data_dir_list
        self.lidar_root = config.lidar_root
        self.image_root = config.image_root
        self.data_len_sequence = [4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        Tr_tmp = []
        data_sum = [0]
        vel_to_cam_Tr = []

        with open('./tools/calib.yaml', "r") as f:
            con = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(11):
            vel_to_cam_Tr.append(np.array(con['Tr{}'.format(i)]))
        for i in self.data_list:
            data_sum.append(data_sum[-1] + self.data_len_sequence[i] + 1)
            Tr_tmp.append(vel_to_cam_Tr[i])

        self.Tr_list = Tr_tmp
        self.data_sum = data_sum
        self.lidar_path = self.lidar_root
        self.image_path = self.image_root

    def get_valid_flag(self, pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE ##__C.PC_AREA_SCOPE = np.array([[-40, 40],[-1, 3],[0, 70.4]])  # x, y, z scope in rect camera coords
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    def __len__(self):
        return self.data_sum[-1]

    def __getitem__(self, index):

        sequence_str_list = []

        for item in self.data_list:
            sequence_str_list.append('{:02d}'.format(item))

        if index in self.data_sum:
            # index_index = self.data_sum.index(index)
            # index_ = 0
            # fn1 = index_
            # fn2 = index_
            index_index = self.data_sum.index(index)  ##返回的是对应此index值所在序列，也就是序列数0~10
            index_ = 0
            fn1 = index_
            fn2 = index_
            fn3 = index_
            fn4 = index_
            c1 = index_
            c2 = index_  ##序列内部的每一帧索引（六位）
            sample_id = index_
        else:
            # index_index, data_begin, data_end = self.get_index(index, self.data_sum)
            # index_ = index - data_begin
            # fn1 = index_ - 1
            # fn2 = index_
            index_index, data_begin, data_end = self.get_index(index, self.data_sum)
            index_ = index - data_begin
            fn1 = index_ - 1
            fn2 = index_ - 1
            fn3 = index_
            fn4 = index_
            c1 = index_ - 1
            c2 = index_  ##取前后两帧
            sample_id = index_

        Tr = self.Tr_list[index_index]
        Tr_inv = np.linalg.inv(Tr)### 
        
        pose_path = 'pose/' + sequence_str_list[index_index] + '_diff.npy'
        lidar_path = os.path.join(self.lidar_path, sequence_str_list[index_index], 'velodyne')
        image_path = os.path.join(self.image_path, sequence_str_list[index_index], 'image_2')
        pose = np.load(pose_path)
        calib = os.path.join(self.image_path, sequence_str_list[index_index], 'calib')

        fn1_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn1))
        fn2_dir = os.path.join(image_path, '{:06d}.png'.format(fn2))
        fn3_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn3))
        fn4_dir = os.path.join(image_path, '{:06d}.png'.format(fn4))
        c1_dir = os.path.join(calib, '{:06d}.txt'.format(c1))
        c2_dir = os.path.join(calib, '{:06d}.txt'.format(c2))

        point1 = np.fromfile(fn1_dir, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(fn3_dir, dtype=np.float32).reshape(-1, 4)

        img1 = Image.open(fn2_dir).convert('RGB')
        img2 = Image.open(fn4_dir).convert('RGB')

        ##图像预处理
        width1, height1 = img1.size  ## matrix(1226,370)
        # print("img1.size: ", img1.size)  #（1241，376）  对应的应该是同一个图像，为什么size不一样
        width2, height2 = img2.size  ## matrix(1226,370)
        # print("img1: ", img1)RGB图像
        img_shape_1 = height1, width1, 3
        img_shape_2 = height2, width2, 3  ##(370,1226,3)
        # print(img_shape_1)

        calib1 = calibration.Calibration(c1_dir)  ##Calibration是类
        calib2 = calibration.Calibration(c2_dir)

        ##图像归一化
        img1 = np.array(img1).astype(np.float)  ##将图像矩阵转化为数组
        img1 = img1 / 255.0  ##使数组每一个元素都位于（0,1）之间
        img1 -= self.mean
        img1 /= self.std  ##每一元素减去均值，除方差
        # print("img1.size: ", img1.shape)    #(376,1241,3)
        # print("img1: ", img1)
        ##img1 (370,1226,3)
        imback1 = np.zeros([384, 1296, 3], dtype=np.float)  ##图像补零
        # print(img1.shape)
        imback1[:img1.shape[0], :img1.shape[1], :] = img1
        imback1 = imback1.transpose(2, 0, 1)  ##（3,384,1296）

        img2 = np.array(img2).astype(np.float)
        img2 = img2 / 255.0
        img2 -= self.mean
        img2 /= self.std  ##img2 (370,1226,3)
        # print("img2.shape: ", img2.shape) (376, 1241, 3)
        imback2 = np.zeros([384, 1296, 3], dtype=np.float)
        imback2[:img2.shape[0], :img2.shape[1], :] = img2
        imback2 = imback2.transpose(2, 0, 1)  ##（3,384,1296）

        # 原始点云补零
        pos1 = np.zeros((self.num_point, 3))
        pos2 = np.zeros((self.num_point, 3))

        pos1[ :point1.shape[0], :] = point1[:, :3]
        pos2[ :point2.shape[0], :] = point2[:, :3]

        T_diff = pose[index_:index_ + 1, :]  ##### read the transformation matrix

        T_diff = T_diff.reshape(3, 4)
        filler = np.array([0.0, 0.0, 0.0, 1.0])
        filler = np.expand_dims(filler, axis=0)  ##1*4
        T_diff = np.concatenate([T_diff, filler], axis=0)  # 4*4

        T_gt = np.matmul(Tr_inv, T_diff)
        T_gt = np.matmul(T_gt, Tr)

        pos1 = pos1.astype(np.float32)
        pos2 = pos2.astype(np.float32)
        
        if self.is_training:
            T_trans = aug_matrix()
        else:
            T_trans = np.eye(4).astype(np.float32)

        T_trans_inv = np.linalg.inv(T_trans)

        # 得到与图像对应的点云在序列中的位置
        pts_rect1 = calib1.lidar_to_rect(point1[:, 0:3])
        # print("fn1: ", fn1)
        # print("point1: ", point1.shape)
        pts_rect2 = calib2.lidar_to_rect(point2[:, 0:3])  # 只取点云前三列xyz，然后转化为参考坐标系（0号相机）下
        pts_img1, pts_rect_depth1 = calib1.rect_to_img(pts_rect1)
        pts_img2, pts_rect_depth2 = calib2.rect_to_img(pts_rect2)  # 将点云进一步转移到2号左侧相机图像上，得到2D坐标与深度


        # get valid point (projected points should be in image)
        pts_valid_flag1 = self.get_valid_flag(pts_rect1, pts_img1, pts_rect_depth1, img_shape_1)
        pts_pure_flag1 = ~ pts_valid_flag1
        pts_valid_flag2 = self.get_valid_flag(pts_rect2, pts_img2, pts_rect_depth2, img_shape_2)
        pts_pure_flag2 = ~ pts_valid_flag2

        # if self.count <= 0:
        #     with open('pts_img1.txt', 'a') as file:
        #         np.savetxt(file, pts_img1[pts_valid_flag1])
        #     self.count = self.count + 1

        # 掩码补零使之与补零后的点云数量对应
        pts_valid1 = np.zeros((self.num_point, 1))
        pts_pure1 = np.zeros((self.num_point, 1))
        pts_valid2 = np.zeros((self.num_point, 1))
        pts_pure2 = np.zeros((self.num_point, 1))
        pts_origin_xy1 = np.zeros((self.num_point, 2))
        pts_origin_xy2 = np.zeros((self.num_point, 2))

        pts_valid1[:pts_valid_flag1.shape[0], :] = pts_valid_flag1.reshape(-1, 1)
        pts_pure1[:pts_pure_flag1.shape[0], :] = pts_pure_flag1.reshape(-1, 1)
        pts_valid2[:pts_valid_flag2.shape[0], :] = pts_valid_flag2.reshape(-1, 1)
        pts_pure2[:pts_pure_flag2.shape[0], :] = pts_pure_flag2.reshape(-1, 1)
        pts_origin_xy1[:pts_img1.shape[0], :] = pts_img1[:, :2]
        pts_origin_xy2[:pts_img2.shape[0], :] = pts_img2[:, :2]

        return pos2, pos1, imback2, imback1, pts_origin_xy2, pts_origin_xy1, T_gt, T_trans, T_trans_inv, Tr, pts_valid2, pts_valid1, sample_id

    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num
