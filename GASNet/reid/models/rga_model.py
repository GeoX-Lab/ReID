# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable

import torchvision
import numpy as np

from .models_utils.rga_branches import RGA_Branch

__all__ = ['resnet50_rga']
WEIGHT_PATH = os.path.join(os.path.dirname(__file__), '../..')+'/weights/pre_train/resnet50-19c8e357.pth'
# WEIGHT_PATH = os.path.join(os.path.dirname(__file__), '../..')+'/checkpoint/chechpoint_0.pth'

# ===================
#   Initialization 
# ===================

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
		nn.init.constant_(m.bias, 0.0)
	elif classname.find('Conv') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0.0)
	elif classname.find('BatchNorm') != -1:
		if m.affine:
			nn.init.constant_(m.weight, 1.0)
			nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.normal_(m.weight, std=0.001)
		if m.bias:
			nn.init.constant_(m.bias, 0.0)
			
# ===============
#    RGA Model 
# ===============

class ResNet50_RGA_Model(nn.Module):
	'''
	Backbone: ResNet-50 + RGA modules.
	'''
	def __init__(self, pretrained=True, num_feat=2048, height=256, width=128, 
		dropout=0, num_classes=0, last_stride=1, branch_name='rgasc', scale=8, d_scale=8,
		model_path=WEIGHT_PATH, use_o_scale=True):
		super(ResNet50_RGA_Model, self).__init__()
		self.pretrained = pretrained
		self.num_feat = num_feat
		self.dropout = dropout
		self.num_classes = num_classes
		self.branch_name = branch_name
		self.use_o_scale = use_o_scale
		print ('Num of features: {}.'.format(self.num_feat))
		
		if 'rgasc' in branch_name:
			spa_on=True 
			cha_on=True
		elif 'rgas' in branch_name:
			spa_on=True
			cha_on=False
		elif 'rgac' in branch_name:
			spa_on=False
			cha_on=True
		else:
			raise NameError
		
		self.backbone = RGA_Branch(pretrained=pretrained, last_stride=last_stride, 
						spa_on=spa_on, cha_on=cha_on, height=height, width=width,
						s_ratio=scale, c_ratio=scale, d_ratio=d_scale, model_path=model_path, use_o_scale=use_o_scale)

		self.feat_bn = nn.BatchNorm1d(self.num_feat)
		self.feat_bn.bias.requires_grad_(False)
		if self.dropout > 0:
			self.drop = nn.Dropout(self.dropout)
		self.cls_ = nn.Linear(self.num_feat, self.num_classes, bias=False)  # 调整特征维度与类别数一致

		self.feat_bn.apply(weights_init_kaiming)
		self.cls_.apply(weights_init_classifier)

	# 将网络出来的特征分成两部分，暂时不知道有什么用
	def _split_feat(self, feature, training):
		feat = self.feat_bn(feature)  # 特征归一化，用在分类特征上
		if self.dropout > 0:
			feat = self.drop(feat)
		# if training and self.num_classes is not None:
		# 	cls_feat = self.cls_(feat)  # 分类特征
		# return feat, cls_feat
		# if training and self.num_classes is not None:
		# 	cls_feat = self.cls_(feat)  # 分类特征
		# 	return feat, cls_feat
		# elif not training:
		# 	return feat
		cls_feat = self.cls_(feat)  # 分类特征
		return feat, cls_feat

	def forward(self, inputs, training=True, use_o_scale=True):
		im_input = inputs[0]
		feat_ = self.backbone(im_input)

		# 用于netron查看模型结构
		# a = feat_[0].size()
		# b = a[2].item()
		# c = a[3].item()
		# f = feat_[1].size()
		# d = f[2].item()
		# e = f[3].item()


		if use_o_scale:
			feat_rga = F.avg_pool2d(feat_[0], feat_[0].size()[2:]).view(feat_[0].size(0), -1)  # 全局注意力
			feat_osc = F.avg_pool2d(feat_[1], feat_[1].size()[2:]).view(feat_[1].size(0), -1)  # 全尺度
			# netron查看模型结构时使用下列两行代码
			# feat_rga = F.avg_pool2d(feat_[0], kernel_size=[b, c]).view(feat_[0].size(0), -1)  # 全局注意力
			# feat_osc = F.avg_pool2d(feat_[1], kernel_size=[d, e]).view(feat_[1].size(0), -1)  # 全尺度
			if training:
				feat_rga_, cls_rga = self._split_feat(feat_rga, True)  # 调用split_feat
				feat_osc_, cls_osc = self._split_feat(feat_osc, True)  # 调用split_feat
				return (feat_rga, feat_rga_, cls_rga, feat_osc, feat_osc_, cls_osc)
			else:
				feat_rga_, cls_rga = self._split_feat(feat_rga, False)  # 调用split_feat, 返回归一化特征
				feat_osc_, cls_osc = self._split_feat(feat_osc, False)  # 调用split_feat， 返回归一化特征
				# return (feat_rga, feat_rga_, feat_osc, feat_osc_)
				return (feat_rga + feat_osc, feat_rga_ + feat_osc_, cls_rga + cls_osc)
		else:
			feat_rga = F.avg_pool2d(feat_, feat_.size()[2:]).view(feat_.size(0), -1)  # 全局注意力
			if training:
				feat_rga_, cls_rga = self._split_feat(feat_rga, True)  # 调用split_feat
				return (feat_rga, feat_rga_, cls_rga)
			else:
				feat_rga_, feat_cls = self._split_feat(feat_rga, False)  # 调用split_feat
				return (feat_rga, feat_rga_, feat_cls)


def resnet50_rga(*args, **kwargs):
	return ResNet50_RGA_Model(*args, **kwargs)

