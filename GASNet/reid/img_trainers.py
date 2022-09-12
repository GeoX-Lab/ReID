from __future__ import print_function, absolute_import

import io
import time
import sys
import os
# import netron

import torch
import torchvision
import numpy as np
from torch.autograd import Variable
from scipy import misc

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter
from .utils.data.transforms import RandomErasing

class BaseTrainer(object):
	def __init__(self, model, criterion, summary_writer, prob=0.5, use_o_scale=True, mean=[0.4914, 0.4822, 0.4465]):
		super(BaseTrainer, self).__init__()
		self.model = model
		self.criterion = criterion
		self.summary_writer = summary_writer
		self.use_o_scale = use_o_scale
		self.normlizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		self.eraser = RandomErasing(probability=prob, mean=[0., 0., 0.])

	def train(self, epoch, data_loader, optimizer, random_erasing, empty_cache=False, print_freq=10):
		self.model.train()

		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		precisions = AverageMeter()

		end = time.time()
		alloss = 0.0
		for i, inputs in enumerate(data_loader):  # inputs[0] 图像的tensor inputs[1] 路径 inputs[2] 人ID inputs[3] 相机ID
			data_time.update(time.time() - end)

			ori_inputs, targets = self._parse_data(inputs)  # 扔掉了相机ID 获取图片车辆ID
			in_size = inputs[0].size()  # (B, C, H, W)
			for j in range(in_size[0]):
				ori_inputs[0][j, :, :, :] = self.normlizer(ori_inputs[0][j, :, :, :])
				if random_erasing:
					ori_inputs[0][j, :, :, :] = self.eraser(ori_inputs[0][j, :, :, :])
			loss, all_loss, prec1 = self._forward(ori_inputs, targets)

			losses.update(loss.data, targets.size(0))
			precisions.update(prec1, targets.size(0))

			# tensorboard
			alloss += loss.item()
			if self.summary_writer is not None:
				if i % 1000 == 0 :
					global_step = epoch * len(data_loader) + i
					self.summary_writer.add_scalar('loss', alloss / 1000, global_step)
					self.summary_writer.add_scalar('rga_loss_cls', all_loss[0], global_step)
					self.summary_writer.add_scalar('rga_loss_tri', all_loss[1], global_step)
					self.summary_writer.add_scalar('oscale_loss_cls', all_loss[2], global_step)
					self.summary_writer.add_scalar('oscale_loss_tri', all_loss[3], global_step)
					alloss = 0.0
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if empty_cache:
				torch.cuda.empty_cache()

			batch_time.update(time.time() - end)
			end = time.time()

			if (i + 1) % print_freq == 0:
				print('Epoch: [{}][{}/{}]\t'
					'Total_Loss: {:.5f}  rgs_cls: {:.5f}  rga_tri: {:.5f}  oscale_cls: {:.5f}  oscale_tri: {:.5f}\t'
					'Prec {:.5%} ({:.5%})\t'
					.format(epoch + 1, i + 1, len(data_loader),
							loss, all_loss[0], all_loss[1], all_loss[2], all_loss[3],
							precisions.val, precisions.avg))

				
	def _parse_data(self, inputs):
		raise NotImplementedError

	def _forward(self, inputs, targets):
		raise NotImplementedError


class ImgTrainer(BaseTrainer):
	def _parse_data(self, inputs):
		imgs, _, pids, _ = inputs
		inputs = [Variable(imgs)]
		targets = Variable(pids.cuda())
		return inputs, targets

	def _forward(self, inputs, targets): # inputs [b, c, h, w]
		# a = inputs
		# b = targets

		# 此处应该返回6个值
		outputs = self.model(inputs, training=True, use_o_scale=self.use_o_scale) # outputs[0]:三元组需要的输出  outputs[1]:未知  outputs[2]:类别输出

		# rga_loss
		rga_loss_cls = self.criterion[0](outputs[2], targets)
		rga_loss_tri = self.criterion[1](outputs[0], targets)

		# oscale_loss
		if self.use_o_scale:
			oscale_loss_cls = self.criterion[0](outputs[5], targets)
			oscale_loss_tri = self.criterion[1](outputs[3], targets)
		else:
			oscale_loss_cls = 0
			oscale_loss_tri = 0

		loss = rga_loss_cls + rga_loss_tri + oscale_loss_cls + oscale_loss_tri

		losses = [rga_loss_cls, rga_loss_tri, oscale_loss_cls, oscale_loss_tri]

		if self.use_o_scale:
			prec, = accuracy(outputs[2].data + outputs[5].data, targets.data)  # 类别特征相加
		else:
			prec, = accuracy(outputs[2].data, targets.data)
		prec = prec[0]
		return loss, losses, prec

