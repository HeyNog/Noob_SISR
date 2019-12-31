import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

class srcnn(nn.Module):
	'''
	Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang
	Image Super-Resolution Using Deep Convolutional Networks
	'''
	def __init__(self, in_channel=3, UPF=4):
		super(srcnn, self).__init__()
		self.name = 'srcnn_955_x%d'%UPF
		self.preprocess = nn.Upsample(scale_factor=UPF, mode='bicubic')
		self.conv0 = nn.Conv2d(in_channel, 64, 9, padding=4)
		self.conv1 = nn.Conv2d(64, 32, 5, padding=2)
		self.conv2 = nn.Conv2d(32, in_channel, 5, padding=2)

	def forward(self, x):
		x = self.preprocess(x)
		x = F.relu(self.conv0(x), inplace=True)
		x = F.relu(self.conv1(x), inplace=True)
		x = self.conv2(x)
		return x

class srcnn915(nn.Module):
	def __init__(self, in_channel=3, UPF=4):
		super(srcnn915, self).__init__()
		self.name = 'srcnn_915_x%d'%UPF
		self.preprocess = nn.Upsample(scale_factor=UPF, mode='bicubic')
		self.conv0 = nn.Conv2d(in_channel, 64, 9, padding=4)
		self.conv1 = nn.Conv2d(64, 32, 1)
		self.conv2 = nn.Conv2d(32, in_channel, 5, padding=2)

	def forward(self, x):
		x = self.preprocess(x)
		x = F.relu(self.conv0(x), inplace=True)
		x = F.relu(self.conv1(x), inplace=True)
		x = self.conv2(x)
		return x