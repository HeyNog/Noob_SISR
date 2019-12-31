import torch
import torch.nn as nn
import torch.nn.functional as F

from .Module.de_subpix import *

class feqe_resblock(nn.Module):
	'''
	FEQE: Vu, Thang and Van Nguyen, Cao and Pham, Trung X. and Luu, Tung M. and Yoo, Chang D.
	Fast and Efficient Image Quality Enhancement via Desubpixel Convolutional Neural Networks
	'''
	def __init__(self,  ich=16,
						kernel_size=3,
						norm_layer=nn.InstanceNorm2d,
						activation=nn.ReLU(inplace=True)):
		super(feqe_resblock, self).__init__()
		self.conv1 = nn.Conv2d(ich, ich, kernel_size, padding=kernel_size//2)
		self.norm1 = norm_layer(ich)
		self.acti1 = activation
		self.conv2 = nn.Conv2d(ich, ich, kernel_size, padding=kernel_size//2)
		self.norm2 = norm_layer(ich)
	def forward(self, x):
		x0 = x
		x = self.acti1(self.norm1(self.conv1(x)))
		x = self.norm2(self.conv2(x))
		return x+x0


class feqe_upsample(nn.Module):
	def __init__(self,  ich=16,
						och=3,
						UPF=4):
		super(feqe_upsample, self).__init__()
		assert UPF==2 or UPF==4
		if UPF==2:
			self.layer = nn.Sequential(nn.Conv2d(ich, och*4, 1),
										nn.PixelShuffle(2))
		elif UPF==4:
			self.layer = nn.Sequential(nn.Conv2d(ich, ich*4, 1),
										nn.PixelShuffle(2),
										nn.Conv2d(ich, och*4, 1),
										nn.PixelShuffle(2))
	def forward(self, x):
		return self.layer(x)


class feqe_downsample(nn.Module):
	def __init__(self, ich=3,
						och=16,
						DF=4):
		super(feqe_downsample, self).__init__()
		assert DF==2 or DF==4
		if DF==2:
			self.layer = nn.Sequential(nn.Conv2d(ich, och//4, 1),
										DeSubPixelShuffle(2))
		elif DF==4:
			self.layer = nn.Sequential(DeSubPixelShuffle(2),
										nn.Conv2d(ich*4, och//4, 1),
										DeSubPixelShuffle(2))
	def forward(self, x):
		return self.layer(x)



class feqe(nn.Module):
	'''

	'''
	def __init__(self,  num_channels=3,
						UPF=4,
						inner_UPF=4,
						n_features=16,
						num_blocks=20,
						preprocess_mode='bicubic'):
		super(feqe, self).__init__()
		self.name = 'feqe_x%d'%UPF
		self.preprocess = nn.Upsample(scale_factor=UPF, mode=preprocess_mode)
		self.ds = feqe_downsample(ich=num_channels, och=n_features, DF=inner_UPF)
		self.mapping = nn.ModuleList([feqe_resblock(ich=n_features) for i in range(num_blocks)])
		self.us = feqe_upsample(ich=n_features, och=num_channels, UPF=inner_UPF)
 
	def forward(self, x):
		x0 = self.preprocess(x)
		x = self.ds(x0)
		for layer in self.mapping:
			x = layer(x)
		x = self.us(x)
		return x+x0