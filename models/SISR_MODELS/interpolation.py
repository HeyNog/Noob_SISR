import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

class interpolation(nn.Module):
	def __init__(self, UPF=4, mode='bicubic'):
		super(interpolation, self).__init__()
		self.name = mode
		self.layer = nn.Upsample(scale_factor=UPF, mode=mode)

	def forward(self, x):
		out = self.layer(x)
		return out

class bicubic(interpolation):
	def __init__(self, UPF=4):
		super(bicubic, self).__init__()
		self.name = 'bicubic'
		self.layer = nn.Upsample(scale_factor=UPF, mode='bicubic')

	def forward(self, x):
		out = self.layer(x)
		return out

class bilinear(interpolation):
	def __init__(self, UPF=4):
		super(bilinear, self).__init__()
		self.name = 'bilinear'
		self.layer = nn.Upsample(scale_factor=UPF, mode='bilinear')

	def forward(self, x):
		out = self.layer(x)
		return out

class nearest(interpolation):
	def __init__(self, UPF=4):
		super(nearest, self).__init__()
		self.name = 'nearest'
		self.layer = nn.Upsample(scale_factor=UPF, mode='nearest')

	def forward(self, x):
		out = self.layer(x)
		return out