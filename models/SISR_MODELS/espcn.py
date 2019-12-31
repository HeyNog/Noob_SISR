import torch
import torch.nn as nn
import torch.nn.functional as F

class espcn(nn.Module):
	'''
	Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang
	Real-Time Single Image and Video Super-Resolution
	Using an Efficient Sub-Pixel Convolutional Neural Network

	Code from Github:
	https://github.com/leftthomas/ESPCN
	'''
	def __init__(self, in_channel=3, UPF=4):
		super(espcn, self).__init__()
		self.name = 'espcn_x%d'%UPF
		self.conv0 = nn.Conv2d(in_channel, 64, 5, padding=2)
		self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
		self.conv2 = nn.Conv2d(32, in_channel*UPF**2, 3, padding=1)
		self.ps1   = nn.PixelShuffle(UPF)

	def forward(self, x):
		out = torch.tanh(self.conv0(x))
		out = torch.tanh(self.conv1(out))
		out = self.conv2(out)
		out = self.ps1(out)
		return out