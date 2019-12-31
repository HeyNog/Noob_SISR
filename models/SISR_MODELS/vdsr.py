import torch
import torch.nn as nn
import torch.nn.functional as F

class vdsr(nn.Module):
	'''
	Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee
	Accurate Image Super-Resolution Using Very Deep Convolutional Networks 
	'''
	def __init__(self, in_channel=3,
						UPF=4,
						num_layers=20,
						num_feats=64,
						activation=nn.ReLU(inplace=True)
						):
		super(vdsr, self).__init__()
		assert num_layers>2
		self.name = 'vdsr_x%d'%UPF
		self.preprocess = nn.Upsample(scale_factor=UPF, mode='bicubic')

		self.input_conv = nn.Sequential(
			nn.Conv2d(in_channel, num_feats, 3, padding=1),
			activation
			)
		
		self.main = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(num_feats, num_feats, 3, padding=1),
				activation
				)  for _ in range(num_layers-2)
			])
		
		self.output_conv = nn.Conv2d(num_feats, in_channel, 3, padding=1)

	def forward(self, x):
		x0 = self.preprocess(x)
		x = self.input_conv(x0)
		for layer in self.main:
			x = layer(x)
		x = self.output_conv(x)
		return x+x0