import torch
import torch.nn as nn
import torch.nn.functional as F

class fsrcnn(nn.Module):
	'''
	FSRCNN : DONG C, CHEN C L, TANG X.
	ACCELERATING THE SUPER-RESOLUTION CONVOLUTIONAL NEURAL NETWORK, ECCV2016
	Code from CSDN :
	https://blog.csdn.net/weixin_38203533/article/details/80704039
	'''
	def __init__(self, num_channels=3, UPF=4, d=64, s=16):
		super(fsrcnn, self).__init__()
		self.name = 'fsrcnn_x%d'%UPF
		# Feature extraction
		self.conv1 = nn.Conv2d(num_channels, d, 5, padding=2)
		self.prelu1 = nn.PReLU()
 
		# Shrinking
		self.conv2 = nn.Conv2d(d, s, 1, padding=0)
		self.prelu2 = nn.PReLU()

		# Non-linear Mapping
		self.conv3 = nn.Conv2d(s, s, 3, padding=1)
		self.prelu3 = nn.PReLU()
		self.conv4 = nn.Conv2d(s, s, 3, padding=1)
		self.prelu4 = nn.PReLU()
		self.conv5 = nn.Conv2d(s, s, 3, padding=1)
		self.prelu5 = nn.PReLU()
		self.conv6 = nn.Conv2d(s, s, 3, padding=1)
		self.prelu6 = nn.PReLU()
		# Expanding
		self.conv7 = nn.Conv2d(s, d, 1, padding=0)
		self.prelu7 = nn.PReLU()
		# Deconvolution
		self.deconv = nn.ConvTranspose2d(d, num_channels, 9,
						stride=UPF, padding=4, output_padding=UPF-1)
 
	def forward(self, x):
		x = self.prelu1(self.conv1(x))
		x = self.prelu2(self.conv2(x))
		x = self.prelu3(self.conv3(x))
		x = self.prelu4(self.conv4(x))
		x = self.prelu5(self.conv5(x))
		x = self.prelu6(self.conv6(x))
		x = self.prelu7(self.conv7(x))
		x = self.deconv(x)
		return x