#------------------
# Derived from: https://github.com/958099161/FEQE-desubpix
#------------------
import torch
import torch.nn as nn


def de_subpix(y, DF):
	'''
	Desubpixel Shuffle in FEQE.
	Args:
		y: The input tensor in the desubpixel shuffle, shape (b, c, h, w)
		DF: The downsampling factor of the desubpixel shuffle.

	Output:
		out: The output tensor of desubpixel shuffle, shape (b, DF^2*c, h//DF, w//DF)
	'''
	(b, c, h, w) = y.shape
	assert (h%DF == 0 and w%DF == 0), 'Input Shape (%d,%d) mismatch with Downsampling-Factor %d'%(h,w,DF)

	d = []
	for i in range(DF):
		for j in range(DF):
			d.append(y[:, :, i::DF, j::DF])
	out = torch.cat(d, dim=1)
	return out


class DeSubPixelShuffle(nn.Module):
	def __init__(self, DF):
		super(DeSubPixelShuffle, self).__init__()
		self.df = DF

	def forward(self, x):
		return de_subpix(x, self.df)

if __name__ == '__main__':
	subpix = nn.functional.pixel_shuffle
	a = torch.rand((1,1,24,24))
	print(a.shape)
	b = de_subpix(a, 4)
	print(b.shape)
	c = subpix(b, 4)
	print(c.shape)
	print(((a-c)**2).sum())

	import time
	a = torch.rand((100, 3, 400, 400)).cuda()
	dps_start = time.time()
	a = de_subpix(a, 4)
	dps_end = time.time()
	print('DeSubPixelShuffle: %.7f s'%(dps_end - dps_start))

	ps_start = time.time()
	a = subpix(a, 4)
	ps_end = time.time()
	print('SubPixelShuffle: %.7f s'%(ps_end - ps_start))

