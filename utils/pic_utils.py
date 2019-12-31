import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import numpy as np
from PIL import Image

import skimage
from skimage.measure import compare_psnr, compare_ssim
from math import log
import os
import os.path as _P

USE_SKI_COMP = False

base_path = _P.split(_P.dirname(__file__))[0]
save_path = _P.join(base_path, 'data', 'save')
test_path = _P.join(base_path, 'data', 'test')

# Functions
def to_psnr(mse, max_range=4):
	psnr = 10*log(max_range/mse, 10)
	return psnr



def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)



def pil_mse(pil1, pil2):
	pil1 = rgb2ycbcr(np.asarray(pil1))
	pil2 = rgb2ycbcr(np.asarray(pil2))
	mse = np.mean((pil1.astype(np.float64) - pil2.astype(np.float64))**2)
	return mse



def pil_psnr(pil1, pil2):
	if USE_SKI_COMP:
		pil1 = np.asarray(pil1)
		pil2 = np.asarray(pil2)
		return compare_psnr(pil1[:,:,0], pil2[:,:,0])
	else:
		mse = pil_mse(pil1, pil2)
		return 10*log(255**2/mse, 10)



def pil_ssim(pil1, pil2):
	if USE_SKI_COMP:
		pil1 = np.array(pil1)
		pil2 = np.array(pil2)
		return compare_ssim(pil1, pil2)
	else:
		return None



def pixel_count(img):
	count = 1
	if isinstance(img, torch.Tensor):
		for i in img.shape:
			count *= i
	elif isinstance(img, Image.Image):
		for i in img.size:
			count *= i
	else:
		count = 0
	return count



def to_pil_image(img_tensor, color_mode='RGB'):
	# [-1,1] tensor Un-Normalize to [0,1] tensor
	# img_tensor = img_tensor/2 + 0.5
	img_tensor = (img_tensor+1)*255
	img_tensor = torch.ceil(img_tensor) // 2
	img_tensor = img_tensor/255
	img_tensor = img_tensor.clamp(0, 1)
	img = T.ToPILImage(mode=color_mode)(img_tensor)
	img = img.convert('RGB')
	return img



def show_image(img_data):
	if isinstance(img_data, (torch.Tensor, np.ndarray)):
		img_data = to_pil_image(img_data)
	img_data.show()



def save_image(img_data, save_file, save_path=save_path,
			sub_path=None, file_type='.png', notice=False):
	if isinstance(img_data, (torch.Tensor, np.ndarray)):
		img_data = to_pil_image(img_data)
	save_file = save_file + file_type
	if sub_path:
		try:
			os.mkdir(os.path.join(save_path, sub_path))
		except FileExistsError as e:
			if notice:
				print('Path %s Exists'%save_path)
			pass
		save_file = os.path.join(save_path, sub_path, save_file)
	else:
		save_file = os.path.join(sub_path, save_file)
	img_data.save(save_file)