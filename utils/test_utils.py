import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image

import time
import matplotlib.pyplot as plt

from . import pic_utils as PU



yes_no_opt = {'y':True, 'Y':True, 'n':False, 'N':False}



def test_model(model, testset, device='cpu', save_output=False, save_origin=False, verbose=True):
	'''
	Testing SISR model.
	Inputs:
		model: Pytorch SISR model, should have attr "name" & "UPF".
		testset: Pytorch Dataset feeding img tensor, should have attr "set_name" & "UPF".
		device: Pytorch device, "cpu" or "cuda"
		save_output: If True, save the output images of the tested model.
		save_origin: If True, save the Groundtruths & LR images

	Outputs:
		run_time: The run time(exclude image loading & loss computation) of the tested model.
		test_time: The total test time.
	'''

	testset_name = testset.set_name
	testld  = DataLoader(testset, batch_size=1)
	model.to(device)
	
	criterion = nn.MSELoss()

	mse, run_time, psnr = 0.0, 0.0, 0.0
	if verbose:
		print('Now Testing on Testset %s'%testset_name)
		print('%d Images'%len(testset))
	test_start = time.time()
	for i, testcp in enumerate(testld):
		# Load Image Couple from Testset
		hrimg, lrimg = testcp
		hrimg, lrimg = hrimg.to(device), lrimg.to(device)
		# Model SR Processing
		sr_start = time.time()
		with torch.no_grad():
			srimg = model(lrimg)
			sr_end = time.time()
			loss = criterion(srimg, hrimg)
		run_time += sr_end - sr_start
		
		# Save SR image
		srimg_cpu = srimg.cpu()
		hrimg_cpu, lrimg_cpu = hrimg.cpu(), lrimg.cpu()
		new_mse = loss.item()
		# mse += new_mse
		# psnr += to_psnr(new_mse)
		hrpil, srpil = PU.to_pil_image(hrimg_cpu[0]), PU.to_pil_image(srimg_cpu[0])
		mse += PU.pil_mse(hrpil, srpil)
		psnr += PU.pil_psnr(hrpil, srpil)
		if save_origin:
			PU.save_image(hrimg_cpu[0], 'HR%s'%(i+1), sub_path='GroundTruth_'+testset_name)
			PU.save_image(lrimg_cpu[0], 'LR%s'%(i+1), sub_path='LowResolution_'+testset_name)
			del hrimg_cpu, lrimg_cpu
		if save_output:
			PU.save_image(srimg_cpu[0], 'SR%s_%s'%(i+1, model.name), sub_path=model.name+'_'+testset_name)
		# Clean up GPU Memory
		del lrimg, hrimg, srimg, srimg_cpu
		torch.cuda.empty_cache()
	test_end = time.time()
	
	mse /= len(testset)
	psnr /= len(testset)
	if verbose:
		print('Test Over')
		print('MSE Loss: %.8f'%mse)
		print('PSNR Score: %.3f'%psnr)
		print('Model Run Time: %.7f'%run_time)
		print('Test Run Time: %.7f'%(test_end-test_start))

	return (run_time, test_end-test_start, mse, psnr)

'''
W.I.P

def img_show(model_list, testset, epochs=None, img_idx=1, axis_flag=False):
	global device

	hrimg, lrimg = testset[int(img_idx)-1]
	tar = PU.to_pil_image(hrimg)
	data_in = torch.Tensor([lrimg.numpy()])
	data_tar = torch.Tensor([hrimg.numpy()])
	data_tar = data_tar.to(device)
	data_in = data_in.to(device)

	model_output = []
	model_psnr = []
	criterion = nn.MSELoss()
	for model_name in model_list:
		model = M.build_model(model_name)
		model.to(device)
		model.eval()
		if epochs:
			total_state_file = M.load_train_state(model, epochs, map_location={'cuda:1':str(device)},notice=True)
		else:
			total_state_file = M.load_train_state(model, '', save_file='min_val_checkpoint.pth', map_location={'cuda:1':str(device)},notice=True)
		if total_state_file:
			try:
				model.load_state_dict(total_state_file['model_state'])
			except KeyError as e:
				model.load_state_dict(total_state_file)
		elif not(isinstance(model, M.interpolation)):
			data_out = torch.ones(data_tar.shape)
			data_out = PU.to_pil_image(data_out[0])
			model_output.append(data_out)
			model_psnr.append('~')
			del model
			continue
		with torch.no_grad():
			data_out = model(data_in)
			loss = criterion(data_out, data_tar)
			data_out = data_out.cpu()
			data_out = PU.to_pil_image(data_out[0])
			model_output.append(data_out)
			# model_psnr.append('%.4f'%to_psnr(loss.item()))
			model_psnr.append('%.4f'%pil_psnr(tar, data_out))
		del model, data_out
		torch.cuda.empty_cache()
	hrimg = PU.to_pil_image(hrimg)
	lrimg = PU.to_pil_image(lrimg)

	plt.subplot(231)
	plt.imshow(np.array(hrimg))
	plt.axis(axis_flag)
	plt.title('GroundTruth')

	# plt.subplot(232)
	# plt.imshow(np.array(lrimg))
	# plt.axis(axis_flag)
	# plt.title('LowResolution')

	for i, model_name in enumerate(model_list):
		plt.subplot(232+i)
		plt.imshow(np.array(model_output[i]))
		plt.axis(axis_flag)
		plt.title(model_name+'   PSNR:%s dB'%model_psnr[i])
	plt.show()
'''