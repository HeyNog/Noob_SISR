import tkinter as tk 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import skimage
from skimage.measure import compare_psnr, compare_ssim

import data.bd_dataset as DS
import models.build_model as M
from math import log

USE_GPU_FLAG = False
USE_SKI_COMP = False

if USE_GPU_FLAG:
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
	device = torch.device('cpu')
print(device)

# Functions
def to_psnr(mse):
	psnr = 10*log(4/mse, 10)
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

def test_model(model_name, testset_name='BSDS200', epochs=None, save_flag=False, save_data=False):
	global device
	if use_gpu.get():
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	else:
		device = torch.device('cpu')
	_print('\n'+str(device))

	testset = DS.Testset(set_name=testset_name)
	testld  = DS.DataLoader(testset, batch_size=1)
	
	# Model Loading
	load_start = time.time()
	model = M.build_model(model_name)
	model.to(device)
	model.eval()
	# Load the min valloss model or load by pretrained epochs
	if epochs:
		total_state_file = M.load_train_state(model, epochs, map_location={'cuda:1':str(device)},notice=True)
	else:
		total_state_file = M.load_train_state(model, '', save_file='min_val_checkpoint.pth', map_location={'cuda:1':str(device)},notice=True)
	if total_state_file:
		try:
			model.load_state_dict(total_state_file['model_state'])
		except KeyError as e:
			model.load_state_dict(total_state_file)
	load_end = time.time()
	_print('Model %s'%model_name)
	_print('Model Loading Completed in %s s'%(load_end - load_start))
	criterion = nn.MSELoss()

	mse, run_time, psnr = 0.0, 0.0, 0.0
	_print('Now Testing on Testset %s'%testset_name)
	_print('%d Images'%len(testset))
	test_start = time.time()
	for i, testcp in enumerate(testld):
		# Load Image Couple from Testset
		hrimg, lrimg = testcp
		hrimg, lrimg = hrimg.to(device), lrimg.to(device)
		# Model SR Processing
		sr_start = time.time()
		with torch.no_grad():
			srimg = model(lrimg)
			loss = criterion(srimg, hrimg)
		sr_end = time.time()
		run_time += sr_end - sr_start
		
		# Save SR image
		srimg_cpu = srimg.cpu()
		hrimg_cpu, lrimg_cpu = hrimg.cpu(), lrimg.cpu()
		new_mse = loss.item()
		# mse += new_mse
		# psnr += to_psnr(new_mse)
		hrpil, srpil = DS.to_pil_image(hrimg_cpu[0]), DS.to_pil_image(srimg_cpu[0])
		mse += pil_mse(hrpil, srpil)
		psnr += pil_psnr(hrpil, srpil)
		if save_data:
			DS.save_image(hrimg_cpu[0], 'HR%s'%(i+1), sub_path='GroundTruth_'+testset_name)
			DS.save_image(lrimg_cpu[0], 'LR%s'%(i+1), sub_path='LowResolution_'+testset_name)
			del hrimg_cpu, lrimg_cpu
		if save_flag:
			DS.save_image(srimg_cpu[0], 'SR%s'%(i+1), sub_path=model.name+'_'+str(epochs)+'_'+testset_name)
		# Clean up GPU Space
		del lrimg, hrimg, srimg, srimg_cpu
		torch.cuda.empty_cache()
	test_end = time.time()
	
	mse /= len(testset)
	psnr /= len(testset)
	_print('Test Over')
	_print('MSE Loss: %.8f'%mse)
	_print('PSNR Score: %.3f'%psnr)
	_print('Model Run Time: %.7f'%run_time)
	_print('Test Run Time: %.7f'%(test_end-test_start))
	del model

def img_show(testset_name='BSDS200', epochs=None, img_idx=1):
	global device
	if use_gpu.get():
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	else:
		device = torch.device('cpu')
	print(device)
	used_device = device
	testset = DS.Testset(set_name=testset_name)
	hrimg, lrimg = testset[int(img_idx)-1]
	tar = DS.to_pil_image(hrimg)
	data_in = torch.Tensor([lrimg.numpy()])
	data_tar = torch.Tensor([hrimg.numpy()])
	data_tar = data_tar.to(device)
	data_in = data_in.to(device)
	if use_gpu.get():
		model_list = ['bicubic','fsrcnn','espcn','RRDBNet','Mynet_add']
	else:
		model_list = ['bicubic','fsrcnn','espcn','espcn_3l','Mynet_add_d16']
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
			data_out = DS.to_pil_image(data_out[0])
			model_output.append(data_out)
			model_psnr.append('~')
			del model
			continue
		with torch.no_grad():
			data_out = model(data_in)
			loss = criterion(data_out, data_tar)
			data_out = data_out.cpu()
			data_out = DS.to_pil_image(data_out[0])
			model_output.append(data_out)
			# model_psnr.append('%.4f'%to_psnr(loss.item()))
			model_psnr.append('%.4f'%pil_psnr(tar, data_out))
		del model, data_out
		torch.cuda.empty_cache()
	hrimg = DS.to_pil_image(hrimg)
	lrimg = DS.to_pil_image(lrimg)

	if axis_on.get():
		axis_flag = 'on'
	else:
		axis_flag = 'off'

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

def show_fun():
	_testset_name = testset_name.get()
	_epochs = epochs.get()
	_img_idx = img_idx.get()
	img_show(testset_name=_testset_name, epochs=_epochs, img_idx=_img_idx)

def button_fun():
	_model_name = model_name.get()
	_testset_name = testset_name.get()
	_epochs = epochs.get()
	_save_flag = save_flag.get()
	_save_data = save_data.get()
	test_model(_model_name, testset_name=_testset_name, epochs=_epochs,
					save_flag=_save_flag, save_data=_save_data)

main_window = tk.Tk(className='Super-Resolution App')
model_name = tk.Variable()
testset_name = tk.Variable()
epochs = tk.Variable()
save_flag = tk.BooleanVar()
save_data = tk.BooleanVar()
use_gpu = tk.BooleanVar()
axis_on = tk.BooleanVar()
img_idx = tk.Variable()

model_label = tk.Label(main_window, text='Model')
model_label.grid(row=1)
testset_label = tk.Label(main_window, text='Testset')
testset_label.grid(row=2)
epochs_label = tk.Label(main_window, text='Epochs')
epochs_label.grid(row=3)
img_idx_label = tk.Label(main_window, text='Img Idx')
img_idx_label.grid(row=4)

model_entry = tk.Entry(main_window, textvariable=model_name)
model_entry.grid(row=1, column=1)
testset_entry = tk.Entry(main_window, textvariable=testset_name)
testset_entry.grid(row=2, column=1)
epochs_entry = tk.Entry(main_window, textvariable=epochs)
epochs_entry.grid(row=3, column=1)
img_idx_entry = tk.Entry(main_window, textvariable=img_idx)
img_idx_entry.grid(row=4, column=1)
save_flag_check = tk.Checkbutton(main_window, variable=save_flag, text='Save Output')
save_flag_check.grid(row=5)
save_data_check = tk.Checkbutton(main_window, variable=save_data, text='Save Origin')
save_data_check.grid(row=5, column=1)
use_gpu_check = tk.Checkbutton(main_window, variable=use_gpu, text='Use GPU')
use_gpu_check.grid(row=6)
axis_on_check = tk.Checkbutton(main_window, variable=axis_on, text='Axis On')
axis_on_check.grid(row=6, column=1)

run_button = tk.Button(main_window, text='Compute whole Testset', command=button_fun)
run_button.grid(row=7, padx=20)
show_button = tk.Button(main_window, text='Show Idx Image Result', command=show_fun)
show_button.grid(row=7, column=1)

printer = tk.Text(main_window)
printer.grid(row=1, column=2, rowspan=7, columnspan=4, padx=20, pady=20)
scroll = tk.Scrollbar()
scroll.grid(row=1, column=6, rowspan=7, pady=20)

scroll.config(command=printer.yview)
printer.config(yscrollcommand=scroll.set)

def _print(print_obj, tktext_obj=printer):
	tktext_obj.config(state=tk.NORMAL)
	tktext_obj.insert(tk.END, str(print_obj)+'\n')
	tktext_obj.config(state=tk.DISABLED)
	tktext_obj.see(tk.END)
	print(print_obj)
# model_label.pack()
# testset_label.pack()
# epochs_label.pack()
# img_idx_label.pack()

# model_entry.pack()
# testset_entry.pack()
# epochs_entry.pack()
# img_idx_entry.pack()
# save_flag_check.pack()
# save_data_check.pack()
# run_button.pack()

main_window.mainloop()
