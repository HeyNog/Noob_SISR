import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data.bd_dataset as DS
import models.build_model as M
from utils.utils import *

import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
epochs_to_save = 10
UPF = 3

if __name__ == '__main__':
	# Train dataset & val dataset
	# trset = DS.Trainset(train_path=DS.train_path)
	trset = DS.Trainset(train_path="/dev/shm/train/", UPF=UPF)
	trloader = DS.DataLoader(trset, batch_size=8, shuffle=True, num_workers=4)
	valset = DS.Testset(set_name='BSDS200', UPF=UPF)
	valloader = DS.DataLoader(valset, batch_size=1)

	model = M.build_model('x%d_fsrcnn'%UPF, UPF=UPF)
	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	model.name += '_batch8_scheduler_small2k_norm'
	print(model)
	print(model.name)
	model.to(device)

	train_loss_loaded, train_loss = [], []
	val_loss, val_loss_loaded = [], []
	min_val_loss = 10.0
	min_val_epoch = 0
	preepochs = 0
	count = 0

	# try to load total_state_dict, t_s_d = False if failed
	t_s_d = M.load_train_state(model, preepochs, save_file='last_checkpoint.pth')
	if t_s_d:
		model.load_state_dict(t_s_d['model_state'])
		optimizer.load_state_dict(t_s_d['optimizer_state'])
		train_loss_loaded = t_s_d['train_loss']
		preepochs = t_s_d['epochs']
		val_loss_loaded = t_s_d['val_loss']
		min_val_loss = (np.array(val_loss_loaded)).min()


	criterion = nn.MSELoss()
	epochs = 0
	lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
									milestones=[50000, 100000, 200000, 300000, 400000],
									gamma=0.5,
									last_epoch=epochs+preepochs-1)
	while(True):
		epochs_loss, epochs_val_loss = 0.0, 0.0
		model.train()
		for i, datacp in enumerate(trloader):
			hrimg, lrimg = datacp
			hrimg, lrimg = hrimg.to(device), lrimg.to(device)

			optimizer.zero_grad()
			srimg = model(lrimg)
			# SmoothL1 loss used for training.
			loss  = criterion(srimg, hrimg)
			loss.backward()
			optimizer.step()
			iter_loss = len(lrimg)*loss.item()
			epochs_loss += iter_loss
			argname = ['MSELoss', 'Step']
			arglist = [loss.item(), i+1]
			argtype = ['%.6f', '%d']
			show_args_status(argname, arglist, args_type=argtype)
		
			del hrimg
			del lrimg
			del srimg
		torch.cuda.empty_cache()
		# Valildation
		model.eval()
		with torch.no_grad():
			for i, valcp in enumerate(valloader):
				hrval, lrval = valcp
				hrval, lrval = hrval.to(device), lrval.to(device)
				srval = model(lrval)
				epochs_val_loss += criterion(srval, hrval).item()
				del srval
				del hrval
				del lrval
		# Note epoch loss
		epochs_loss /= len(trset)
		epochs_val_loss /= len(valset)
		val_loss.append(epochs_val_loss)
		train_loss.append(epochs_loss)
		train_loss_loaded.append(epochs_loss)
		val_loss_loaded.append(epochs_val_loss)
		# Print epoch messege
		print('\nEpoch %d train loss: %.7f'%(preepochs+epochs+1, epochs_loss))
		print('Epoch %d val   loss: %.7f'%(preepochs+epochs+1, epochs_val_loss))
		torch.cuda.empty_cache()
		lr_scheduler.step()
		# Save train state to state dict file
		if epochs % epochs_to_save == epochs_to_save-1:
			M.save_train_state(model, optimizer, preepochs+epochs+1,
						train_loss_loaded, save_file='last_checkpoint.pth', val_loss=val_loss_loaded)
		if min_val_loss >= epochs_val_loss:
			min_val_loss = epochs_val_loss
			min_val_epoch = epochs + preepochs + 1
			count = 0
			print('Save min val loss checkpoint in epoch%d'%min_val_epoch)
			M.save_train_state(model, optimizer, preepochs+epochs+1,
					train_loss_loaded, save_file='min_val_checkpoint.pth', val_loss=val_loss_loaded)
		elif count <= 500:
			count += 1
			# if count == 100:
			# 	print('Learning Rate Decayed in Epoch%d.'%(preepochs+epochs+1))
			# 	optimizer = M.lr_decay(optimizer, decay_rate=2)
			# if count == 300:
			# 	print('Learning Rate Decayed in Epoch%d.'%(preepochs+epochs+1))
			# 	optimizer = M.lr_decay(optimizer, decay_rate=2)
		else:
			print('End Training in Epoch %d'%(epochs+1+preepochs))
			print('Min_val_loss: %.7f in Epoch %d'%(min_val_loss, min_val_epoch))
			break
		epochs += 1



		
