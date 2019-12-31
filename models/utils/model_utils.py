# /Noob_SISR/models/utils/model_utils.py
import torch
import os

# Default save model path
this_location = os.path.dirname(os.path.dirname(__file__))
save_path = os.path.join(this_location, 'torch_models_save')

def save_train_state(model, optimizer, epochs, train_loss,
			save_path=save_path, save_file=None, **kw):
	'''
	Save pytorch train state.
	Including model state, optimizer state, epochs, train loss(list).
	The save model should have attr 'name'.
	Save additional status by using **kw:
		save_train_state(..., key1=value1, key2=value2)
		or save_train_state(..., **additional_state_dict)

	When save file successfully, return True.
	Else return False.
	'''
	# Make state dict
	total_state_dict = {'epochs': epochs,
						'model_state': model.state_dict(),
						'optimizer_state': optimizer.state_dict(),
						'train_loss': train_loss}
	for new_keys in kw.keys():
		if total_state_dict.get(new_keys)==None:
			total_state_dict[new_keys] = kw.get(new_keys)
		else:
			old_value, new_value = total_state_dict[new_keys], kw[new_keys]
			print('Warning: Key %s Exists (Value: %s)'%(new_keys, old_value))
			print('Replace %s:%s by %s:%s'%(new_keys, old_value, new_keys, new_value))
			total_state_dict[new_keys] = kw.get(new_keys)
	# Save state dict file
	save_path = os.path.join(save_path, model.name)
	try:
		os.mkdir(save_path)
	except FileExistsError as e:
		print('Path %s Exists'%save_path)
	if save_file == None:
		save_file = 'Train_State%d.pth'%(epochs)
	save_file = os.path.join(save_path, save_file)
	try:
		torch.save(total_state_dict, save_file)
	except Exception as e:
		print('Save state failed %s'%e)
		return False
	else:
		print('Saved %s successfully'%save_file)
		return True

def load_train_state(model, preepochs, notice=False,
					save_path=save_path, save_file='Train_State',
					map_location=None):
	'''
	Load pytorch train state
	including model state, optimizer state, epochs, train loss(list)
	the input model of this function should have attr 'name'

	when save file successfully, return total_state_dict
	else return False
	'''
	save_path = os.path.join(save_path, model.name)
	if save_file == 'Train_State':
		save_file += '%s.pth'%(preepochs)
	save_file = os.path.join(save_path, save_file)
	try:
		total_state_dict = torch.load(save_file, map_location=map_location)
	except Exception as e:
		if notice:
			print('Load state failed %s'%e)
		return False
	else:
		print('Loaded %s successfully'%save_file)
		return total_state_dict

