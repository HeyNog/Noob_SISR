import torch
import os
import json

import data.bd_dataset as DS
import models.build_model as M
from utils.test_utils import test_model
from utils.utils import asking, print_args_status
import utils.pic_utils as PU

yes_no_opt = {'y':True, 'Y':True, 'n':False, 'N':False}
# Location of trained models
models_location = os.path.join(PU.base_path, 'models', 'torch_models_save')

if __name__ == '__main__':
	# Read Test Options
	opt_file = os.path.join(PU.base_path, 'utils', 'test_option.json')
	with open(opt_file, 'r') as f:
		test_opt = json.load(f)

	# Parallelism Configuration.
	gpu_flag = test_opt['gpu_flag']
	if gpu_flag:
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		_device = device
	else:
		device = torch.device('cpu')
		num_threads = test_opt['num_threads']
		torch.set_num_threads(num_threads)
		# _device for print function.
		_device = str(device) + '_%d'%num_threads


	# Args & Options
	UPF = test_opt['Scale Factor']
	testset_name = test_opt['Testset']
	save_output = test_opt['save_output']
	save_origin = test_opt['save_origin']
	num_loops = test_opt['num_loops']


	# Load trained model
	model_list = test_opt['model_list']


	# Testset
	testset = DS.Testset(set_name=testset_name, UPF=UPF)


	# Test Ready
	print('\n\n')
	print('>>> SISR Options <<<')
	print('Device: %s'%_device)
	print('Scale Factor: %d'%UPF)
	print('Testset: %s'%testset_name)
	print('Model List: %s'%model_list)
	print('')
	print('>>> Test Options <<<')
	print('Save outputs: %s'%save_output)
	print('Save GT & LQ: %s'%save_origin)
	print('Test Loop: %d'%num_loops)

	input('Press any key to start...')



	# Test Part
	for model_name in model_list:
		print('')
		model = M.build_model(model_name, UPF=UPF)
		total_state_file = M.load_train_state(model, '',
												save_file='min_val_checkpoint.pth',
												map_location={'cuda:1':str(device)},
												notice=True)
		if total_state_file:
			try:
				model.load_state_dict(total_state_file['model_state'])
			except KeyError as e:
				model.load_state_dict(total_state_file)
		# Configure save flag before test
		save_output = test_opt['save_output']
		save_origin = test_opt['save_origin']
		# Test Repeatedly: num_loops > 1
		if num_loops > 1:
			run_time, test_time = [], []
			for i in range(num_loops):
				run_time_i, test_time_i, _, psnr_i = test_model(model, testset,
															device=device,
															save_output=save_output,
															save_origin=save_origin,
															verbose=False)
				run_time.append(run_time_i)
				test_time.append(test_time_i)
				# Avoid over-writing
				save_output, save_origin = False, False
				
				# Print Status
				args_name = ['Test Loop', 'Run time', 'Test time', 'PSNR']
				args_list = [i+1, run_time_i, test_time_i, psnr_i]
				args_type = ['%d', '%.4f', '%.2f', '%.3f']
				args_unit = ['', 's', 's', 'dB']
				print_args_status(args_name, args_list, args_type=args_type, args_unit=args_unit)

			# Exclude the fastest & the slowest
			if num_loops > 2:
				run_time.sort()
				test_time.sort()
				run_time, test_time = run_time[1:-1], test_time[1:-1]

			avg_run_time = sum(run_time)/len(run_time)
			avg_test_time = sum(test_time)/len(test_time)

			# Print Conclusion
			args_name = ['Test Loop', 'Avg Run time', 'Avg Test time', 'PSNR']
			args_list = [i+1, avg_run_time, avg_test_time, psnr_i]
			args_type = ['%d', '%.4f', '%.4f', '%.3f']
			print('%s :: %s'%(model.name, type(model)))
			print_args_status(args_name, args_list, args_type=args_type, args_unit=args_unit)

		# Single Test: num_loop == 1
		else:
			test_model(model, testset,
						device=device,
						save_output=save_output,
						save_origin=save_origin,
						verbose=True)