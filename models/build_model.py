'''
Module: /Noob_SISR/models/build_model.py

Methods:
	build_model
	save_train_state (from .utils.model_utils)
	load_train_state (from .utils.model_utils)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os

from .utils.model_utils import *
from . import SISR_MODELS
from .SISR_MODELS import *

# Search for Module Class when init.
attr_list = dir(SISR_MODELS)
model_list = []
for attr in attr_list:
	tmp = getattr(SISR_MODELS, attr)
	if hasattr(tmp, 'forward'):
		model_list.append(attr)

def build_model(model_name, *args, **kw):
	model_class = ''
	for model_type in model_list:
		try:
			idx = model_name.index(model_type)
		except ValueError as e:
			pass
		else:
			# Update with the longer name(Using more advanced model)
			if len(model_class) < len(model_type):
				model_class = model_type
	assert model_class != '' , 'Model Name Not Found'
	model_builder = getattr(SISR_MODELS, model_class)
	model = model_builder(*args, **kw)

	model.name = model_name
	return model

if __name__ == '__main__':
	model = input('Build Model:')
	model = build_model(model)