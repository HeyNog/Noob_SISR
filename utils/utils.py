#! /usr/bin/env python
#-*- coding:utf-8 -*-

def show_args_status(args_name, args_list, args_type=None, args_unit=None):
	assert len(args_list) == len(args_name)
	if args_type != None:
		assert len(args_type) == len(args_list)
	else:
		args_type = ['%s'] * len(args_list)

	if args_unit != None:
		assert len(args_unit) == len(args_list)
	else:
		args_unit = [''] * len(args_list)

	status_str = '\r'
	for i, argsn in enumerate(args_name):
		argstr = args_type[i]%args_list[i] + args_unit[i]
		status_str += argsn + ':' + argstr + '    '
	print(status_str, end=' ', flush=True)




def print_args_status(args_name, args_list, args_type=None, args_unit=None):
	assert len(args_list) == len(args_name)

	if args_type != None:
		assert len(args_type) == len(args_list)
	else:
		args_type = ['%s'] * len(args_list)

	if args_unit != None:
		assert len(args_unit) == len(args_list)
	else:
		args_unit = [''] * len(args_list)

	status_str = ''
	for i, argsn in enumerate(args_name):
		argstr = args_type[i]%args_list[i] + args_unit[i]
		status_str += argsn + ':' + argstr + '    '
	print(status_str)




def asking(ask, supposed_answer, fun=None):
	'''
	function:: asking(ask, supposed_answer, fun=None)

	Keep asking until input a supposed answer.
	
	Inputs:
		ask: The question that printed.
		supposed_answer: A list/tuple of the supposed input; or a dict of supposed inputs & corresponding answers; or a type that the inputs should be.
		fun(optional): A function that transforms the keyboard string inputs. Default: None.
	
	Outputs:
		answer: the output depends on the keyboard inputs & supposed_answer.
	
	'''
	while(True):
		answer = input(ask)
		if fun != None:
			try:
				answer = fun(answer)
			except Exception as e:
				print(e)
				continue

		if isinstance(supposed_answer, (list, tuple)):
			if set(supposed_answer).issuperset(set([answer])):
				break
		elif isinstance(supposed_answer, dict):
			answer = supposed_answer.get(answer, '__Loop Continue.')
			if answer != '__Loop Continue.':
				break
		elif isinstance(supposed_answer, type):
			try:
				answer = supposed_answer(answer)
				break
			except ValueError as e:
				print(e)
	return answer