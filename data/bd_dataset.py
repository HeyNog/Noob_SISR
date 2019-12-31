#Noob_SISR/data/dataset.py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from PIL import Image
import numpy as np
import os
import random

# default path of train/val/test dataset
data_base = os.path.dirname(__file__)
train_path = os.path.join(data_base, 'train')
val_path = os.path.join(data_base, 'val')
test_path = os.path.join(data_base, 'test')
save_path = os.path.join(data_base, 'save')

class Trainset(Dataset):
	def __init__(self,
					train_path=train_path,
					UPF=4,
					simple=False,
					color_mode='RGB'):
		self.path = train_path
		# img transforms for data augmentation
		self.da = T.Compose([T.RandomCrop(240),
			T.RandomHorizontalFlip(),
			T.RandomVerticalFlip()])
		# img transforms for preprocessing
		self.pp = T.Compose([T.ToTensor(),
								T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
		# Randomly resize HR Images
		self.randomresize = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
		self.randomrotate = [0, 90, 180, 270]
		self.UPF = UPF
		self.simple = simple
		self.color_mode = color_mode

	def __len__(self):
		# DIV2K 800 + Flickr2K 2650
		return 3450

	def __getitem__(self, idx):
		UPF = self.UPF
		# get the img file 'train (idx+1).png'
		file_name = 'train (%d).png'%(idx+1)
		file_name = os.path.join(self.path, file_name)
		hrimg = Image.open(file_name).convert(self.color_mode)
		# Data Augmentation
		# PIL.Image.size is (W,H) while T.Resize.size should be (H,W)
		if not(self.simple):
			# Random Resize
			rand_scale = random.choice(self.randomresize)
			hrimg = T.Resize(size=(int(hrimg.size[1]*rand_scale), int(hrimg.size[0]*rand_scale)),
								interpolation=Image.BILINEAR)(hrimg)
			
			# Data Augmentation
			hrimg = self.da(hrimg)
			
			# Random Rotate
			rotate_angle = random.choice(self.randomrotate)
			hrimg = hrimg.rotate(rotate_angle)
		
		# Ensure that the size of HR img can be divided
		height, width = hrimg.size[1]//UPF, hrimg.size[0]//UPF
		hrimg = T.CenterCrop(size=(height*UPF, width*UPF))(hrimg)
		lrimg = T.Resize(size=(height, width),
							interpolation=Image.BICUBIC)(hrimg)
		hrimg = self.pp(hrimg)
		lrimg = self.pp(lrimg)
		return (hrimg, lrimg)

class Valset(Dataset):
	def __init__(self,
					val_path=val_path,
					UPF=4,
					color_mode='RGB'):
		self.path = val_path
		# same as the Trainset
		self.da = T.Compose([T.RandomCrop(640),
			T.RandomHorizontalFlip(),
			T.RandomVerticalFlip()])
		self.pp = T.Compose([T.ToTensor(),
			T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

	def __len__(self):
		# DIV2K val 100
		return 100

	def __getitem__(self, idx):
		file_name = 'val (%d).png'%(idx+1)
		file_name = os.path.join(self.path, file_name)
		hrimg = self.da(Image.open(file_name).convert(self.color_mode))
		lrimg = T.Resize(size=(hrimg.size[1]//4,hrimg.size[0]//4),
							interpolation=Image.BICUBIC)(hrimg)
		hrimg = self.pp(hrimg)
		lrimg = self.pp(lrimg)
		return (hrimg, lrimg)

class Testset(Dataset):
	def __init__(self, base_path=test_path,
					set_name='BSDS200',
					file_type='.png',
					UPF=4,
					color_mode='RGB'):
		self.file_type = file_type
		self.set_name = set_name
		self.path = os.path.join(base_path, set_name)
		# self.pp = T.Compose([T.ToTensor()])
		self.pp = T.Compose([T.ToTensor(),
			T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
		self.upf = UPF
		self.color_mode = color_mode

	def __len__(self):
		# length of classical dataset
		switch_setname = {
			'BSDS100': 100,
			'BSDS200': 200,
			'General100': 100,
			'historical': 10,
			'manga109': 109,
			'Set5': 5,
			'Set14': 14,
			'T91': 91,
			'urban100': 100,
			'LowResolution_BSDS100': 100
		}
		try:
			length = switch_setname[self.set_name]
		except KeyError as e:
			print('Undefined set name, set length = 1')
			print('Please define dataset length in %s'%__file__)
			length = 1
		return length

	def __getitem__(self, idx):
		file_name = 'test (%d)%s'%(idx+1, self.file_type)
		file_name = os.path.join(self.path, file_name)
		hrimg = Image.open(file_name)
		hrimg = hrimg.convert(mode=self.color_mode)
		# the height and width of lr img
		height, width = hrimg.size[1]//self.upf, hrimg.size[0]//self.upf
		# CenterCrop to ensure lr img can be x4 reconstructed
		hrimg = T.CenterCrop(size=(height*self.upf, width*self.upf))(hrimg)
		lrimg = T.Resize(size=(height, width),
							interpolation=Image.BICUBIC)(hrimg)
		hrimg = self.pp(hrimg)
		lrimg = self.pp(lrimg)
		return (hrimg, lrimg)

def to_pil_image(img_tensor, color_mode='RGB'):
	# [-1,1] tensor Un-Normalize to [0,1] tensor
	# img_tensor = img_tensor/2 + 0.5
	img_tensor = (img_tensor+1)*255
	img_tensor = torch.ceil(img_tensor) // 2
	img_tensor = img_tensor/255
	img_tensor = img_tensor.clamp(0, 1)
	img = T.ToPILImage(mode=color_mode)(img_tensor)
	img = img.convert(color_mode)
	return img

def show_image(img_data):
	if isinstance(img_data, (torch.Tensor, np.ndarray)):
		img_data = to_pil_image(img_data)
	img_data.show()

def save_image(img_data, save_file,
				save_path=save_path,
				sub_path=None,
				file_type='.png',
				notice=False):
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

if __name__ == '__main__':
	dataset = Trainset()
	print('form dataset')
	print(dataset[10][0].shape)
	print(dataset[10][1].shape)
	dl = DataLoader(dataset, batch_size=64, shuffle=True)
	print('from DataLoader')
	for i, cp in enumerate(dl):
		print(i, cp[0].shape, cp[1].shape)
		if i == 100:
			break
	print('test over')
