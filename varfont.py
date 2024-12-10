'''
DCGAN with 3 font size digits
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

DATA_FOLDER = './tf_data/VGAN/MNIST'
INPUT_BATCH_SIZE = 60

def mnist_data():
	compose = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
	out_dir = '{}/dataset/images'.format(DATA_FOLDER)
	return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

##### Data Preparation
data = mnist_data()
data_loader = DataLoader(data, batch_size=INPUT_BATCH_SIZE, shuffle=True)
tf.InteractiveSession()

paddings = tf.constant([[14, 14,], [14, 14]])



for n_batch, (batch,_) in enumerate(data_loader):

	x_train = batch.permute(0, 2, 3, 1).numpy()
	x_train = tf.reshape(x_train,(-1,28,28))
	y = tf.reshape(x_train,(-1,28,28,1))
	#withput scaling
	no_scaling = []
	for i in range(20):
		q  =np.full((64,64),-1.) #tf.image.resize_image_with_crop_or_pad(x,56,56)
		x = tf.image.resize_images(y[i], (32, 32))
		q[16:48,16:48] = tf.reshape(x,(32, 32)).eval()
		no_scaling.append(q)

	no_scaling = np.dstack(no_scaling)
	no_scaling = np.rollaxis(no_scaling,-1)
	no_scaling = no_scaling.astype(np.float32)

	#with scale = 0.5


	with_half_scaling = []
	for i in range(20,40):
		x = tf.image.resize_images(y[i], (16, 16))
		q  =np.full((64,64),-1.) #tf.image.resize_image_with_crop_or_pad(x,56,56)
		q[24:40,24:40] = tf.reshape(x,(16,16)).eval()
		#r = np.copy(tf.reshape(q,(56,56)).eval())
		with_half_scaling.append(q)
		# s = (r+1.)/2
		# plt.imshow(s)
		# plt.axis('on')
		# plt.show()

	with_half_scaling = np.dstack(with_half_scaling)
	with_half_scaling = np.rollaxis(with_half_scaling,-1)
	with_half_scaling = with_half_scaling.astype(np.float32)
	kernel = np.ones((3,3),np.uint8)
	with_double_scaling = []
	for i in range(40,60):
		x = tf.image.resize_images(y[i], (64, 64))
		r = np.copy(tf.reshape(x,(64,64)).eval())
		r = cv2.erode(r,kernel,iterations = 1)
		with_double_scaling.append(r)


	with_double_scaling = np.dstack(with_double_scaling)
	with_double_scaling = np.rollaxis(with_double_scaling,-1)
	with_double_scaling = with_double_scaling.astype(np.float32)


	if n_batch == 0:
 		dataset = tf.concat([no_scaling,with_half_scaling,with_double_scaling],axis = 0)
	else:
		dataset = tf.concat([dataset,no_scaling,with_half_scaling,with_double_scaling],axis = 0)
	if (n_batch+1)%100 == 0:
		d = dataset.eval()
		d = np.reshape(d,(-1,64*64))
		file_name = "train"+n_batch+".txt"
		np.savetxt(file_name,d,fmt='%f')
	print(n_batch)

dataset = dataset.eval()
print('dataset',dataset.shape)
