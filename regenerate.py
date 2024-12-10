import tensorflow as tf
import numpy as np
from IPython import display
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from  utils.utils import Logger
from matplotlib import pyplot as plt
import random
DATA_FOLDER = './tf_data/VGAN/MNIST'

MASK_PIXELS = 10*10
IMAGE_PIXELS = 28*28

NOISE_SIZE = 100
BATCH_SIZE = 16

IMAGE_DIM = 28
MASK_DIM = 10


def noise(n_rows, n_cols):
	return np.random.normal(size=(n_rows, n_cols))

def xavier_init(size):
	in_dim = size[0] if len(size) == 1 else size[1]
	stddev = 1. / np.sqrt(float(in_dim))
	return tf.random_uniform(shape=size, minval=-stddev, maxval=stddev)

def images_to_vectors(images):
	return images.reshape(images.shape[0], IMAGE_PIXELS)

def vectors_to_images(vectors):
	return vectors.reshape(vectors.shape[0], IMAGE_DIM, IMAGE_DIM,1)
def noisy_images(images):

	mask = np.ones_like(images,dtype = np.float32)
	c = 0
	imgs = np.copy(images)
	for s in imgs:
		
		n = noise(10,10)
		start_x = random.randint(0,18)
		start_y = random.randint(0,18)
		for i in range(start_x,start_x+10):
			for j in range(start_y,start_y+10):
				#print(start_x,start_y,s.shape)
				s[i,j] = n[(i-start_x),(j-start_y)]
				mask[c,i,j] = 0.0
		c = c+1

	return imgs, mask

def mnist_data():
	compose = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
	out_dir = '{}/dataset/images'.format(DATA_FOLDER)
	return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

def merge_images(image_batch, size):
    image_batch = np.reshape(image_batch,(BATCH_SIZE,28,28))
    h,w = image_batch.shape[1], image_batch.shape[2]
    
    img = np.zeros((int(h*size[0]), w*size[1]))
    res = np.zeros((int(h*size[0]), w*size[1]))
    for idx, im in enumerate(image_batch):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = im
        
#    for i in range(int(h*size[0])):
#         for j in range(int(w*size[0])):
#         res[i,j] = 
         
    return img

data = mnist_data()
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

#Generator Weights

with tf.name_scope("generator_weights"):
    # Layer 1 Variables
	G_W1 = tf.Variable(xavier_init([NOISE_SIZE, 256]))
	G_B1 = tf.Variable(xavier_init([256]))

    # Layer 2 Variables
	G_W2 = tf.Variable(xavier_init([256, 512]))
	G_B2 = tf.Variable(xavier_init([512]))

    # Layer 3 Variables
	G_W3 = tf.Variable(xavier_init([512, 1024]))
	G_B3 = tf.Variable(xavier_init([1024]))

    # Out Layer Variables
	G_W4 = tf.Variable(xavier_init([1024, IMAGE_PIXELS]))
	G_B4 = tf.Variable(xavier_init([IMAGE_PIXELS]))

# Store Variables in list
G_var_list = [G_W1, G_B1, G_W2, G_B2, G_W3, G_B3, G_W4, G_B4]


#Discriminator

X = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS))
with tf.name_scope("discriminator_weights"):
	# Layer 1 Variables
	D_W1 = tf.Variable(xavier_init([IMAGE_PIXELS, 1024]))
	D_B1 = tf.Variable(xavier_init([1024]))

	# Layer 2 Variables
	D_W2 = tf.Variable(xavier_init([1024, 512]))
	D_B2 = tf.Variable(xavier_init([512]))

	# Layer 3 Variables
	D_W3 = tf.Variable(xavier_init([512, 256]))
	D_B3 = tf.Variable(xavier_init([256]))

	# Out Layer Variables
	D_W4 = tf.Variable(xavier_init([256, 1]))
	D_B4 = tf.Variable(xavier_init([1]))

# Store Variables in list
D_var_list = [D_W1, D_B1, D_W2, D_B2, D_W3, D_B3, D_W4, D_B4]

# Generator input
Z = tf.placeholder(tf.float32, shape=(None, NOISE_SIZE))

def discriminator(x):
	l1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(x,   D_W1) + D_B1, .2), .3)
	l2 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(l1,  D_W2) + D_B2, .2), .3)
	l3 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(l2,  D_W3) + D_B3, .2), .3)
	out = tf.matmul(l3, D_W4) + D_B4
	return out

def generator(z):

	l1 = tf.nn.leaky_relu(tf.matmul(z,  G_W1) + G_B1, .2)
	l2 = tf.nn.leaky_relu(tf.matmul(l1, G_W2) + G_B2, .2)
	l3 = tf.nn.leaky_relu(tf.matmul(l2, G_W3) + G_B3, .2)
	out = tf.nn.tanh(tf.matmul(l3, G_W4) + G_B4)
	return out


#G_opt = tf.train.AdamOptimizer(2e-4).minimize(G_loss, var_list=G_var_list)

G_sample = generator(Z)
D_fake = discriminator(G_sample)
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

session = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(session, "../init_gan_weights/final_weights")

for n_batch, (batch,_) in enumerate(data_loader):
	print(batch.shape)
	y = batch[0:BATCH_SIZE]
	break
	
#print("ts shape ",test_samples.shape)
test_samples = y.permute(0, 1, 3, 2).numpy()
test_samples = images_to_vectors(test_samples)
test_samples = np.reshape(test_samples,(-1,28,28))
print(test_samples.shape)
fi,batch_mask = noisy_images(test_samples)
test_samples_nhwc = vectors_to_images(test_samples)
test_samples = images_to_vectors(test_samples)

logger = Logger(model_name='test', data_name='generator_output')
logger1 =  Logger(model_name='test', data_name='true_image')
logger2 = Logger(model_name='test', data_name='noisy_image')

logger1.log_images(test_samples_nhwc,1, 1, 1, 1, '1',format='NHWC');

test_noise_reshape = images_to_vectors(fi)
test_noise_reshape = vectors_to_images(test_noise_reshape)
logger2.log_images(test_noise_reshape,1, 1, 1, 1,'1' ,format='NHWC');

#batch_mask = np.ones((BATCH_SIZE,28,28),np.float32)
#batch_mask[:,9:19,9:19] = 0.0
batch_mask = images_to_vectors(batch_mask)
images = tf.placeholder(tf.float32,shape = (BATCH_SIZE,28*28))
recovered_images = tf.reshape(G_sample,(BATCH_SIZE,28*28))
contextual_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(tf.multiply(images,batch_mask) - tf.multiply(G_sample, batch_mask))), 1)

perceptual_loss = G_loss
complete_loss = contextual_loss + 0.001 * perceptual_loss
grad_complete_loss = tf.gradients(complete_loss, Z)
zhats = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))


vel = 0
momentum = 0.9
learning_rate = 0.01
for i in range(10000):
    fd = {
        Z: zhats,
        images: test_samples
    }
    run = [complete_loss, grad_complete_loss, G_sample]
    loss, g, G_imgs = session.run(run, feed_dict=fd)

    if (i%500 is 0):
        print("loss in iteration: "+ str(i) + " is: " + str(np.mean(loss)))
        logger.log_images(vectors_to_images(G_imgs),1, 1, 1, 1, '1',format='NHWC');
    prev_vel = np.copy(vel)
    vel = momentum*vel - learning_rate*g[0]
    zhats += -momentum* prev_vel + (1+momentum)*vel
    zhats = np.clip(zhats, -1, 1)

created_images = (G_imgs + 1.)/2
im = merge_images(created_images, [4,4])
plt.imshow(im)
plt.axis('on')
#plt.show()






