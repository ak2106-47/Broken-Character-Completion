'''
Original code with noise vector of size 100
'''

from IPython import display
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from  utils.utils import Logger
import tensorflow as tf
import numpy as np

DATA_FOLDER = './tf_data/VGAN/MNIST'
IMAGE_PIXELS = 28*28
NOISE_SIZE = 100
BATCH_SIZE = 100
IMAGE_DIM = 28
SUBIMAGE_DIM = 10
num_epochs = 200

def noise(n_rows, n_cols):
    return np.random.normal(size=(n_rows, n_cols))

def xavier_init(size):
    in_dim = size[0] if len(size) == 1 else size[1]
    stddev = 1. / np.sqrt(float(in_dim))
    return tf.random_uniform(shape=size, minval=-stddev, maxval=stddev)

def images_to_vectors(images):
    return images.reshape(images.shape[0], IMAGE_PIXELS)

def vectors_to_images(vectors):
    return vectors.reshape(vectors.shape[0], IMAGE_DIM, IMAGE_DIM, 1)

def mnist_data():
    compose = transforms.Compose(
        [transforms.CenterCrop(IMAGE_DIM),transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


# Load data
data = mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
# Num batches
num_batches = len(data_loader)

## Discriminator

# Input
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


## Generator

# Input
Z = tf.placeholder(tf.float32, shape=(None, NOISE_SIZE))
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


def discriminator(x):
    #print("x shape ", x.shape)
    l1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(x,   D_W1) + D_B1, .2), .3)
    #print("l1 shape ", l1.shape)
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
G_sample = generator(Z)
D_real = discriminator(X)
D_fake = discriminator(G_sample)

# Losses
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

# Optimizers
D_opt = tf.train.AdamOptimizer(2e-4).minimize(D_loss, var_list=D_var_list)
G_opt = tf.train.AdamOptimizer(2e-4).minimize(G_loss, var_list=G_var_list)



# Start interactive session
session = tf.InteractiveSession()
saver = tf.train.Saver()

# Init Variables
tf.global_variables_initializer().run()
# Init Logger
logger = Logger(model_name='DCGAN1', data_name='CIFAR10')

for epoch in range(num_epochs):
    for n_batch, (batch,_) in enumerate(data_loader):

        # 1. Train Discriminator
        y = batch.permute(0, 1, 3, 2).numpy()
        #print('batch ' , y.shape)
        
        Z_batch = noise(BATCH_SIZE, NOISE_SIZE)
        X_batch = images_to_vectors(y)

        feed_dict = {X: X_batch, Z: Z_batch}

        _, d_error, d_pred_real, d_pred_fake = session.run([D_opt, D_loss, D_real, D_fake], feed_dict=feed_dict)

        # 2. Train Generator
        feed_dict = {Z: noise(BATCH_SIZE, NOISE_SIZE)}

        _, g_error = session.run([G_opt, G_loss], feed_dict=feed_dict)

        if n_batch % 100 == 0:
            display.clear_output(True)
            # Generate images from test noise
            test_images = session.run(
                G_sample, feed_dict={Z: noise(16, NOISE_SIZE)}
            )
            test_images = vectors_to_images(test_images)
            # Log Images
            logger.log_images(test_images, 16, epoch, n_batch, num_batches,'1',format='NHWC');
            # Log Status
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
save_path = saver.save(session, "./init_gan_weights/final_weights")
