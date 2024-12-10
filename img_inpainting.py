import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import random
import glob

filelist = glob.glob('./datasetL/*.png')
train_set = np.array([((np.array(Image.open(fname))/255)-0.5)/0.5 for fname in filelist])
print(train_set.shape)
train_set = np.reshape(train_set,(-1,28,28,1))


def save_original(images):
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(images[k], (28, 28)), cmap='gray')

    label = 'Original'
    fig.text(0.5, 0.04, label, ha='center')
    path = '_original_noisy.png'
    plt.savefig(path)

def noisy_images(images):
	copy = np.copy(images)
	
	for c in copy:
		x = random.randint(0,18)
		y = random.randint(0,18)
		l = random.randint(7,10)
		b = random.randint(7,10)
		for i in range(x,x+l):
			for j in range(y,y+b):
				c[i,j]  = 1.0
		#plt.imshow(c,cmap="gray")
		#plt.show()
	return copy
	
	
# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        print('gen....\n')
        conv1 = tf.layers.conv2d(x, 512, [4, 4], strides=(1, 1), padding='same')
        lrelu1 =  tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)
        print('\nconv1 shape',conv1.shape)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        print('\nconv2 shape',conv2.shape)
        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 64, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        print('\nconv3 shape',conv3.shape)
        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 32, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        print('\nconv4 shape',conv4.shape)
        # output layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [2, 2], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv5)
        print('\n\n\n g ',o.shape,"conv5 ",conv5,'\n\n\n')
        
        return o

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        print('discriminator')
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = tf.nn.leaky_relu(conv1, 0.2)
        print("\n\n\nconv 1 ",conv1.shape,"\n\n\n")
        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        print("\n\n\nconv 2 ",conv2.shape,"\n\n\n")
        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        print("\n\n\nconv 3 ",conv3.shape,"\n\n\n")
        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        print("\n\n\nconv 4 ",conv4.shape,"\n\n\n")
        conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.sigmoid(conv5)
        print('\n\n\n dc ',o.shape,"conv5 ",conv5.shape,'\n\n\n')
        return o, conv5

fixed_z_ = noisy_images(train_set[0:25])
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 70
lr = 0.0002
train_epoch = 200000

# variables : input
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
z = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
isTrain = tf.placeholder(dtype=tf.bool)
#sess = tf.InteractiveSession()
# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
print("\nreal\n")
D_real, D_real_logits = discriminator(x, isTrain)
print("\nfake\n")
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver(max_to_keep = 100)

# results save folder
root = 'results/'
model = 'CHARS_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

save_original(fixed_z_)


train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()


for epoch in range(train_epoch):
    G_losses = []
    D_losses = []

    epoch_start_time = time.time()
    for iter in range(len(train_set) // batch_size):
        # update discriminator


        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        z_ = noisy_images(x_)
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        index = random.randint(0,(len(train_set) // batch_size)-1)
        z_ = noisy_images(train_set[index*batch_size:(index+1)*batch_size])
        x_ = train_set[index*batch_size:(index+1)*batch_size]
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_,x:x_,isTrain: True})
        G_losses.append(loss_g_)
        if iter%10 == 0:
        	print(iter," ",end=' ')
    
    saver.save(sess, './weights/weight',global_step=epoch)
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

saver.save(sess, "./weights/final_weights")
sess.close()
