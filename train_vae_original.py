# Simple VAE to see if VAEs can learn sparsity inducing distributions
# Kerem Tezcan, CVL
# initial: 28.05.2017
# last mod: 30.05.2017 

from __future__ import division
from __future__ import print_function
# import os.path

import numpy as np
import time as tm
import tensorflow as tf
import os
import sys
from datetime import datetime
import random
import time

t0 = time.time()

SEED = 1001
seed = 1
np.random.seed(seed=1)

import argparse

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--mode', type=str, default='jonatank')
parser.add_argument('--modality', type=str, default='FLAIR')

args = parser.parse_args()

# mode=sys.argv[2]
mode = args.mode  # 'MRIunproc'
modality = args.modality # 'FLAIR' 'T1_' 'T1POST' 'T1PRE' 'T2'

print('Modality: ', modality)

logdir = "/scratch_net/bmicdl03/jonatank/logs/ddp/vae/" + modality + datetime.now().strftime("%Y%m%d-%H%M%S")
print(logdir)
#os.environ["CUDA_VISIBLE_DEVICES"]=os.environ['SGE_GPU']

# parameters
# ==============================================================================
# ==============================================================================

user = 'jonatank'  # 'Ender'

# mode=sys.argv[2]
mode = args.mode  # 'MRIunproc'

# mode='MRIunproc' #'sparse', 'nonsparse', 'MNIST', 'circ', 'Lshape', 'edge', 'Lapedge', 'spiketrain', 'Gaussedge'

# ndims=int(sys.argv[3])
ndims = 28
useMixtureScale = True
noisy = 50
batch_size = 1000
usebce = False
kld_div = 25.0
nzsamp = 1

train_size = 5000
test_size = 1000

if useMixtureScale:
    kld_div = 1.

std_init = 0.05

input_dim = ndims * ndims
fcl_dim = 500

# lat_dim=int(sys.argv[1])
lat_dim = 60
print(">>> lat_dim value: " + str(lat_dim))
print(">>> mode is: " + mode)

lat_dim_1 = max(1, np.floor(lat_dim / 2))
lat_dim_2 = lat_dim - lat_dim_1

num_inp_channels = 1

# make a dataset to use later
# ==============================================================================
# ==============================================================================
#DS = Dataset(train_size, test_size, ndims, noisy, seed, mode, downscale=True)
from dataloader import MR_image_data
MRi = MR_image_data(dirname='/scratch_net/bmicdl03/jonatank/data/', trainset_ratio = 1, noiseinvstd=0, patchsize=28, modality='AXFLAIR_')

print('CAME HERE!! 1')

# make a simple fully connected network
# ==============================================================================
# ==============================================================================

tf.reset_default_graph()

print('CAME HERE!! 11')

sess = tf.InteractiveSession()

print('CAME HERE!! 12')


# define the activation function to use:
def fact(x):
    # return tf.nn.tanh(x)
    return tf.nn.relu(x)


# define the input place holder
x_inp = tf.placeholder("float", shape=[None, input_dim])
# x_rec = tf.placeholder("float", shape=[None, input_dim])
l2_loss = tf.constant(0.0)

# define the network layer parameters
intl = tf.truncated_normal_initializer(stddev=std_init, seed=SEED)
intl_cov = tf.truncated_normal_initializer(stddev=std_init, seed=SEED)

with tf.variable_scope("VAE") as scope:
    enc_conv1_weights = tf.get_variable("enc_conv1_weights", [3, 3, num_inp_channels, 32], initializer=intl)
    enc_conv1_biases = tf.get_variable("enc_conv1_biases", shape=[32], initializer=tf.constant_initializer(value=0))

    enc_conv2_weights = tf.get_variable("enc_conv2_weights", [3, 3, 32, 64], initializer=intl)
    enc_conv2_biases = tf.get_variable("enc_conv2_biases", shape=[64], initializer=tf.constant_initializer(value=0))

    enc_conv3_weights = tf.get_variable("enc_conv3_weights", [3, 3, 64, 64], initializer=intl)
    enc_conv3_biases = tf.get_variable("enc_conv3_biases", shape=[64], initializer=tf.constant_initializer(value=0))

    mu_weights = tf.get_variable(name="mu_weights", shape=[int(input_dim * 64), lat_dim], initializer=intl)
    mu_biases = tf.get_variable("mu_biases", shape=[lat_dim], initializer=tf.constant_initializer(value=0))

    logVar_weights = tf.get_variable(name="logVar_weights", shape=[int(input_dim * 64), lat_dim], initializer=intl)
    logVar_biases = tf.get_variable("logVar_biases", shape=[lat_dim], initializer=tf.constant_initializer(value=0))

    if useMixtureScale:

        dec_fc1_weights = tf.get_variable(name="dec_fc1_weights", shape=[int(lat_dim), int(input_dim * 48)],
                                          initializer=intl)
        dec_fc1_biases = tf.get_variable("dec_fc1_biases", shape=[int(input_dim * 48)],
                                         initializer=tf.constant_initializer(value=0))

        dec_conv1_weights = tf.get_variable("dec_conv1_weights", [3, 3, 48, 48], initializer=intl)
        dec_conv1_biases = tf.get_variable("dec_conv1_biases", shape=[48], initializer=tf.constant_initializer(value=0))

        dec_conv2_weights = tf.get_variable("decc_conv2_weights", [3, 3, 48, 90], initializer=intl)
        dec_conv2_biases = tf.get_variable("dec_conv2_biases", shape=[90], initializer=tf.constant_initializer(value=0))

        dec_conv3_weights = tf.get_variable("dec_conv3_weights", [3, 3, 90, 90], initializer=intl)
        dec_conv3_biases = tf.get_variable("dec_conv3_biases", shape=[90], initializer=tf.constant_initializer(value=0))

        dec_out_weights = tf.get_variable("dec_out_weights", [3, 3, 90, 1], initializer=intl)
        dec_out_biases = tf.get_variable("dec_out_biases", shape=[1], initializer=tf.constant_initializer(value=0))

        dec1_out_cov_weights = tf.get_variable("dec1_out_cov_weights", [3, 3, 90, 1], initializer=intl)
        dec1_out_cov_biases = tf.get_variable("dec1_out_cov_biases", shape=[1],
                                              initializer=tf.constant_initializer(value=0))

    else:

        pass

######## TWO LAYER
# a. build the encoder layers

x_inp_ = tf.reshape(x_inp, [batch_size, ndims, ndims, 1])

enc_conv1 = tf.nn.conv2d(x_inp_, enc_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu1 = fact(tf.nn.bias_add(enc_conv1, enc_conv1_biases))

enc_conv2 = tf.nn.conv2d(enc_relu1, enc_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu2 = fact(tf.nn.bias_add(enc_conv2, enc_conv2_biases))

enc_conv3 = tf.nn.conv2d(enc_relu2, enc_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu3 = fact(tf.nn.bias_add(enc_conv3, enc_conv3_biases))

flat_relu3 = tf.contrib.layers.flatten(enc_relu3)

# b. get the values for drawing z
mu = tf.matmul(flat_relu3, mu_weights) + mu_biases
mu = tf.tile(mu, (nzsamp, 1))  # replicate for number of z's you want to draw
logVar = tf.matmul(flat_relu3, logVar_weights) + logVar_biases
logVar = tf.tile(logVar, (nzsamp, 1))  # replicate for number of z's you want to draw
std = tf.exp(0.5 * logVar)

# c. draw an epsilon and get z
epsilon = tf.random_normal(tf.shape(logVar), name='epsilon')
z = mu + tf.multiply(std, epsilon)

if useMixtureScale:

    indices1 = tf.range(start=0, limit=lat_dim_1, delta=1, dtype='int32')
    indices2 = tf.range(start=lat_dim_1, limit=lat_dim, delta=1, dtype='int32')

    z1 = tf.transpose(tf.gather(tf.transpose(z), indices1))
    z2 = tf.transpose(tf.gather(tf.transpose(z), indices2))

    # d. build the decoder layers from z1 for mu(z)
    dec_L1 = fact(tf.matmul(z, dec_fc1_weights) + dec_fc1_biases)
else:
    pass

dec_L1_reshaped = tf.reshape(dec_L1, [batch_size, int(ndims), int(ndims), 48])

dec_conv1 = tf.nn.conv2d(dec_L1_reshaped, dec_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu1 = fact(tf.nn.bias_add(dec_conv1, dec_conv1_biases))

dec_conv2 = tf.nn.conv2d(dec_relu1, dec_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu2 = fact(tf.nn.bias_add(dec_conv2, dec_conv2_biases))

dec_conv3 = tf.nn.conv2d(dec_relu2, dec_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu3 = fact(tf.nn.bias_add(dec_conv3, dec_conv3_biases))

# e. build the output layer w/out activation function
dec_out = tf.nn.conv2d(dec_relu3, dec_out_weights, strides=[1, 1, 1, 1], padding='SAME')
y_out_ = tf.nn.bias_add(dec_out, dec_out_biases)

y_out = tf.contrib.layers.flatten(y_out_)

# e.2 build the covariance at the output if using mixture of scales
if useMixtureScale:
    # e. build the output layer w/out activation function
    dec_out_cov = tf.nn.conv2d(dec_relu3, dec1_out_cov_weights, strides=[1, 1, 1, 1], padding='SAME')
    y_out_prec_log = tf.nn.bias_add(dec_out_cov, dec1_out_cov_biases)

    y_out_prec_ = tf.exp(y_out_prec_log)

    y_out_prec = tf.contrib.layers.flatten(y_out_prec_)

print('CAME HERE!! 2')

# build the loss functions and the optimizer
# ==============================================================================
# ==============================================================================

# KLD loss per sample in the batch
KLD = -0.5 * tf.reduce_sum(1 + logVar - tf.pow(mu, 2) - tf.exp(logVar), reduction_indices=1)

x_inp_ = tf.tile(x_inp, (nzsamp, 1))

# L2 loss per sample in the batch
if useMixtureScale:
    l2_loss_1 = tf.reduce_sum(tf.multiply(tf.pow((y_out - x_inp_), 2), y_out_prec), axis=1)
    l2_loss_2 = tf.reduce_sum(tf.log(y_out_prec), axis=1)  # tf.reduce_sum(tf.log(y_out_cov),axis=1)
    l2_loss_ = l2_loss_1 - l2_loss_2
else:
    l2_loss_ = tf.reduce_sum(tf.pow((y_out - x_inp_), 2), axis=1)
    if usebce:
        l2_loss_ = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=x_inp_),
                                 reduction_indices=1)

# take the total mean loss of this batch
loss_tot = tf.reduce_mean(1 / kld_div * KLD + 0.5 * l2_loss_)

# get the optimizer
if useMixtureScale:
    train_step = tf.train.AdamOptimizer(5e-4).minimize(loss_tot)
else:
    train_step = tf.train.AdamOptimizer(5e-3).minimize(loss_tot)

## cost functions for reconstruction after training
##==============================================================================
##==============================================================================
# x_rec=tf.get_variable('x_rec',shape=[5000,784],initializer=tf.constant_initializer(value=0.0))
#
# prior_cost =  - 0.5 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_rec),2), y_out_prec),axis=1) \
#             + 0.5 * tf.reduce_sum(tf.log(y_out_prec), axis=1) - 0.5*784*tf.log(2*np.pi)


# start session
# ==============================================================================
# ==============================================================================
print('CAME HERE!! 3')

print('CAME HERE!! 31')

sess.run(tf.global_variables_initializer())

print('CAME HERE!! 32')

print("Initialized parameters")

saver = tf.train.Saver()

print('CAME HERE!! 33')

ts = tm.time()

print('CAME HERE!! 4')

# do the training
# ==============================================================================
# ==============================================================================


# LOG
writer = tf.summary.FileWriter(logdir)
loss_tot_tb = tf.Variable(0, dtype=tf.float32)
loss_tot_summ = tf.summary.scalar('Tot Loss', loss_tot_tb)
loss_l2_tb = tf.Variable(0, dtype=tf.float32)
loss_l2_summ = tf.summary.scalar('L2 Loss', loss_l2_tb)
loss_kld_tb = tf.Variable(0, dtype=tf.float32)
loss_kld_summ = tf.summary.scalar('Kdl Loss', loss_kld_tb)
loss_valid_tb = tf.Variable(0, dtype=tf.float32)
loss_valid_summ = tf.summary.scalar('Valid tot Loss', loss_valid_tb)

input_img = tf.Variable(tf.zeros([10, ndims, ndims, 1]), dtype=tf.float32)
input_img_s = tf.summary.image('Input image', input_img)
rec_img = tf.Variable(tf.zeros([10, ndims, ndims, 1]), dtype=tf.float32)
rec_img_s = tf.summary.image('Valid tot Loss', rec_img)
sampled_img = tf.Variable(tf.zeros([10, ndims, ndims, 1]), dtype=tf.float32)
sampled_img_s = tf.summary.image('Valid tot Loss', sampled_img)

t1 = time.time()
total = t1-t0
print('TIME TO TRAIN START: ', total)

with tf.device('/GPU:0'):
    # train for N steps
    for step in range(0, 500001):  # 500k
        #t0 = time.time()

        batch = MRi.get_patch(batch_size, test=False)

        batch = np.reshape(batch, [batch_size, ndims*ndims])
        # batch = MRi.get_train_batch(batch_size)

        #t1 = time.time()
        #total = t1 - t0
        #print('TIME TO LOAD DATA: ', total)

        #t0 = time.time()
        # run the training step
        sess.run([train_step], feed_dict={x_inp: batch})

        #t1 = time.time()
        #total = t1 - t0
        #print('TIME TO TRAIN: ', total)

        # print some stuf...
        if step % 500 == 0:  # 500
            test_batch = MRi.get_patch(batch_size, test=True)
            test_batch = np.reshape(test_batch, [batch_size, ndims*ndims])


            if useMixtureScale:
                loss_l2_1 = l2_loss_1.eval(feed_dict={x_inp: batch})
                loss_l2_2 = l2_loss_2.eval(feed_dict={x_inp: batch})
                loss_l2_ = l2_loss_.eval(feed_dict={x_inp: batch})
                loss_kld = KLD.eval(feed_dict={x_inp: batch})
                std_val = std.eval(feed_dict={x_inp: batch})
                mu_val = mu.eval(feed_dict={x_inp: batch})
                loss_tot_ = loss_tot.eval(feed_dict={x_inp: batch})

            rec_test_batch = y_out.eval(feed_dict={x_inp: test_batch})
            test_loss_l2 = np.mean(np.sum(np.power((rec_test_batch[0:test_batch.shape[0], :] - test_batch), 2), axis=1))

            sess.run(loss_tot_tb.assign(loss_tot_))
            writer.add_summary(sess.run(loss_tot_summ), step)
            sess.run(loss_l2_tb.assign(np.mean(loss_l2_)))
            writer.add_summary(sess.run(loss_l2_summ), step)
            sess.run(loss_kld_tb.assign(np.mean(loss_kld)))
            writer.add_summary(sess.run(loss_kld_summ), step)
            sess.run(loss_valid_tb.assign(np.mean(test_loss_l2)))
            writer.add_summary(sess.run(loss_valid_summ), step)

            #img_summary = tf.summary.image("Test reconstructions", xh[0:test_batch.shape[0], :], max_outputs=16)
            #x = sess.run(img_summary)
            #writer.add_summary(x)

            writer.flush()

            if useMixtureScale:
                print(
                    "Step {0} | L2 Loss: {1:.3f} | KLD Loss: {2:.3f} | L2 Loss_1: {3:.3f} | L2 Loss_2: {4:.3f} | loss_tot: {5:.3f} | L2 Loss test: {6:.3f}" \
                        .format(step, np.mean(loss_l2_1 - loss_l2_2), np.mean(loss_kld), np.mean(loss_l2_1),
                                np.mean(loss_l2_2), np.mean(loss_tot_), np.mean(test_loss_l2)))
            else:
                print(
                    "Step {0} | L2 Loss: {1:.3f} | KLD Loss: {2:.3f} | L2 Loss test: {3:.3f} | std: {4:.3f} | mu: {5:.3f}" \
                        .format(step, np.mean(loss_l2_), np.mean(loss_kld), np.mean(test_loss_l2), np.mean(std_val),
                                np.mean(mu_val)))

        if step % 5000 == 0:
            input_img_re = np.reshape(test_batch[:10], [10, ndims, ndims])
            out_img_re = np.reshape(rec_test_batch[:10], [10, ndims, ndims])

            sess.run(input_img.assign(input_img_re[:,:,:,np.newaxis]))
            writer.add_summary(sess.run(input_img_s), step)
            sess.run(rec_img.assign(out_img_re[:,:,:,np.newaxis]))
            writer.add_summary(sess.run(rec_img_s), step)

            #sess.run(sampled_img.assign())
            #writer.add_summary(sess.run(loss_valid_summ), step)

            writer.flush()

            saver.save(sess, logdir + '/' + str(mode) + '_lat' + str(lat_dim) + '_ns' + str(noisy) + '_ps' + str(ndims) + '_modality' + modality + '_step' + str(
                step) + '.ckpt')

            print(logdir + '/' + str(mode) + '_lat' + str(lat_dim) + '_ns' + str(noisy) + '_ps' + str(ndims) + '_modality' + modality + '_step' + str(
                step) + '.ckpt')

print("elapsed time: {0}".format(tm.time() - ts))