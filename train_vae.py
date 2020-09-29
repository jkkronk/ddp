# Simple VAE to see if VAEs can learn sparsity inducing distributions
# Kerem Tezcan, CVL
# initial: 28.05.2017
# last mod: 30.05.2017 

## direk precision optimize etmek daha da iyi olabilir. 

from __future__ import division
from __future__ import print_function
#import os.path

import numpy as np
import time as tm
import tensorflow as tf
import os
import sys
from datetime import datetime
import random


SEED=1001
seed=1 
np.random.seed(seed=1)



import argparse

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--mode', type=str, default='jonatank')

args=parser.parse_args()


logdir = "/scratch_net/bmicdl03/jonatank/logs/ddp/vae/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# parameters
#==============================================================================
#==============================================================================

user='jonatank'

#mode=sys.argv[2]
mode= args.mode # 'MRIunproc'

#mode='MRIunproc' #'sparse', 'nonsparse', 'MNIST', 'circ', 'Lshape', 'edge', 'Lapedge', 'spiketrain', 'Gaussedge' 

#ndims=int(sys.argv[3])
ndims=28
useMixtureScale=True
noisy=50
batch_size = 50 #1000
usebce=False
kld_div=25.0
nzsamp=1

train_size=5000
test_size=1000  

if useMixtureScale:
     kld_div=1.

std_init=0.05               

input_dim=ndims*ndims
fcl_dim=500 

#lat_dim=int(sys.argv[1])
lat_dim=60
print(">>> lat_dim value: "+str(lat_dim))
print(">>> mode is: " + mode)
     
lat_dim_1 = max(1, np.floor(lat_dim/2))
lat_dim_2 = lat_dim - lat_dim_1
#
#if user=='Kerem':
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
#     from tensorflow.python.client import device_lib
#     print (device_lib.list_local_devices())
#     
#     print( os.environ['SGE_GPU'])


num_inp_channels=1

#make a dataset to use later
#==============================================================================
#==============================================================================
datapath = '/srv/beegfs02/scratch/fastmri_challenge/data/brain'
#DS = SliceData(datapath, transform, sample_rate=0.1) # sample_rate = how much ratio of data to use
    #(train_size, test_size, ndims, noisy, seed, mode, downscale=True)
from dataloader import MR_image_data
MRi = MR_image_data(dirname=datapath, trainset_ratio=0.5, noiseinvstd=50, patchsize=28)


print('CAME HERE!! 1')

#make a simple fully connected network
#==============================================================================
#==============================================================================

tf.compat.v1.reset_default_graph()

print('CAME HERE!! 11')

sess = tf.compat.v1.InteractiveSession()

print('CAME HERE!! 12')

#define the activation function to use:
def fact(x):
     #return tf.compat.v1.nn.tanh(x)
     return tf.compat.v1.nn.relu(x)


#define the input place holder
x_inp =tf.compat.v1.placeholder("float", shape=[None, input_dim])
#x_rec = tf.compat.v1.placeholder("float", shape=[None, input_dim])
l2_loss =tf.compat.v1.constant(0.0)

#define the network layer parameters
intl =tf.compat.v1.truncated_normal_initializer(stddev=std_init, seed=SEED)
intl_cov =tf.compat.v1.truncated_normal_initializer(stddev=std_init, seed=SEED)

with tf.compat.v1.variable_scope("VAE") as scope:
     
     
     
    enc_conv1_weights =tf.compat.v1.get_variable("enc_conv1_weights", [3, 3, num_inp_channels, 32], initializer=intl)
    enc_conv1_biases =tf.compat.v1.get_variable("enc_conv1_biases", shape=[32], initializer=tf.compat.v1.constant_initializer(value=0))
     
    enc_conv2_weights =tf.compat.v1.get_variable("enc_conv2_weights", [3, 3, 32, 64], initializer=intl)
    enc_conv2_biases =tf.compat.v1.get_variable("enc_conv2_biases", shape=[64], initializer=tf.compat.v1.constant_initializer(value=0))
     
    enc_conv3_weights =tf.compat.v1.get_variable("enc_conv3_weights", [3, 3, 64, 64], initializer=intl)
    enc_conv3_biases =tf.compat.v1.get_variable("enc_conv3_biases", shape=[64], initializer=tf.compat.v1.constant_initializer(value=0))
         
    mu_weights =tf.compat.v1.get_variable(name="mu_weights", shape=[int(input_dim*64), lat_dim], initializer=intl)
    mu_biases =tf.compat.v1.get_variable("mu_biases", shape=[lat_dim], initializer=tf.compat.v1.constant_initializer(value=0))
    
    logVar_weights =tf.compat.v1.get_variable(name="logVar_weights", shape=[int(input_dim*64), lat_dim], initializer=intl)
    logVar_biases =tf.compat.v1.get_variable("logVar_biases", shape=[lat_dim], initializer=tf.compat.v1.constant_initializer(value=0))
    
    
    if useMixtureScale:
         
         dec_fc1_weights =tf.compat.v1.get_variable(name="dec_fc1_weights", shape=[int(lat_dim), int(input_dim*48)], initializer=intl)
         dec_fc1_biases =tf.compat.v1.get_variable("dec_fc1_biases", shape=[int(input_dim*48)], initializer=tf.compat.v1.constant_initializer(value=0))
         
         dec_conv1_weights =tf.compat.v1.get_variable("dec_conv1_weights", [3, 3, 48, 48], initializer=intl)
         dec_conv1_biases =tf.compat.v1.get_variable("dec_conv1_biases", shape=[48], initializer=tf.compat.v1.constant_initializer(value=0))
          
         dec_conv2_weights =tf.compat.v1.get_variable("decc_conv2_weights", [3, 3, 48, 90], initializer=intl)
         dec_conv2_biases =tf.compat.v1.get_variable("dec_conv2_biases", shape=[90], initializer=tf.compat.v1.constant_initializer(value=0))
          
         dec_conv3_weights =tf.compat.v1.get_variable("dec_conv3_weights", [3, 3, 90, 90], initializer=intl)
         dec_conv3_biases =tf.compat.v1.get_variable("dec_conv3_biases", shape=[90], initializer=tf.compat.v1.constant_initializer(value=0))
         
         dec_out_weights =tf.compat.v1.get_variable("dec_out_weights", [3, 3, 90, 1], initializer=intl)
         dec_out_biases = tf.compat.v1.get_variable("dec_out_biases", shape=[1], initializer=tf.compat.v1.constant_initializer(value=0))
         
         dec1_out_cov_weights = tf.compat.v1.get_variable("dec1_out_cov_weights", [3, 3, 90, 1], initializer=intl)
         dec1_out_cov_biases = tf.compat.v1.get_variable("dec1_out_cov_biases", shape=[1], initializer=tf.compat.v1.constant_initializer(value=0))
         
    else:
         
         pass
    
######## TWO LAYER 
# a. build the encoder layers

x_inp_ = tf.compat.v1.reshape(x_inp, [batch_size,ndims,ndims,1])

enc_conv1 = tf.compat.v1.nn.conv2d(x_inp_, enc_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu1 = fact(tf.compat.v1.nn.bias_add(enc_conv1, enc_conv1_biases))

enc_conv2 = tf.compat.v1.nn.conv2d(enc_relu1, enc_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu2 = fact(tf.compat.v1.nn.bias_add(enc_conv2, enc_conv2_biases))

enc_conv3 = tf.compat.v1.nn.conv2d(enc_relu2, enc_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
enc_relu3 = fact(tf.compat.v1.nn.bias_add(enc_conv3, enc_conv3_biases))
      
flat_relu3 = tf.contrib.layers.flatten(enc_relu3)

# b. get the values for drawing z
mu = tf.compat.v1.matmul(flat_relu3, mu_weights) + mu_biases
mu = tf.compat.v1.tile(mu, (nzsamp, 1)) # replicate for number of z's you want to draw
logVar = tf.compat.v1.matmul(flat_relu3, logVar_weights) + logVar_biases
logVar = tf.compat.v1.tile(logVar,  (nzsamp, 1))# replicate for number of z's you want to draw
std = tf.compat.v1.exp(0.5 * logVar)

# c. draw an epsilon and get z
epsilon = tf.compat.v1.random_normal(tf.compat.v1.shape(logVar), name='epsilon')
z = mu + tf.compat.v1.multiply(std, epsilon)


if useMixtureScale:

     
     indices1=tf.compat.v1.range(start=0, limit=lat_dim_1, delta=1, dtype='int32')
     indices2=tf.compat.v1.range(start=lat_dim_1, limit=lat_dim, delta=1, dtype='int32')
     
     z1 = tf.compat.v1.transpose(tf.compat.v1.gather(tf.compat.v1.transpose(z),indices1))
     z2 = tf.compat.v1.transpose(tf.compat.v1.gather(tf.compat.v1.transpose(z),indices2))
     
     
     # d. build the decoder layers from z1 for mu(z)
     dec_L1 = fact(tf.compat.v1.matmul(z, dec_fc1_weights) + dec_fc1_biases)
else:
     pass
    
dec_L1_reshaped = tf.compat.v1.reshape(dec_L1 ,[batch_size,int(ndims),int(ndims),48])

dec_conv1 = tf.compat.v1.nn.conv2d(dec_L1_reshaped, dec_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu1 = fact(tf.compat.v1.nn.bias_add(dec_conv1, dec_conv1_biases))

dec_conv2 = tf.compat.v1.nn.conv2d(dec_relu1, dec_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu2 = fact(tf.compat.v1.nn.bias_add(dec_conv2, dec_conv2_biases))

dec_conv3 = tf.compat.v1.nn.conv2d(dec_relu2, dec_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
dec_relu3 = fact(tf.compat.v1.nn.bias_add(dec_conv3, dec_conv3_biases))

# e. build the output layer w/out activation function
dec_out = tf.compat.v1.nn.conv2d(dec_relu3, dec_out_weights, strides=[1, 1, 1, 1], padding='SAME')
y_out_ = tf.compat.v1.nn.bias_add(dec_out, dec_out_biases)

y_out = tf.contrib.layers.flatten(y_out_)
                 
# e.2 build the covariance at the output if using mixture of scales
if useMixtureScale:
     
     # e. build the output layer w/out activation function
     dec_out_cov = tf.compat.v1.nn.conv2d(dec_relu3, dec1_out_cov_weights, strides=[1, 1, 1, 1], padding='SAME')
     y_out_prec_log = tf.compat.v1.nn.bias_add(dec_out_cov, dec1_out_cov_biases)
     
     y_out_prec_ = tf.compat.v1.exp(y_out_prec_log)
     
     y_out_prec=tf.contrib.layers.flatten(y_out_prec_)
     
#     #DBG # y_out_cov=tf.compat.v1.ones_like(y_out)

#####################################

######### One LAYER 
## a. build the encoder layers
#enc_L1 = fact(tf.compat.v1.nn.bias_add( tf.compat.v1.matmul(x_inp, enc_fc1_weights),  enc_fc1_biases))
#
## b. get the values for drawing z
#mu = tf.compat.v1.matmul(enc_L1, mu_weights) + mu_biases
#mu = tf.compat.v1.tile(mu, (nzsamp, 1))# replicate for number of z's you want to draw
#logVar = tf.compat.v1.matmul(enc_L1, logVar_weights) + logVar_biases
#logVar = tf.compat.v1.tile(logVar,  (nzsamp, 1))# replicate for number of z's you want to draw
#std = tf.compat.v1.exp(0.5 * logVar)
#
## c. draw an epsilon and get z
#epsilon = tf.compat.v1.random_normal(tf.compat.v1.shape(logVar), name='epsilon')
#z = mu + tf.compat.v1.multiply(std, epsilon)
#
#
#if useMixtureScale:
#     
#     indices1=tf.compat.v1.range(start=0, limit=lat_dim_1, delta=1, dtype='int32')
#     indices2=tf.compat.v1.range(start=lat_dim_1, limit=lat_dim, delta=1, dtype='int32')
#     
#     z1 = tf.compat.v1.transpose(tf.compat.v1.gather(tf.compat.v1.transpose(z),indices1))
#     z2 = tf.compat.v1.transpose(tf.compat.v1.gather(tf.compat.v1.transpose(z),indices2))
#     
#     
#     # d. build the decoder layers from z1 for mu(z)
#     dec_L1 = fact(tf.compat.v1.matmul(z1, dec_fc1_weights) + dec_fc1_biases)
#else:
#     dec_L1 = fact(tf.compat.v1.matmul(z, dec_fc1_weights) + dec_fc1_biases)
#     
#
## e. build the output layer w/out activation function
#y_out = tf.compat.v1.matmul(dec_L1, dec_out_weights) + dec_out_biases
#                 
## e.2 build the covariance at the output if using mixture of scales
#if useMixtureScale:
#     dec2_L1 = fact(tf.compat.v1.matmul(z2, dec2_fc1_weights) + dec2_fc1_biases)
#     
#     y_out_prec_log = tf.compat.v1.matmul(dec2_L1, dec_out_cov_weights) + dec_out_cov_biases
#     
#     #y_out_prec = tf.compat.v1.exp(y_out_prec_log) / (tf.compat.v1.exp(y_out_prec_log)*0.00001 + 1.) + 0.01
#     y_out_prec = tf.compat.v1.exp(y_out_prec_log)
#     
#     #DBG # y_out_cov=tf.compat.v1.ones_like(y_out)
######################################
     
print('CAME HERE!! 2')


# build the loss functions and the optimizer
#==============================================================================
#==============================================================================

# KLD loss per sample in the batch
KLD = -0.5 * tf.compat.v1.reduce_sum(1 + logVar - tf.compat.v1.pow(mu, 2) - tf.compat.v1.exp(logVar), reduction_indices=1)

x_inp_ = tf.compat.v1.tile(x_inp, (nzsamp, 1))

# L2 loss per sample in the batch
if useMixtureScale:
     l2_loss_1 = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(tf.compat.v1.pow((y_out - x_inp_),2), y_out_prec),axis=1)
     l2_loss_2 = tf.compat.v1.reduce_sum(tf.compat.v1.log(y_out_prec), axis=1) #tf.compat.v1.reduce_sum(tf.compat.v1.log(y_out_cov),axis=1)
     l2_loss_ = l2_loss_1 - l2_loss_2
else:
     l2_loss_ = tf.compat.v1.reduce_sum(tf.compat.v1.pow((y_out - x_inp_),2), axis=1)
     if usebce:
          l2_loss_ = tf.compat.v1.reduce_sum(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=x_inp_), reduction_indices=1)
     
# take the total mean loss of this batch
loss_tot = tf.compat.v1.reduce_mean(1/kld_div*KLD + 0.5*l2_loss_)

# get the optimizer
if useMixtureScale:
     train_step = tf.compat.v1.train.AdamOptimizer(5e-4).minimize(loss_tot)
else:
     train_step = tf.compat.v1.train.AdamOptimizer(5e-3).minimize(loss_tot)
     
## cost functions for reconstruction after training
##==============================================================================
##==============================================================================
#x_rec=tf.compat.v1.get_variable('x_rec',shape=[5000,784],initializer=tf.compat.v1.constant_initializer(value=0.0))
#
#prior_cost =  - 0.5 * tf.compat.v1.reduce_sum(tf.compat.v1.multiply(tf.compat.v1.pow((y_out - x_rec),2), y_out_prec),axis=1) \
#             + 0.5 * tf.compat.v1.reduce_sum(tf.compat.v1.log(y_out_prec), axis=1) - 0.5*784*tf.compat.v1.log(2*np.pi)


# start session
#==============================================================================
#==============================================================================
print('CAME HERE!! 3')



print('CAME HERE!! 31')

sess.run(tf.compat.v1.global_variables_initializer())

print('CAME HERE!! 32')

print("Initialized parameters")

saver = tf.compat.v1.train.Saver()

print('CAME HERE!! 33')


ts=tm.time()

print('CAME HERE!! 4')

# do the training
#==============================================================================
#==============================================================================
test_batch = MRi.get_patch(batch_size, test=True)
test_batch = np.transpose(np.reshape(test_batch, [-1, batch_size]))
#test_batch = DS.get_test_batch(batch_size)

# LOG
writer = tf.compat.v1.summary.FileWriter(logdir)

#summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

with tf.compat.v1.device('/gpu:0'):
     
     #train for N steps
    for step in range(0, 500001): # 500k
        batch = MRi.get_patch(batch_size, test=False)
        batch = np.transpose(np.reshape(batch, [-1, batch_size]))
        # batch = MRi.get_train_batch(batch_size)

        # run the training step
        sess.run([train_step], feed_dict={x_inp: batch})
         
    
        #print some stuf...
        if step % 500 == 0: # 500

            if useMixtureScale:
                loss_l2_1 = l2_loss_1.eval(feed_dict={x_inp: batch})
                loss_l2_2 = l2_loss_2.eval(feed_dict={x_inp: batch})
                loss_l2_ = l2_loss_.eval(feed_dict={x_inp: batch})
                loss_kld = KLD.eval(feed_dict={x_inp: batch})
                std_val = std.eval(feed_dict={x_inp: batch})
                mu_val = mu.eval(feed_dict={x_inp: batch})
                loss_tot_ = loss_tot.eval(feed_dict={x_inp: batch})

            xh = y_out.eval(feed_dict={x_inp: test_batch})
            test_loss_l2 = np.mean(np.sum(np.power((xh[0:test_batch.shape[0], :] - test_batch), 2), axis=1))

            #tf.compat.v1.summary.scalar('loss_l2', np.mean(loss_l2_1 - loss_l2_2))
            #tf.compat.v1.summary.scalar('KLD Loss', np.mean(loss_kld))
            #tf.compat.v1.summary.scalar('loss_tot', np.mean(loss_tot_))

            #tf.compat.v1.summary.scalar('test_recloss', np.mean(test_loss_l2))


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


            #tf.compat.v1.summary.image("Input data", test_batch)
            #tf.compat.v1.summary.image("Recdata data", xh)

        if step % 100000 == 0:

            saver.save(sess, os.path.join('/scratch_net/bmicdl03/jonatank/logs/ddp/vae/', str(mode) + '_fcl' + str(
                fcl_dim) + '_lat' + str(lat_dim) + '_ns' + str(noisy) + '_ps' + str(ndims) + '_step' + str(step) + '.ckpt'))

     
print("elapsed time: {0}".format(tm.time()-ts))
