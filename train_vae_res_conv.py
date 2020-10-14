# Simple VAE to see if VAEs can learn sparsity inducing distributions
# Kerem Tezcan, CVL
# initial: 28.05.2017
# last mod: 30.05.2017 

## direk precision optimize etmek daha da iyi olabilir. 

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

from vae_models.vae_res_conv import VariationalAutoencoder
from vae_models.definevae_res_conv import VAEModel
import time


# parameters
# ==============================================================================
# ==============================================================================


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
vae_name = modality + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "/scratch_net/bmicdl03/jonatank/logs/ddp/vae/" + vae_name
print(logdir)

user = 'jonatank'

# ndims=int(sys.argv[3])
patch_size = 32
useMixtureScale = True
noisy = 0
batch_size = 512 #1024
usebce = False
nzsamp = 1
lr_rate = 5e-4
kld_div = 1.

std_init = 0.05
weight = 1

fcl_dim = 500
lat_dim = 60
print(">>> lat_dim value: " + str(lat_dim))
print(">>> mode is: " + mode)

lat_dim_1 = max(1, np.floor(lat_dim / 2))
lat_dim_2 = lat_dim - lat_dim_1

num_inp_channels = 1

# make a dataset to use later
# ==============================================================================
# ==============================================================================
datapath = '/scratch_net/bmicdl03/jonatank/data'#'/srv/beegfs02/scratch/fastmri_challenge/data/brain'
# DS = SliceData(datapath, transform, sample_rate=0.1) # sample_rate = how much ratio of data to use
# (train_size, test_size, ndims, noisy, seed, mode, downscale=True)
from dataloader import MR_image_data

MRi = MR_image_data(dirname=datapath, trainset_ratio=1, noiseinvstd=0, patchsize=32, modality=modality)

print('CAME HERE!! 1')

# make a simple fully connected network
# ==============================================================================
# ==============================================================================

print(tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
))

tf.reset_default_graph()

# do the training
# ==============================================================================
# ==============================================================================

# LOG
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter(logdir)
loss_tot_tb = tf.Variable(0, dtype=tf.float32)
loss_tot_summ = tf.summary.scalar('Tot Loss', loss_tot_tb)
loss_l2_tb = tf.Variable(0, dtype=tf.float32)
loss_l2_summ = tf.summary.scalar('L2 Loss', loss_l2_tb)
loss_kld_tb = tf.Variable(0, dtype=tf.float32)
loss_kld_summ = tf.summary.scalar('Kdl Loss', loss_kld_tb)
loss_valid_tb = tf.Variable(0, dtype=tf.float32)
loss_valid_summ = tf.summary.scalar('Valid tot Loss', loss_valid_tb)

input_img = tf.Variable(tf.zeros([10, patch_size, patch_size, 1]), dtype=tf.float32)
input_img_s = tf.summary.image('Input image', input_img)
rec_img = tf.Variable(tf.zeros([10, patch_size, patch_size, 1]), dtype=tf.float32)
rec_img_s = tf.summary.image('Valid tot Loss', rec_img)
sampled_img = tf.Variable(tf.zeros([10, patch_size, patch_size, 1]), dtype=tf.float32)
sampled_img_s = tf.summary.image('Valid tot Loss', sampled_img)

t1 = time.time()
total = t1-t0
print('TIME TO TRAIN START: ', total)

with tf.device('/GPU:0'):
    vae_network = VariationalAutoencoder
    model = VAEModel(vae_network, batch_size, patch_size, lr_rate, modality, logdir)
    model.initialize()

    # train for N steps
    for step in range(0, 500001):  # 500k
        #t0 = time.time()

        batch = MRi.get_patch(batch_size, test=False)

        #t1 = time.time()
        #total = t1 - t0
        #print('TIME TO LOAD DATA: ', total)

        #t0 = time.time()

        # run the training step
        model.train(batch[:,:,:,np.newaxis],weight)

        #t1 = time.time()
        #total = t1 - t0
        #print('TIME TO TRAIN: ', total)

        # print some stuf...
        if step % 500 == 0:  # 500
            val_batch = MRi.get_patch(batch_size, test=True)

            model.validate(val_batch[:, :, :, np.newaxis], weight)
            # model.visualize(modality, step)
            gen_loss, lat_loss = model.sess.run([model.autoencoder_loss,
                                                            model.latent_loss], {model.image_matrix: batch[:,:,:,np.newaxis]})

            gen_loss_valid, lat_loss_valid = model.sess.run([model.autoencoder_loss_test,
                                                            model.latent_loss_test], {model.image_matrix: val_batch[:,:,:,np.newaxis]})
            print(("epoch %d: train_gen_loss %f train_lat_loss %f total train_loss %f") % (
                step, gen_loss.mean(), lat_loss.mean(),
                gen_loss.mean() + lat_loss.mean()))

            print(("epoch %d: test_gen_loss %f test_lat_loss %f total loss %f") % (
                step, gen_loss_valid.mean(), lat_loss_valid.mean(),
                gen_loss_valid.mean() + lat_loss_valid.mean()))


            sess.run(loss_tot_tb.assign(gen_loss.mean() + lat_loss.mean()))
            writer.add_summary(sess.run(loss_tot_summ), step)
            sess.run(loss_l2_tb.assign(gen_loss.mean()))
            writer.add_summary(sess.run(loss_l2_summ), step)
            sess.run(loss_kld_tb.assign(lat_loss.mean()))
            writer.add_summary(sess.run(loss_kld_summ), step)
            sess.run(loss_valid_tb.assign(gen_loss_valid.mean() + lat_loss_valid.mean()))
            writer.add_summary(sess.run(loss_valid_summ), step)

            #img_summary = tf.summary.image("Test reconstructions", xh[0:test_batch.shape[0], :], max_outputs=16)
            #x = sess.run(img_summary)
            #writer.add_summary(x)

            writer.flush()

        if step % 5000 == 0:
            input_img_re = np.reshape(val_batch[:10], [10, patch_size, patch_size])
            rec_val_batch = model.sess.run([model.decoder_output_test], {model.image_matrix: val_batch[:,:,:,np.newaxis]})
            out_img_re = np.reshape(rec_val_batch[0], [batch_size, patch_size, patch_size])[:10]

            sess.run(input_img.assign(input_img_re[:,:,:,np.newaxis]))
            writer.add_summary(sess.run(input_img_s), step)
            sess.run(rec_img.assign(out_img_re[:,:,:,np.newaxis]))
            writer.add_summary(sess.run(rec_img_s), step)

            model.save(vae_name, step)
