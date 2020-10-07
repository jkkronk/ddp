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
from models.definevae_res_conv import VAEModel

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

user = 'jonatank'

# mode='MRIunproc' #'sparse', 'nonsparse', 'MNIST', 'circ', 'Lshape', 'edge', 'Lapedge', 'spiketrain', 'Gaussedge'

# ndims=int(sys.argv[3])
ndims = 28
useMixtureScale = True
noisy = 0
batch_size = 1024
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
datapath = '/srv/beegfs02/scratch/fastmri_challenge/data/brain'
# DS = SliceData(datapath, transform, sample_rate=0.1) # sample_rate = how much ratio of data to use
# (train_size, test_size, ndims, noisy, seed, mode, downscale=True)
from dataloader import MR_image_data

MRi = MR_image_data(dirname=datapath, trainset_ratio=0.5, noiseinvstd=50, patchsize=28, modality=modality)

print('CAME HERE!! 1')

# make a simple fully connected network
# ==============================================================================
# ==============================================================================

vae_network = VariationalAutoencoder
model = VAEModel(vae_network, batch_size, ndims, lr_rate,modality, logdir)
model.initialize()

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

# summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

with tf.device('/gpu:0'):
    # train for N steps
    for step in range(0, 500001):  # 500k
        batch = MRi.get_patch(batch_size, test=False)
        # batch = np.transpose(np.reshape(batch, [-1, batch_size]))
        # batch = MRi.get_train_batch(batch_size)

        # run the training step
        model.train(batch,weight)

        # print some stuf...
        if step % 10 == 0:  # 500
            validate_images = MRi.get_patch(batch_size, test=True)

            model.validate(validate_images, weight)
            # model.visualize(modality, step)
            gen_loss, res_loss, lat_loss = model.sess.run([model.autoencoder_loss,
                                                           model.autoencoder_res_loss,
                                                           model.latent_loss], {model.image_matrix: input_images})
            gen_loss_valid, res_loss_valid, lat_loss_valid = model.sess.run([model.autoencoder_loss_test,
                                                                             model.autoencoder_res_loss_test,
                                                                             model.latent_loss_test],
                                                                            {model.image_matrix: validate_images})
            print(("epoch %d: train_gen_loss %f train_lat_loss %f train_res_loss %f total train_loss %f") % (
                ep, gen_loss.mean(), lat_loss.mean(), res_loss.mean(),
                gen_loss.mean() + lat_loss.mean() + res_loss.mean()))

            print(("epoch %d: test_gen_loss %f test_lat_loss %f res_loss %f total loss %f") % (
                ep, gen_loss_valid.mean(), lat_loss_valid.mean(), res_loss.mean(),
                gen_loss_valid.mean() + lat_loss_valid.mean() + res_loss.mean()))


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

        if step % 500 == 0:
            model.save(modality, step)
