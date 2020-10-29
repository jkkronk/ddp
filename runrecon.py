#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:31:54 2020

@author: ktezcan
"""

import numpy as np
import pickle
from datetime import datetime
import tensorflow as tf
import h5py
import argparse
from skimage.measure import compare_ssim

from US_pattern import US_pattern
from dataloader import MR_image_data, MR_kspace_data
import vaerecon
from utils import tFT, FT

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--subj', type=str, default='file_brain_AXFLAIR_200_6002462.h5')
parser.add_argument('--sli', type=int, default=5)
parser.add_argument('--usfact', type=int, default=4)
parser.add_argument('--contrun', type=int, default=0)
parser.add_argument('--skiprecon', type=int, default=1)
parser.add_argument('--directapprox', type=int, default=0)

args=parser.parse_args()
subj=args.subj
sli=args.sli
usfact = args.usfact
contrun = args.contrun
direct_approx = args.directapprox

vae_model = 'FLAIR20201020-125121/jonatank_lat60_ns50_ps28_modalityFLAIR_step100000.ckpt'
datapath = '/scratch_net/bmicdl03/jonatank/data/'#'/srv/beegfs02/scratch/fastmri_challenge/data/brain/'
sensmap_path = '/scratch_net/bmicdl03/jonatank/data/est_coilmaps_cal/' #'/srv/beegfs02/scratch/fastmri_challenge/data/brain_sensmap_espirit/multicoil_val'
basefolder = '/scratch_net/bmicdl03/jonatank/logs/ddp/'
logdir = basefolder + "restore/" + datetime.now().strftime("%Y%m%d-%H%M%S")
patch_sz=28
lat_dim=60
mode = 'MRIunproc'#'Melanie_BFC'
noise=0
rmses=np.zeros((1, 1, 4))
regtype='reg2_dc'
reg=0
dcprojiter=1

print('Under sample factor: ', usfact)
R=4
n=10
if R<=3:
     num_iter = 100 # 302
else:
     num_iter = 100 # 602

if contrun == 0:
     contRec = ''
else:
     contRec = basefolder+'MAPestimation/rec_us'+str(R)+'_vol'+subj+'_sli'+str(sli)+'_regtype_'+regtype+'_dcprojiter_'+str(dcprojiter)
     numiter = 302

print(vae_model)

## CREATE DATA

USp = US_pattern()
MRi = MR_kspace_data(dirname=datapath)

ksp_subj = MRi.get_subj(subj)
GT_img = MRi.get_gt(subj)
ksp_subj = np.moveaxis(ksp_subj, 1, -1) # [sli,coils,w,h] --> [sli,w,h,coils]

img_sizex = ksp_subj.shape[1]
img_sizey = ksp_subj.shape[2]

normalize = True
if normalize:
     img_ch = np.abs(np.fft.ifftshift(np.fft.ifft2(ksp_subj[sli], axes=(0, 1)), axes=(0, 1)))
     norm_fac = 1 / np.percentile(
          np.sqrt(np.sum(np.square(img_ch), axis=-1)) * np.sqrt(ksp_subj[sli].shape[0] * ksp_subj[sli].shape[1]), 99)
     ksp = ksp_subj[sli] * norm_fac * np.sqrt(img_sizex * img_sizey)
     GT_img = GT_img[sli] * norm_fac
else:
     ksp = ksp_subj[sli] * 1000
     GT_img = GT_img[sli] * 1000

gt_pad = np.zeros((img_sizex, img_sizey))
gt_pad[160:-160] = GT_img

try:
     with h5py.File(sensmap_path + 'coilmap_r_' + subj, 'r') as fdset:
          coilmap_r = fdset['coilmaps'][sli]  # (number of slices, number of coils, height, width)

     with h5py.File(sensmap_path + 'coilmap_i_' + subj, 'r') as fdset:
          coilmap_i = fdset['coilmaps'][sli]  # (number of slices, number of coils, height, width)

     sensmap = np.transpose(coilmap_r + coilmap_i * 1j, (1, 2, 0))

     print("Read from existing sensmap pattern file")
except:
     sensmap = np.ones_like(ksp)
     print('Warning sensmaps is not found. Continues with zero sensitivitly maps.')

sensmaps = np.fft.fftshift(sensmap, axes=(0,1))

try:
     uspat = np.load(basefolder+'uspats/uspat_us'+str(R)+'_vol'+subj+'_sli'+str(sli) + '.npy')
     print("Read from existing u.s. pattern file")
except:
     USp=US_pattern()
     uspat = USp.generate_opt_US_pattern_1D(ksp.shape[:2], R=R, max_iter=100, no_of_training_profs=15)
     np.save(basefolder+'uspats/uspat_us'+str(R)+'_vol'+subj+'_sli'+str(sli), uspat)
     print('Creating new undersampling file')

uspat_allcoils = np.repeat(uspat[:, :, np.newaxis], ksp.shape[-1], axis=2)
usksp = uspat_allcoils * ksp

#import matplotlib.image as mpimg
#mpimg.imsave("tmp/ksp.png", 20*np.log(np.abs(ksp[-1])))

tmp = np.fft.ifft2(np.fft.ifftshift(usksp, axes=(0, 1)), axes=(0, 1))
rss = np.sqrt(np.sum(np.square(np.abs(tmp)), axis=2))

#pickle.dump(rss, open(logdir + '_zerofilled', 'wb'))

###################
###### RECON ######
###################

if not args.skiprecon:
     rec_vae, __, __, __ = vaerecon.vaerecon(usksp, sensmaps, n, lat_dim=lat_dim, patchsize=patch_sz, contRec=contRec, parfact=25, num_iter=num_iter, regiter=10, reglmb=0.0, regtype=regtype, mode=mode, directapprox=direct_approx, vae_model=vae_model, logdir=logdir, directapp=direct_approx, gt=gt_pad.flatten())

     rec_vae[:,-1] = gt_pad.flatten()

     recon_sli = np.reshape(rec_vae[:, num_iter-2], (img_sizex, img_sizey))

     rss = np.sqrt(np.sum(np.square(np.abs(sensmaps * np.tile(recon_sli[:, :, np.newaxis], [1, 1, sensmaps.shape[2]]))), axis=-1))

     rec_vae[:, -2] = np.reshape(rss, [-1])

     temp = np.fft.ifft2(np.fft.ifftshift(ksp, axes=(0, 1)), axes=(0, 1))

     img_space = np.sum(temp, axis=2)

     rec_vae[:, -3] = np.reshape(img_space, [-1])

     pickle.dump(rec_vae, open(
          basefolder + 'rec/rec' + str(args.contrun) + '_us' + str(R) + '_vol' + subj + '_sli' + str(
               sli) + '_directapprox_' + str(direct_approx) + '_VAEDC',
          'wb'))

     rec_gt = abs(np.reshape(rec_vae[:,-1], (img_sizex, img_sizey)))[img_sizey/2:-img_sizey/2, :]
     rec_last = abs(np.fft.fftshift(np.reshape(rec_vae[:,-2], (img_sizex, img_sizey))))[img_sizey/2:-img_sizey/2, :]

     nmse = np.sqrt(((rec_gt - rec_last) ** 2).mean()) / np.sqrt(((rec_gt) ** 2).mean())

     rms = np.sqrt(((rec_gt - rec_last) ** 2).mean())

     (ssim_rec, diff) = compare_ssim(rec_gt, rec_last, full=True)

     print('<<RECONSTRUCTION DONE>>', '     Subject: ', subj, '     NMSE = ', nmse, '      RMS = ', rms ,'     SSIM = ', ssim_rec)
















