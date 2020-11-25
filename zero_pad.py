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
parser.add_argument('--sli', type=int, default=1)
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
     img_ch = np.abs(np.fft.ifftshift(np.fft.ifft2(ksp_subj, axes=(1, 2)), axes=(1, 2)))
     norm_fac = 1 / np.percentile(
          np.sqrt(np.sum(np.square(img_ch), axis=-1)) * np.sqrt(ksp_subj.shape[1] * ksp_subj.shape[2]), 99)
     ksp = ksp_subj * norm_fac * np.sqrt(img_sizex * img_sizey)
     #GT_img = GT_img * norm_fac
else:
     ksp = ksp_subj * 1000
     #GT_img = GT_img * 1000


recs  = []

for i in range(ksp.shape[0]):
     ksp_sli = ksp[i]
     
     USp = US_pattern()
     uspat = USp.generate_US_pattern_pytorch(ksp_sli.shape, R=R)
     print('Creating new undersampling file')
     uspat_allcoils = np.repeat(uspat[:, :, np.newaxis], ksp_sli.shape[-1], axis=2)
     usksp = uspat_allcoils * ksp_sli

     #import matplotlib.image as mpimg
     #mpimg.imsave("tmp/ksp.png", 20*np.log(np.abs(ksp[-1])))

     tmp = np.fft.ifft2(np.fft.ifftshift(usksp, axes=(0, 1)), axes=(0, 1))
     rss = np.sqrt(np.sum(np.square(np.abs(tmp)), axis=2))

     recs.append(rss/norm_fac)

pickle.dump(GT_img, open(basefolder + 'rec/' + 'gt_' + '_us' + str(R) + '_vol' + subj, 'wb'))
pickle.dump(recs, open(basefolder + 'rec/' + 'zero_pad' + '_us' + str(R) + '_vol' + subj, 'wb'))















