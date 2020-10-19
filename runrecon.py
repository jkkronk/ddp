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

from US_pattern import US_pattern
from dataloader import MR_image_data, MR_kspace_data
import vaerecon
import vaerecon_direct
from utils import tFT

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
direct_approx = args.direct_approx

vae_model = 'FLAIR20201016-113444/jonatank_lat60_ns50_ps28_modalityFLAIR_step100000.ckpt'
datapath = '/scratch_net/bmicdl03/jonatank/data/'#'/srv/beegfs02/scratch/fastmri_challenge/data/brain/'
sensmap_path = '/scratch_net/bmicdl03/jonatank/data/est_coilmaps_cal/' #'/srv/beegfs02/scratch/fastmri_challenge/data/brain_sensmap_espirit/multicoil_val'
basefolder = '/scratch_net/bmicdl03/jonatank/logs/ddp/'
logdir = basefolder + "restore/" + datetime.now().strftime("%Y%m%d-%H%M%S")
patch_sz=28
lat_dim=60
mode = 'MRIunproc'#'Melanie_BFC'
noise=0
rmses=np.zeros((1, 1, 4))
regtype='reg2_proj'
reg=0
dcprojiter=1

print('Under sample factor: ', usfact)
R=4

if R<=3:
     num_iter = 300 # 302
else:
     num_iter = 400 # 602

if contrun == 0:
     contRec = ''
else:
     contRec = basefolder+'MAPestimation/rec_us'+str(R)+'_vol'+subj+'_sli'+str(sli)+'_regtype_'+regtype+'_dcprojiter_'+str(dcprojiter)
     numiter = 302

## CREATE DATA

USp = US_pattern()
MRi = MR_kspace_data(dirname=datapath)

ksp_subj = MRi.get_subj(subj)
print(vae_model)
ksp = ksp_subj[sli] * 1000
ksp = np.moveaxis(ksp, 0, -1) # [w,h,coils]

img_sizex = ksp.shape[0]
img_sizey = ksp.shape[1]

try:
     uspat = np.load(basefolder+'uspats/uspat_us'+str(R)+'_vol'+subj+'_sli'+str(sli) + '.npy')
     print("Read from existing u.s. pattern file")
except:
     USp=US_pattern()
     uspat = USp.generate_opt_US_pattern_1D(ksp.shape[1:], R=R, max_iter=100, no_of_training_profs=15)
     np.save(basefolder+'uspats/uspat_us'+str(R)+'_vol'+subj+'_sli'+str(sli), uspat)
     print('Creating new undersampling file')

uspat_allcoils = np.repeat(uspat[:, :, np.newaxis], ksp.shape[-1], axis=2)
usksp = uspat_allcoils * ksp

try:
     with h5py.File(sensmap_path + 'coilmap_r_' + subj, 'r') as fdset:
          coilmap_r = fdset['coilmaps'][sli]  # (number of slices, number of coils, height, width)

     with h5py.File(sensmap_path + 'coilmap_i_' + subj, 'r') as fdset:
          coilmap_i = fdset['coilmaps'][sli]  # (number of slices, number of coils, height, width)

     sensmap = np.transpose(coilmap_r + coilmap_i * 1j, (1, 2, 0))

     print("Read from existing sensmap pattern file")
except:
     sensmap = np.ones_like(usksp)
     print('Warning sensmaps is not found. Continues with zero sensitivitly maps.')

#import matplotlib.image as mpimg
#mpimg.imsave("tmp/ksp.png", 20*np.log(np.abs(ksp[-1])))

###################
###### RECON ######
###################

if not args.skiprecon:
     if direct_approx:
          rec_vae, phaseregvals = vaerecon_direct.vaerecon_direct(usksp, sensmap, uspat_allcoils, lat_dim=lat_dim, patchsize=patch_sz,
                                                    contRec=contRec, parfact=25, num_iter=num_iter, regiter=10,
                                                    reglmb=reg, regtype=regtype, mode=mode, vae_model=vae_model,
                                                    logdir=logdir)
     else:
          rec_vae, phaseregvals = vaerecon.vaerecon(usksp, sensmap, uspat_allcoils, lat_dim=lat_dim, patchsize=patch_sz, contRec=contRec, parfact=25, num_iter=num_iter, regiter=10, reglmb=reg, regtype=regtype, mode=mode, vae_model=vae_model, logdir=logdir)

     GT_img = MRi.get_gt(subj)[sli]

     #lastiter = int((np.floor(rec_vae.shape[-1]/13)-2)*13)
     maprecon = rec_vae[:, -1].reshape((img_sizex, img_sizey)) # this is the final reconstructed image

     mse_rec = ((maprecon[(img_sizey/2):(-img_sizey/2),:] - GT_img) ** 2).mean()

     ssim_rec = -1
     #ssim_rec = ssim(maprecon, GT_img, data_range=maprecon.max() - maprecon.min())

     print('<<RECONSTRUCTION DONE>>', '     Subject: ', subj, '     MSE = ', mse_rec, '     SSIM = ', ssim_rec)

     pickle.dump(rec_vae, open(basefolder+'rec/rec'+str(args.contrun)+'_us'+str(R)+'_vol'+subj+'_sli'+str(sli)+'_regtype_'+regtype+'_dcprojiter_'+str(dcprojiter) ,'wb'))















