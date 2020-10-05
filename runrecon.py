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

from US_pattern import US_pattern
from dataloader import MR_image_data, MR_kspace_data
import vaerecon

import argparse

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--subj', type=str, default='file_brain_AXFLAIR_200_6002462.h5')
parser.add_argument('--sli', type=int, default=10)
parser.add_argument('--usfact', type=float, default=4) 
parser.add_argument('--contrun', type=int, default=0) 
parser.add_argument('--skiprecon', type=int, default=1) 

args=parser.parse_args()
subj=args.subj
sli=args.sli
usfact = args.usfact
contrun = args.contrun

vae_model = '20201005-091052/jonatank_fcl500_lat60_ns0_ps28_step0.ckpt'
datapath = '/srv/beegfs02/scratch/fastmri_challenge/data/brain/'
basefolder = '/scratch_net/bmicdl03/jonatank/logs/ddp/'
logdir = "/scratch_net/bmicdl03/jonatank/logs/ddp/restore/" + datetime.now().strftime("%Y%m%d-%H%M%S")
patch_sz=28
lat_dim=60
mode = 'MRIunproc'#'Melanie_BFC'
noise=0
rmses=np.zeros((1,1,4))
regtype='reg2_dc'
reg=0.1
dcprojiter=1

if np.floor(usfact) == usfact: # if it is already an integer
     usfact = int(usfact)
print('Under sample factor: ', usfact)
R=usfact
if R<=3:
     num_iter = 402 # 302
else:
     num_iter = 602 # 602

if contrun == 0:
     contRec = ''
else:
     contRec = basefolder+'MAPestimation/rec_us'+str(R)+'_vol'+str(vol)+'_sli'+str(sli)+'_regtype_'+regtype+'_dcprojiter_'+str(dcprojiter)
     numiter = 302

## CREATE DATA

USp = US_pattern()
MRi = MR_kspace_data(dirname=datapath)

ksp_subj = MRi.get_subj(subj)
print('Shape ksp_subj: ', ksp_subj.shape)
ksp = ksp_subj[sli]

try:
     uspat = np.load(basefolder+'uspats/uspat_us'+str(R)+'_vol'+subj+'_sli'+str(sli) + '.npy')
     print("Read from existing u.s. pattern file")
except:
     USp=US_pattern()
     uspat = USp.generate_opt_US_pattern_1D(ksp.shape[1:], R=R, max_iter=100, no_of_training_profs=15)
     np.save(basefolder+'uspats/uspat_us'+str(R)+'_vol'+subj+'_sli'+str(sli), uspat)

usksp = np.moveaxis(ksp, 0, -1) # [w,h,coils]
uspat_allcoils = np.repeat(uspat[:, :, np.newaxis], usksp.shape[-1], axis=2)
usksp = uspat_allcoils * usksp

try:
     with h5py.File(basefolder + 'sensmaps/' + coilmap_r_ + subj_name, 'r') as fdset:
          coilmap_r = fdset['coilmaps'][sli]  # (number of slices, number of coils, height, width)

     with h5py.File(basefolder + 'sensmaps/' + coilmap_i_ + subj_name, 'r') as fdset:
          coilmap_i = fdset['coilmaps'][sli]  # (number of slices, number of coils, height, width)

     sensmap = coilmap_r + coilmap_i * 1j

     print("Read from existing u.s. pattern file")
except:
     sensmap = np.ones_like(usksp)

#import matplotlib.image as mpimg
#mpimg.imsave("tmp/ksp.png", 20*np.log(np.abs(ksp[-1])))

###################
###### RECON ######
###################

if not args.skiprecon:
     print(usksp.shape)
     rec_vae = vaerecon.vaerecon(usksp, sensmap, uspat_allcoils, lat_dim=lat_dim, patchsize=patch_sz, contRec=contRec, parfact=25, num_iter=num_iter, regiter=10, reglmb=reg, regtype=regtype, mode=mode, vae_model=vae_model)
     rec_vae = rec_vae[0]
     pickle.dump(rec_vae, open(basefolder+'MAPestimation/rec'+str(args.contrun)+'_us'+str(R)+'_vol'+str(vol)+'_sli'+str(sli)+'_regtype_'+regtype+'_dcprojiter_'+str(dcprojiter) ,'wb')   )
    
     lastiter = int((np.floor(rec_vae.shape[1]/13)-2)*13)
     maprecon = rec_vae[:,lastiter].reshape([252, 308]) # this is the final reconstructed image














