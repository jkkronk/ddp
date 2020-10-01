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
parser.add_argument('--subj', type=str, default='multicoil_val/file_brain_AXFLAIR_200_6002462.h5')
parser.add_argument('--sli', type=int, default=150) 
parser.add_argument('--usfact', type=float, default=4) 
parser.add_argument('--contrun', type=int, default=0) 
parser.add_argument('--skiprecon', type=int, default=1) 

args=parser.parse_args()
subj=args.subj
sli=args.sli
usfact = args.usfact
contrun = args.contrun

basefolder = '/scratch_net/bmicdl03/jonatank/logs/dpp/vae/'
logdir = "/scratch_net/bmicdl03/jonatank/logs/ddp/restore/" + datetime.now().strftime("%Y%m%d-%H%M%S")
ndims=28
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
MRi = MR_kspace_data(dirname='/srv/beegfs02/scratch/fastmri_challenge/data/brain')

ksp_subj = MRi.get_image(subj)
ksp = ksp_subj[sli]

#nx, ny = (252, 308)
#x = np.linspace(0, 1, nx)
#y = np.linspace(0, 1, ny)
#xv, yv = np.meshgrid(x, y)
#synthph = np.transpose(np.pi*(xv+yv)-np.pi)

try:
     uspat = np.load(basefolder+'uspats/uspat_us'+str(R)+'_vol'+subj+'_sli'+str(sli) + '.npy')
     print("Read from existing u.s. pattern file")
except:
     USp=US_pattern()
     uspat = USp.generate_opt_US_pattern_1D(ksp.shape[1:], R=R, max_iter=100, no_of_training_profs=15)
     np.save(basefolder+'uspats/uspat_us'+str(R)+'_vol'+subj+'_sli'+str(sli), uspat)

usksp = np.zeros(ksp.shape)

for i in range(ksp.shape[0]):
    for j in range(ksp.shape[1]):
        usksp[i,j] = uspat*ksp[i,j]

###################
###### RECON ######
###################

if not args.skiprecon:
     rec_vae = vaerecon.vaerecon(usksp, sensmaps=np.ones_like(usksp), dcprojiter=dcprojiter, lat_dim=lat_dim, patchsize=ndims, contRec=contRec, parfact=25, num_iter=num_iter, regiter=10, reglmb=reg, regtype=regtype, half=True, mode=mode, chunks40=chunks40, writer=file_writer)
     rec_vae = rec_vae[0]
     pickle.dump(rec_vae, open(basefolder+'MAPestimation/rec'+str(args.contrun)+'_us'+str(R)+'_vol'+str(vol)+'_sli'+str(sli)+'_regtype_'+regtype+'_dcprojiter_'+str(dcprojiter) ,'wb')   )
    
     lastiter = int((np.floor(rec_vae.shape[1]/13)-2)*13)
     maprecon = rec_vae[:,lastiter].reshape([252, 308]) # this is the final reconstructed image














