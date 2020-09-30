#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:31:54 2020

@author: ktezcan
"""

import numpy as np
import pickle
from datetime import datetime
import argparse
import torch
import torch.utils.data as data
from tensorboardX import SummaryWriter

from US_pattern import US_pattern
from dataloader_pytorch import fastMRI_kspace
import vaerecon

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--subj', type=str, default='file_brain_AXFLAIR_200_6002462.h5')
parser.add_argument('--slice', type=int, default=5)
parser.add_argument('--usfact', type=float, default=4) 
parser.add_argument('--contrun', type=int, default=0) 
parser.add_argument('--skiprecon', type=int, default=1) 

args=parser.parse_args()
subj=args.subj
sli=args.slice
R = args.usfact

basefolder = '/scratch_net/bmicdl03/jonatank/logs/ddp/vae_pytorch'
logdir = "/scratch_net/bmicdl03/jonatank/logs/ddp/restore/" + datetime.now().strftime("%Y%m%d-%H%M%S")
path = '/srv/beegfs02/scratch/fastmri_challenge/data/brain'
mode = 'MRIunproc'#'Melanie_BFC'
patch_size = 28
USp = US_pattern()

# Cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device: ' + str(device))

# Init logging with Tensorboard
file_writer = SummaryWriter(logdir + '/writer/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

if np.floor(R) == R: # if it is already an integer
     R = int(R)
print('Undersample', R)

rmses=np.zeros((1,1,4))

validation_dataset = fastMRI_kspace(dirname=path, subj=subj)
valid_data_loader  = data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1)
print('Data loaded')

ksp_all = next(iter(valid_data_loader))
ksp = ksp_all[slice]

try:
     uspat = np.load(basefolder+'uspats/uspat_us'+str(R)+'_vol'+subj+'_sli'+str(sli) + '.npy')
     print("Read from existing u.s. pattern file")
except:
     USp=US_pattern()
     uspat = USp.generate_opt_US_pattern_1D(ksp[1, 2].shape, R=R, max_iter=100, no_of_training_profs=15)
     np.save(basefolder+'uspats/uspat_us'+str(R)+'_vol'+str(vol)+'_sli'+str(sli), uspat)

usksp = np.zeros(ksp.shape)

for i in range(ksp.shape[1]):
    for j in range(ksp.shape[2]):
        usksp[:,i,j] = uspat*ksp[:,i,j]

###################
###### RECON ######
###################

nx, ny = (252, 308)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)
synthph = np.transpose(np.pi*(xv+yv)-np.pi)

regtype='reg2_dc'
reg=0.1
dcprojiter=10
chunks40=True

if R<=3:
     num_iter = 402 # 302
else:
     num_iter = 602 # 602

if not args.skiprecon:
     rec_vae = vaerecon.vaerecon(usksp, sensmaps=np.ones_like(usksp), dcprojiter=dcprojiter, lat_dim=lat_dim, patchsize=ndims, contRec=contRec, parfact=25, num_iter=num_iter, regiter=10, reglmb=reg, regtype=regtype, half=True, mode=mode, chunks40=chunks40, writer=file_writer)
     rec_vae = rec_vae[0]
     pickle.dump(rec_vae, open(basefolder+'MAPestimation/rec'+str(args.contrun)+'_us'+str(R)+'_vol'+str(vol)+'_sli'+str(sli)+'_regtype_'+regtype+'_dcprojiter_'+str(dcprojiter) ,'wb')   )
    
     lastiter = int((np.floor(rec_vae.shape[1]/13)-2)*13)
     maprecon = rec_vae[:,lastiter].reshape([252, 308]) # this is the final reconstructed image














