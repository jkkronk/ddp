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
parser.add_argument('--vol', type=int, default=5)
parser.add_argument('--sli', type=int, default=150) 
parser.add_argument('--usfact', type=float, default=4) 
parser.add_argument('--contrun', type=int, default=0) 
parser.add_argument('--skiprecon', type=int, default=1) 

args=parser.parse_args()

basefolder = '/scratch_net/bmicdl03/jonatank/logs/dpp/vae/'

logdir = "/scratch_net/bmicdl03/jonatank/logs/ddp/restore/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)
file_writer.set_as_default()

def FT (x, normalize=False):
     #inp: [nx, ny]
     #out: [nx, ny]
     if normalize:
          return np.fft.fftshift(    np.fft.fft2(  x , axes=(0,1)  ),   axes=(0,1)    ) / np.sqrt(252*308)
     else:
          return np.fft.fftshift(    np.fft.fft2(  x , axes=(0,1)  ),   axes=(0,1)    ) 

def tFT (x, normalize=False):
     #inp: [nx, ny]
     #out: [nx, ny]
     if normalize:
          return np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )   * np.sqrt(252*308)
     else:
          return np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )   

def UFT(x, uspat, normalize=False):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]
     
     return uspat*FT(x, normalize)

def tUFT(x, uspat, normalize=False):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]
     return  tFT( uspat*x ,normalize)


def calc_rmse(rec,imorig):
     return 100*np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))) / np.sum(np.square(np.abs(imorig))) )

ndims=28
lat_dim=60

mode = 'MRIunproc'#'Melanie_BFC'


USp = US_pattern()


#make a dataset to load images
noise=0
#DS = Dataset(-1, -1, ndims,noise, 1, mode)
MRi = MR_kspace_data(dirname='/srv/beegfs02/scratch/fastmri_challenge/data/brain')

rmses=np.zeros((1,1,4))

vol=args.vol
sli=args.sli

usfact = args.usfact

print(usfact)
if np.floor(usfact) == usfact: # if it is already an integer
     usfact = int(usfact)
print(usfact)



###################
###### RECON ######
###################
          

R=usfact

#               orim = DS.MRi.pad_image(DS.MRi.d_brains_test[imix,:,:].copy(), [252, 308] )
ksp = MRi.get_image()

nx, ny = (252, 308)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xv, yv = np.meshgrid(x, y)
synthph = np.transpose(np.pi*(xv+yv)-np.pi)

#          synthph = 1.5

#          orim = orima*np.exp(1j*synthph*(orima>0.1))



try:
     uspat = np.load('sample_data_and_uspat/uspat_np_fastmri.npy')
     print("Read from existing u.s. pattern file")
except:
     USp=US_pattern()
     uspat = USp.generate_opt_US_pattern_1D(ksp[0, 0].shape, R=R, max_iter=100, no_of_training_profs=15)
     np.save(basefolder+'uspats/uspat_us'+str(R)+'_vol'+str(vol)+'_sli'+str(sli), uspat)

usksp = np.zeros(ksp.shape)

for i in range(ksp.shape[0]):
    for j in range(ksp.shape[1]):
        usksp[i,j] = uspat*ksp[i,j]

#usksp = UFT(orim,uspat, normalize=False)/np.percentile( np.abs(tUFT(UFT(orim,uspat, normalize=False),uspat, normalize=False).flatten())  ,99)

slice = 50

usksp=usksp[slice]

regtype='reg2_dc'
reg=0.1
dcprojiter=10
chunks40=True

if R<=3:
     num_iter = 402 # 302
else:
     num_iter = 602 # 602
     
if args.contrun == 0:
     contRec = ''
else:
     contRec = basefolder+'MAPestimation/rec_us'+str(R)+'_vol'+str(vol)+'_sli'+str(sli)+'_regtype_'+regtype+'_dcprojiter_'+str(dcprojiter)
     numiter = 302

if not args.skiprecon:
     rec_vae = vaerecon.vaerecon(usksp, sensmaps=np.ones_like(usksp), dcprojiter=dcprojiter, lat_dim=lat_dim, patchsize=ndims, contRec=contRec, parfact=25, num_iter=num_iter, regiter=10, reglmb=reg, regtype=regtype, half=True, mode=mode, chunks40=chunks40, writer=file_writer)
     rec_vae = rec_vae[0]
     pickle.dump(rec_vae, open(basefolder+'MAPestimation/rec'+str(args.contrun)+'_us'+str(R)+'_vol'+str(vol)+'_sli'+str(sli)+'_regtype_'+regtype+'_dcprojiter_'+str(dcprojiter) ,'wb')   )
    
     lastiter = int((np.floor(rec_vae.shape[1]/13)-2)*13)
     maprecon = rec_vae[:,lastiter].reshape([252, 308]) # this is the final reconstructed image














