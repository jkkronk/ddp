import numpy as np
import pickle
import vaerecon5


from US_pattern import US_pattern

import argparse

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--sli', type=int, default=3) 
parser.add_argument('--base', default="sess_02_07_2018/CK/")
parser.add_argument('--usfact', type=float, default=2) 
parser.add_argument('--contrun', type=int, default=0) 

args=parser.parse_args()



def FT (x):
     #inp: [nx, ny]
     #out: [nx, ny, ns]
     return np.fft.fftshift(    np.fft.fft2( sensmaps*np.tile(x[:,:,np.newaxis],[1,1,sensmaps.shape[2]]) , axes=(0,1)  ),   axes=(0,1)    ) #  / np.sqrt(x.shape[0]*x.shape[1])

def tFT (x):
     #inp: [nx, ny, ns]
     #out: [nx, ny]
     
     temp = np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )
     return np.sum( temp*np.conjugate(sensmaps) , axis=2)  / np.sum(sensmaps*np.conjugate(sensmaps),axis=2) #  * np.sqrt(x.shape[0]*x.shape[1])


def UFT(x, uspat):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny, ns]
     
     return np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])*FT(x)

def tUFT(x, uspat):
     #inp: [nx, ny], [nx, ny]
     #out: [nx, ny]
     
     tmp1 = np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])
#     print(x.shape)
#     print(tmp1.shape)
     
     return  tFT( tmp1*x )


#############################
#
#
#def FT (x):
#     #inp: [nx, ny]
#     #out: [nx, ny]
#     return np.fft.fftshift(    np.fft.fft2(  x , axes=(0,1)  ),   axes=(0,1)    ) / np.sqrt(252*308)
#
#def tFT (x):
#     #inp: [nx, ny]
#     #out: [nx, ny]
#     return np.fft.ifft2(  np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )   * np.sqrt(252*308)
#
#def UFT(x, uspat):
#     #inp: [nx, ny], [nx, ny]
#     #out: [nx, ny]
#     
#     return uspat*FT(x)
#
#def tUFT(x, uspat):
#     #inp: [nx, ny], [nx, ny]
#     #out: [nx, ny]
#     return  tFT( uspat*x )



def calc_rmse(rec,imorig):
     return 100*np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))) / np.sum(np.square(np.abs(imorig))) )

ndims=28
lat_dim=60

mode = 'MRIunproc'#'Melanie_BFC'

USp = US_pattern()

usfact = args.usfact

print(usfact)
if np.floor(usfact) == usfact: # if it is already an integer
     usfact = int(usfact)
print(usfact)

###################
###### RECON ######
###################

#dirbase = '/usr/bmicnas01/data-biwi-01/ktezcan/measured_data/measured2/the_h5_files/'+args.base
#sli= args.sli
#
#import h5py
#f = h5py.File(dirbase+'ddr_sl'+str(sli)+'.h5', 'r')
#ddr = np.array((f['DS1']))
#f = h5py.File(dirbase+'ddi_sl'+str(sli)+'.h5', 'r')
#ddi = np.array((f['DS1']))
#dd= ddr+1j*ddi
#dd=np.transpose(dd)
##          dd=np.transpose(dd,axes=[1,0,2])
#dd=np.rot90(np.rot90(np.rot90(dd,3),3),3)
#
#dd = np.fft.ifftn(np.fft.fftshift(np.fft.fftn(dd,axes=[0,1]),axes=[0,1]),axes=[0,1])
#
#
#f = h5py.File(dirbase+'espsi_sl'+str(sli)+'.h5', 'r')
#espsi = np.array((f['DS1']))
#f = h5py.File(dirbase+'espsr_sl'+str(sli)+'.h5', 'r')
#espsr = np.array((f['DS1']))
#esps= espsr+1j*espsi
#esps = np.transpose(esps)
##               esps=np.transpose(esps,axes=[1,0,2])
#esps=np.rot90(np.rot90(np.rot90(esps,3),3),3)
#esps=np.fft.fftshift(esps,axes=[0,1])
#sensmaps = esps.copy()
#sensmaps = np.fft.fftshift(sensmaps,axes=[0,1])
#          
#
#R=usfact
#
#ddimc = tFT(dd)
#dd=dd/np.percentile(  np.abs(ddimc).flatten()   ,99)

#################################################################################

dirbase = '/usr/bmicnas01/data-biwi-01/ktezcan/measured_data/measured2/the_h5_files/'+args.base
sli= args.sli

import h5py
f = h5py.File(dirbase+'ddr_sl'+str(sli)+'.h5', 'r')
ddr = np.array((f['DS1']))
f = h5py.File(dirbase+'ddi_sl'+str(sli)+'.h5', 'r')
ddi = np.array((f['DS1']))

dd= ddr+1j*ddi
dd=np.transpose(dd)
#          dd=np.transpose(dd,axes=[1,0,2])
dd=np.rot90(dd,3)

dd = np.fft.ifftn(np.fft.fftshift(np.fft.fftn(dd,axes=[0,1]),axes=[0,1]),axes=[0,1])



f = h5py.File(dirbase+'espsi_sl'+str(sli)+'.h5', 'r')
espsi = np.array((f['DS1']))
f = h5py.File(dirbase+'espsr_sl'+str(sli)+'.h5', 'r')
espsr = np.array((f['DS1']))

esps= espsr+1j*espsi
esps = np.transpose(esps)
#               esps=np.transpose(esps,axes=[1,0,2])
esps=np.rot90(esps,3)
esps=np.fft.fftshift(esps,axes=[0,1])
sensmaps = esps.copy()
sensmaps = np.fft.fftshift(sensmaps,axes=[0,1])

sensmaps=np.rot90(np.rot90(sensmaps))
dd=np.rot90(np.rot90(dd))
#          sensmaps=np.swapaxes(sensmaps,0,1)
#          dd=np.swapaxes(dd,0,1)



sensmaps=sensmaps/np.tile(np.sum(sensmaps*np.conjugate(sensmaps),axis=2)[:,:,np.newaxis],[1, 1, sensmaps.shape[2]])

ddimc = tFT(dd)
dd=dd/np.percentile(  np.abs(ddimc).flatten()   ,99)
ddimc = tFT(dd)

          


###################################################

R=usfact

try:
     uspat = np.load('/usr/bmicnas01/data-biwi-01/ktezcan/reconsampling/uspats/uspat_realim_us'+str(R)+'_base_'+dirbase[-19:-1].replace("/","_")+'_sli'+str(sli)+'.npy')
     print("Read from existing u.s. pattern file")
except:
     uspat = USp.generate_opt_US_pattern_1D(dd.shape[0:2], R=R, max_iter=100, no_of_training_profs=15)
     np.save('/usr/bmicnas01/data-biwi-01/ktezcan/reconsampling/uspats/uspat_realim_us'+str(R)+'_base_'+dirbase[-19:-1].replace("/","_")+'_sli'+str(sli), uspat)
     print("Generated a new u.s. pattern file")
     
     
usksp = dd*np.tile(uspat[:,:,np.newaxis], [1, 1, dd.shape[2]])




if R<=3:
     num_iter = 602 # 302
else:
     num_iter = 602 # 602
     
num_iter = 402
     
regtype='reg2'
reg=0 # no phase regulization!
dcprojiter=10
onlydciter=10 # do firist iterations only SENSE reconstruction
chunks40=True
mode = 'MRIunproc'
  
#rec_vae = vaerecon5.vaerecon(usksp, sensmaps=sensmaps, dcprojiter=10, onlydciter=onlydciter, lat_dim=lat_dim, patchsize=ndims, parfact=20, num_iter=302, rescaled = rescaled, half=half,regiter=15, reglmb=reg, regtype=regtype)
          
rec_vae = vaerecon5.vaerecon(usksp, sensmaps=sensmaps, dcprojiter=10, onlydciter=10, lat_dim=lat_dim, patchsize=ndims, contRec='' ,parfact=25, num_iter=302, regiter=10, reglmb=reg, regtype=regtype, half=True, mode=mode, chunks40=chunks40)
pickle.dump(rec_vae[0], open('/usr/bmicnas01/data-biwi-01/ktezcan/reconsampling/MAPestimation/rec_meas_us'+str(R)+'_sli'+str(sli)+'_regtype_'+regtype+'_dcprojiter_'+str(dcprojiter) ,'wb')   )















