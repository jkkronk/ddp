import numpy as np
import nibabel as nib
import glob, os
import h5py
import scipy.misc as smi
#import matplotlib.pyplot as plt
from US_pattern import US_pattern
import time as tm
import random


from skimage.util.shape import view_as_blocks


class MR_image_data:
    # Class that reads the MR images, prepares them for further work, and generates batches of images/patches
    #====================================================================
    def __init__(self, dirname='/srv/beegfs02/scratch/fastmri_challenge/data/brain', trainset_ratio = 1, noiseinvstd=0, patchsize=28, modality='AXFLAIR_'):
        self.dirname = dirname
        self.noise = noiseinvstd
        self.patchsize=patchsize

        self.allfiles = os.listdir(dirname+'/multicoil_train/h5_slices/')
        self.datafiles = [s for s in self.allfiles if modality in s]
        self.data_size_train = int(trainset_ratio*len(self.datafiles))
        self.datafiles_train = self.datafiles[:self.data_size_train]

        with h5py.File(self.dirname + '/multicoil_train/' + self.datafiles_train[0], 'r') as fdset:
             self.reconstruction_rss_size = fdset['reconstruction_rss'].shape #  The shape of the reconstruction_rss tensor is (number of slices, r_height, r_width)

        self.allfiles_test = os.listdir(dirname + '/multicoil_val/h5_slices/')
        self.datafiles_test = [s for s in self.allfiles_test if modality in s]
        self.data_size_test = len(self.datafiles_test)

    #====================================================================
    def get_batch(self, batchsize, test=False): # rixsb = np.sort(np.random.choice(self.nstrain*len(self.useSlice), batchsize, replace=False))
        btch = np.zeros([batchsize, self.reconstruction_rss_size[-1], self.reconstruction_rss_size[-2]])

        if not test: # If training data
            random.shuffle(self.datafiles_train)


            for ix in range(batchsize):
                volindex = np.sort(np.random.choice(self.data_size_train, 1, replace=False))
                file = self.datafiles_train[volindex[0]]

                with h5py.File(self.dirname + '/multicoil_train/h5_slices/' + file, 'r') as fdset:
                    h5data = fdset['reconstruction_rss'][:]

                sliceindex = np.sort(np.random.choice(h5data.shape[0], 1, replace=False))
                #print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))
                btch[ix,:,:] = h5data[sliceindex[0], :,:] # vol, sli, x, y

        else: # If test data
            for ix in range(batchsize):
                volindex = np.sort(np.random.choice(self.data_size_test, 1, replace=False))
                file = self.datafiles_test[volindex[0]]

                with h5py.File(self.dirname + '/multicoil_val/h5_slices/' + file, 'r') as fdset:
                    h5data = fdset['reconstruction_rss'][:]

                sliceindex = np.sort(np.random.choice(h5data.shape[0], 1, replace=False))
                # print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))
                btch[ix, :, :] = h5data[sliceindex[0], :, :]  # vol, sli, x, y

        if self.noise ==0:
             return btch
        elif self.noise > 0:
             return btch + np.random.normal(loc=0, scale=1/self.noise, size=btch.shape)
        
    def get_patch(self, batchsize, test=False): # rixsb = np.sort(np.random.choice(self.nstrain*len(self.useSlice), batchsize, replace=False))
        btch = np.zeros([batchsize, self.patchsize, self.patchsize])

        if not test: # If train
            random.shuffle(self.datafiles_train)

            for ix in range(batchsize):
                volindex = np.sort(np.random.choice(self.data_size_train, 1, replace=False))
                subj_file = self.datafiles_train[volindex[0]]

                with h5py.File(self.dirname + '/multicoil_train/h5_slices/' + subj_file, 'r') as fdset:
                    h5data = fdset['reconstruction_rss'][:]

                sliceindex = np.sort(np.random.choice(h5data.shape[0], 1, replace=False))
                patchindexr = np.random.randint(0, h5data[sliceindex[0]].shape[0] - self.patchsize)
                patchindexc = np.random.randint(h5data[sliceindex[0]].shape[1] - self.patchsize)
                # print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))
                btch[ix, :, :] = h5data[sliceindex[0],
                                     patchindexr:patchindexr + self.patchsize,
                                     patchindexc:patchindexc + self.patchsize]  # sli, x, y

        else:
            for ix in range(batchsize):
                volindex = np.sort(np.random.choice(self.data_size_test, 1, replace=False))
                subj_file = self.datafiles_test[volindex[0]]
                with h5py.File(self.dirname + '/multicoil_val/h5_slices/' + subj_file, 'r') as fdset:
                    h5data = fdset['reconstruction_rss'][:]

                sliceindex = np.sort(np.random.choice(h5data.shape[0], 1, replace=False))
                patchindexr = np.random.randint(0, h5data[sliceindex[0]].shape[0] - self.patchsize)
                patchindexc = np.random.randint(h5data[sliceindex[0]].shape[1] - self.patchsize)
                # print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))
                btch[ix, :, :] = h5data[sliceindex[0],
                                     patchindexr:patchindexr + self.patchsize,
                                     patchindexc:patchindexc + self.patchsize]  # sli, x, y

        #print("KCT-dbg: time for a batch: " + str(tm.time()-tms))
        
        if self.noise ==0:
             return btch
        elif self.noise > 0:
             return btch + np.random.normal(loc=0, scale=1/self.noise, size=btch.shape)
        

    def get_image(self, subj, slice):

         with h5py.File(self.dirname + '/multicoil_train/' + subj, 'r') as fdset:
              if self.noise == 0:
                   return fdset['reconstruction_rss'][slice, :,:] # sli, x, y
              elif self.noise >0:
                   return fdset['reconstruction_rss'][slice, :,:] + np.random.normal(loc=0, scale=1/self.noise, size=[self.imgSize[0],self.imgSize[1]])
         
         
     
    def pad_image(self, x, newsize, mode='edge'):
         # or mode='constant'
         assert(newsize[0]>=x.shape[0] and newsize[1]>=x.shape[1])
         
         pr=int( (newsize[0]-x.shape[0])/2 )
         pc=int( (newsize[1]-x.shape[1])/2 )
         
         return np.pad(x, ((pr,pr),(pc,pc)), mode=mode)# mode='constant', constant_values=x.min())
    
    def pad_batch(self, x, newsize, mode='edge'):
         
         tmp=[]
         for ix in range(x.shape[0]): # loop on batch dimension
              tmp.append(self.pad_image(x[ix, :, :], newsize, mode) )
         
         return np.array(tmp)
    
     

    #copy pasted from carpedm20's dcgan implementation
    def center_crop(self, x, crop_h, crop_w, resize_h=64, resize_w=64, offset=0):
          
          
          if crop_w is None:
               crop_w = crop_h
          h, w = x.shape[:2]
          j = int(round((h - crop_h)/2.))
          i = int(round((w - crop_w)/2.))
          
          #somehow scipy rescales images to unit8 range, make sure to rescale after interpolation
          #(https://github.com/scipy/scipy/issues/4458)
          prevmax=x.max()
          
          if (resize_h==crop_h) and (resize_w==crop_w) :
               return x[j+offset:j+offset+crop_h, i:i+crop_w]
          else:
               return smi.imresize(x[j+offset:j+offset+crop_h, i:i+crop_w], [resize_h, resize_w], interp='lanczos') *prevmax /255


class MR_kspace_data:
    # Class that reads the MR kspace images, prepares them for further work, and generates batches of images/patches

    # ====================================================================
    def __init__(self, dirname='/srv/beegfs02/scratch/fastmri_challenge/data/brain/'):
        self.dirname = dirname
        self.allfiles = os.listdir(dirname + 'multicoil_val/')

    def get_subj(self, subj_name):

        with h5py.File(self.dirname + 'multicoil_val/' + subj_name, 'r') as fdset:
            kspace_img = fdset['kspace'][:]  # The shape of kspace tensor is (number of slices, number of coils, height, width)

        return kspace_img

    def get_gt(self, subj_name):
        with h5py.File(self.dirname + 'multicoil_val/' + subj_name, 'r') as fdset:
            gt_img = fdset['reconstruction_rss'][:]  # The shape of kspace tensor is (number of slices, number of coils, height, width)

        return gt_img

