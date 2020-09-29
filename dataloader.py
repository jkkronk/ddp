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
    def __init__(self, dirname='/srv/beegfs02/scratch/fastmri_challenge/data/brain', trainset_ratio = 0.5, noiseinvstd=0, patchsize=28):
        self.dirname = dirname
        #self.imgSize = imgSize
        self.noise = noiseinvstd
        
        self.patchsize=patchsize
        
        
        #self.testchunks = testchunks
        
        self.hdfFileName = []
         

        self.files_brain = []
        self.files_segms = []
        
        self.allfiles = os.listdir(dirname+'/multicoil_train')

        with h5py.File(self.dirname + '/multicoil_train/' + self.allfiles[0], 'r') as fdset:
             self.reconstruction_rss_size = fdset['reconstruction_rss'].shape #  The shape of the reconstruction_rss tensor is (number of slices, r_height, r_width)

        self.data_size = round(trainset_ratio*len(self.allfiles))

        self.allfiles_test = os.listdir(dirname + '/multicoil_val_old')
        self.data_size_test = len(self.allfiles_test)

        #self.nochunks = 0
     
        #for ix in range(len(allfiles)):
        #     if (allfiles[ix])[:3] == 'chu':
        #          self.nochunks = self.nochunks + 1
                  
        #self.notrainchunks = self.nochunks - len(self.testchunks)
        #self.notestchunks = len(self.testchunks)
        
        #self.imgSizeCh = self.imgSize.copy()
        #self.imgSizeCh.append(self.nochunks)
        
        #with h5py.File(self.dirname+"chunk"+str(39), 'r') as fdset:
        #     self.imgSizeCh = fdset['mydataset'].shape

    
    #====================================================================
    def get_batch(self, batchsize, test=False): # rixsb = np.sort(np.random.choice(self.nstrain*len(self.useSlice), batchsize, replace=False))

        btch = np.zeros([self.reconstruction_rss_size[-1], self.reconstruction_rss_size[-2], batchsize])

        if not test:
            random.shuffle(self.allfiles)
            train_files = self.allfiles[:self.data_size]

            #chunkindices = np.sort(np.random.choice(self.notrainchunks, batchsize, replace=True))
             
            for ix in range(batchsize):
                volindex = np.sort(np.random.choice(self.data_size, 1, replace=False))
                file = train_files[volindex[0]]

                with h5py.File(self.dirname + '/multicoil_train/' + file, 'r') as fdset:
                    sliceindex = np.sort(np.random.choice(fdset.shape[0], 1, replace=False))
                    #print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))
                    btch[:,:,ix] = fdset['reconstruction_rss'][sliceindex[0], :,:] # vol, sli, x, y
                      
        else:
            # To be implemented
            #print('EXIT: Test dataloader not implemented...')
            #quit()

            # chunkindices = np.sort(np.random.choice(self.notrainchunks, batchsize, replace=True))

            for ix in range(batchsize):
                volindex = np.sort(np.random.choice(self.data_size_test, 1, replace=False))
                file = self.allfiles_test[volindex[0]]

                with h5py.File(self.dirname + '/multicoil_val_old/' + file, 'r') as fdset:
                    sliceindex = np.sort(np.random.choice(fdset.shape[0], 1, replace=False))
                    # print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))
                    btch[:, :, ix] = fdset['reconstruction_rss'][sliceindex[0], :, :]  # vol, sli, x, y

            #chunkindices = np.sort(np.random.choice(self.notestchunks, batchsize, replace=True)) + self.notrainchunks
             
             #for ix, ixr in enumerate(chunkindices):
             #     with h5py.File(self.dirname+"chunk"+str(ixr), 'r') as fdset:
             #         volindex = np.sort(np.random.choice(self.imgSizeCh[0], 1, replace=False))
             #         sliceindex = np.sort(np.random.choice(self.imgSizeCh[1], 1, replace=False))
             #         #print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))
             #         btch[:,:,ix] = fdset['mydataset'][volindex[0], sliceindex[0], :,:]
                      
                     
        #print("KCT-dbg: time for a batch: " + str(tm.time()-tms))   
        
        if self.noise ==0:
             return btch
        elif self.noise > 0:
             return btch + np.random.normal(loc=0, scale=1/self.noise, size=btch.shape)
        
    def get_patch(self, batchsize, test=False): # rixsb = np.sort(np.random.choice(self.nstrain*len(self.useSlice), batchsize, replace=False))
        btch = np.zeros([self.patchsize, self.patchsize, batchsize])
        #btch = np.zeros([self.patchsize, self.patchsize, batchsize ])

        if not test:
            #chunkindices = np.sort(np.random.choice(self.notrainchunks, batchsize, replace=True))

            random.shuffle(self.allfiles)
            train_files = self.allfiles[:self.data_size]

            # chunkindices = np.sort(np.random.choice(self.notrainchunks, batchsize, replace=True))

            for ix in range(batchsize):
                volindex = np.sort(np.random.choice(self.data_size, 1, replace=False))
                file = train_files[volindex[0]]

                with h5py.File(self.dirname + '/multicoil_train/' + file, 'r') as fdset:
                    sliceindex = np.sort(np.random.choice(fdset['reconstruction_rss'].shape[0], 1, replace=False))
                    patchindexr = np.random.randint(0, fdset['reconstruction_rss'][sliceindex[0]].shape[
                        0] - self.patchsize)
                    patchindexc = np.random.randint(
                        fdset['reconstruction_rss'][sliceindex[0]].shape[1] - self.patchsize)
                    # print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))

                    btch[:, :, ix] = fdset['reconstruction_rss'][sliceindex[0],
                                     patchindexr:patchindexr + self.patchsize,
                                     patchindexc:patchindexc + self.patchsize]  # sli, x, y

        else:
            # To be implemented

            for ix in range(batchsize):
                volindex = np.sort(np.random.choice(self.data_size_test, 1, replace=False))
                file = self.allfiles_test[volindex[0]]

                with h5py.File(self.dirname + '/multicoil_val_old/' + file, 'r') as fdset:
                    sliceindex = np.sort(np.random.choice(fdset['reconstruction_rss'].shape[0], 1, replace=False))
                    patchindexr = np.random.randint(0, fdset['reconstruction_rss'][sliceindex[0]].shape[0] - self.patchsize)
                    patchindexc = np.random.randint(fdset['reconstruction_rss'][sliceindex[0]].shape[1] - self.patchsize)
                    # print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))

                    btch[:, :, ix] = fdset['reconstruction_rss'][sliceindex[0],
                                     patchindexr:patchindexr + self.patchsize,
                                     patchindexc:patchindexc + self.patchsize]  # sli, x, y


            #chunkindices = np.sort(np.random.choice(self.notestchunks, batchsize, replace=True)) + self.notrainchunks
            #
            #for ix, ixr in enumerate(chunkindices):
            #     with h5py.File(self.dirname+"chunk"+str(ixr), 'r') as fdset:
            #        volindex = np.sort(np.random.choice(self.imgSizeCh[0], 1, replace=False))
            #          sliceindex = np.sort(np.random.choice(self.imgSizeCh[1], 1, replace=False))
            #          patchindexr = np.random.randint(0, self.imgSizeCh[2] - self.patchsize)
            #          patchindexc =  np.random.randint(self.imgSizeCh[3] - self.patchsize)
            #          btch[:,:,ix] = fdset['mydataset'][volindex[0], sliceindex[0], patchindexr:patchindexr+self.patchsize, patchindexc:patchindexc+self.patchsize] # vol, sli, x, y
                      
                     
        #print("KCT-dbg: time for a batch: " + str(tm.time()-tms))   
        
        if self.noise ==0:
             return btch
        elif self.noise > 0:
             return btch + np.random.normal(loc=0, scale=1/self.noise, size=btch.shape)
        

    def get_image(self, subj, slice):

         with h5py.File(self.dirname + '/multicoil_train/' + subj + '.h5', 'r') as fdset:
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
    def __init__(self, dirname='/srv/beegfs02/scratch/fastmri_challenge/data/brain'):
        self.dirname = dirname
        self.allfiles = os.listdir(dirname + '/multicoil_valid')
        self.data_size = len(self.allfiles)


    # ====================================================================
    def get_batch(self, batchsize, subj=0):  # rixsb = np.sort(np.random.choice(self.nstrain*len(self.useSlice), batchsize, replace=False))
        # NOT IMPLEMENTED FOR KSPACE YET
        print('NOT IMPLEMENTED FOR KSPACE YET. EXIT')
        quit()


        subj_name = self.allfiles[subj]

        with h5py.File(self.dirname + '/multicoil_valid/' + subj_name, 'r') as fdset:
            self.kspace_size = fdset[
                'kspace'].shape  # The shape of kspace tensor is (number of slices, number of coils, height, width)

        btch = np.zeros([self.reconstruction_rss_size[-1], self.reconstruction_rss_size[-2], self.reconstruction_rss_size[1], batchsize])

        for ix in range(batchsize):
            volindex = np.sort(np.random.choice(self.data_size_test, 1, replace=False))
            file = self.allfiles_test[volindex[0]]

            with h5py.File(self.dirname + '/multicoil_valid/' + file, 'r') as fdset:
                sliceindex = np.sort(np.random.choice(fdset.shape[0], 1, replace=False))
                # print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))
                btch[:, :, ix] = fdset['reconstruction_rss'][sliceindex[0], :, :]  # vol, sli, x, y

            # chunkindices = np.sort(np.random.choice(self.notestchunks, batchsize, replace=True)) + self.notrainchunks

            # for ix, ixr in enumerate(chunkindices):
            #     with h5py.File(self.dirname+"chunk"+str(ixr), 'r') as fdset:
            #         volindex = np.sort(np.random.choice(self.imgSizeCh[0], 1, replace=False))
            #         sliceindex = np.sort(np.random.choice(self.imgSizeCh[1], 1, replace=False))
            #         #print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))
            #         btch[:,:,ix] = fdset['mydataset'][volindex[0], sliceindex[0], :,:]

        # print("KCT-dbg: time for a batch: " + str(tm.time()-tms))


    def get_subj(self, subj):
        subj_name = self.allfiles[subj]

        with h5py.File(self.dirname + '/multicoil_valid/' + subj_name, 'r') as fdset:
            kspace_img = fdset['kspace']  # The shape of kspace tensor is (number of slices, number of coils, height, width)

        return kspace_img

