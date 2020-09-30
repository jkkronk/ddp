__author__ = 'jonatank'
from torch.utils.data import Dataset
from torchvision import transforms

import h5py
import torch
import numpy as np
from skimage.transform import resize
from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import time
import imgaug as ia
import glob, os
import random

class fastMRI_patch(Dataset):
    def __init__(self, dirname='/srv/beegfs02/scratch/fastmri_challenge/data/brain', trainset_ratio=0.5,
                 patchsize=28, modality='AXFLAIR_' ,train=True):
        self.dirname = dirname
        self.patchsize = patchsize
        self.train = train

        if train:
            self.folderpath = '/multicoil_train/'
        else:
            self.folderpath = '/multicoil_val_old/'

        self.allfiles = os.listdir(dirname + self.folderpath)
        self.data_size = round(trainset_ratio * len(self.allfiles))
        self.files = self.allfiles[:self.data_size]

        self.datafiles = [s for s in self.files if modality in s]

        self.size = len(self.datafiles)

        print(self.datafiles)

    def __getitem__(self, index):
        with h5py.File(self.dirname + self.folderpath + self.datafiles[index], 'r') as fdset:
            file = fdset['reconstruction_rss'][:]
            sliceindex = np.sort(np.random.choice(file.shape[0], 1, replace=False))
            patchindexr = np.random.randint(0, file[sliceindex[0]].shape[
                0] - self.patchsize)
            patchindexc = np.random.randint(
                file[sliceindex[0]].shape[1] - self.patchsize)
            # print("KCT-dbg: vol index: "+str(volindex[0])+" sliceindex: "+str(sliceindex[0]))

            img = file[sliceindex[0],
                             patchindexr:patchindexr + self.patchsize,
                             patchindexc:patchindexc + self.patchsize]  # sli, x, y

        fdset.close()

        return torch.from_numpy(img).unsqueeze(0)

    def __len__(self):
        return self.size

class fastMRI_kspace():
    def __init__(self, dirname='/srv/beegfs02/scratch/fastmri_challenge/data/brain', subj='file_brain_AXFLAIR_200_6002462.h5'):
        self.dirname = dirname
        self.folderpath = '/multicoil_val_old/'

        self.file = dirname + self.folderpath + subj
        with h5py.File(self.file, 'r') as fdset:
            self.kspace = fdset['kspace'][:] # [slice,coil,sx,sy]
        fdset.close()

        self.size = self.kspace.shape[0]

    def getitem(self):
        kspace_slice = self.kspace

        return kspace_slice # [Slice, Coil, Sx, Sy]

    def len(self):
        return self.size
