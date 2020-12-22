import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import json
import pandas as pd
from utils import *


class IBSRdataset(Dataset):
    """ISBR Brain Segmentation Dataset"""
    # TODO Now load everything during initialization, should change to lazy loading
    def __init__(self, config_file, base_dir='../../data/IBSR_nifti_stripped/processed/', transforms_=None):
        """
        Args:
            config_file (string): path to the .json config file.
            base_dir (string): path to the base directory storing the processed data and annotation
            transform (List[torchvision.transforms]): a list of transformation Objects
        """
        self.transform = transforms.Compose(transforms_)
        self.config_file = config_file
        self.base_dir = base_dir

        # read config file
        with open(config_file) as f:
            dataset = json.load(f)

        vox_list = []
        for obj in dataset['data']:
            vox_list.append((obj['image'], obj['label']))

        # print(vox_list)

        # read images and labels
        img_list = []
        label_list = []
        for img_dir, label_dir in vox_list:
            img_dir = os.path.join(base_dir, img_dir)
            label_dir = os.path.join(base_dir, label_dir)

            img_list.append(load_nifti(img_dir)[:,:,:,1])
            label_list.append(load_nifti(label_dir))
        
        # print(len(img_list))

        self.img_slices = []
        self.label_slices = []
        for i in range(len(img_list)):
            img = img_list[i]
            label = label_list[i]
            assert img.shape == label.shape
            for i in range(img.shape[2]):
                # take slices
                img_slice = img[:,:,i]
                label_slice = label[:,:,i]
                if img_slice.min() != img_slice.max() and label_slice.min() != label_slice.max(): # take slices with contents only
                    self.img_slices.append(img_slice)
                    self.label_slices.append(label_slice)
                assert len(self.img_slices) == len(self.label_slices)
        
        # print(len(img_slices))

        
    def __len__(self):
        return len(self.img_slices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_slices[idx]
        label = self.label_slices[idx]

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(label).type(torch.FloatTensor)
        }

