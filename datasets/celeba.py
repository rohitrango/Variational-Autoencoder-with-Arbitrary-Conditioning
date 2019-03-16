'''
CelebA Dataset for PyTorch
Author: Rohit Jena
'''
import os
from os.path import join, exists
import shutil
import cv2

import torch
from torch.utils.data import Dataset
import numpy as np

def divide_dataset_basic(path):
    '''
    Check if path exists or not
    and then create subsets
    '''
    if exists(join(path, 'train')):
        print 'Path {} exists'.format(path)
        return

    # Divide into train, val and test
    all_files = None
    for root, _, all_files in os.walk(join(path, 'img_align_celeba')):
        all_files = map(lambda x: join(root, x), all_files)
        break
    np.random.shuffle(all_files)

    # Divide the dataset
    # They report only validation scores, which is 15% of the dataset
    divide = int(len(all_files)*0.85)
    mode = {
        'train' : all_files[:divide],
        'val'   : all_files[divide:],
    }

    # Copy the files
    for key, files in mode.items():
        cur_path = join(path, key)
        os.makedirs(cur_path)
        for filename in files:
            shutil.copy(filename, cur_path)
            print 'copied {}'.format(filename)
        print "Copied all {} files to {}".format(key, cur_path)


class CelebA(Dataset):

    '''
    Generalized dataset for CelebA
    Inputs: path: path to files
            mode: train/val modes
            crop: Crop sizes (default: 64)
    '''
    def __init__(self, mode, cfg, crop_size=64):
        super(CelebA, self).__init__()
        self.mode = mode
        self.path = cfg['dataset']['path']

        self.crop_size = crop_size
        assert (mode in ['train', 'val']), 'mode in {} should be train/val'.format(self.__name__)
        self._get_files()
        raise NotImplementedError


    def _get_files(self):
        # Get all image file paths from root path
        self.files = []
        assert exists(join(self.path, self.mode)), \
                'Path {} does not exist'.format(join(self.path, self.mode))

        for root, _, files in os.walk(join(self.path, self.mode)):
            self.files.extend(sorted(list(map(lambda x: join(root, x), files))))


    def __len__(self):
        # Return length
        assert self.files != [], 'Empty file list'
        return len(self.files)


    def _get_task_image(self, image, idx):
        # Get the image corresponding to the task (removepixel, etc)
        rng = np.random.RandomState(idx)
        task_image = image + 0
        raise NotImplementedError


    def __getitem__(self, idx):
        # Get the idx' valued item
        filename = self.files[idx]

        image = cv2.imread(filename)
        assert (image is not None), filename
        # Convert image to right format (Crop and scale)
        image = cv2.resize(image, (self.crop_size, self.crop_size))
        image = (image[:, :, ::-1]/255.0)*2 - 1
        task_image = self._get_task_image(image, idx)

        return {
            'image': torch.Tensor(image.transpose(2, 0, 1)),
            'taskimage': torch.Tensor(task_image.transpose(2, 0, 1)),
        }

