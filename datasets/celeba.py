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
from matplotlib import pyplot as plt

def divide_dataset_basic(path, fraction=0.85):
    '''
    Helper function for dividing the CelebA dataset into train or test sets
    
    Check if path exists or not
    and then create subsets accordingly

    The paper creates a 85-15 % split. I use the same split here
    Change the `fraction` if you want to change the split
    '''
    if exists(join(path, 'train')):
        print 'Path {} exists'.format(path)
        return

    # Divide into train and val
    all_files = None
    for root, _, all_files in os.walk(join(path, 'img_align_celeba')):
        all_files = map(lambda x: join(root, x), all_files)
        break
    np.random.shuffle(all_files)

    # Divide the dataset
    # They report only validation scores, which is 15% of the dataset
    divide = int(len(all_files)*fraction)
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
            cfg : config file
            crop: Crop sizes (default: 64)
    '''
    def __init__(self, mode, cfg, crop_size=64):
        super(CelebA, self).__init__()
        self.mode = mode
        self.path = cfg['dataset']['path']
        self.height = cfg['dataset']['h']
        self.type = cfg['dataset'].get('type', None)
        self.P = cfg['dataset'].get('p', 1)
        self.crop_size = crop_size
        assert (mode in ['train', 'val']), 'mode in {} should be train/val'.format(self.__name__)
        self._get_files()


    def _get_files(self):
        # Get all image file paths from root path and store in a list
        self.files = []
        if not exists(join(self.path, self.mode)):
            divide_dataset_basic(self.path)

        for root, _, files in os.walk(join(self.path, self.mode)):
            self.files.extend(sorted(list(map(lambda x: join(root, x), files))))


    def __len__(self):
        # Return length
        assert self.files != [], 'Empty file list'
        return len(self.files)


    def _get_mask(self, idx, image):
        # Get mask given in the config by checking type
        if self.type is None:
            mask = np.ones(image.shape)[:, :, :1]
            x_start, y_start = np.random.randint(image.shape[0] - self.height, size=(2, ))
            width, height = self.height, self.height
            mask[y_start:y_start+height, x_start:x_start+width] = 0
        # Center mask, create a mask of height H * H from center
        elif self.type == 'center':
            mask = np.ones(image.shape)[:, :, :1]
            c_y, c_x = image.shape[0]//2, image.shape[1]//2
            mask[c_y - self.height//2 : c_y + self.height//2, \
                 c_x - self.height//2 : c_x + self.height//2] = 0
        # Random mask, drop pixels randomly
        elif self.type == 'random':
            mask = np.random.rand(*image.shape)[:, :, :1]
            mask = (mask < self.P).astype(float)
        # Half mask, randomly pick one from left, right top bottom
        elif self.type == 'half':
            mask = np.ones(image.shape)[:, :, :1]
            # Get which half is to be masked in case one is chosen
            # and then choose at random
            leftStart, topStart = 32*np.random.randint(2, size=(2, ))
            goLeft = np.random.rand() < 0.5
            if goLeft:
                mask[:, leftStart:leftStart+32] = 0
            else:
                mask[topStart:topStart+32, :] = 0
        # rest are not implemented for now
        else:
            raise NotImplementedError
        return mask


    def __getitem__(self, idx):
        # Get the idx' valued item
        # First fetch the image, then get the mask
        # Return the final image
        filename = self.files[idx]
        image = cv2.imread(filename)
        assert (image is not None), filename

        # Convert image to right format (Crop and scale)
        image = cv2.resize(image, (self.crop_size, self.crop_size))
        image = (image[:, :, ::-1]/255.0)*2 - 1

        mask = self._get_mask(idx, image)
        observed = mask*image

        return {
            'image': torch.Tensor(image.transpose(2, 0, 1)),
            'mask' : torch.Tensor(mask.transpose(2, 0, 1)),
            'observed': torch.Tensor(observed.transpose(2, 0, 1)),
        }



if __name__ == '__main__':
    CFG = {
        'dataset': {
            'path': '/home/rohitrango/datasets/CelebA',
            'type': 'half',
            'h'   : 20,
        }
    }
    dataset = CelebA('train', CFG)
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        plt.subplot(1, 3, 1)
        plt.imshow((data['image'].numpy().transpose(1, 2, 0) + 1)/2.0)
        plt.subplot(1, 3, 2)
        plt.imshow((data['mask'][0] + 1)/2.0, 'gray')
        plt.subplot(1, 3, 3)
        plt.imshow((data['observed'].numpy().transpose(1, 2, 0) + 1)/2.0)
        plt.show()
