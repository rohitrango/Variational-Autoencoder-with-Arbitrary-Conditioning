'''
MNIST Dataset for PyTorch
Author: Rohit Jena
'''
from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class MNISTRowDeleted(Dataset):
    '''
    Dataset of MNIST digits with only a few rows remaining
    '''
    def __init__(self, mode, cfg):
        super(MNISTRowDeleted, self).__init__()
        self.path = cfg['dataset']['path']
        self.horiz = cfg['dataset']['h']
        self.mode = mode
        assert mode in ['train', 'test'], 'Mode should be train/test'
        self._get_files()

    def _get_files(self):
        # Get the files from path
        mode = 'training' if self.mode == 'train' else 'test'
        path = join(self.path, 'processed', mode + '.pt')
        self.images, self.labels = torch.load(path)
        self.images = self.images.float()
        self.images = ((self.images)/255.0)*2 - 1

    def __len__(self):
        return int(self.images.shape[0])

    def __getitem__(self, idx):
        # image is 1*28*28
        image = self.images[idx:idx+1].float()

        # Get index and mask
        start = 10 + idx%10
        mask = np.zeros(image.shape)
        mask[:, start:start+self.horiz, :] = 1
        observed = (mask*image).float()

        # Return
        # image < [-1, 1]
        # mask < [0, 1]
        # observed < [-1, 1]
        return {
            'image'     : torch.Tensor(image),
            'mask'      : torch.Tensor(mask),
            'observed'  : torch.Tensor(observed)
        }


if __name__ == '__main__':
    CFG = {
        'dataset': {
            'path': '/home/rohitrango/datasets/MNIST',
            'h' : 3,
        }
    }
    dataset = MNISTRowDeleted('train', CFG)
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        plt.subplot(1, 3, 1)
        plt.imshow((data['image'][0] + 1)/2.0, 'gray')
        plt.subplot(1, 3, 2)
        plt.imshow((data['mask'][0] + 1)/2.0, 'gray')
        plt.subplot(1, 3, 3)
        plt.imshow((data['observed'][0] + 1)/2.0, 'gray')
        plt.show()
