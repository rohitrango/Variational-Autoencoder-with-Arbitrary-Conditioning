'''
Parts for networks
Author: Rohit Jena
'''
import torch
import torch.nn.functional as F
from torch import nn

class ResBlock(nn.Module):
    '''
    Residual block without downsampling
    '''
    def __init__(self,
                 in_ch,
                 out_ch,
                 hid_ch=None,
                 kernel_size=3,
                 padding=1,
                 activation=F.leaky_relu
                ):
        super(ResBlock, self).__init__()
        hid_ch = out_ch if hid_ch is None else hid_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hid_ch = hid_ch
        self.activation = activation
        # Define modules
        self.conv1 = nn.Conv2d(in_ch, hid_ch, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hid_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm2d(hid_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, image, concat=None):
        # get output from image
        if concat is None:
            outx = image
        else:
            outx = torch.cat([image, concat], 1)

        out = self.bn1(self.conv1(outx))
        out = self.activation(out)
        out = self.bn2(self.conv2(out))
        out = self.activation(out)
        out = out + self.residual(outx)
        return out


class InConv(nn.Module):
    '''
    Just the initial layers from an image to features
    '''
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 padding=1,
                 activation=F.leaky_relu
                ):
        super(InConv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        # Modules
        self.bn0 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.activation = activation

    def forward(self, image):
        out = self.bn0(image)
        out = self.bn1(self.conv1(out))
        out = self.activation(out)
        return out
