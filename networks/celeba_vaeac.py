'''
Network for MNIST
Author: Rohit Jena
'''
import torch
import torch.nn.functional as F
from torch import nn

import parts as parts
CHAN = 8

class ProposalNet(nn.Module):
    '''
    This network is the prior net for MNIST
    Proposal network tries to learn p(z|x, b)
    '''
    def __init__(self,
                 inp_channels=3,
                 n_hidden=16,
                 fc_hidden=50,
                 fc_out=16,
                 activation=F.relu,
                ):
        super(ProposalNet, self).__init__()
        self.inp_channels = inp_channels
        self.activation = activation
        # Get modules
        self.in_conv = parts.InConv(inp_channels, n_hidden)
        self.resblock1 = parts.ResBlock(n_hidden, n_hidden)
        self.resblock2 = parts.ResBlock(n_hidden, n_hidden)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        # fc layers
        self.fc1 = nn.Linear(n_hidden * 16 * 16, fc_hidden)
        self.fc_mean = nn.Linear(fc_hidden, fc_out)
        self.fc_sigma = nn.Linear(fc_hidden, fc_out)

    def forward(self, image):
        outx = torch.cat([image['image'], image['mask']], 1)
        out = outx
        out = self.in_conv(out)
        out = self.resblock1(out)
        out = self.maxpool1(out)
        out = self.resblock2(out)
        out = self.maxpool2(out)
        out = out.view(out.shape[0], -1)
        # fully connected parts
        out = self.activation(self.fc1(out))
        mean = self.fc_mean(out)
        logs = self.fc_sigma(out)
        return mean, logs


class EncoderDecoder(nn.Module):
    '''
    Prior and generator network
    Prior network tries to learn p(z|x_{1-b}, b)
    Generator tries to learn p(x_b | z, x_{1-b}, b)
    '''
    def __init__(self,
                 cfg,
                 activation=F.relu,
                ):
        super(EncoderDecoder, self).__init__()
        model_cfg = cfg['model']
        n_hidden = model_cfg['n_hidden']
        fc_hidden = model_cfg['fc_hidden']
        fc_out = model_cfg['fc_out']
        inp_channels = model_cfg['inp_channels']
        last_layer = model_cfg['last_layer']

        # Store the last layer activation
        if last_layer == 'tanh':
            self.last_activation = F.tanh
        elif last_layer == 'sigmoid':
            self.last_activation = F.sigmoid
        else:
            print("Using no activation at final layer...")
            self.last_activation = None

        # Proposal Net will take input channels + 1 for mask
        self.proposal_net = ProposalNet(
            inp_channels=inp_channels+1,
            n_hidden=n_hidden,
            fc_hidden=fc_hidden,
            fc_out=fc_out
            )

        self.n_hidden = n_hidden
        # Prior and generator net
        self.inp_channels = inp_channels
        self.activation = activation
        # Get modules
        # In Conv gets inp_channels + 1 for mask
        self.in_conv = parts.InConv(inp_channels + 1, n_hidden)
        self.resblock1 = parts.ResBlock(n_hidden, n_hidden)
        self.resblock2 = parts.ResBlock(n_hidden, n_hidden)
        self.resblock3 = parts.ResBlock(n_hidden, n_hidden)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        # fc layers
        self.fc1 = nn.Linear(n_hidden * CHAN * CHAN, fc_hidden)
        self.fc_mean = nn.Linear(fc_hidden, fc_out)
        self.fc_sigma = nn.Linear(fc_hidden, fc_out)

        # Decoder network
        self.out_fc1 = nn.Linear(fc_out, fc_hidden)
        self.out_fc2 = nn.Linear(fc_hidden, n_hidden * CHAN * CHAN)
        # resblocks
        self.out_resblock1 = parts.ResBlock(2 * n_hidden, n_hidden)
        self.out_resblock2 = parts.ResBlock(2 * n_hidden, n_hidden)
        self.out_resblock3 = parts.ResBlock(2 * n_hidden, n_hidden)
        self.upsample = nn.Upsample(scale_factor=2)
        self.out_resblock4 = parts.ResBlock(n_hidden, n_hidden)
        # Final output (will have inp_channels)
        self.out_conv = nn.Conv2d(n_hidden, 2*inp_channels, 3, padding=1)


    def forward(self, image):
        # Proposal net outputs
        # proposal net will take care of what inputs to take
        prop_mean, prop_logs = self.proposal_net(image)
        # Prior net
        # concatenate the observed image and mask together
        outx = torch.cat([image['observed'], image['mask']], 1)
        out = outx + 0
        out = self.in_conv(out)
        out = self.resblock1(out)           # 64
        out = self.maxpool1(out)            # 32
        out1 = out + 0
        out = self.resblock2(out)           # 32
        out = self.maxpool2(out)            # 16
        out2 = out + 0
        out = self.resblock3(out)
        out = self.maxpool3(out)            # 8
        out3 = out + 0

        out = out.view(out.shape[0], -1)
        # fully connected parts
        out = self.activation(self.fc1(out))
        mean = self.fc_mean(out)
        logs = self.fc_sigma(out)
        sample = mean + torch.exp(logs)*torch.randn(mean.shape).to(mean.device)
        # Decoder net
        out = self.activation(self.out_fc1(sample))
        out = self.activation(self.out_fc2(out))
        out = out.view(out.shape[0], self.n_hidden, CHAN, CHAN)
        # Conv
        out = self.out_resblock1(out, out3)
        out = self.upsample(out)
        out = self.out_resblock2(out, out2)
        out = self.upsample(out)
        out = self.out_resblock3(out, out1)
        out = self.upsample(out)

        out = self.out_resblock4(out)
        out = self.out_conv(out)
        if self.last_activation is not None:
            out = self.last_activation(out)

        return {
            'out'       : out,
            'prop_mean' : prop_mean,
            'prop_logs' : prop_logs,
            'mean'      : mean,
            'logs'      : logs,
        }


# if __name__ == '__main__':
#     CFG = {
#         'model': {
#             'n_hidden': 16,
#             'fc_hidden': 50,
#             'fc_out': 16,
#             'inp_channels':3,
#             'last_layer': 'tanh',
#         }
#     }
#     network = EncoderDecoder(CFG)
#     inputs = {
#         'image': torch.rand(8, 3, 64, 64),
#         'mask': torch.rand(8, 1, 64, 64),
#         'observed': torch.rand(8, 3, 64, 64),
#     }
#     outputs = network(inputs)
#     print 'inputs', inputs
#     for key, value in outputs.items():
#         print key, value.shape
