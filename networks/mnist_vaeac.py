'''
Network for MNIST
Author: Rohit Jena
'''
import torch
import torch.nn.functional as F
from torch import nn
from networks import parts

class ProposalNetMini(nn.Module):
    '''
    This network is the prior net for MNIST
    '''
    def __init__(self,
                 inp_channels=3,
                 n_hidden=16,
                 fc_hidden=50,
                 fc_out=16,
                 activation=F.leaky_relu,
                ):
        super(ProposalNetMini, self).__init__()
        self.inp_channels = inp_channels
        self.activation = activation
        # Get modules
        self.in_conv = parts.InConv(inp_channels, n_hidden)
        self.resblock1 = parts.ResBlock(n_hidden, n_hidden) # 14 * 14
        self.resblock2 = parts.ResBlock(n_hidden, n_hidden) # 7 * 7
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        # fc layers
        self.fc1 = nn.Linear(n_hidden * 7 * 7, fc_hidden)
        self.fc_mean = nn.Linear(fc_hidden, fc_out)
        self.fc_sigma = nn.Linear(fc_hidden, fc_out)

    def forward(self, image):
        outx = torch.cat([image['image'], image['mask']], 1)
        out = outx
        out = self.in_conv(out)
        out = self.resblock1(out)
        out = self.pool1(out)
        out = self.resblock2(out)
        out = self.pool2(out)
        out = out.view(out.shape[0], -1)
        # fully connected parts
        out = self.activation(self.fc1(out))
        mean = self.fc_mean(out)
        logs = self.fc_sigma(out)
        return mean, logs


class EncoderDecoderNetMini(nn.Module):
    '''
    Prior and generator network combined into one network
    '''
    def __init__(self,
                 cfg,
                 activation=F.relu,
                ):
        super(EncoderDecoderNetMini, self).__init__()
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
        self.proposal_net = ProposalNetMini(
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
        self.resblock1 = parts.ResBlock(n_hidden, n_hidden) # 14 * 14
        self.resblock2 = parts.ResBlock(n_hidden, n_hidden) # 7 * 7
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        # fc layers
        self.fc1 = nn.Linear(n_hidden * 7 * 7, fc_hidden)
        self.fc_mean = nn.Linear(fc_hidden, fc_out)
        self.fc_sigma = nn.Linear(fc_hidden, fc_out)

        # Decoder network
        self.out_fc1 = nn.Linear(fc_out, fc_hidden)
        self.out_fc2 = nn.Linear(fc_hidden, n_hidden * 7 * 7)
        # resblocks
        self.out_resblock1 = parts.ResBlock(2 * n_hidden, n_hidden)
        self.out_resblock2 = parts.ResBlock(2 * n_hidden, n_hidden)
        self.upsample = nn.Upsample(scale_factor=2)
        self.out_resblock3 = parts.ResBlock(n_hidden, n_hidden)
        # Final output (will have inp_channels)
        self.out_conv = nn.Conv2d(n_hidden, 2*inp_channels, 3, padding=1)


    def forward(self, image):
        # Proposal net outputs
        # proposal net will take care of what inputs to take
        prop_mean, prop_logs = self.proposal_net(image)
        # Prior net
        # concatenate the observed image and mask together
        outx = torch.cat([image['observed'], image['mask']], 1)
        out = outx
        out = self.in_conv(out)
        out = self.resblock1(out)           # 28
        out = self.pool1(out)            # 14
        out1 = out
        out = self.resblock2(out)           # 14
        out = self.pool2(out)            # 7
        out2 = out
        out = out.view(out.shape[0], -1)
        # fully connected parts
        out = self.activation(self.fc1(out))
        mean = self.fc_mean(out)
        logs = self.fc_sigma(out)

        if not self.training:
            sample = mean + F.softplus(logs)*torch.randn(mean.shape).to(mean.device)
        else:
            sample = prop_mean + F.softplus(prop_logs)*torch.randn(mean.shape).to(mean.device)

        # Decoder net
        out = self.activation(self.out_fc1(sample))
        out = self.activation(self.out_fc2(out))
        out = out.view(out.shape[0], self.n_hidden, 7, 7)
        # Conv
        out = self.out_resblock1(out, out2)
        out = self.upsample(out)
        out = self.out_resblock2(out, out1)
        out = self.upsample(out)
        out = self.out_resblock3(out)
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
