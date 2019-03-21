'''
losses.py: Contains loss functions
Author: Rohit Jena
'''
import torch
import torch.nn as nn
from torch.nn import functional as F

def rgb2y(image):
    '''
    Take input as RGB image batch and return batch of Y images
    '''
    img = image[:, 0:1]*0.299 + image[:, 1:2]*0.587 + image[:, 2:3]*0.114
    return img


class MSERecon(nn.Module):
    '''
    Loss function with Gaussian log-likelihood
    '''
    def forward(self, outputs, inputs, cfg, val=False):
        # There will be 3 terms,
        # first term: reconstruction loss
        # second term: KL divergence
        # third term: regularizer (optional)
        channels = cfg['model']['inp_channels']

        mask = (1 - inputs['mask'])
        recon_loss = mask*((outputs['out'][:, :channels] - inputs['image'])**2)
        recon_loss = (recon_loss/mask.mean() + ((outputs['out'][:, channels:] - inputs['image'])**2))
        recon_loss = recon_loss.mean()/2.0
        # 2nd term
        mu1, logs1 = outputs['prop_mean'], outputs['prop_logs']
        mu2, logs2 = outputs['mean'], outputs['logs']
        sigma1 = torch.exp(logs1)
        sigma2 = torch.exp(logs2)
        # calculate kl div
        kl_div = (logs2 - logs1) + 0.5*(sigma1**2 + (mu1 - mu2)**2)/(sigma2**2) - 0.5
        kl_div = kl_div.mean()
        # 3rd optional term
        loss_reg = 0
        cfg_reg = cfg['reg']
        if cfg_reg['apply_reg']:
            loss_reg = mu2**2 / (2 * cfg_reg['sigma_m']**2) - cfg_reg['sigma_s']*(logs2 - sigma2)
            loss_reg = loss_reg.mean()

        # If not validation, use the 3-termed loss,
        # else, just use the MSE between generated and ground truth
        # print(float(recon_loss), float(kl_div), float(loss_reg))
        if not val:
            loss_val = recon_loss + cfg_reg['lambda_kl']*kl_div + cfg_reg['lambda_reg']*loss_reg
        else:
            # loss_val = ((outputs['out'][:, channels:] - inputs['image'])**2).mean()
            # Get PSNR here
            imageY = inputs['image']
            outY = outputs['out'][:, channels:]

            loss_val = ((outY - imageY)**2).mean(1).mean(1).mean(1)
            loss_val = 10*torch.log10(4.0/loss_val)
            loss_val = loss_val.mean()

        return loss_val


class BCERecon(nn.Module):
    '''
    Similar to MSERecon but with Bernoulli loglikelihood
    '''
    def forward(self, outputs, inputs, cfg=None, val=False, eps=1e-10):
        # There will be 3 terms,
        # first term: reconstruction loss
        # second term: KL divergence
        # third term: regularizer (optional)
        channels = cfg['model']['inp_channels']
        mask = 1 - inputs['mask']
        # Get recon loss
        image = (inputs['image'] + 1.0)/2
        image[image > 0.5] = 1
        image[image <= 0.5] = 0
        # print(image.shape)
        # print(outputs['out'].shape)

        # First part is for x_b, second is for y
        # First part (unobserved) will consist of first 'C' channels
        # Second part (entire image) will consist of next 'C' channels
        # print(outputs['out'][:, :channels].max(), outputs['out'][:, :channels].min())
        # print(outputs['out'][:, channels:].max(), outputs['out'][:, channels:].min())

        recon_loss = F.binary_cross_entropy(outputs['out'][:, :channels] + eps, image, weight=mask)
        # recon_loss = mask*(-image*torch.log(eps + outputs['out'][:, :channels]) \
        #              -(1 - image)*torch.log(1 + eps - outputs['out'][:, :channels]))
        # recon_loss = recon_loss + (-image*torch.log(eps + outputs['out'][:, channels:])\
        #              -(1 - image)*torch.log(1 + eps - outputs['out'][:, channels:]))
        recon_loss = recon_loss.mean()
        # 2nd term
        mu1, logs1 = outputs['prop_mean'], outputs['prop_logs']
        mu2, logs2 = outputs['mean'], outputs['logs']
        sigma1 = torch.exp(logs1)
        sigma2 = torch.exp(logs2)
        # calculate kl div
        kl_div = (logs2 - logs1) + 0.5*(sigma1**2 + (mu1 - mu2)**2)/(sigma2**2) - 0.5
        kl_div = kl_div.mean()
        # 3rd optional term
        loss_reg = 0
        cfg_reg = cfg['reg']
        if cfg_reg['apply_reg']:
            loss_reg = mu2**2 / (2 * cfg_reg['sigma_m']**2) - cfg_reg['sigma_s']*(logs2 - sigma2)
            loss_reg = loss_reg.mean()

        # print(float(recon_loss), float(kl_div), float(loss_reg))
        if not val:
            loss_val = recon_loss + cfg_reg['lambda_kl']*kl_div + cfg_reg['lambda_reg']*loss_reg
        else:
            loss_val = recon_loss

        return loss_val
