'''
losses.py: Contains loss functions
Author: Rohit Jena
'''
import torch
import torch.nn as nn

class MSERecon(nn.Module):
    '''
    Loss function with Gaussian log-likelihood
    '''
    def forward(self, outputs, inputs, cfg):
        # There will be 3 terms,
        # first term: reconstruction loss
        # second term: KL divergence
        # third term: regularizer (optional)
        channels = cfg['model']['inp_channels']

        mask = 1 - inputs['mask']
        recon_loss = mask*((outputs['out'][:, :channels] - inputs['image'])**2)
        recon_loss = (recon_loss + ((outputs['out'][:, channels:] - inputs['image'])**2))
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

        # print(float(recon_loss), float(kl_div), float(loss_reg))
        loss_val = recon_loss + cfg_reg['lambda_kl']*kl_div + cfg_reg['lambda_reg']*loss_reg
        return loss_val


class BCERecon(nn.Module):
    '''
    Similar to MSERecon but with Bernoulli loglikelihood
    '''
    def forward(self, outputs, inputs, cfg=None):
        # There will be 3 terms,
        # first term: reconstruction loss
        # second term: KL divergence
        # third term: regularizer (optional)
        channels = cfg['model']['inp_channels']
        mask = 1 - inputs['mask']
        # Get recon loss
        image = (inputs['image'] + 1.0)/2
        # print(image.shape)
        # print(outputs['out'].shape)

        # First part is for x_b, second is for y
        recon_loss = mask*( -image*torch.log(outputs['out'][:, :channels]) \
                     -(1 - image)*torch.log(1 - outputs['out'][:, :channels]))
        recon_loss = recon_loss + (-image*torch.log(outputs['out'][:, channels:])\
                     -(1 - image)*torch.log(1 - outputs['out'][:, channels:]))
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

        # print(float(recon_loss), float(kl_div), float(loss_reg))
        loss_val = recon_loss + cfg_reg['lambda_kl']*kl_div + cfg_reg['lambda_reg']*loss_reg
        return loss_val
