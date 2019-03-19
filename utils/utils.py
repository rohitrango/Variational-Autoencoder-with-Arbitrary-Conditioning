'''
utils.py: utility functions for loading/saving checkpoints,
get optimizers, schedulers, etc
Author: Rohit Jena
'''
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim import lr_scheduler
from skimage import io
import numpy as np

from datasets.celeba import CelebA
from datasets.mnist import MNISTRowDeleted

from networks import mnist_vaeac, celeba_vaeac
from losses import MSERecon, BCERecon

# Dictionary of loss functions
LOSS_DICT = {
    'mse': MSERecon(),
    'bce': BCERecon(),
}

def convert_to_lower(cfg):
    '''
    Convert all key values in config file to lower case
    Only exceptions are path files
    '''
    if not isinstance(cfg, dict):
        if hasattr(cfg, 'lower'):
            cfg = cfg.lower()
        return cfg

    new_cfg = dict()
    for key, val in cfg.items():
        if hasattr(key, 'lower'):
            key = key.lower()
        if ('path' in key) or (key == 'pretrained'):
            new_cfg[key] = val
        else:
            new_cfg[key] = convert_to_lower(val)

    return new_cfg


def get_model(cfg):
    '''
    Get the model from name
    '''
    if cfg['model']['name'] == 'EncoderDecoderNetMini'.lower():
        model = mnist_vaeac.EncoderDecoderNetMini(cfg)
    elif cfg['model']['name'] == 'EncoderDecoderNet'.lower():
        model = celeba_vaeac.EncoderDecoder(cfg)
    else:
        raise NotImplementedError
    return model


def get_data_loaders(cfg):
    '''
    Get the train and test datasets from config dict
    '''
    name = cfg['dataset']['name']
    if name == 'mnist':
        train_dataset = MNISTRowDeleted('train', cfg)
        val_dataset = MNISTRowDeleted('test', cfg)

    elif name == 'celeba':
        train_dataset = CelebA('train', cfg)
        val_dataset = CelebA('val', cfg)
    else:
        raise NotImplementedError

    # Generate train and test loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch-size'] \
        , shuffle=cfg['train']['shuffle'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['val']['batch-size'] \
        , shuffle=cfg['val']['shuffle'])
    return train_dataset, val_dataset, train_loader, val_loader


def load_ckpt(model, optim, cfg):
    '''
    Load the checkpoints for generator and discriminator (and optimizers) from config
    Also load the checkpoint and random seed for reproducibility
    '''
    data = None
    if cfg['pretrained'] is not None:
        data = torch.load(cfg['pretrained'])
        model.load_state_dict(data['model'])
        optim.load_state_dict(data['optim'])
        del data['model'], data['optim']

    return model, optim, data


def save_ckpt(model_and_optim, cfg, ckpt, seed, override=False):
    '''
    Save the checkpoint which is described by `ckpt`, and parameters, and also the seed used
    for reproducibility
    '''
    model, optim = model_and_optim
    if override or (ckpt) % cfg['train']['save-freq'] == 0:
        state_dict = dict()
        state_dict['seed'] = seed
        state_dict['ckpt'] = ckpt
        state_dict['model'] = model.state_dict()
        state_dict['optim'] = optim.state_dict()

        if not os.path.exists(cfg['save-path']):
            os.makedirs(cfg['save-path'])

        path = os.path.join(cfg['save-path'], 'model_checkpoint_{}.pt'.format(ckpt))
        torch.save(state_dict, path)
        print 'Saved checkpoint at {}\n'.format(path)


def get_losses(cfg):
    '''
    Get loss functions from the loss dictionary
    '''
    loss = cfg['model']['loss']
    loss = LOSS_DICT[loss]

    return loss


def get_optimizers(model, cfg):
    '''
    Get optimizers for generator and discriminator
    '''
    subcfg = cfg['model']
    # Get the optimizer with all parameters for discriminator
    if subcfg['optimizer'] == 'adam':
        optim = Adam(model.parameters(), lr=subcfg['lr'], \
            betas=(subcfg['beta1'], subcfg['beta2']), weight_decay=subcfg['weight-decay'])
    elif subcfg['optimizer'] == 'sgd':
        optim = SGD(model.parameters(), lr=subcfg['lr'], \
            weight_decay=subcfg['weight-decay'])
    else:
        raise NotImplementedError
    return optim

def get_schedulers(optim, cfg, ckpt):
    '''
    Get scheduler for the optimizers
    '''
    subcfg = cfg['model']

    if subcfg['scheduler'] == 'step':
        scheduler = lr_scheduler.StepLR(optim, subcfg['decay-steps'], \
            gamma=subcfg['decay-factor'], last_epoch=ckpt-1)
    else:
        raise NotImplementedError

    # override lr here if specified
    if cfg['model'].get('lr-override', False):
        for param_group in optim.param_groups:
            param_group['lr'] = cfg['model']['lr']

    return scheduler


def init_weights(model, cfg):
    '''
    Initialize weights using the initializations given
    The paper uses orthogonal init.
    '''
    def init_helper(mod, subcfg):
        '''
        Module agnostic helper function for initializing
        '''
        init = subcfg['init']
        if init == 'orthogonal':
            initializer_ = nn.init.orthogonal
        else:
            raise NotImplementedError

        # Update params
        for param in list(mod.parameters()):
            if isinstance(param, (nn.Conv2d, nn.Linear)):
                initializer_(param.weight)

    # Get sub configs and update
    subcfg = cfg['model']
    # Get init
    init_helper(model, subcfg)
    return model


def re_normalize(image, cfg):
    '''
    Renormalize to [0, 1]
    useful for plotting and saving
    '''
    min_val = cfg['dataset']['min']*1.0
    max_val = cfg['dataset']['max']*1.0
    return (image - min_val)/(max_val - min_val)


def save_images(data, outs, cfg, save_index, suffix=''):
    '''
    Save black-and-white or colored images
    '''
    if cfg['model']['inp_channels'] == 1:
        io.imsave('{}/{}_img_{}.png'.format(cfg['save-path'], save_index, suffix), \
            data['image'][0, 0].data.cpu().numpy())
        io.imsave('{}/{}_obs_{}.png'.format(cfg['save-path'], save_index, suffix), \
            data['observed'][0, 0].data.cpu().numpy())
        io.imsave('{}/{}_out_{}.png'.format(cfg['save-path'], save_index, suffix), \
            outs['out'][0, 1].data.cpu().numpy())
        print "Saved for ckpt: {} {}".format(save_index, suffix)

    elif cfg['model']['inp_channels'] == 3:
        io.imsave('{}/{}_img_{}.png'.format(cfg['save-path'], save_index, suffix), \
            re_normalize(data['image'][0].data.cpu().numpy().transpose(1, 2, 0), cfg))
        io.imsave('{}/{}_obs_{}.png'.format(cfg['save-path'], save_index, suffix), \
            re_normalize(data['observed'][0].data.cpu().numpy().transpose(1, 2, 0), cfg))
        io.imsave('{}/{}_out_{}.png'.format(cfg['save-path'], save_index, suffix), \
            re_normalize(outs['out'][0].data.cpu().numpy().transpose(1, 2, 0)[:, :, :3], cfg))
        print "Saved for ckpt: {} {}".format(save_index, suffix)

    else:
        raise NotImplementedError


def save_val_images(data, outs, cfg, save_index, suffix='val'):
    '''
    Save images from validation for results
    The image is the same repeated N times for N samples
    '''
    N, C, H, W = outs['out'].shape
    if cfg['model']['inp_channels'] == 1:
        out_img = np.zeros((H, W*(N + 2)))
        out_img[:, :W] = data['observed'][0, 0].data.cpu().numpy()
        for i in range(N):
            out_img[:, (i+1)*W:(i+2)*W] = outs['out'][i, 1].data.cpu().numpy()
        out_img[:, -W:] = data['image'][0, 0].data.cpu().numpy()
        # Save it
        io.imsave('{}/{}_val_out_{}.png'.format(cfg['save-path'], save_index, suffix),\
            out_img)
        print "Saved for ckpt: {} {}".format(save_index, suffix)
    # Treat colored images a little differently
    elif cfg['model']['inp_channels'] == 3:
        out_img = np.zeros((H, W*(N + 2), 3))
        out_img[:, :W] = data['observed'][0].data.cpu().numpy().transpose(1, 2, 0)
        for i in range(N):
            out_img[:, (i+1)*W:(i+2)*W] = outs['out'][i, :3].data.cpu().numpy().transpose(1, 2, 0)
        out_img[:, -W:] = data['image'][0].data.cpu().numpy().transpose(1, 2, 0)
        # Save it
        io.imsave('{}/{}_val_out_{}.png'.format(cfg['save-path'], save_index, suffix),\
            (out_img + 1.0)/2.0)
        print "Saved for ckpt: {} {}".format(save_index, suffix)
    else:
        raise NotImplementedError


def repeat_data(data, cfg):
    '''
    Repeat data along the batch axis to get more samples
    '''
    samples = cfg['val']['num-samples']
    for key, value in data.items():
        data[key] = data[key].repeat(samples, 1, 1, 1)
    return data
