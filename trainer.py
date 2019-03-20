'''
trainer.py : Contains train and validation loops
Author: Rohit Jena
'''
import os
import json

import numpy as np
import torch
from torch.autograd import Variable

from utils import utils

def validate(cfg):
    '''
    Main loop for validation, load the dataset, model, and
    other things. Run validation on the validation set
    '''
    print json.dumps(cfg, sort_keys=True, indent=4)

    use_cuda = cfg['use-cuda']
    _, _, _, val_dl = utils.get_data_loaders(cfg)

    model = utils.get_model(cfg)
    if use_cuda:
        model = model.cuda()
    model = utils.init_weights(model, cfg)

    # Get pretrained models, optimizers and loss functions
    optim = utils.get_optimizers(model, cfg)
    model, _, metadata = utils.load_ckpt(model, optim, cfg)
    loss_fn = utils.get_losses(cfg)

    # Set up random seeds
    ckpt = 0
    if metadata is not None:
        seed = metadata['seed']
        ckpt = metadata['ckpt']
    # Validation code, reproducibility is required
    seed = 42

    # Random seed according to what the saved model is
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Run test loop
    losses_list = []
    # Run the main training loop
    for idx, data in enumerate(val_dl):
        # Change to required device
        for key, value in data.items():
            data[key] = Variable(value)
            if use_cuda:
                data[key] = data[key].cuda()

        data = utils.repeat_data(data, cfg)

        # Get all outputs
        outputs = model(data)
        loss_val = loss_fn(outputs, data, cfg, val=True)
        losses_list.append(float(loss_val))
        # print it
        print('Step: {}, val_loss: {}'.format(
            idx, loss_val.data.cpu().numpy()
        ))

        if cfg['val']['save-img']:
            print outputs['out'].shape
            utils.save_val_images(data, outputs, cfg, idx)

    print("""
        Summary:
        Mean:   {},
        Std:    {},
        25per:  {},
        50per:  {},
        75per:  {},
    """.format(
        np.mean(losses_list),
        np.std(losses_list),
        np.percentile(losses_list, 25),
        np.percentile(losses_list, 50),
        np.percentile(losses_list, 75),
    ))

    print("PSNR: {}".format(10*np.log10(4/np.mean(losses_list))))



def train(cfg):
    '''
    This is the main loop for training
    Loads the dataset, model, and other things
    '''
    print json.dumps(cfg, sort_keys=True, indent=4)

    use_cuda = cfg['use-cuda']

    _, _, train_dl, val_dl = utils.get_data_loaders(cfg)

    model = utils.get_model(cfg)
    if use_cuda:
        model = model.cuda()
    model = utils.init_weights(model, cfg)

    # Get pretrained models, optimizers and loss functions
    optim = utils.get_optimizers(model, cfg)
    model, optim, metadata = utils.load_ckpt(model, optim, cfg)
    loss_fn = utils.get_losses(cfg)

    # Set up random seeds
    seed = np.random.randint(2**32)
    ckpt = 0
    if metadata is not None:
        seed = metadata['seed']
        ckpt = metadata['ckpt']

    # Get schedulers after getting checkpoints
    scheduler = utils.get_schedulers(optim, cfg, ckpt)
    # Print optimizer state
    print optim

    # Get loss file handle to dump logs to
    if not os.path.exists(cfg['save-path']):
        os.makedirs(cfg['save-path'])
    lossesfile = open(os.path.join(cfg['save-path'], 'losses.txt'), 'a+')

    # Random seed according to what the saved model is
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Run training loop
    num_epochs = cfg['train']['num-epochs']
    for epoch in range(num_epochs):
        # Run the main training loop
        model.train()
        for data in train_dl:
            # zero out the grads
            optim.zero_grad()

            # Change to required device
            for key, value in data.items():
                data[key] = Variable(value)
                if use_cuda:
                    data[key] = data[key].cuda()

            # Get all outputs
            outputs = model(data)
            loss_val = loss_fn(outputs, data, cfg)

            # print it
            print('Epoch: {}, step: {}, loss: {}'.format(
                epoch, ckpt, loss_val.data.cpu().numpy()
            ))

            # Log into the file after some epochs
            if ckpt % cfg['train']['step-log'] == 0:
                lossesfile.write('Epoch: {}, step: {}, loss: {}\n'.format(
                    epoch, ckpt, loss_val.data.cpu().numpy()
                ))

            # Backward
            loss_val.backward()
            optim.step()

            # Update schedulers
            scheduler.step()

            # Peek into the validation set
            ckpt += 1
            if ckpt % cfg['peek-validation'] == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_dl:
                        # Change to required device
                        for key, value in val_data.items():
                            val_data[key] = Variable(value)
                            if use_cuda:
                                val_data[key] = val_data[key].cuda()

                        # Get all outputs
                        outputs = model(val_data)
                        loss_val = loss_fn(outputs, val_data, cfg)

                        print 'Validation loss: {}'.format(loss_val.data.cpu().numpy())

                        lossesfile.write('Validation loss: {}\n'.format(\
                            loss_val.data.cpu().numpy()))
                        utils.save_images(val_data, outputs, cfg, ckpt)
                        break
                model.train()
            # Save checkpoint
            utils.save_ckpt((model, optim), cfg, ckpt, seed)

    lossesfile.close()
