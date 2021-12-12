'''
Main program for the repository
Author: Rohit Jena
'''
import argparse
import os
import yaml

import torch

from utils import utils
from trainer import train, validate

def get_args():
    '''
    Get arguments for running the main code
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, \
        help='Mode (training v/s validation)', choices=['train', 'val'])
    parser.add_argument('--config', type=str, required=True, \
        help='Config file to read from.')
    return parser


if __name__ == '__main__':

    args = get_args().parse_args()
    if not os.path.exists(args.config):
        print('Config file {} does not exist.'.format(args.config))

    with open(args.config, 'r') as fi:
        CONFIG = yaml.safe_load(fi.read())
        CONFIG = utils.convert_to_lower(CONFIG)

    if args.mode == 'train':
        train(CONFIG)
    else:
        with torch.no_grad():
            validate(CONFIG)
