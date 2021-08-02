import argparse
import os
import numpy as np
import math
from tqdm import tqdm

import random
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import sys


parser = argparse.ArgumentParser(description='BEGAN')
parser.add_argument('--dataset', required=True, help='CelebA', default='CelebA')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=16, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=0.001')
parser.add_argument('--lr_update_step', type=float, default=100000, help='Reduce learning rate by factor of 2 every n iterations. Default=1')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--poll_step', default=1000, help="how often to poll if training has plateaued")
parser.add_argument('--patience', default=10, help="how long to wait before reducing lr")


parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--log_step', default=10, help="logging frequency")
parser.add_argument('--tb_log_step', default=100, help="tensorboard logging frequency")
parser.add_argument('--visualize_step', default=500, help="display image frequency")
parser.add_argument('--checkpoint_step', default=50000, help="checkpoint frequency")


parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--h', type=int, default=64, help="h value ( size of noise vector )")
parser.add_argument('--n', type=int, default=128, help="n value")
parser.add_argument('--lambda_k', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.5)
opt = parser.parse_args()


    


