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
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import options 


os.makedirs("/mnist", exist_ok=True)
opt = options.Option().parse()

dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        drop_last=True,
        batch_size=opt.batch_size,
        shuffle=True,
    )
Tensor = torch.FloatTensor

for i, (imgs, _) in enumerate(dataloader):  
    print('iter')          
    # Configure input
    # real_imgs = Variable(imgs.type(Tensor))
    real_imgs = imgs

    save_image(real_imgs.data[:10], f"real.png", nrow=5, normalize=True)
    print('save')
    exit()