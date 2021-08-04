import torch
import os
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def load_data(opt):

    if opt.dataroot =='':
        opt.dataroot = f'../../data/{opt.dataset}'
    
    if opt.dataset in ['mnist']:
        opt.dataroot = f'../../data/{opt.dataset}'
        opt.abnormal_idx = int(opt.abnormal_idx)

        transform = transforms.Compose(
            [
                transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset = MNIST(root= opt.dataroot , train=opt.isTrain, download=True, transform=transform)
        tmp_data = dataset.data.numpy()
        tmp_label = dataset.targets.numpy()

        normal_idx = np.where(tmp_data != opt.abnormal_idx)[0]
        abnormal_idx = np.where(tmp_data == opt.abnormal_idx)[0]
        
        # normal_data = tmp_data[normal_idx]
        # abnormal_data = tmp_data[abnormal_idx]
        # normal_label = tmp_label[normal_idx]
        # abnormal_label = tmp_label[abnormal_idx]

        #normal class: 0, abnormal class: 1
        tmp_label[normal_idx] = 0
        tmp_label[abnormal_idx] = 1
        dataset.targets = torch.from_numpy(tmp_label)
        

        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=opt.batch_size,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    drop_last=True,
                                                    worker_init_fn=(None if opt.manual_seed == -1
                                                    else lambda x: np.random.seed(opt.manual_seed)))

        return dataloader