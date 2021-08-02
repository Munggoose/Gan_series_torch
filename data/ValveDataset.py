import torchvision 
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import reculsive_file_load


class ValveDataset(Dataset):
    def __init__(self,train=True, transforms = None):
        super(ValveDataset,self).__init__()
        self.transform = transforms
        if train:
            self.root = 'C:\\Users\\LMH\Desktop\data_list\Valve_seat\\train\\normal'
        self.file_paths = reculsive_file_load(self.root,'bmp')
        # self.x = []
        # self.y = []
        # for file in file_paths:
        #     self.x.append(Image.open(file))
        #     self.y.append(1)

    def __len__(self):
        return len(self.file_paths)    

    def __getitem__(self,idx):
        self.x = Image.open(self.file_paths[idx])
        self.y = 1
        if self.transform:
            return self.transform(self.x),torch.FloatTensor(self.y)
        
        self.x = np.array(self.x)
        return torch.FloatTensor(self.x), torch.FloatTensor(self.y)
        