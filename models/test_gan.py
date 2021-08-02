import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import datasets
from torchvision.utils import save_image
from torchvision.transforms import transforms
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from options import Option

opt_cl = Option()
opt = opt_cl.parse()

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
opt.device = device
# class Generator(nn.Module):
#     def __init__(self,opt):
#         super(Generator,self).__init__()
#         self.first_l = nn.Sequential(
#             nn.ConvTranspose2d(opt.latent_dim,opt.ngf*8, 4,1,0),
#             nn.BatchNorm2d(opt.ngf*8),
#             nn.ReLU(),
#             nn.ConvTranspose2d(opt.ngf*8,opt.ngf*4, 4,1,0),
#             nn.BatchNorm2d(opt.ngf*4),
#             nn.ReLU())

#         def convt_block(in_fe,out_fe,normal=True):
#             layer = [nn.ConvTranspose2d(in_fe,out_fe,4,2,1)]
#             if normal:
#                 layer.append(nn.BatchNorm2d(out_fe))
#             layer.append(nn.ReLU())
#             return layer
#         #7x7 -> 14->28
#         self.main_l = nn.Sequential(
#             *convt_block(opt.ngf*4,opt.ngf*2,opt),  #->14
#             *convt_block(opt.ngf*2,opt.ngf,opt), #->28
#         )
#         self.last_l = nn.Sequential(
#             nn.ConvTranspose2d(opt.ngf,opt.channels,3,1,1),
#             nn.Tanh()
#         )
    
#     def forward(self,x):
#         x = self.first_l(x)
#         x = self.main_l(x)
#         x = self.last_l(x)

#         return x




# class Discriminator(nn.Module):
#     def __init__(self,opt):
#         super(Discriminator,self).__init__()
#         #input image shape = bt,1,28,28
#         # self.first_layer = nn.Linear(opt.channels, opt.ndf)
#         self.first_l = nn.Conv2d(opt.channels,opt.ndf,5,1,0) #->24
        
#         def conv_block(in_fe,out_fe,normal =True): #-> //2
#             layer = [nn.Conv2d(in_fe, out_fe,4,2,1)]
#             if normal:
#                 layer.append(nn.BatchNorm2d(out_fe))
#             layer.append(nn.LeakyReLU(0.2))
#             return layer

#         self.main_l = nn.Sequential(
#             *conv_block(opt.ndf,opt.ndf*2),  #->12
#             *conv_block(opt.ndf*2,opt.ndf*4),  #->6
#             *conv_block(opt.ndf*4,opt.ndf*8)   #->3
#         )

#         self.last_l = nn.Sequential(
#             nn.Conv2d(opt.ndf*8, 1,3,1,0),
#             nn.Sigmoid()
#         )

#     def forward(self,x):
#         x = self.first_l(x)
#         x = self.main_l(x)
#         x = self.last_l(x)
#         x= x.view(-1,1)
#         # print(x)
#         # exit()
#         return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)

        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)

        
        out = out.view(out.shape[0], -1)
        
        validity = self.adv_layer(out)
        return validity

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':

    os.makedirs("images", exist_ok=True)
    dataloader = DataLoader(dataset=datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),batch_size=opt.batch_size,shuffle=True)

    # net_g = Generator(opt)
    # net_d = Discriminator(opt)
    net_g = Generator()
    net_d = Discriminator()

    net_g.apply(weights_init_normal)
    net_d.apply(weights_init_normal)

    net_g.cuda()
    net_d.cuda()

    optim_G = torch.optim.RMSprop(net_g.parameters(),lr=opt.lr)
    optim_D = torch.optim.RMSprop(net_d.parameters(),lr=opt.lr)

    batches_done = 0
    #train step
    for epoch in tqdm(range(opt.n_epochs)):
        for i, (img,_) in enumerate(dataloader):
            
            optim_D.zero_grad
            real_img = img.to(device)
            
            # z = torch.randn(img.shape[0],100,1,1).to(device)
            # z = Variable(torch.Tensor(np.random.normal(0, 1, (opt.batch_size, 100,1,1)))).cuda()
            z = torch.randn((opt.batch_size, 100)).view(-1, 100, 1, 1).cuda()
            fake_img = net_g(z)

            real_validation = net_d(real_img)
            fake_validation = net_d(fake_img)

            d_loss =  torch.mean(fake_validation)-torch.mean(real_validation)
            d_loss.backward()
            optim_D.step()

            # Clip weights of discriminator
            for p in net_d.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
            
            optim_G.zero_grad()
            if i % opt.n_critic == 0:
            # Train Generator
                fake_img = net_g(z)
                fake_validation = net_d(fake_img)
                g_loss = -torch.mean(fake_validation)
                g_loss.backward()
                optim_G.step()
            

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(fake_img.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                

