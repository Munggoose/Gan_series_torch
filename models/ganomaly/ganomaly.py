import os
import time
import numpy as np
from tqdm import tqdm
import random
import argparse

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from network import NetG, NetD, weights_init
from collections import OrderedDict
from custom_loader import load_data
from evaluate import evaluate

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
# from options import Option
from utils.visualizer import Visualizer

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--n_epochs",type=int,default=1, help = 'number of epochs of training')
        parser.add_argument("--batch_size",type=int, default=64, help = 'size of the batches')
        parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
        parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
        parser.add_argument("--device",type=str, default='cuda',help='device cuda or cpu')
        parser.add_argument("--ngf",type=int, default=64,help='number of generator featmap')
        parser.add_argument("--ndf",type=int, default=64,help='number of discriminator featmap')
        parser.add_argument("--channels", type=int, default=1, help="number of image channels")
        parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
        parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
        parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
        parser.add_argument("--isTrain",type=bool, default=True, help= 'check is trainning mode')

        #Ganomaly option
        parser.add_argument("--w_adv",type=float,default=1, help = 'weight of discrminator loss')
        parser.add_argument("--w_con",type=float,default=1, help = 'weight of image loss')
        parser.add_argument("--w_enc",type=float,default=1, help = 'weight of feature loss')
        parser.add_argument('--abnormal_idx',type = int, default = 0,help= 'Abnomral')
        parser.add_argument('--dataroot',type=str, default='',help = 'dataset root folder')
        parser.add_argument('--dataset',type=str, default='mnist',help='dataset folder name')
        parser.add_argument('--manual_seed',type=int, default =-1, help = 'seed number')
        parser.add_argument('--ngpu',type=int, default=1, help ='number or gpus')
        parser.add_argument('--extralayers',type=int, default=0, help='add extralayer')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_image_freq', type=int, default=100, help='frequency of saving real and fake images')
        parser.add_argument('--metric',type=str,default='roc',help='Evaluation metric.')
        
        self.opt =parser.parse_args()
    
    def parse(self):
        """ Parse Arguments.
        """
        args = vars(self.opt)
        return self.opt



class Ganomaly():
    
    def __init__(self, opt):
        self.seed(opt.manual_seed)

        #Initalize variable
        self.opt = opt
        self.device = torch.device("cuda" if self.opt.device != 'cpu' else "cpu")

        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        #criterion
        self.l_adv = nn.MSELoss() #L2 loss
        self.l_con = nn.L1Loss()
        self.l_enc = nn.MSELoss() #L2 loss
        self.l_bce = nn.BCELoss()

        self.input = torch.empty(size=(self.opt.batch_size,self.opt.channels,self.opt.img_size,self.opt.img_size),device=self.device)
        self.label = torch.empty(size=(self.opt.batch_size,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(self.opt.batch_size,),dtype=torch.long, device=self.device)

        self.fixed_input = torch.empty(size=(self.opt.batch_size,self.opt.channels,self.opt.img_size,self.opt.img_size),dtype=torch.float32, device=self.device)
        self.real_label = torch.ones(size=(self.opt.batch_size,),dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batch_size,),dtype=torch.float32, device=self.device)

        with torch.no_grad():
            self.an_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=opt.device)
            self.gt_labels = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long,    device=opt.device)
            self.latent_i  = torch.zeros(size=(len(dataloader.dataset), opt.latent_dim), dtype=torch.float32, device=opt.device)
            self.latent_o  = torch.zeros(size=(len(dataloader.dataset), opt.latent_dim), dtype=torch.float32, device=opt.device)

        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr = self.opt.lr, betas=(0.5, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr = self.opt.lr, betas=(0.5, 0.999))


    def set_input(self, input:torch.Tensor):
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # if self.total_steps == self.opt.batch_size:
            #     self.fixed_input.resize_(input[0].size()).copy_(input[0])

    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    def get_current_images(self):
        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    def save_weight(self,epoch):
        weight_dir = './weight/ganomaly/'
        
        if not os.exists(weight_dir):
            os.mkdirs(weight_dir)

        torch.save({'epoch':epoch + 1,'state_dict':self.netg.state_dict()},f'{weight_dir}netG.pth')
        torch.save({'epoch':epoch + 1,'state_dict':self.netD.state_dict()},f'{weight_dir}netD.pth')
    
    
    #===========model train=========
    def forward(self):
        """[summary]
        Model Forward propagation
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    def backward_g(self):
        """[summary]
        Generator backpropagation
        """
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)

    def backward_d(self):

        """[summary]
        Discriminator backpropagation
        """
    
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()


def train(opt, model ,dataloader):
    total_steps = 0
    for epoch in range(opt.n_epochs):
        for data  in tqdm(dataloader,bar_format='{desc:<0.5}{percentage:1.0f}%|{bar:60}{r_bar}'):
            total_steps += opt.batch_size
            model.set_input(data)

            model.forward()

            #netG
            model.optimizer_g.zero_grad()
            model.backward_g()
            model.optimizer_g.step()

            #netD
            model.optimizer_d.zero_grad()
            model.backward_d()
            model.optimizer_d.step()

            # netD weight_re_init
            if model.err_d.item() < 1e-5 : 
                model.netd.apply(weights_init)


            if total_steps % opt.print_freq == 0:
                errors = OrderedDict([
                    ('err_d',model.err_d.item()),
                    ('err_g',model.err_g.item()),
                    ('err_g_adv',model.err_g_adv.item()),
                    ('err_g_con',model.err_g_con.item()),
                    ('err_g_enc',model.err_g_enc.item())
                ])

                print(f"  err_d:{errors['err_g']:.4f} err_g:{errors['err_d']:.4f}",end='')
                # if self.opt.display:
                #     counter_ratio = float(epoch_iter) / len(self.dataloader.dataset)
                #     self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if total_steps % opt.save_image_freq == 0:
                reals, fakes, fixed = model.get_current_images()
            
            
            
                # self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                # if self.opt.display:
                #     self.visualizer.display_current_images(reals, fakes, fixed)

        # print(f">> Training model ganomaly. Epoch {epoch}/{opt.n_epochs}")

        # self.visualizer.print_current_errors(self.epoch, errors)

def test(opt, model, dataloader,path):

    # with torch.no_grad():
    #     pretrained_dict = torch.load(path)['state_dict']
    #     try:
    #         model.netg.load_state_dict(pretrained_dict)

    #     except IOError:
    #         raise IOError("netG weights not found")
    #     print('   Loaded weights.')

        # opt.phase = 'test'
    times = []
    total_steps = 0
    epoch_iter = 0
    
    for i, data in enumerate(dataloader, 0):
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        time_i = time.time()
        model.set_input(data)
        model.forward()
        model.fake, latent_i, latent_o = model.netg(model.input)

        error = torch.mean(torch.pow((model.latent_i-model.latent_o), 2), dim=1)
        time_o = time.time()
        with torch.no_grad():
            model.an_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=opt.device)
            model.gt_labels = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long,    device=opt.device)
            model.latent_i  = torch.zeros(size=(len(dataloader.dataset), opt.latent_dim), dtype=torch.float32, device=opt.device)
            model.latent_o  = torch.zeros(size=(len(dataloader.dataset), opt.latent_dim), dtype=torch.float32, device=opt.device)


        model.an_scores[i*opt.batch_size : i*opt.batch_size+error.size(0)] = error.reshape(error.size(0))
        model.gt_labels[i*opt.batch_size : i*opt.batch_size+error.size(0)] = model.gt.reshape(error.size(0))
        # print(latent_i.reshape(error.size(0), opt.latent_dim).shape)
        # print(model.latent_i [i*opt.batch_size : i*opt.batch_size+error.size(0), :].shape)
        # exit()
        model.latent_i [i*opt.batch_size : i*opt.batch_size+error.size(0), :] = latent_i.reshape(error.size(0), opt.latent_dim)
        model.latent_o [i*opt.batch_size : i*opt.batch_size+error.size(0), :] = latent_o.reshape(error.size(0), opt.latent_dim)

        times.append(time_o - time_i)

        # Save test images.
        # if opt.save_test_images:
        #     dst = os.path.join(opt.outf, opt.name, 'test', 'images')
        #     if not os.path.isdir(dst):
        #         os.makedirs(dst)
        #     real, fake, _ = model.get_current_images()
            # vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
            # vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

    # Measure inference time.
    times = np.array(times)
    times = np.mean(times[:100] * 1000)

    # Scale error vector between [0, 1]
    model.an_scores = (model.an_scores - torch.min(model.an_scores)) / (torch.max(model.an_scores) - torch.min(model.an_scores))
    # auc, eer = roc(self.gt_labels, self.an_scores)
    auc = evaluate(model.gt_labels, model.an_scores, metric=opt.metric)
    performance = OrderedDict([('Avg Run Time (ms/batch)', times), (opt.metric, auc)])

    return performance


if __name__ =='__main__':
    opt = Option().parse()
    dataloader = load_data(opt)
    print('load finish')
    model =  Ganomaly(opt)
    print('__train__start____')
    train(opt, model, dataloader)
    result=test(opt,model,dataloader,'./')
