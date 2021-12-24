import os, sys
import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_De import faces_data, High_Data, Low_Data
from model_De import High2Low, Discriminator
from model_utils import GEN_DEEP
from dataset import get_loader
from tensorboardX import SummaryWriter
import tensorlayer as tl

model_file = './your path/'
tl.files.exists_or_mkdir(model_file)

writer = SummaryWriter('./your path/')

import argparse
import time
import math


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_num = 2020
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    max_epoch = 30000
    learn_rate = 1e-4
    alpha, beta = 1, 0.005
    

    g_h2l = High2Low()
    G_h2l = High2Low().to(device)
    
    D_h2l = Discriminator().to(device)

    mse = nn.MSELoss()

    optim_D_h2l = optim.Adam(filter(lambda p: p.requires_grad, D_h2l.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_h2l = optim.Adam(G_h2l.parameters(), lr=learn_rate, betas=(0.0, 0.9))

    #img_h, img_w = 128, 256
    data = faces_data(High_Data, Low_Data)
    batch_size =25
    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    for ep in range(1, max_epoch+1):
        G_h2l.train()
        D_h2l.train()
        loss_g_t, loss_d_t = 0 , 0
        for i, batch in enumerate(loader):
            n_step = math.floor(100/batch_size)
            step_time = time.time()
            optim_D_h2l.zero_grad()
            optim_G_h2l.zero_grad()

            zs = batch["z"].to(device)
            lrs = batch["lr"].to(device)
            hrs = batch["hr"].to(device)
            downs = batch["hr_down"].to(device)

            lr_gen = G_h2l(hrs)
            lr_gen_detach = lr_gen.detach()
            loss_D_h2l = nn.ReLU()(1.0 - D_h2l(lrs)).mean() + nn.ReLU()(1 + D_h2l(lr_gen_detach)).mean()
            
            loss_D_h2l.backward()
            
            optim_D_h2l.step()
            

            # update generator
            optim_D_h2l.zero_grad()
            gan_loss_h2l = -D_h2l(lr_gen).mean()
            mse_loss_h2l = mse(lr_gen, downs)

            loss_G_h2l = alpha * mse_loss_h2l + beta * gan_loss_h2l
            loss_G_h2l.backward()
            optim_G_h2l.step()

            loss_g_t += loss_G_h2l.item()
            loss_d_t += loss_D_h2l.item()
            #print(" {}({}) D_h2l: {:.3f}, D_l2h: {:.3f}, G_h2l: {:.3f}, G_l2h: {:.3f} \r".format(i+1, ep, loss_D_h2l.item(), loss_D_l2h.item(), loss_G_h2l.item(), loss_G_l2h.item()), end=" ")
            print(" Epoch: [{}/{}] step: [{}/{}] D_h2l: {:.6f},  G_h2l: {:.6f}, time: {:.6f} ".format(ep , max_epoch,  i+1, n_step, loss_D_h2l.item(),  loss_G_h2l.item(), time.time() - step_time))
            writer.add_scalar('loss_G_h2l', loss_G_h2l.item(), global_step=(ep-1)*n_step+i)
            writer.add_scalar('loss_D_h2l', loss_D_h2l.item(), global_step=(ep-1)*n_step+i)
        writer.add_scalar('total_loss_G_h2l', loss_g_t/n_step, global_step=ep)
        writer.add_scalar('total_loss_D_h2l', loss_d_t/n_step, global_step=ep)


        print("\n Testing and saving...")
        G_h2l.eval()
        D_h2l.eval()
        if ep >= 1000 and ep % 20 ==0: 
            save_file = model_file + 'model_epoch_{:04d}.pth'.format(ep)
            torch.save({"G_h2l": G_h2l.state_dict(), "D_h2l": D_h2l.state_dict()},save_file)
            print("saved: ", save_file)
    print('the process is finished')