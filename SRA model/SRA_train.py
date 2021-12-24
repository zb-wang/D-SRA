import torch
import os, sys
import numpy as np
import cv2
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Data_load import Dataset_111, l2h_High_Data, l2h_Low_Data
from model_SRA import SRA
from dataset import get_loader
import argparse
import time
import math
import tensorlayer as tl
from tensorboardX import SummaryWriter
writer = SummaryWriter('./your path/')
model_path = './your path/'
tl.files.exists_or_mkdir(model_path)
weights_path =  './your path/'
save_dir = './your path/'
tl.files.exists_or_mkdir(save_dir)
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available else "cpu")


    seed_num = 2020
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True  #limit the choice of algorithm
    torch.backends.cudnn.benchmark = False  #set for the reproduction

    max_epoch = 5000
    learn_rate = 1e-4
    alpha, beta = 1, 0.005
    
    G_l2h = SRA().to(device)
    D_l2h = Discriminator().to(device)
    d_l2h = Discriminator()
    a = torch.load(weights_path)
    G_l2h.load_state_dict(a["G_l2h"])
    D_l2h.load_state_dict(a['D_l2h'])
    mse = nn.MSELoss()
    optim_D_l2h = optim.Adam(filter(lambda p: p.requires_grad, D_l2h.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_l2h = optim.Adam(G_l2h.parameters(), lr=learn_rate, betas=(0.0, 0.9))

    #load data
    data = Dataset_111(l2h_High_Data, l2h_Low_Data)
    batch_size = 5

    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)


    for ep in range(1,max_epoch+1):
        G_l2h.train()
        D_l2h.train()
        loss_g_t, loss_d_t =0, 0
        for i, batch in enumerate(loader):
            n_step = math.floor(200/batch_size)
            step_time = time.time()
            optim_D_l2h.zero_grad()
            optim_G_l2h.zero_grad()
            lrs = batch['lr'].to(device)
            hrs = batch['hr'].to(device)

            hr_gen = G_l2h(lrs)
            lr_detach = lrs.detach()
            hr_gen_detach = hr_gen.detach()
            loss_D_l2h = nn.ReLU()(1.0 - D_l2h(hrs)).mean() + nn.ReLU()(1 + D_l2h(hr_gen_detach)).mean()
            loss_D_l2h.backward()
            optim_D_l2h.step()

            optim_D_l2h.zero_grad()
            gan_loss_l2h = -D_l2h(hr_gen).mean()
            mse_loss_l2h = mse(hr_gen, hrs)

            loss_G_l2h = alpha* mse_loss_l2h + beta*gan_loss_l2h
            loss_G_l2h.backward()
            optim_G_l2h.step()

            loss_g_t += loss_G_l2h.item()
            loss_d_t += loss_D_l2h.item()
            print(" Epoch: [{}/{}] step: [{}/{}] D_l2h: {:.6f},  G_l2h: {:.6f}, time: {:.6f} ".format(ep , max_epoch,  i+1, n_step, loss_D_l2h.item(),  loss_G_l2h.item(), time.time() - step_time))
            writer.add_scalar('loss_G_l2h', loss_G_l2h.item(), global_step=(ep-1)*n_step+i)
            writer.add_scalar('loss_D_l2h', loss_D_l2h.item(), global_step=(ep-1)*n_step+i)
        writer.add_scalar('total_loss_G_l2h', loss_g_t/n_step, global_step=ep)
        writer.add_scalar('total_loss_D_l2h', loss_d_t/n_step, global_step=ep)

        print("\n Testing and saving...")
        G_l2h.eval()
        D_l2h.eval()
        if ep % 10 ==0: 
            ####real-time output
            lr_out = lr_detach.cpu().numpy()
            hr_out = hr_gen_detach.cpu().numpy()
            lr_out = (0.5* lr_out + 0.5)*255
            hr_out = (0.5* hr_out + 0.5)*255
            cv2.imwrite(save_dir + 'lr_1{0:05d}.bmp'.format(ep+1030), lr_out[-1,0, :,:])
            cv2.imwrite(save_dir + 'hr_1{0:05d}.bmp'.format(ep+1030), hr_out[-1,0, :,:])
            
            ###########real-time save the model

            save_file = model_path + "model_epoch_1{:04d}.pth".format(ep+1030)
            torch.save({"G_l2h": G_l2h.state_dict(), "D_l2h": D_l2h.state_dict()},save_file)
            print("saved: ", save_file)
    print('the process is finished')
