import os, sys
import numpy as np
import cv2
import tensorlayer as  tl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as nnF
import torch
import tensorflow.compat.v1 as tf

l2h_High_Data = ['./your path/']
l2h_Low_Data = ['./your path/']


class Dataset_111():
    def __init__(self, hr_data, lr_data):
        self.hr_imgs = [os.path.join(d, i) for d in hr_data for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.lr_imgs = [os.path.join(d, i) for d in lr_data for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])    

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, index):
        data = {}
        hr = cv2.imread(self.hr_imgs[index],cv2.IMREAD_GRAYSCALE)
        lr = cv2.imread(self.lr_imgs[index],cv2.IMREAD_GRAYSCALE)
        data["hr"] = self.preproc(hr)
        data['lr'] = self.preproc(lr)
        data["hr_down"] = nnF.avg_pool2d(data["hr"], 4, 4)
        return data


if __name__ == "__main__":
    
    data = Dataset_111(l2h_High_Data, l2h_Low_Data)
    batch_size = 20
    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    for i, batch in enumerate(loader):
        print("batch: ", i)
        lrs = batch["lr"].numpy()       
        hrs = batch["hr"].numpy()
        downs = batch["hr_down"].numpy()      
        for b in range(batch_size):
        
            lr = lrs[b]
            hr = hrs[b]
            down = downs[b]
            lr = lr.transpose(1, 2, 0)
            hr = hr.transpose(1, 2, 0)
            down = down.transpose(1, 2, 0)
            lr = (0.5* lr + 0.5)*255
            hr = (0.5* hr + 0.5)*255
            down = (down - down.min()) / (down.max() - down.min())
            cv2.imwrite(save_dir + "lr_{}_{}.bmp".format(i, b), lr)
            cv2.imwrite(save_dir + "hr_{}_{}.bmp".format(i, b), hr)
    print("finished.")
