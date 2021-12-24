from torch.utils.data import DataLoader
import cv2
from torchvision import transforms
import os,time,sys
from torch.utils.data import Dataset, DataLoader
import tensorlayer as  tl
import torch
from PIL import Image
import numpy as np
from model_SRA import SRA

class load_test(Dataset):
    def __init__(self,data_test):
        
        self.test_imgs = [os.path.join(d, i) for d in data_test for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.preproc = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])

        

    def __len__(self):
        return len(self.test_imgs)
    
    def __getitem__(self,index):
        data = {}
        test_img = cv2.imread(self.test_imgs[index],cv2.IMREAD_GRAYSCALE)
        data['test'] = self.preproc(test_img)
        return data



def l2h_predict(l2h_test_data_path, save_dir, model_path,img_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_l2h = SRA()
    medel_l2h = model_l2h.to(device)
    a = torch.load(model_path)
    model_l2h.load_state_dict(a["G_l2h"])
    model_l2h = model_l2h.eval()
    dataset = load_test(l2h_test_data_path)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    out_mean = 0
    for i, data in enumerate(loader):
        img_test = data['test'].to(device)
        with torch.no_grad():
            data_out = model_l2h(img_test)
        out = data_out.cpu().numpy().squeeze(0)
        out = (out*0.5+0.5)*255
        out_mean += out/1000
        cv2.imwrite(save_dir + '{0:04d}.bmp'.format(i+1), out[0,:,:])
    cv2.imwrite(img_name, out_mean[0,:,:])   

if __name__=='__main__':
    model_path = 'your model path'
    data_list = [ your data_list path ]   
    mean_list = [your mean_list name]
    for i in range(1):
        test_data = 'your test data path'  
        save_data =  'your save data path'  
        tl.files.exists_or_mkdir(save_data)
        l2h_predict(test_data, save_data, model_path, mean_list)