from torch.utils.data import DataLoader
import cv2
from torchvision import transforms
import os,time,sys
from torch.utils.data import Dataset, DataLoader
from model_De import High2Low
import tensorlayer as  tl
import torch
from PIL import Image
import numpy as np
from model_utils import GEN_DEEP

test_data_path = ["./your path/"]
model_path = './your file'
test_result_dir = './your path/'
tl.files.exists_or_mkdir(test_result_dir)
class load_test(Dataset):
    def __init__(self,data_test):
        self.test_imgs = [os.path.join(d, i) for d in data_test for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.preproc = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
            #transforms.Normalize((0.5), (0.5))
        

    def __len__(self):
        return len(self.test_imgs)
    
    def __getitem__(self,index):
        data = {}
        test_img = cv2.imread(self.test_imgs[index],cv2.IMREAD_GRAYSCALE)
        data['test'] = self.preproc(test_img)

        return data

def predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    


    model_h2l = High2Low()
    medel_h2l = model_h2l.to(device)
    a = torch.load(model_path)
    model_h2l.load_state_dict(a["G_h2l"])
    model_h2l = model_h2l.eval()
    dataset = load_test(test_data_path)
    
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        img_test = data['test'].to(device)
        

        with torch.no_grad():
            data_out = model_h2l(img_test)

        out = data_out.cpu().numpy().squeeze(0)
        out = (out*0.5+0.5)*255
        cv2.imwrite(test_result_dir + '0-1_{0:04d}.bmp'.format(i+1), out[0,:,:])

    

if __name__=='__main__':
    #l2h_predict()
    predict()
