import torch.nn as nn
import torch
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class CA(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(CA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, inplanes//reduction, 1, padding = 0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inplanes//reduction, inplanes, 1, padding=0, bias=True),
            nn.Sigmoid()            
        )

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv(out)
        out = x * out
        return out

class SA(nn.Module):
    def __init__(self, inplanes, reduction=2):
        super(SA, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, inplanes*reduction, 1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inplanes * reduction, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.convs(x)
        out = x * out
        return out

class CSAR(nn.Module):
    def __init__(self, inplanes):
        super(CSAR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=True)
        )
        self.CA = CA(inplanes, reduction = 16)
        self.SA = SA(inplanes, reduction = 2)
        self.conv_1 = nn.Conv2d(2*inplanes, inplanes, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        y = self.conv(x)
        y_ca = self.CA(y)
        y_sa = self.SA(y)
        y = self.conv_1(torch.cat([y_ca, y_sa],1))
        out = x + y
        return out

class CSAR_group(nn.Module):
    def __init__(self, inplanes, num_CSAR=12):
        super(CSAR_group, self).__init__()
        CSAR_block = []
        for i in range(num_CSAR):
            CSAR_block.append(CSAR(inplanes))
        self.CSAR_chain = nn.Sequential(*CSAR_block)

    def forward(self, x):
        y = self.CSAR_chain(x)
        return y 

class FT(nn.Module):
    def __init__(self, inplanes):
        super(FT, self).__init__()
        self.conv_head = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=True)
        self.ffm = CSAR_group(inplanes)
        self.node1 = nn.Conv2d(2*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node2 = nn.Conv2d(3*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node3 = nn.Conv2d(4*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node4 = nn.Conv2d(5*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node5 = nn.Conv2d(6*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node6 = nn.Conv2d(7*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node7 = nn.Conv2d(8*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node8 = nn.Conv2d(9*inplanes,inplanes, kernel_size=1, padding=0, bias=True)

        self.conv_tail = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=True)
    
    def forward(self, x):
        y0 = self.conv_head(x)
        y1 = self.node1(torch.cat([y0, self.ffm(y0)], 1))
        y2 = self.node2(torch.cat([y0, y1, self.ffm(y1)], 1))
        y3 = self.node3(torch.cat([y0, y1, y2, self.ffm(y2)], 1))
        y4 = self.node4(torch.cat([y0, y1, y2, y3, self.ffm(y3)], 1))
        y5 = self.node5(torch.cat([y0, y1, y2, y3, y4, self.ffm(y4)], 1))
        y6 = self.node6(torch.cat([y0, y1, y2, y3, y4, y5, self.ffm(y5)], 1))
        y7 = self.node7(torch.cat([y0, y1, y2, y3, y4, y5, y6, self.ffm(y6)], 1))
        y8 = self.node8(torch.cat([y0, y1, y2, y3, y4, y5, y6, y7, self.ffm(y7)], 1))
        out = self.conv_tail(y8)
        out = x + out
        return out

class CSFM(nn.Module):
    def __init__(self, inplanes=64, scale=4):
        super(CSFM, self).__init__()
        self.conv0 = nn.Conv2d(1, inplanes, kernel_size=3, padding=1, bias=True)
        self.FT_net = FT(inplanes)
        self.Up = nn.Sequential(
            nn.PixelShuffle(scale),
            nn.Conv2d(int(inplanes/scale**2), 1, kernel_size=3, padding=1, bias=True)
        )


    def forward(self, x):
        x = self.conv0(x)
        x = self.FT_net(x)
        out = self.Up(x)
        return out

class FT1_net(nn.Module):
    def __init__(self, inplanes):
        super(FT1_net, self).__init__()
        scale = 2
        self.conv_head = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=True)
        self.ffm0 = CSAR_group(inplanes)
        self.ffm1 = CSAR_group(inplanes)
        self.ffm2 = CSAR_group(inplanes)
        self.ffm3 = CSAR_group(inplanes)
        self.ffm4 = CSAR_group(inplanes)
        self.ffm5 = CSAR_group(inplanes)
        self.node1 = nn.Conv2d(2*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node2 = nn.Conv2d(3*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node3 = nn.Conv2d(4*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node4 = nn.Conv2d(5*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node5 = nn.Conv2d(6*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node6 = nn.Conv2d(7*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        
        self.conv_tail = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=False)
        # self.upscale = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
        #     nn.BatchNorm2d(inplanes),
        #     nn.LeakyReLU(True),
        #     nn.ConvTranspose2d(inplanes, inplanes, kernel_size=1, stride=1)
        # )
        self.upscale = nn.Sequential(
            nn.Conv2d(inplanes, inplanes*scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.up_skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            conv3x3(inplanes,int(inplanes/2))
        )
        


        
        


    def forward(self, x):
        y0 = self.conv_head(x)
        y1 = self.node1(torch.cat([y0, self.ffm0(y0)], 1))
        y2 = self.node2(torch.cat([y0, y1, self.ffm1(y1)], 1))
        y3 = self.node3(torch.cat([y0, y1, y2, self.ffm2(y2)], 1))
        y4 = self.node4(torch.cat([y0, y1, y2, y3, self.ffm3(y3)], 1))
        y5 = self.node5(torch.cat([y0, y1, y2, y3, y4, self.ffm4(y4)], 1))
        y6 = self.node6(torch.cat([y0, y1, y2, y3, y4, y5, self.ffm5(y5)], 1))

        out = self.conv_tail(y6)
        out = self.upscale(out)
        x = self.up_skip(x)
        out = x + out
        return out

class FT2_net(nn.Module):
    def __init__(self, inplanes):
        super(FT2_net, self).__init__()
        scale = 2 
        #self.conv0 = conv3x3(inplanes, int(inplanes/2))
        self.conv_head = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=True)
        self.ffm0 = CSAR_group(inplanes)
        self.ffm1 = CSAR_group(inplanes)
        self.node1 = nn.Conv2d(2*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.node2 = nn.Conv2d(3*inplanes,inplanes, kernel_size=1, padding=0, bias=True)
        self.conv_tail = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=True)
        # self.upscale = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
        #     nn.BatchNorm2d(inplanes),
        #     nn.LeakyReLU(True),
        #     nn.ConvTranspose2d(inplanes, inplanes, kernel_size=1, stride=1)
        # )
        self.upscale = nn.Sequential(
            nn.Conv2d(inplanes, inplanes *scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

        self.up_skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
            conv3x3(inplanes,int(inplanes/2))
        )

    def forward(self, x):
        
        y0 = self.conv_head(x)
        y1 = self.node1(torch.cat([y0, self.ffm0(y0)], 1))
        y2 = self.node2(torch.cat([y0, y1, self.ffm1(y1)], 1))
        out = self.conv_tail(y2)
        out = self.upscale(out)
        x = self.up_skip(x)
        out = x + out
        return out


class SRA(nn.Module):
    def __init__(self, inplanes_ft1=64, inplanes_ft2=32, inplanes_final=16):
        super(CSFM_L2H, self).__init__()
        self.FE_net = conv3x3(1, inplanes_ft1)
        self.FT1_net = FT1_net(inplanes_ft1)
        self.FT2_net = FT2_net(inplanes_ft2)


        self.final = nn.Sequential(
            conv3x3(inplanes_final,inplanes_final),
            nn.LeakyReLU(True),
            nn.Conv2d(inplanes_final, int(inplanes_final/2), kernel_size=1, stride=1),
            nn.LeakyReLU(True),
            nn.Conv2d(int(inplanes_final/2), 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.FE_net(x)
        x = self.FT1_net(x)
        x = self.FT2_net(x)
        out = self.final(x)

        return out



