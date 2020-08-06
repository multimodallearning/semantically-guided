import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from src.utils import *

# weight initialisation
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight, gain=np.sqrt(2))

# Segmentation Network: UNet
class UNet2D(nn.Module):
    ''' n_classes: number of labels '''
    def __init__(self, n_classes):
        super(UNet2D, self).__init__()
       
        self.conv0 = nn.Conv2d(1, 5, 3, padding=1)
        self.batch0 = nn.BatchNorm2d(5)
        
        # downsampling w/ stride=2
        self.conv1 = nn.Conv2d(5, 16, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.conv11 = nn.Conv2d(16, 16, 3, padding=1)
        self.batch11 = nn.BatchNorm2d(16)
        
        # downsampling w/ stride=2
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(32)
        self.conv22 = nn.Conv2d(32, 32, 3, padding=1)
        self.batch22 = nn.BatchNorm2d(32)
        
        # downsampling w/ stride=2
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2,padding=1)
        self.batch3 = nn.BatchNorm2d(64)
        self.conv33 = nn.Conv2d(64, 64, 3, padding=1)
        self.batch33 = nn.BatchNorm2d(64)

        self.conv6bU = nn.Conv2d(96, 32, 3, padding=1)
        self.batch6bU = nn.BatchNorm2d(32)
        
        self.conv6U = nn.Conv2d(48, 32, 3, padding=1)
        self.batch6U = nn.BatchNorm2d(32)
        
        self.conv7U = nn.Conv2d(32, n_classes, 3, padding=1)
        
    def forward(self, inputImg):
        H,W = inputImg.shape[2:]
        H_grid = H//2; W_grid = W//2;
        H_grid2 = H_grid//2; W_grid2 = W_grid//2;
        
        x1 = F.leaky_relu(self.batch0(self.conv0(inputImg)), 0.1)
        
        x = F.leaky_relu(self.batch1(self.conv1(x1)),0.1)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)),0.1)
        
        x = F.leaky_relu(self.batch2(self.conv2(x2)),0.1)
        x3 = F.leaky_relu(self.batch22(self.conv22(x)),0.1)
        
        x = F.leaky_relu(self.batch3(self.conv3(x3)),0.1)
        x = F.leaky_relu(self.batch33(self.conv33(x)),0.1)
        
        # upsampling and skip connection
        x = F.upsample(x, size=[H_grid2,W_grid2], mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x,x3),1))),0.1)
        
        # upsampling and skip connection
        x = F.upsample(x, size=[H_grid,W_grid], mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x,x2),1))),0.1)

        return self.conv7U(x)
    
    

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, inch, outch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inch, outch, 3, padding=1),
            nn.BatchNorm2d(outch),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(outch, outch, 3, padding=1),
            nn.BatchNorm2d(outch),
            nn.LeakyReLU(0.1,inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x   
    
class down(nn.Module):
    def __init__(self, inch, outch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(inch, outch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
       
# registration network, regent
class RegNet(nn.Module):
    ''' n_classes: number image dimensions
        n_input: number of labels x2
        ch0: output channel of the first conv. layer
    '''
    def __init__(self, inch=16, ch0=16):
        super(RegNet, self).__init__()    
        ch1 = 2*ch0 
        ch2 = 2*ch1
        ch3 = 2*ch2
        ch4 = 2*ch3
               
        # down stream
        self.conv1 = double_conv(inch, ch0)   
        self.down1 = down(ch0,ch1)   
        self.down2 = down(ch1,ch2)
        self.down3 = down(ch2,ch3)
        self.down4 = down(ch3,ch4)
        self.outconv1 = nn.Conv2d(ch4,ch2,1)
        self.bn1 = nn.BatchNorm2d(ch2)
        self.relu1 = nn.PReLU()
        self.outconv2 = nn.Conv2d(ch2,2,1)

    def forward(self, x):
        
        # down stream
        x1 = self.conv1(x)  
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x5 = self.down4(x4) 
        
        y1 = self.relu1(self.bn1(self.outconv1(x5))) 
        output = self.outconv2(y1) 
        
        return output   
