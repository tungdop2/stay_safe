import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

def conv_block(in_channels, out_channels, pooling=False):
        '''
        params: in_channels: (int) number of input channels
        params: out_channels: (int) number of output channels
        params: pooling: (bool) use pooling or not
        return: convolutional layers
        '''
        conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )
        if pooling: 
            conv_layers.add_module('max_pooling',nn.MaxPool2d(2))
        return conv_layers

class ResNet9x128(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        #1st Block
        self.conv1 = conv_block(in_channels, 64)#input size 1*128*128
        self.conv2 = conv_block(64, 128, True) #After pooling 64*64*64
        #Residual layer
        self.res1 = nn.Sequential(conv_block(128,128), conv_block(128,128))
        
        #2nd Block
        self.conv3 = conv_block(128, 256, True) #After pooling 256*32*32
        self.conv4 = conv_block(256, 512, True) #After pooling 512*16*16
        #Residual layer
        self.res2 = nn.Sequential(conv_block(512,512), conv_block(512,512))
        
        #Linear Network
        self.linear = nn.Sequential(
            nn.MaxPool2d(16), #After pooling 512*1*1
            nn.Flatten(), # 512
            nn.Linear(512, num_classes),
            )

    def forward(self,x):
        #Block-1
        out = self.conv1(x)
        out = self.conv2(out)
        res1 = self.res1(out) + out

        # #Block-2
        out = self.conv3(res1)
        out = self.conv4(out)
        res2 = self.res2(out) + out

        # #Linear network
        out = self.linear(res2)
        return out
    
    def model_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Grayscale(1),
            transforms.Normalize(0.5, 0.5)
        ])

def face_mask_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Grayscale(1),
        transforms.Normalize(0.5, 0.5)
    ])

class ResNet9x64(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        #1st Block
        self.conv1 = conv_block(in_channels, 64)#input size 1*128*128
        self.conv2 = conv_block(64, 128, True) #After pooling 64*64*64
        #Residual layer
        self.res1 = nn.Sequential(conv_block(128,128), conv_block(128,128))
        
        #2nd Block
        self.conv3 = conv_block(128, 256, True) #After pooling 256*32*32
        self.conv4 = conv_block(256, 512, True) #After pooling 512*16*16
        #Residual layer
        self.res2 = nn.Sequential(conv_block(512,512), conv_block(512,512))
        
        #Linear Network
        self.linear = nn.Sequential(
            nn.MaxPool2d(16), #After pooling 512*1*1
            nn.Flatten(), # 512
            nn.Linear(512, num_classes),
            )

    def forward(self,x):
        #Block-1
        out = self.conv1(8)
        out = self.conv2(out)
        res1 = self.res1(out) + out

        # #Block-2
        out = self.conv3(res1)
        out = self.conv4(out)
        res2 = self.res2(out) + out

        # #Linear network
        out = self.linear(res2)
        return out

def resnet9():
    model = ResNet9x128(1, 2)
    ckpt = 'facemask/x128.pt'
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model
