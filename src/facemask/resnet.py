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
        conv_layers.add_module('max_pooling', nn.MaxPool2d(2))
    return conv_layers

class ResNet9(nn.Module):
    def __init__(self, size, in_channels, num_classes):
        super().__init__()

        # 1st Block
        self.conv1 = conv_block(in_channels, int(size / 2))  # input size 1*128*128
        self.conv2 = conv_block(int(size / 2), size , True)  # After pooling 64*64*64
        # Residual layer
        self.res1 = nn.Sequential(conv_block(size, size), conv_block(size, size))

        # 2nd Block
        self.conv3 = conv_block(size, size * 2, True)  # After pooling 256*32*32
        self.conv4 = conv_block(size * 2, size * 4, True)  # After pooling 512*16*16
        # Residual layer
        self.res2 = nn.Sequential(conv_block(size * 4, size * 4), conv_block(size * 4, size * 4))

        # Linear Network
        self.linear = nn.Sequential(
            nn.MaxPool2d(int(size / 8)),  # After pooling 512*1*1
            nn.Flatten(),  # 512
            nn.Linear(size * 4, num_classes),
        )

    def forward(self, x):
        # Block-1
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

def model(size=128):
    model = ResNet9(size, 1, 2)
    ckpt = 'facemask/x{}.pt'.format(size)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model, transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Grayscale(1),
        transforms.Normalize(0.5, 0.5)
    ])
