import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class ModelM5(nn.Module):
    def __init__(self):
        super(ModelM5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 5, bias=False)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, 5, bias=False)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 160, 5, bias=False)
        self.conv5_bn = nn.BatchNorm2d(160)
        self.fc1 = nn.Linear(10240, 2, bias=False)
        self.fc1_bn = nn.BatchNorm1d(2)
    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        flat5 = torch.flatten(conv5.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat5))
        return logits
    def forward(self, x):
        logits = self.get_logits(x)
        return logits

def face_mask_model():
    ckpt = torch.load('facemask/best_m5_1.pt', map_location='cpu')
    model = ModelM5()
    model.load_state_dict(ckpt)
    return model


def face_mask_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Grayscale(1),
        transforms.Normalize(0.5, 0.5)
    ])