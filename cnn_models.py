import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from torchvision import models
# load the binarized labels file
#lb = joblib.load('../outputs/lb.pkl')
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        #self.fc2 = nn.Linear(256, len(lb.classes_))
        self.fc2 = nn.Linear(256, 39)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        net = models.resnet18(weights=False)
        net.fc = nn.Linear(512, 39)
        self.net = net
    def forward(self, x):
        x = self.net(x)
        return x

class CustomResNet34(nn.Module):
    def __init__(self):
        super(CustomResNet34, self).__init__()
        net = models.resnet34(weights=False)
        net.fc = nn.Linear(512, 39)
        self.net = net
    def forward(self, x):
        x = self.net(x)
        return x