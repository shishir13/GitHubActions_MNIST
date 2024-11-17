import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Input: 28x28x1
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 28x28x16
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)  # 14x14x16
        
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)  # 14x14x16
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)  # 14x14x16
        self.bn4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)  # 7x7x16
        
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3)  # 5x5x32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 10, kernel_size=3)  # 3x3x10
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def get_model_parameters():
    model = MNISTModel()
    return sum(p.numel() for p in model.parameters())
