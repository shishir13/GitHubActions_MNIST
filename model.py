import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # (batch_size, 8, 28, 28)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # (batch_size, 16, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)  # After first pool: (batch_size, 16, 14, 14)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)  # (batch_size, 32, 12, 12)
        self.fc1 = nn.Linear(32 * 5 * 5, 10)  # Reduced size of fully connected layer
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 8, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (batch_size, 16, 7, 7)
        x = F.relu(self.conv3(x))  # (batch_size, 32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)
        x = self.dropout(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

def get_model_parameters():
    model = MNISTModel()
    return sum(p.numel() for p in model.parameters())
