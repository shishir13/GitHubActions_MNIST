import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)   # Output: (batch_size, 8, 24, 24)
        self.pool = nn.MaxPool2d(2, 2)                # Output: (batch_size, 8, 12, 12)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)  # Output: (batch_size, 16, 8, 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 64)          # Adjusted to reduce parameters
        self.fc2 = nn.Linear(64, 10)                  # Output layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)                              # Output: (batch_size, 16, 4, 4)
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_model_parameters():
    model = MNISTModel()
    return sum(p.numel() for p in model.parameters())
