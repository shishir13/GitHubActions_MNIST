import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import MNISTModel
from tqdm import tqdm
import random

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(f'Epoch: {epoch} Loss: {loss.item():.6f} Accuracy: {100*correct/processed:0.2f}%')
    
    return 100*correct/processed

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Training transformations with augmentation
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST(
        './data', train=True, download=True,
        transform=train_transforms
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = MNISTModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Train for one epoch
    accuracy = train_model(model, device, train_loader, optimizer, 1)
    print(f"Final Accuracy: {accuracy}%")
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_model.pth')
    return accuracy

if __name__ == '__main__':
    main()
