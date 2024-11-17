import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTModel

# Set random seed for reproducibility
torch.manual_seed(0)

# Define transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 100 batches
        if total % (100 * labels.size(0)) == 0:
            print(f'Progress: [{total:>5d}/60000] Loss: {loss.item():.4f}')

    accuracy = 100 * correct / total
    return accuracy

def main():
    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = MNISTModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)  # Adjusted learning rate

    # Train the model for one epoch and measure accuracy
    accuracy = train_model(model, train_loader, criterion, optimizer)
    print(f"Training Accuracy after 1 epoch: {accuracy:.2f}%")

    # Calculate the number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_params}")
    
    # Save the model
    torch.save(model.state_dict(), 'mnist_model.pth')

if __name__ == '__main__':
    main()
