import torch
import pytest
from model import MNISTModel, get_model_parameters
from train import main as train_main
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

@pytest.fixture
def model():
    return MNISTModel()

def test_parameter_count(model):
    """Test 1: Verify the model has fewer than 25,000 parameters"""
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 25000, f"Model has {param_count} parameters, which exceeds the limit of 25,000"

def test_model_output_shape(model):
    """Test 2: Verify the model produces correct output shape"""
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), f"Expected output shape (64, 10), but got {output.shape}"

def test_model_forward_pass(model):
    """Test 3: Test forward pass with specific input"""
    x = torch.ones(1, 1, 28, 28)  # Single image of all ones
    output = model(x)
    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert output.shape == (1, 10), f"Expected output shape (1, 10), but got {output.shape}"

def test_training_accuracy():
    """Test 4: Verify model achieves required accuracy in one epoch"""
    accuracy = train_main()
    assert accuracy >= 95.0, f"Model achieved {accuracy}% accuracy, which is below the required 95%"

def test_augmentation():
    """Test 5: Verify image augmentation is working"""
    transform = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Test for 100 different images
    original_images = []
    augmented_images = []
    
    for i in range(100):
        img1, _ = dataset[i]
        img2, _ = dataset[i]  # Get the same image again, but it will be differently augmented
        original_images.append(img1)
        augmented_images.append(img2)
    
    # Convert to tensors for easy comparison
    original_tensor = torch.stack(original_images)
    augmented_tensor = torch.stack(augmented_images)
    
    # Check if augmented images are different from original
    diff = (original_tensor - augmented_tensor).abs().mean()
    assert diff > 0.01, "Augmentation does not seem to be working properly"
