import torch
from torchvision import datasets, transforms

def get_mnist_digits(target_digit):
    transform = transforms.Compose([transforms.ToTensor()])
    # Download MNIST to a ./data folder
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    idx = dataset.targets == target_digit
    images = dataset.data[idx].float() / 255.0
    return images.unsqueeze(1)