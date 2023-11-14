from random import Random
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import RandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.models import LinearNet, ConvNet

if __name__ == '__main__':

    # Set the batch size
    batch_size = 1024

    # Check if CUDA is available (Note: It is not currently benificial to use GPU acceleration of the Raspberry Pi)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device: {}".format(device))

    # Load the MNIST dataset
    transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('/home/pi/ee347/challenge-lab/data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10('/home/pi/ee347/challenge-lab/data', train=False, transform=transform)
    train_sampler = RandomSampler(trainset, num_samples=10000)
    test_sampler = RandomSampler(testset, num_samples=2000)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, sampler=test_sampler)

    # Create the model and optimizer
    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Start timer
    t = time.time_ns()

    # Train the model
    model.train()
    train_loss = 0

    # Batch Loop
    for i, (images, labels) in enumerate(tqdm(train_loader, leave=False)):

        # Move the data to the device (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Accumulate the loss
        train_loss = train_loss + loss.item()

    # Test the model
    model.eval()
    correct = 0
    total = 0

    # Batch Loop
    for images, labels in tqdm(test_loader, leave=False):

        # Move the data to the device (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get the predicted class from the maximum value in the output-list of class scores
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        # Accumulate the number of correct classifications
        correct += (predicted == labels).sum().item()

    # Print the epoch statistics
    print("Training Loss: {:.4f}, Test Accuracy: {:.2f}%, Time Taken: {:.2f}s".format(train_loss / len(train_loader), 100 * correct / total, (time.time_ns() - t) / 1e9))