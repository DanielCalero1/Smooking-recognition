from torchvision import models, datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
import csv

# Parameters
batch_size = [128]
learning_rate = [0.01,0.001]
num_epochs = 10

# Available models
available_models = {
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnet50': models.resnet50,
    'resnext101_64x4d': models.resnext101_64x4d, 
    'resnext101_32x8d': models.resnext101_32x8d,
    'mobilenet_v3_large': models.mobilenet_v3_large,
}

# Select model
model_name = 'resnext101_64x4d'
model = available_models[model_name](pretrained=True)

# Directories
train_dir = '/Users/alexandre/Projects/Resnet/Dataset-1/training_data'
val_dir = '/Users/alexandre/Projects/Resnet/Dataset-1/validation_data'
test_dir = '/Users/alexandre/Projects/Resnet/Dataset-1/testing_data'

# Data augmentation for training data
train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),  # Random crop and resize
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
    transforms.RandomRotation(degrees=15),  # Rotate up to 15 degrees
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Transforms for validation and testing data
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

# Training, validation, and test loaders
for bs in batch_size:
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    num_classes = len(train_dataset.classes)

    for lr in learning_rate:
        # Initialize the model
        model = available_models[model_name](pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Freeze the convolutional base
        for param in model.parameters():
            param.requires_grad = False

        # Modify the classifier for the custom dataset
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

        # Output directory
        CKPT_DIR = f'Data_augmented_Results/{model_name}_model_{bs}bs_{lr}lr'
        cwd = os.getcwd()
        rwd = os.path.join(cwd, CKPT_DIR)
        if not os.path.exists(rwd):
            os.makedirs(rwd)

        # Save job parameters
        job_params = {
            'model_name': model_name,
            'epochs': num_epochs,
            'batch_size': bs,
            'lr': lr,
        }
        with open(f"{CKPT_DIR}/job_params.json", "w") as outfile:
            json.dump(job_params, outfile, indent=4)

        # CSV file for results
        csv_file = f"{CKPT_DIR}/batch_accuracy_results.csv"

        # Training function
        def train(model, loader, criterion, optimizer):
            model.train()
            running_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                labels = F.one_hot(labels, num_classes=num_classes).float()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            return running_loss / len(loader)

        # Validation function
        def validate(model, loader, criterion):
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    labels = F.one_hot(labels, num_classes=num_classes).float()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.argmax(1)).sum().item()
            accuracy = 100 * correct / total
            return running_loss / len(loader), accuracy

        # Initialize CSV
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train_loss", "Validation_loss", "Validation_Accuracy", "Test_Accuracy"])

        # Training loop
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy = validate(model, val_loader, criterion)

            print(f"Epoch: {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.2f}%")

            test_loss, test_accuracy = validate(model, test_loader, criterion)
            print(f"Test Accuracy: {test_accuracy:.2f}%")

            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([epoch+1, train_loss, val_loss, val_accuracy, test_accuracy])
