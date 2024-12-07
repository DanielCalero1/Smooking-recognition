from torchvision import models
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,TensorDataset
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import os 
import csv
import json
# for number-crunching
#import numpy as np
#import scipy.stats as stats

# for dataset management
#import pandas as pd

# for data visualization
#import matplotlib.pyplot as plt
#from IPython import display
# print(dir(models))
# assert 0 
# Parameters
# batch_size = [2, 4, 8, 16, 32, 64, 128, 256,512] 
batch_size = [128] 
# learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
learning_rate = [0.01]
num_epochs = 1


# model = models.resnext101_32x8d()
available_models = {
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnet50': models.resnet50,
    'resnext101_64x4d': models.resnext101_64x4d, 
    'resnext101_32x8d': models.resnext101_32x8d,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    # Add more models as needed
}

# Select a model by name
model_name = 'resnext101_64x4d'

# Initialize the model
model = available_models[model_name](pretrained=True)


# Print the model's name
# print(f"Selected model: {model_name}")
# assert 0 
# model = models.resnet50(pretrained=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# train_dir = '/Users/alexandre/Projects/Resnet/Dataset/Training'
train_dir = '/Users/alexandre/Projects/Resnet/Dataset-1/training_data'
val_dir = '/Users/alexandre/Projects/Resnet/Dataset-1/validation_data'
# val_dir = '/Users/alexandre/Projects/Resnet/Dataset/Validation'
test_dir = '/Users/alexandre/Projects/Resnet/Dataset-1/testing_data'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)


for bs in batch_size:

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    num_classes = len(train_dataset.classes)

    for lr in learning_rate:
        model = available_models[model_name](pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        model.eval()

        predictions = []
        labels = []
        images_list = []

        #Freeze the convolutional base
        for param in model.parameters():
            param.requires_grad = False

        # Modify the classifier for the custom dataset
        # resnet50.fc = nn.Linear(2048, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)


        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)

        # Loss and optimizer
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        # optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr)
        optimizer = torch.optim.RMSprop(model.fc.parameters(),lr=lr)

        # Load the datasets
        CKPT_DIR = f'Results/{model_name}model_{bs}bs_{lr}lr'
        cwd = os.getcwd()
        rwd = os.path.join(cwd, CKPT_DIR)
        if not os.path.exists(rwd):
            os.makedirs(rwd)

        job_params ={'model_name': model_name,               
                        'epochs': num_epochs,
                        'batch_size': bs,
                        'lr': lr,
                        }
        with open(f"{CKPT_DIR}/job_params.json", "w") as outfile:
            json.dump(job_params, outfile, indent=4)

        csv_file = f"{CKPT_DIR}/batch_accuracy_results.csv"
        # Training function
        def train(model, loader, criterion, optimizer):
            model.train()
            running_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                # One-hot encode labels for BCEWithLogitsLoss
                labels = F.one_hot(labels, num_classes=num_classes).float()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            return running_loss / len(loader)

        def validate(model, loader, criterion):
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    # One-hot encode labels for BCEWithLogitsLoss
                    labels = F.one_hot(labels, num_classes=num_classes).float()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    # For accuracy, compare with predicted class
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.argmax(1)).sum().item()
            accuracy = 100 * correct / total
            return running_loss / len(loader), accuracy


        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train_loss", "Validation_loss", "Validation_Accuracy", "Test_Accuracy"])

        # Training loop
        for epoch in range(num_epochs+1):
            train_loss = train(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy = validate(model, val_loader, criterion)
            
            print(f"Epoch:, {epoch+1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2f}%")

            # Test the model
            test_loss, test_accuracy = validate(model, test_loader, criterion)
            # print(f"Test Accuracy: {test_accuracy:.2f}%")
            # assert 0 
            with open(csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch+1, train_loss, val_loss, val_accuracy, test_accuracy])
# print(f'Finished Training for model {model_name} with batch size {bs} and learning rate {lr}')
# assert 0 

# Save the model

# fwd = os.path.join(cwd, FIG_DIR)
# if not os.path.exists(fwd):
#     os.makedirs(fwd)

#torch.save(model.state_dict(), f"{CKPT_DIR}")



