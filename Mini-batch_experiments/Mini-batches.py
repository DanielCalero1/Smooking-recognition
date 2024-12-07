from torchvision import models, datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Entrenamiento de ResNet50 con diferentes tamaños de mini-batch.")
parser.add_argument('--lr', type=float, required=True, help="Learning rate para el optimizador")
parser.add_argument('--output_prefix', type=str, required=True, help="Prefijo para archivos de salida")
args = parser.parse_args()

learning_rate = args.lr
output_prefix = args.output_prefix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet50 = models.resnet50(pretrained=True)

num_features = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Linear(512, 1)
)
resnet50.to(device)

for param in resnet50.parameters():
    param.requires_grad = False
for param in resnet50.fc.parameters():
    param.requires_grad = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = '/ruta/en/cluster/training_data'
val_dir = '/ruta/en/cluster/validation_data'
rain_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

optimizer = torch.optim.SGD(resnet50.fc.parameters(), lr=learning_rate, momentum=0.9)

all_val_accuracies = []
all_epoch_accuracies = []
all_confusion_matrices = []

batch_sizes = [32, 64, 128, 256]

for batch_size in batch_sizes:
    print(f"Entrenando con batch size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 8

    epoch_accuracy = []
    val_accuracies = []

    for epoch in range(num_epochs):
        resnet50.train()
        correct, total, running_loss = 0, 0, 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = resnet50(images)
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()
            predicted_class = (outputs.squeeze() > 0.5).long()
            correct += (predicted_class == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item()
        epoch_accuracy.append(correct / total * 100)

        resnet50.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for val_images, val_targets in val_loader:
                val_images, val_targets = val_images.to(device), val_targets.to(device)
                val_outputs = resnet50(val_images)
                val_predicted_class = (val_outputs.squeeze() > 0.5).long()
                val_correct += (val_predicted_class == val_targets).sum().item()
                val_total += val_targets.size(0)
        val_accuracy = val_correct / val_total * 100
        val_accuracies.append(val_accuracy)

    all_epoch_accuracies.append(epoch_accuracy)
    all_val_accuracies.append(val_accuracies)

    # Matriz de confusión
    resnet50.eval()
    val_predictions, val_labels = [], []
    with torch.no_grad():
        for val_images, val_targets in val_loader:
            val_images, val_targets = val_images.to(device), val_targets.to(device)
            val_outputs = resnet50(val_images)
            predicted_class = (val_outputs.squeeze() > 0.5).long()
            val_predictions.extend(predicted_class.cpu().numpy())
            val_labels.extend(val_targets.cpu().numpy())
    cm = confusion_matrix(val_labels, val_predictions)
    all_confusion_matrices.append(cm)

with open(f"{output_prefix}_accuracy_results.txt", "w") as file:
    for i, batch_size in enumerate(batch_sizes):
        file.write(f"Batch Size: {batch_size}\n")
        file.write("Epoch\tTraining Accuracy (%)\tValidation Accuracy (%)\n")
        for epoch in range(num_epochs):
            file.write(f"{epoch + 1}\t{all_epoch_accuracies[i][epoch]:.2f}\t{all_val_accuracies[i][epoch]:.2f}\n")
        file.write("\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

markers = ['o', 's', 'D', '^', 'v', '<', '>', '*', 'p', 'h'] 
for i, batch_size in enumerate(batch_sizes):
    marker = markers[i % len(markers)]  
    ax1.plot(range(1, num_epochs + 1), all_epoch_accuracies[i], label=f'batch_size={batch_size}', marker=marker)
    ax2.plot(range(1, num_epochs + 1), all_val_accuracies[i], label=f'batch_size={batch_size}', marker=marker)

ax1.set_title('Training accuracy vs epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Accuracy (%)')
ax1.legend()
ax1.grid(False)

ax2.set_title('Validation accuracy vs epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Accuracy (%)')
ax2.legend()
ax2.grid(False)

plt.tight_layout()
plt.savefig(f'{output_prefix}_accuracy_per_batch_size.png')
