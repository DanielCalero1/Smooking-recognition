from torchvision import models, datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Learning rates para probar
learning_rates = [0.23,0.34,0.45,0.56,0.67]

# Configurar tamaño de batch
batch_size = 256

# Cargar modelo pre-entrenado
resnet50 = models.resnet50(pretrained=True)

# Ajustar la última capa del modelo
num_features = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Linear(512, 1)
)
resnet50.to(device)

# Congelar capas excepto la última capa completamente conectada
for param in resnet50.parameters():
    param.requires_grad = False
for param in resnet50.fc.parameters():
    param.requires_grad = True

# Transformaciones de datos
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Rutas de los datos (modifica según tu clúster)
train_dir = '/home/dan27/projects/def-ayers/dan27/courses/Project/Dataset-1/training_data'
val_dir = '/home/dan27/projects/def-ayers/dan27/courses/Project/Dataset-1/validation_data'

# Datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# DataLoaders con batch size fijo
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Criterio de pérdida
criterion = nn.BCEWithLogitsLoss()

# Variables para almacenar resultados
all_epoch_accuracies = []
all_val_accuracies = []

# Entrenamiento y evaluación para cada learning rate
for lr in learning_rates:
    print(f"Entrenando con learning rate: {lr}")

    # Configurar optimizador
    optimizer = torch.optim.SGD(resnet50.fc.parameters(), lr=lr, momentum=0.9)

    epoch_accuracy = []
    val_accuracies = []

    # Número de épocas
    num_epochs = 8

    for epoch in range(num_epochs):
        resnet50.train()
        correct, total, running_loss = 0, 0, 0.0

        # Entrenamiento
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

        # Validación
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

    # Guardar resultados en archivo individual
    with open(f"results_lr_{lr:.3f}.txt", "w") as file:
        file.write(f"Learning Rate: {lr}\n")
        file.write("Epoch\tTraining Accuracy (%)\tValidation Accuracy (%)\n")
        for epoch in range(num_epochs):
            file.write(f"{epoch + 1}\t{epoch_accuracy[epoch]:.2f}\t{val_accuracies[epoch]:.2f}\n")

# Opcional: graficar resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for i, lr in enumerate(learning_rates):
    ax1.plot(range(1, num_epochs + 1), all_epoch_accuracies[i], label=f'lr={lr:.3f}')
    ax2.plot(range(1, num_epochs + 1), all_val_accuracies[i], label=f'lr={lr:.3f}')

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
plt.savefig('accuracy_per_learning_rate.png')
