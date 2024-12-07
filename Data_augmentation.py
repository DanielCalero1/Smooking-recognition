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
learning_rate = 0.067

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
# Transformaciones de datos con Data Augmentation para el conjunto de entrenamiento
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # Volteo horizontal aleatorio
    transforms.RandomRotation(degrees=15),  # Rotación aleatoria hasta 15 grados
    transforms.ToTensor(),  # Convertir a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización
])

# Transformaciones de datos para validación (sin data augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar para validación
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = '/home/dan27/projects/def-ayers/dan27/courses/Project/Dataset-1/training_data'
val_dir = '/home/dan27/projects/def-ayers/dan27/courses/Project/Dataset-1/validation_data'

# Datasets con transformaciones correspondientes
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

# DataLoaders con batch size fijo
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Restante del código sigue igual...

# Criterio de pérdida
criterion = nn.BCEWithLogitsLoss()

# Variables para almacenar resultados
all_epoch_accuracies = []
all_val_accuracies = []
Experiments=5
num_epochs=5
# Inicializar variables para almacenar accuracies acumulados
cumulative_train_accuracy = [0] * num_epochs  # Para acumular accuracies de entrenamiento
cumulative_val_accuracy = [0] * num_epochs    # Para acumular accuracies de validación

# Número de experimentos
Experiments = 5

# Entrenamiento y evaluación para cada experimento
for i in range(Experiments):
    print(f"Experiment {i + 1}")

    # Configurar optimizador
    optimizer = torch.optim.SGD(resnet50.fc.parameters(), lr=learning_rate, momentum=0.9)

    epoch_accuracy = []  # Para almacenar accuracies de cada época en este experimento
    val_accuracies = []  # Para almacenar accuracies de validación en este experimento

    # Número de épocas
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

    # Acumular accuracies de este experimento
    for epoch in range(num_epochs):
        cumulative_train_accuracy[epoch] += epoch_accuracy[epoch]
        cumulative_val_accuracy[epoch] += val_accuracies[epoch]

# Calcular promedios de accuracy por época
average_train_accuracy = [acc / Experiments for acc in cumulative_train_accuracy]
average_val_accuracy = [acc / Experiments for acc in cumulative_val_accuracy]

# Guardar resultados promediados en archivo
with open("Date_augmentation.txt", "w") as file:
    file.write("Epoch\tAvg Training Accuracy (%)\tAvg Validation Accuracy (%)\n")
    for epoch in range(num_epochs):
        file.write(f"{epoch + 1}\t{average_train_accuracy[epoch]:.2f}\t{average_val_accuracy[epoch]:.2f}\n")

print("Resultados promediados guardados en 'Date_augmentation.txt'.")
