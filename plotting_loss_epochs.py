import pandas as pd
import matplotlib.pyplot as plt
import os
from torchvision import models

available_models = {
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnet50': models.resnet50,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    # Add more models as needed
}

# batch_size = [2, 4, 16, 32, 64, 128, 256] 
# learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
batch_size = 4
learning_rate = 0.01 

# Select a model by name
model_name = 'resnext50_32x4d'

CKPT_DIR = f'/Users/alexandre/Projects/Resnet/results_{model_name}modell/{model_name}model_{batch_size}bs_{learning_rate}lr'
cwd = os.getcwd()
rwd = os.path.join(cwd, CKPT_DIR)
if not os.path.exists(rwd):
    os.makedirs(rwd)

# Load the CSV file
#file_path = f"/Users/alexandre/Projects/Resnet/Results/{model_name}model_2bs_0.1lr/batch_accuracy_results.csv"  # Update with the correct path to your file
file_path = f"{CKPT_DIR}/batch_accuracy_results.csv"  # Update with the correct path to your file
data = pd.read_csv(file_path)

# Extract data
epochs = data['Epoch']
train_loss = data['Train_loss']
validation_loss = data['Validation_loss']
validation_accuracy = data['Validation_Accuracy']
test_accuracy = data['Test_Accuracy']

# Create a mosaic layout for the plots
fig, axes = plt.subplot_mosaic(
    [
        ["train_loss", "validation_loss"],
        ["validation_accuracy", "test_accuracy"]
    ],
    figsize=(12, 10)
)


# Plot Train Loss
axes["train_loss"].plot(epochs, train_loss, label="Train Loss", color="blue", marker="o")
axes["train_loss"].set_title("Train Loss vs Epochs")
axes["train_loss"].set_xlabel("Epochs")
axes["train_loss"].set_ylabel("Loss")
axes["train_loss"].legend()

# Plot Validation Loss
axes["validation_loss"].plot(epochs, validation_loss, label="Validation Loss", color="orange", marker="o")
axes["validation_loss"].set_title("Validation Loss vs Epochs")
axes["validation_loss"].set_xlabel("Epochs")
axes["validation_loss"].set_ylabel("Loss")
axes["validation_loss"].legend()

# Plot Validation Accuracy
axes["validation_accuracy"].plot(epochs, validation_accuracy, label="Validation Accuracy", color="green", marker="o")
axes["validation_accuracy"].set_title("Validation Accuracy vs Epochs")
axes["validation_accuracy"].set_xlabel("Epochs")
axes["validation_accuracy"].set_ylabel("Accuracy")
axes["validation_accuracy"].legend()

# Plot Test Accuracy
axes["test_accuracy"].plot(epochs, test_accuracy, label="Test Accuracy", color="red", marker="o")
axes["test_accuracy"].set_title("Test Accuracy vs Epochs")
axes["test_accuracy"].set_xlabel("Epochs")
axes["test_accuracy"].set_ylabel("Accuracy")
axes["test_accuracy"].legend()

# Ensure tight layout to align grids properly
plt.tight_layout()

# batch_size = [2, 4, 8, 16, 32, 64, 128, 256,512] 
# learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

# bs = 2 
# lr = 0.1 





# plt.savefig(f"{CKPT_DIR}/loss_epochs_{learning_rate}.png")
# Show the plots
plt.show()
assert 0 
