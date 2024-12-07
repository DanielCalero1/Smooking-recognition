import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

model_name = 'resnext101_64x4d'
bs = 128
learning_rate = [ 0.01, 0.001]

# CKPT_DIR_save = f'/Users/alexandre/Projects/Resnet/Resnext{model_name}
# CKPT_DIR_save = f'/Users/alexandre/Projects/Resnet/SGD_Resnext{model_name}'
CKPT_DIR_save = f'/Users/alexandre/Projects/Resnet/static_Data_aug_Results'
# CKPT_DIR_save = f'/Users/alexandre/Projects/Resnet/RMSProp_Results_resnext101_64x4d'
cwd = os.getcwd()
rwd = os.path.join(cwd, CKPT_DIR_save)
if not os.path.exists(rwd):
    os.makedirs(rwd)


plt.figure(figsize=(14, 6))
for lr in learning_rate: 
    # CKPT_DIR = f'/Users/alexandre/Projects/Resnet/Results_{model_name}modell/{model_name}model_{bs}bs_{lr}lr'
    # CKPT_DIR = f'/Users/alexandre/Projects/Resnet/SGD_Results_{model_name}/{model_name}model_{bs}bs_{lr}lr'
    CKPT_DIR = f'{CKPT_DIR_save}/{model_name}model_{bs}bs_{lr}lr'
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    # Load the CSV file for this configuration
    file_path = f"{CKPT_DIR}/batch_accuracy_results.csv"

    data = pd.read_csv(file_path)

    # Extract data
    epochs = data['Epoch']
    train_loss = data['Train_loss']
    validation_loss = data['Validation_loss']
    validation_accuracy = data['Validation_Accuracy']
    test_accuracy = data['Test_Accuracy']

    # fig, axes = plt.subplot_mosaic(
    #         [
    #             ["Training Accuracy vs Epochs"]
    #         ],
    #         figsize=(12, 6)

    #     )
    plt.subplot(1, 2, 1)
    plt.tick_params(axis='x', labelsize=14)  # Change x-axis tick label size
    plt.tick_params(axis='y', labelsize=14)

    if lr == 0.1:
        plt.plot(epochs, test_accuracy, label='lr = 0.1', marker='o')
    elif lr == 0.01:
        plt.plot(epochs, test_accuracy, label='lr = 0.01', marker='s')
    elif lr == 0.001:
        plt.plot(epochs, test_accuracy, label='lr = 0.001', marker='^')
    elif lr == 0.0001:
        plt.plot(epochs, test_accuracy, label='lr = 0.0001', marker='d')
    elif lr == 0.00001:
        plt.plot(epochs, test_accuracy, label='lr = 0.00001', marker='x')
    
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Training Accuracy',fontsize=16)
    plt.title('Training Accuracy vs Epochs',fontsize=16)
    plt.legend(fontsize=14)


    plt.subplot(1, 2, 2)
    plt.tick_params(axis='x', labelsize=14)  # Change x-axis tick label size
    plt.tick_params(axis='y', labelsize=14)

    if lr == 0.1:
        plt.plot(epochs, validation_accuracy, label='lr = 0.1', marker='o')
    elif lr == 0.01:
        plt.plot(epochs, validation_accuracy, label='lr = 0.01', marker='s')
    elif lr == 0.001:
        plt.plot(epochs, validation_accuracy, label='lr = 0.001', marker='^')
    elif lr == 0.0001:
        plt.plot(epochs, validation_accuracy, label='lr = 0.0001', marker='d')
    elif lr == 0.00001:
        plt.plot(epochs, validation_accuracy, label='lr = 0.00001', marker='x')
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Validation Accuracy',fontsize=16)
    plt.title('Validation Accuracy vs Epochs',fontsize=16)
    plt.legend(fontsize=14)

plt.tight_layout()
plt.savefig(f"{CKPT_DIR_save}/Trainig_acc_x_validation_acc_bs{bs}.png")
plt.show()