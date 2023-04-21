# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from NeuralNets import Net, MediumweightCNN, ConvNeuralNet, MediumweightCNN_MFCC, MediumweightCNN_padded

from torch.optim.lr_scheduler import ExponentialLR

torch.manual_seed(432)

# Load the data from the npz files
data = np.load('log_mel_spectro_data_min_max_norm_augmented_v2.npz')
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']

# Print the shape of the data
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
    
# Convert the numpy arrays to torch tensors
train_X = torch.from_numpy(X_train).float()
train_y = torch.from_numpy(y_train).float()
val_X = torch.from_numpy(X_val).float()
val_y = torch.from_numpy(y_val).float()

# Create a neural network
model = MediumweightCNN(use_maxpool=True)

# Check to see what device is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using the Apple MPS backend")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using the CUDA backend")
else:
    device = torch.device("cpu")
    print("Using the CPU backend")

# Send the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=4e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.09, weight_decay=1e-5)

# Define the ExponentialLR scheduler
use_scheduler = True

if use_scheduler:
    gamma = 0.78
    scheduler = ExponentialLR(optimizer, gamma=gamma, verbose=True)

# Create the datasets
train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
val_dataset = torch.utils.data.TensorDataset(val_X, val_y)

# Print the number of training and validation examples
print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of validation examples: {len(val_dataset)}")

# Create the data loaders
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize lists to store loss and accuracy for each epoch
train_losses = []
train_accs = []
val_losses = []
val_accs = []

# Set the number of epochs
num_epochs = 30

# Increment epoch counter
epochs = 0

ncols_pbar = 125

# Train the model

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    # Customize tqdm progress bar
    pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training", ncols=ncols_pbar, unit="batch", ascii=True)

    for i, data in enumerate(pbar_train):

        # Send the data to mps
        images, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = (outputs > 0.5).float()

        
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Update tqdm progress bar with additional information
        pbar_train.set_postfix(loss=train_loss/(i+1), accuracy=100*correct_train/total_train)
        
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    train_acc = 100 * correct_train / total_train
    train_accs.append(train_acc)
    
    # Validate
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Validating", ncols=ncols_pbar, unit="batch", ascii=True)
        for i, data in enumerate(pbar_val):

            # Send the data to mps
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()

            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            pbar_val.set_postfix(loss=val_loss/(i+1), accuracy=100*correct_val/total_val)

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_acc = 100 * correct_val / total_val
    val_accs.append(val_acc)

    
    # Print statistics
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%'.format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

    # Decay the learning rate
    if use_scheduler:
        scheduler.step()

    print()

    # Increment epoch counter
    epochs += 1


# Plot loss and accuracy curves for training, validation, and testing sets
plt.plot(range(epochs), train_losses, label='Training Loss')
plt.plot(range(epochs), val_losses, label='Validation Loss')
# plt.plot(range(epochs), test_losses, label='Testing Loss')
plt.legend()
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()

# Save the loss curves to a file
plt.savefig('loss_curves.png')
plt.show()

plt.plot(range(epochs), train_accs, label='Training Accuracy')
plt.plot(range(epochs), val_accs, label='Validation Accuracy')
# plt.plot(range(epochs), test_accs, label='Testing Accuracy')
plt.legend()
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()

# Save the accuracy curves to a file
plt.savefig('accuracy_curves.png')

plt.show()

# Make a single plit with both accuracy and loss curves on it as subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(range(epochs), train_losses, label='Training Loss')
ax1.plot(range(epochs), val_losses, label='Validation Loss')

ax1.legend()
ax1.set_title('Loss Curves')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid()

ax2.plot(range(epochs), train_accs, label='Training Accuracy')
ax2.plot(range(epochs), val_accs, label='Validation Accuracy')

ax2.legend()
ax2.set_title('Accuracy Curves')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.grid()

# Also save this plot to a file
plt.savefig('loss_accuracy_curves.png')
plt.show()




