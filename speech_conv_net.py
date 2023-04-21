# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from NeuralNets import MediumweightCNN

def main(name, isNorm):
    """
    Runs the program.

    :param name: The name of the datafile
    :type name: string
    :param isNorm: If the data is normal or not
    :type isNorm: boolean
    :return: None
    """
    if isNorm:
        epochs, train_losses, val_losses, train_accs, val_accs = run_normal(name)
    else:
        epochs, train_losses, val_losses, train_accs, val_accs = run_augmen(name)

    plot(epochs, train_losses, val_losses, train_accs, val_accs)

def run_augmen(name):
    """
    The main function if we're using the augmented dataset.

    :param name: The name of the datafile
    :type name: string
    :return: epochs, train_losses, val_losses, train_accs, and val_accs
    :rtype: ndarrays
    """

    torch.manual_seed(432)
    train_X, train_y, val_X, val_y = load_data_augmen(name)
    return make_model(train_X, train_y, val_X, val_y, False, 16, 100)

def run_normal(name):
    """
    The main function if we're using the normal dataset.

    :param name: The name of the datafile
    :type name: string
    :return: epochs, train_losses, val_losses, train_accs, and val_accs
    :rtype: ndarrays
    """

    torch.manual_seed(432)
    train_X, train_y, val_X, val_y = load_data_norm(name)
    return make_model(train_X, train_y, val_X, val_y, True, 32, 5000)

def load_data_augmen(name):
    """
    Loads data for an augmented dataset.

    :param name: The name of the datafile
    :type name: string
    :return: The training data and labels and the validation data and labels
    :rtype: ndarrays
    """

    # Load the data from the npz files
    data = np.load(name) #'log_mel_spectro_data_min_max_norm_augmented_v1.npz'
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

    return train_X, train_y, val_X, val_y

def load_data_norm(name):
    """
    Loads data for an normal dataset.

    :param name: The name of the datafile
    :type name: string
    :return: The training data and labels and the validation data and labels
    :rtype: ndarrays
    """

    # Load the data from the npz files
    data = np.load(name) #'mel_spectro_data_min_max_norm_augmented.npz'
    X = data['X']
    y = data['y']

    # Print the shape of the data
    print(X.shape)
    print(y.shape)

        
    # Convert the numpy arrays to torch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    # Split the dataset into training, and validation sets
    train_split = 0.90
    train_size = int(train_split * len(X))
    val_size = len(X) - train_size
    train_indices, val_indices = torch.utils.data.random_split(torch.arange(len(X)), [train_size, val_size])

    # Create the training and validation sets
    train_X = X[train_indices.indices]
    train_y = y[train_indices.indices]

    val_X = X[val_indices.indices]
    val_y = y[val_indices.indices]

    return train_X, train_y, val_X, val_y

def make_model(train_X, train_y, val_X, val_y, isNorm, batch_size, num_epochs): #epochs: 5000 norm, 100 augmen # batch size 32 norm, 16 augmen
    """
    Trains and validates the model.

    :param train_X: The training data
    :type train_X: ndarray
    :param train_y: The training labels
    :type train_y: ndarray
    :param val_X: The validation data
    :type val_X: ndarray
    :param val_y: The validation labels
    :type val_y: ndarray
    :param isNorm: If the data is normal or not
    :type isNorm: boolean
    :param batch_size: The size of the batch
    :type batch_size: int
    :param num_epochs: The number of epochs
    :type num_epochs: int
    :return: epochs, train_losses, val_losses, train_accs, and val_accs
    :rtype: ndarrays
    """
    
    # Create a neural network
    model = MediumweightCNN(use_maxpool=isNorm)

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

    if isNorm:
        # TODO: set optimizer to the normal
        print()
    else:
        #set optimizer to the augmen
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # ptimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.01, weight_decay=1e-5)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)

    # Print the number of training and validation examples
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Initialize lists to store loss and accuracy for each epoch
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

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

        print()

        # Increment epoch counter
        epochs += 1

        return epochs, train_losses, val_losses, train_accs, val_accs

def plot(epochs, train_losses, val_losses, train_accs, val_accs):
    """
    Creates the plots

    :param epochs: The epochs
    :type epochs: ndarray
    :param train_losses: The train losses
    :type train_losses: ndarray
    :param val_losses: the validation losses
    :type val_losses: ndarray
    :param train_accs: The training accuracies
    :type train_accs: ndarray
    :param val_accs: The validation accuracies
    :type val_accs: ndarray
    :return: None
    """
    
    # Plot loss and accuracy curves for training, validation, and testing sets
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    # plt.plot(range(epochs), test_losses, label='Testing Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    # Save the loss curves to the TensorBoard directory
    # plt.savefig(os.path.join(new_path, 'loss_curves.png'))
    plt.show()

    plt.plot(range(epochs), train_accs, label='Training Accuracy')
    plt.plot(range(epochs), val_accs, label='Validation Accuracy')
    # plt.plot(range(epochs), test_accs, label='Testing Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()

    plt.show()