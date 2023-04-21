import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size=13):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x.squeeze()
        
    
class MediumweightCNN(nn.Module):
    def __init__(self, use_maxpool=True):
        super(MediumweightCNN, self).__init__()
        self.use_maxpool = use_maxpool
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 9), stride=2, padding=(1, 2))
        if self.use_maxpool:
            self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 4), stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 7), stride=2, padding=(1, 2))
        if self.use_maxpool:
            self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 4), stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=2, padding=(1, 2))
        if self.use_maxpool:
            self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 4), stride=2)

        num_nodes = 5120 if self.use_maxpool else 182272
        self.fc1 = nn.Linear(num_nodes, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.use_maxpool:
            x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        if self.use_maxpool:
            x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        if self.use_maxpool:
            x = self.maxpool3(x)

        # Reshape for fully connected layer
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x.squeeze()
    
