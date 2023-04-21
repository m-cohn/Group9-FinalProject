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
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5), stride=2, padding=(1, 2))
        if self.use_maxpool:
            self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 4), stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), stride=2, padding=(1, 2))
        if self.use_maxpool:
            self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 4), stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=2, padding=(1, 2))
        if self.use_maxpool:
            self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 4), stride=2)

        num_nodes = 2560 if self.use_maxpool else 182272
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

    
class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(322944, 1024)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)

        return out.squeeze()
