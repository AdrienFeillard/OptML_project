import torch
import torch.nn as nn
import torch.nn.functional as F

# Overfitting model - DeepCNN
class DeepCNN(nn.Module):
    def __init__(self, num_classes=10): # 'use_regularization' removed
        super(DeepCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(-1, 512 * 2 * 2)
        x = F.relu(self.fc1(x))
        # Dropout removed
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def deep_cnn(num_classes=10, **kwargs): # 'use_regularization' removed
    return DeepCNN(num_classes=num_classes, **kwargs)

# Medium model - ModerateCNN
class ModerateCNN(nn.Module):
    def __init__(self, num_classes=10): # 'use_regularization' removed
        super(ModerateCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        # Dropout removed
        x = self.fc2(x)
        return x

def moderate_cnn(num_classes=10, **kwargs): # 'use_regularization' removed
    return ModerateCNN(num_classes=num_classes, **kwargs)

# Underfitting model - MiniCNN
class MiniCNN(nn.Module):
    def __init__(self, num_classes=10): # 'use_regularization' removed
        super(MiniCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(8 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        # Dropout removed
        x = self.fc1(x)
        return x

def mini_cnn(num_classes=10, **kwargs): # 'use_regularization' removed
    return MiniCNN(num_classes=num_classes, **kwargs)