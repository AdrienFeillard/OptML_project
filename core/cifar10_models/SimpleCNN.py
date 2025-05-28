import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32->16
        x = self.pool(F.relu(self.conv2(x)))  # 16->8
        x = self.pool(F.relu(self.conv3(x)))  # 8->4

        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def simple_cnn(num_classes=10, **kwargs):
    return SimpleCNN(num_classes=num_classes, **kwargs)

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)   # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 16x16 -> 16x16
        self.pool = nn.MaxPool2d(2, 2)                # réduit taille par 2

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32->16
        x = self.pool(F.relu(self.conv2(x)))  # 16->8
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def tiny_cnn(num_classes=10, **kwargs):
    return TinyCNN(num_classes=num_classes, **kwargs)


class BabyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BabyCNN, self).__init__()
        # Juste une seule couche conv toute simple
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)  # 3x32x32 -> 8x32x32
        self.pool = nn.MaxPool2d(2, 2)  # 8x32x32 -> 8x16x16

        # Couche fully connected minuscule
        self.fc1 = nn.Linear(8 * 16 * 16, num_classes)  # Directement 10 sorties

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> batch x 8 x 16 x 16
        x = x.view(x.size(0), -1)  # aplatissement
        x = self.fc1(x)  # sortie brute (logits)
        return x


def baby_cnn(num_classes=10, **kwargs):
    return BabyCNN(num_classes=num_classes, **kwargs)