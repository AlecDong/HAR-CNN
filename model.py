# Our model
import torch.nn as nn
import torchvision.models
alexnet = torchvision.models.alexnet(pretrained=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Transfer learning from alexnet
        self.alexnet = alexnet.features
        self.fc1 = nn.Linear(256*3*3, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 15)

    def forward(self, x):
        x = self.alexnet(x).detach()
        x = x.view(-1, 256*3*3)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)