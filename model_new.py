# Our model
import torch.nn as nn
import torchvision.models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Transfer learning from alexnet
        self.fc1 = nn.Linear(256*6*6, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 15)

    def forward(self, x):
        x = self.dropout(x).detach()
        x = x.view(-1, 256*6*6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)