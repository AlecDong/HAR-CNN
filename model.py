# Our model
import torch.nn as nn
import torchvision.models
alexnet = torchvision.models.alexnet(pretrained=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Transfer learning from alexnet
        self.alexnet = alexnet.features
        self.fc1 = nn.Linear(256*6*6, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 15)

    def forward(self, x):
        x = self.dropout(self.alexnet(x)).detach()
        x = x.view(-1, 256*6*6)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)