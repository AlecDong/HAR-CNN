# Our model
import torch.nn as nn
import torchvision.models

class CNN2(nn.Module):
    def __init__(self, fc1_out = 128, fc2_out = 32, dropout = 0.3):
        super(CNN2, self).__init__()
        # Transfer learning from alexnet
        self.fc1 = nn.Linear(256*6*6, fc1_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, 15)

    def forward(self, x):
        x = self.dropout(x)
        x = x.view(-1, 256*6*6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)