# Our model
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(0.4) # remember to use model.eval() at inference
        self.fc1 = nn.Linear(3*150*150, 15)

    def forward(self, x):
        x = x.view(-1, 3*150*150)
        return self.fc1(x)