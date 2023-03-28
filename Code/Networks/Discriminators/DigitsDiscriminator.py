import torch.nn as nn

class DigitsDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(500, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x