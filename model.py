import torch
import torch.nn as nn
'''
created Simple neural Net
'''
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
