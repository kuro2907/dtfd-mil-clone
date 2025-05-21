import os
import torch
import torch.nn as nn


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, c_dim):
        super(residual_block, self).__init__()
        self.linear1 = nn.Linear(c_dim, c_dim, bias=False)
        self.linear2 = nn.Linear(c_dim, c_dim, bias=False)
        # Make sure ReLUs aren't in-place
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        # Use non-in-place operations
        identity = x.clone()
        out = self.relu1(self.linear1(x))
        out = self.linear2(out)
        out = out + identity  # Not in-place addition
        out = self.relu2(out)
        return out


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        # Change inplace=True to inplace=False to avoid breaking computation graph
        self.relu1 = nn.ReLU(inplace=False)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):
        # Create new tensor for each operation
        out = self.fc1(x.clone())
        out = self.relu1(out)  # Already using inplace=False

        if self.numRes > 0:
            out = self.resBlocks(out)

        return out
