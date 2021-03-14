
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.emb = nn.Embedding(500, 4)
        self.l1 = nn.Linear(4, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, outputs)

    def forward(self, x):
        x = F.relu(self.l1(self.emb(x)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
        # return x.view(x.size(0), -1)