from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv,GINConv,ChebConv
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from monai.networks.nets import SEresnet50



class Seresnet50_Contrastive(SEresnet50):
    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.flatten(x)
        feature = F.normalize(self.head(x), dim=1)

        x = self.last_linear(x.detach())

        return x, feature

    def flatten(self, x: torch.Tensor):
        x = self.adaptive_avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = torch.flatten(x, 1)

        return x
    def __init__(self,head,feat_dim,**kwargs):
        super().__init__(**kwargs)
        # dim_in is set to 2048
        if head == 'linear':
            self.head = nn.Linear(2048, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, feat_dim)
            )
