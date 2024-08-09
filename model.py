import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


# 实例化放入 SAGE+MLP 组合模型
class SageConv(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()
        self.proj = nn.Linear(in_features * 2, out_features, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.normal_(self.proj.weight)

        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, features, adj):
        """
        Args:
            adj: can be sparse or dense matrix.
        """

        if adj.layout != torch.sparse_coo:  # adj.layout: torch.strided
            if len(adj.shape) == 3:
                neigh_feature = torch.bmm(adj, features) / (
                        adj.sum(dim=1).reshape(
                            (adj.shape[0], adj.shape[1], -1)) + 1)
            else:
                neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        else:
            neigh_feature = torch.spmm(adj, features) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)

        # perform conv
        data = torch.cat([features, neigh_feature], dim=-1)
        combined = self.proj(data)
        return combined


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, out_dim):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim2, out_dim),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Sage_En(nn.Module):
    def __init__(self, nfeat, nembed, dropout, input_dim, hidden_dim1, hidden_dim2, out_dim):
        super(Sage_En, self).__init__()
        self.sage1 = SageConv(nfeat, nembed)  # 聚合图中节点的邻居特征
        self.mlp = MLP(input_dim, hidden_dim1, hidden_dim2, out_dim)  # 学习节点特征到类别标签的映射
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        # print(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        preds_probability = F.softmax(x, dim=1)

        return preds_probability


