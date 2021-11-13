# import tensorflow as tf
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(
            -1, step_dim
        )

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class DRRAveStateRepresentation(nn.Module):
    def __init__(self, embedding_dim, n_groups=None):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.drr_ave = torch.nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)

        self.initialize()

    def initialize(self):
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()

    def forward(self, x):
        drr_ave = self.drr_ave(x[1]).squeeze(1)
        output = torch.cat(
            (
                x[0],
                x[0] * drr_ave,
                drr_ave,
            ),
            1,
        )
        return output


class FairRecStateRepresentation(nn.Module):
    def __init__(self, embedding_dim, n_groups):
        super(FairRecStateRepresentation, self).__init__()
        self.act = nn.ReLU()

        self.embedding_dim = embedding_dim

        self.fav = nn.Linear(n_groups, embedding_dim)

        self.attention_layer = Attention(embedding_dim, 5)

    def forward(self, x):
        group_mean = []
        for group in x[1]:
            group_mean.append(torch.mean(group / self.embedding_dim, axis=0))
        group_mean = torch.stack(group_mean)

        items = torch.add(x[0], group_mean).squeeze()
        ups = self.attention_layer(items)
        fs = self.act(self.fav(x[2]))

        return torch.cat((ups, fs), 1)
