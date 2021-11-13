import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    def __init__(self, embedding_dim, srm_size, hidden_dim):
        super(CriticNetwork, self).__init__()

        self.act = nn.ReLU()
        self.fc1 = nn.Linear(embedding_dim * srm_size, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

        self.initialize()

    def initialize(self):
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight)
        nn.init.kaiming_uniform_(self.out.weight)

    def forward(self, x):
        s = self.act(self.fc1(x[1]))
        s = torch.cat([x[0], s], 1)
        s = self.act(self.fc2(s))
        s = self.act(self.fc3(s))
        s = self.act(self.fc4(s))
        s = self.out(s)
        return s


class Critic(object):
    def __init__(self, hidden_dim, learning_rate, embedding_dim, srm_size, tau, device):

        self.embedding_dim = embedding_dim
        self.srm_size = srm_size
        self.network = CriticNetwork(embedding_dim, srm_size, hidden_dim).to(device)
        self.target_network = CriticNetwork(embedding_dim, srm_size, hidden_dim).to(
            device
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), learning_rate)
        self.loss = nn.MSELoss()
        self.tau = tau

        self.update_target_network()

    def update_target_network(self):
        for target_param, param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save_weights(self, path):
        torch.save(self.network.state_dict(), path)

    def load_weights(self, path):
        self.network.load_state_dict(torch.load(path))
