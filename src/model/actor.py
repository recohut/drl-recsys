# mport tensorflow as tf
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, embedding_dim, srm_size, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * srm_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),
        )

        self.initialize()

    def initialize(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, state):
        return self.layers(state)


class Actor(object):
    def __init__(
        self,
        embedding_dim,
        srm_size,
        hidden_dim,
        learning_rate,
        state_size,
        tau,
        device,
    ):

        self.embedding_dim = embedding_dim
        self.srm_size = srm_size
        self.state_size = state_size
        self.network = ActorNetwork(embedding_dim, srm_size, hidden_dim).to(device)
        self.target_network = ActorNetwork(embedding_dim, srm_size, hidden_dim).to(
            device
        )

        self.optimizer = torch.optim.Adam(self.network.parameters(), learning_rate)
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
