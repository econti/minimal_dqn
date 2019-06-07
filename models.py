#!/usr/bin/env python3

import torch
import torch.nn as nn


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, bayesian_dqn):
        """
        If `bayesian_dqn=True` remove last linear layer and use model as a feature
        extractor for Bayesian Linear Regression.
        """
        super(CnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(self.feature_size(), 512), nn.ReLU())
        if not bayesian_dqn:
            self.fc.add_module("final_fc", nn.Linear(512, self.num_actions))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, bayesian_dqn):
        """
        If `bayesian_dqn=True` remove last linear layer and use model as a feature
        extractor for Bayesian Linear Regression.
        """
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.layers = nn.Sequential(nn.Linear(input_shape[0], 64), nn.ReLU())
        if not bayesian_dqn:
            self.fc.add_module("final_fc", nn.Linear(64, self.num_actions))

    def forward(self, x):
        return self.layers(x)
