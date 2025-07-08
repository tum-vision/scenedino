import torch
from torch import nn
import torch.nn.functional as F


class NoDimReduction(nn.Module):
    def __init__(self, full_channels, reduced_channels):
        super().__init__()
        assert full_channels == reduced_channels

    def forward(self, features):
        return features


class MlpDimReduction(nn.Module):
    def __init__(self, full_channels, reduced_channels, latent_channels):
        super().__init__()
        self.linear_in = nn.Linear(reduced_channels, latent_channels)
        self.linear_out = nn.Linear(latent_channels, full_channels)
        self.relu = nn.ReLU()

    def transform_expand(self, features):
        latent = self.relu(self.linear_in(features))
        output = self.linear_out(latent)
        return F.normalize(output, dim=-1)


class OrthogonalLinearDimReduction(nn.Module):
    def __init__(self, full_channels, reduced_channels):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(full_channels))
        self.weights = torch.nn.Parameter(torch.eye(full_channels, reduced_channels))

    def transform_expand(self, features):
        output = features @ self.weights.transpose(0, 1) + self.bias
        return F.normalize(output, dim=-1)
