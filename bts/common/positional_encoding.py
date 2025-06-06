from typing import Callable
from numpy import pi
import torch
import torch.nn as nn
import numpy as np
import torch.autograd.profiler as profiler


# TODO: rethink encoding mode
def encoding_mode(
    encoding_mode: str, d_min: float, d_max: float, inv_z: bool, EPS: float
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def _z(xy: torch.Tensor, z: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        if inv_z:
            z = (1 / z.clamp_min(EPS) - 1 / d_max) / (1 / d_min - 1 / d_max)
        else:
            z = (z - d_min) / (d_max - d_min)
        z = 2 * z - 1
        return torch.cat(
            (xy, z), dim=-1
        )  ## concatenates the normalized x, y, and z coordinates

    def _distance(xy: torch.Tensor, z: torch.Tensor, distance: torch.Tensor):
        if inv_z:
            distance = (1 / distance.clamp_min(EPS) - 1 / d_max) / (
                1 / d_min - 1 / d_max
            )
        else:
            distance = (distance - d_min) / (d_max - d_min)
        distance = 2 * distance - 1
        return torch.cat(
            (xy, distance), dim=-1
        )  ## Apply the positional encoder to the concatenated xy and depth/distance coordinates (it enables the model to capture more complex spatial dependencies without a significant increase in model complexity or training data)

    match encoding_mode:
        case "z":
            return _z
        case "distance":
            return _distance
        case _:
            return _z


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1)
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get("num_freqs", 6),
            d_in,
            conf.get("freq_factor", np.pi),
            conf.get("include_input", True),
        )


def token_decoding(filter: nn.Module, pos_offset: float = 0.0):
    def _decode(xyz: torch.Tensor, tokens: torch.Tensor):
        """Decode tokens into density for given points

        Args:
            x (torch.Tensor): points in xyz n_pts, 3
            tokens (torch.Tensor): tokens n_pts, n_tokens, d_in + 2
        """
        n_pts, n_tokens = tokens.shape

        with profiler.record_function("positional_enc"):
            z = xyz[..., 3]
            scale = tokens[..., 0]  # n_pts, n_tokens
            token_pos_offset = tokens[..., 1]  # n_pts, n_tokens
            weights = tokens[..., 2:]  # n_pts, n_tokens, d_in
            positions = (
                2.0
                * (z.unsqueeze(1).unsqueeze(2).repeat(1, n_tokens) - token_pos_offset)
                / scale
                - 1.0
            )  # n_pts, n_tokens ((z - t_o) / s) * 2.0 - 1.0  t_o => -1.0 t_o + s => 1.0

            individual_densities = filter(positions, weights)  # n_pts, n_tokens

            densities = individual_densities.sum(-1)  # n_pts

            return densities

    return _decode


class FourierFilter(nn.Module):
    # TODO: add filter functions
    def __init__(
        self,
        num_freqs=6,
        d_in=3,
        freq_factor=np.pi,
        include_input=True,
        filter_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))
        self.filter_fn = filter_fn

    def forward(self, positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Predict density for given normalized points using Fourier features

        Args:
            positions (torch.Tensor): normalized positions between -1 and 1, (n_pts, n_tokens)
            weights (torch.Tensor): weights for each point (n_pts, n_tokens, num_freqs * 2)

        Returns:
            torch.Tensor: aggregated density for each point (n_pts)
        """
        with profiler.record_function("positional_enc"):
            positions = positions.unsqueeze(1).repeat(
                1, self.num_freqs * 2, 1
            )  # n_pts, num_freqs * 2, n_tokens
            densities = weights.permute(0, 2, 1) * torch.sin(
                torch.addcmul(self._phases, positions, self._freqs)
            )  # n_pts, num_freqs * 2, n_tokens

            if self.filter_fn is not None:
                densities = self.filter_fn(densities, positions)

            return densities.sum(-2)  # n_pts, n_tokens

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get("num_freqs", 6),
            d_in,
            conf.get("freq_factor", np.pi),
        )


class LogisticFilter(nn.Module):
    def __init__(self, slope: float) -> None:
        super().__init__()
        self.slope = slope

    def forward(self, positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Predict the density as sum of weighted logistic functions

        Args:
            positions (torch.Tensor): normalized positions between -1 and 1, (n_pts, n_tokens)
            weights (torch.Tensor): weights for each point (n_pts, n_tokens, d_in)

        Returns:
            torch.Tensor: density for each point (n_pts, n_tokens)
        """
        with profiler.record_function("positional_enc"):
            weights = weights.squeeze(-1)  # n_pts, n_tokens

            sigmoid_pos = self.slope * positions + 1.0
            return (
                weights * torch.sigmoid(sigmoid_pos) * torch.sigmoid(-sigmoid_pos)
            )  # n_pts, n_tokens

    @classmethod
    def from_conf(cls, conf):
        # PyHocon construction
        return cls(conf.get("slope", 10.0))
