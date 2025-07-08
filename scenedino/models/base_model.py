from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, xyz: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        pass
