from abc import ABC, abstractmethod

import torch


class BaseLoss(ABC):
    def __init__(self, config) -> None:
        super().__init__()

    @abstractmethod
    def get_loss_metric_names(self) -> list[str]:
        ...

    @abstractmethod
    def __call__(self, data) -> dict[str, torch.Tensor]:
        ...
