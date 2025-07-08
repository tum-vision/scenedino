from .base_loss import BaseLoss
from .reconstruction_loss import ReconstructionLoss
from .stego_loss import StegoLoss


def make_loss(config) -> BaseLoss:
    loss_type = config["type"]
    match loss_type:
        case "reconstruction":
            return ReconstructionLoss(config)
        case "stego":
            return StegoLoss(config)
        case _:
            raise ValueError(f"Unknown loss type {loss_type}")
