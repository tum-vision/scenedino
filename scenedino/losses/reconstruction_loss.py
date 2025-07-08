import math
from typing import Any, Callable, Protocol

import torch
import kornia
from torch import profiler

import torch.nn.functional as F

from scenedino.common.util import kl_div, normalized_entropy
from scenedino.losses.base_loss import BaseLoss
from scenedino.common.errors import (
    alpha_consistency_uncert,
    compute_l1ssim,
    compute_edge_aware_smoothness,
    compute_3d_smoothness,
    compute_normalized_l1,
    depth_smoothness_regularization,
    depth_regularization,
    alpha_regularization,
    flow_regularization,
    kl_prop,
    max_alpha_inputframe_regularization,
    surfaceness_regularization,
    sdf_eikonal_regularization,
    weight_entropy_regularization,
    max_alpha_regularization,
    density_grid_regularization,
    alpha_consistency,
    entropy_based_smoothness,
)


EPS = 1e-5


# TODO: need wrappers around the different losses as an interface to the data variable
def make_reconstruction_error(
    criterion: str,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    match criterion:
        case "l1":
            return lambda a, b: torch.mean(torch.nn.L1Loss(reduction="none")(a, b), dim=1)
        case "l1+ssim":
            return compute_l1ssim
        case "l2":
            return lambda a, b: torch.mean(torch.nn.MSELoss(reduction="none")(a, b) / 2, dim=1)
        case "cosine":
            return lambda a, b: 1 - torch.nn.CosineSimilarity(dim=1)(a, b)
        case _:
            raise ValueError(f"Unknown reconstruction error: {criterion}")


def make_regularization(
    config, ignore_invalid: bool
) -> Callable[[Any, int], torch.Tensor]:
    """Make a regularization function from the config.

    Args:
        config (dict): config dict

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: regularization function
    """
    match config["type"]:
        case "edge_aware_smoothness":

            def _wrapper(data, scale):
                gt_img = data["rgb_gt"][..., :3]
                depth = data["coarse"][scale]["depth"].permute(1, 0, 2, 3)
                _, _, h, w = depth.shape
                gt_img = (
                    gt_img.unsqueeze(-2).permute(0, 1, 4, 5, 2, 3).reshape(-1, 3, h, w)
                )
                depth_input = 1 / depth.reshape(-1, 1, h, w).clamp(1e-3, 80)
                depth_input = depth_input / torch.mean(depth_input, dim=[2, 3], keepdim=True)

                return compute_edge_aware_smoothness(
                    gt_img, depth_input, temperature=1
                ).mean()

            return _wrapper

        case "dino_edge_aware_smoothness":

            def _wrapper(data, scale):
                gt_img = data["rgb_gt"][..., :3]
                dino = data["coarse"][scale]["dino_features"]

                _, _, h, w, _, c_dino = dino.shape
                gt_img = gt_img.unsqueeze(-2).permute(0, 1, 4, 5, 2, 3).reshape(-1, 3, h, w)
                dino_input = dino.permute(0, 1, 4, 5, 2, 3).reshape(-1, c_dino, h, w)

                return compute_edge_aware_smoothness(
                    gt_img, dino_input, temperature=25
                ).mean()

            return _wrapper
            
        case _:
            raise ValueError(f"Unknown regularization type: {config['type']}")


class PolicyCallable(Protocol):
    def __call__(self, invalids: torch.Tensor, **kwargs) -> torch.Tensor:
        ...


def strict_policy(invalids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    invalid = torch.all(torch.any(invalids > 0.5, dim=-2), dim=-1).unsqueeze(-1)
    return invalid


def weight_guided_policy(invalids: torch.Tensor, **kwargs) -> torch.Tensor:
    weights = kwargs["weights"]
    invalid = torch.all(
        (invalids.to(weights.dtype) * weights.unsqueeze(-1)).sum(-2) > 0.9,
        dim=-1,
        keepdim=True,
    )
    return invalid


def occ_and_weight_guided_policy(invalids: torch.Tensor, **kwargs) -> torch.Tensor:
    weight_guided_invalid = weight_guided_policy(invalids, **kwargs)

    # occs = 1 indicates that there can be a valid reprojection. Therefore, we have to negate it
    occ = kwargs["occ"]

    invalid = weight_guided_invalid | (~(occ.to(kwargs["weights"].dtype) > 0.5))

    return invalid


def weight_guided_diverse_policy(invalids: torch.Tensor, **kwargs) -> torch.Tensor:
    rgb_samps = kwargs["rgb_samps"]
    ray_std = torch.std(rgb_samps, dim=-3).mean(-1)
    weights = kwargs["weights"]
    invalid = torch.all(
        ((invalids.to(torch.float32) * weights.unsqueeze(-1)).sum(-2) > 0.9)
        | (ray_std < 0.01),
        dim=-1,
        keepdim=True,
    )
    return invalid


def no_policy(invalids: torch.Tensor, **kwargs) -> torch.Tensor:
    invalid = torch.zeros_like(
        torch.all(torch.any(invalids > 0.5, dim=-2), dim=-1).unsqueeze(-1),
        dtype=torch.bool,
    )
    return invalid


def invalid_policy(
    invalid_policy: str,
) -> PolicyCallable:
    match invalid_policy:
        case "strict":
            return strict_policy
        case "weight_guided":
            return weight_guided_policy
        case "weight_guided_diverse":
            return weight_guided_diverse_policy
        case "occ_weight_guided":
            return occ_and_weight_guided_policy
        case None | "none":
            return no_policy
        case _:
            raise ValueError(f"Unknown invalid policy: {invalid_policy}")


# TODO: scale all of them with a lambda factor
class ReconstructionLoss(BaseLoss):
    def __init__(self, config, use_automasking: bool = False) -> None:
        super().__init__(config)
        if config.get("fine", None) is None:
            self.rgb_fine_crit = None
        else:
            self.rgb_fine_crit = make_reconstruction_error(
                config["fine"].get("criterion", "l2")
            )
            self.dino_fine_crit = make_reconstruction_error(
                config["fine"].get("dino_criterion", "l2")
            )
            self.lambda_fine = config["fine"].get("lambda", 1)
        if config.get("coarse", None) is None:
            self.rgb_coarse_crit = None
        else:
            self.rgb_coarse_crit = make_reconstruction_error(
                config["coarse"].get("criterion", "l2")
            )
            self.dino_coarse_crit = make_reconstruction_error(
                config["coarse"].get("dino_criterion", "l2")
            )
            self.lambda_coarse = config["coarse"].get("lambda", 1)
        self.invalid_policy = invalid_policy(config.get("invalid_policy", "strict"))
        self.ignore_invalid = self.invalid_policy is not no_policy

        self.regularizations: list[tuple] = []
        for regularization_config in config["regularizations"]:
            reg_fn = make_regularization(regularization_config, self.ignore_invalid)
            self.regularizations.append(
                (regularization_config["type"], reg_fn, regularization_config["lambda"])
            )

        self.median_thresholding = config.get("median_thresholding", False)

        self.reconstruct_dino = config.get("reconstruct_dino", False)
        self.lambda_dino_coarse = config.get("lambda_dino_coarse", 1)
        self.lambda_dino_fine = config.get("lambda_dino_fine", 1)
        self.temperature_dino = config.get("temperature_dino", 1)

    def get_loss_metric_names(self) -> list[str]:
        loss_metric_names = ["rec_loss"]
        if self.rgb_fine_crit is not None:
            loss_metric_names.append("loss_rgb_fine")
            if self.reconstruct_dino:
                loss_metric_names.append("loss_dino_fine")
        if self.rgb_coarse_crit is not None:
            loss_metric_names.append("loss_rgb_coarse")
            if self.reconstruct_dino:
                loss_metric_names.append("loss_dino_coarse")
        for regularization in self.regularizations:
            loss_metric_names.append(regularization[0])
        return loss_metric_names

    def __call__(self, data) -> dict[str, torch.Tensor]:
        # print(data["dino_gt"].shape)
        # print(data["coarse"][0]["dino_features"].shape)
        with profiler.record_function("loss_computation"):
            n_scales = len(data["coarse"])

            if self.rgb_coarse_crit is not None:
                invalid_coarse = self.invalid_policy(
                    data["coarse"][0]["invalid"],
                    weights=data["coarse"][0]["weights"],
                    # rgb_samps=data["coarse"][0]["rgb_samps"],
                )
                loss_device = invalid_coarse.device

            if self.rgb_fine_crit is not None:
                invalid_fine = self.invalid_policy(
                    data["fine"][0]["invalid"],
                    weights=data["fine"][0]["weights"],
                    # rgb_samps=data["fine"][0]["rgb_samps"],
                )
                loss_device = invalid_fine.device

            losses = {
                name: torch.tensor(0.0, device=loss_device)
                for name in self.get_loss_metric_names()
            }

            for scale in range(n_scales):
                if self.rgb_coarse_crit is not None:
                    coarse = data["coarse"][scale]
                    rgb_coarse = coarse["rgb"]
                    if "dino_features_downsampled" in coarse:
                        dino_coarse = coarse["dino_features_downsampled"]
                    else:
                        dino_coarse = coarse["dino_features"]

                if self.rgb_fine_crit is not None:
                    fine = data["fine"][scale]
                    rgb_fine = fine["rgb"]
                    if "dino_features_downsampled" in fine:
                        dino_fine = fine["dino_features_downsampled"]
                    else:
                        dino_fine = fine["dino_features"]

                if "dino_artifacts" in data:
                    dino_artifacts = data["dino_artifacts"].unsqueeze(-2).expand(dino_coarse.shape)
                    dino_coarse = dino_coarse + dino_artifacts

                rgb_gt = data["rgb_gt"].unsqueeze(-2).expand(rgb_coarse.shape)
                dino_gt = data["dino_gt"].unsqueeze(-2).expand(dino_coarse.shape)

                def rgb_loss(pred, gt, invalid, criterion):
                    # TODO: move the reshaping and selection to the wrapper, maybe other functions as well
                    b, pc, h, w, num_views, channels = pred.shape
                    loss = (
                        criterion(
                            pred.permute(0, 1, 4, 5, 2, 3).reshape(-1, channels, h, w),
                            gt.permute(0, 1, 4, 5, 2, 3).reshape(-1, channels, h, w),
                        )
                        .view(b, pc, num_views, h, w)
                        .permute(0, 1, 3, 4, 2)
                        .unsqueeze(-1)
                    )
                    loss = loss.amin(-2)

                    if self.ignore_invalid and invalid is not None:
                        loss = loss * (1 - invalid.to(torch.float32))

                    if self.median_thresholding:
                        threshold = torch.median(loss.view(b, -1), dim=-1)[0].view(
                            -1, 1, 1, 1, 1
                        )
                        loss = loss[loss <= threshold]

                    return loss.mean()

                def dino_loss(pred, gt, invalid, criterion):
                    # TODO: move the reshaping and selection to the wrapper, maybe other functions as well
                    channels = pred.shape[-1]
                    loss = (
                        criterion(
                            pred.reshape(-1, channels),
                            gt.reshape(-1, channels),
                        )
                    )
                    # TODO: invalid feature handling
                    return loss.nanmean()

                if self.rgb_coarse_crit is not None:
                    loss_coarse = rgb_loss(
                        rgb_coarse, rgb_gt, invalid_coarse, self.rgb_coarse_crit
                    )
                    losses["loss_rgb_coarse"] += loss_coarse.item()
                    losses["rec_loss"] += loss_coarse * self.lambda_coarse

                    if self.reconstruct_dino:
                        loss_coarse = dino_loss(
                            self.temperature_dino * dino_coarse, self.temperature_dino * dino_gt,
                            None, self.dino_coarse_crit
                        )
                        losses["loss_dino_coarse"] += loss_coarse.item()
                        losses["rec_loss"] += loss_coarse * self.lambda_coarse * self.lambda_dino_coarse

                if self.rgb_fine_crit is not None:
                    loss_fine = rgb_loss(
                        rgb_fine, rgb_gt, invalid_fine, self.rgb_fine_crit
                    )
                    losses["loss_rgb_fine"] += loss_fine.item()
                    losses["rec_loss"] += loss_fine * self.lambda_fine

                    if self.reconstruct_dino:
                        loss_fine = dino_loss(
                            dino_fine, dino_gt, invalid_fine.unsqueeze(-1), self.dino_fine_crit
                        )
                        losses["loss_dino_fine"] += loss_fine.item()
                        losses["rec_loss"] += loss_fine * self.lambda_fine * self.lambda_dino_fine

                for regularization in self.regularizations:
                    # TODO: make it properly work with the different scales
                    reg_loss = regularization[1](data, scale)

                    if reg_loss:
                        losses[regularization[0]] += reg_loss.item()
                        losses["rec_loss"] += reg_loss * regularization[2]

            losses = {name: value / n_scales for name, value in losses.items()}

            return losses
