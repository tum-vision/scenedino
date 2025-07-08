import math
import torch
import torch.nn.functional as F

# TODO: check if the functions can be moved somewhere else
from scenedino.common.util import kl_div, normalized_entropy
from scenedino.models.prediction_heads.layers import ssim, geo


# TODO: have two signatures with override. One for mask, one without mask
# NOTE: what is the purpose of the mask. Ask Felix
def compute_l1ssim(
    img0: torch.Tensor, img1: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:  ## (img0 == pred, img1 == GT)
    """Calculate the L1-SSIM error between two images. Use a mask if provided to ignore certain pixels.

    Args:
        img0 (torch.Tensor): torch.Tensor of shape (B, c, h, w) containing the predicted images.
        img1 (torch.Tensor): torch.Tensor of shape (B, c, h, w) containing the ground truth images.
        mask (torch.Tensor | None, optional): torch.Tensor of shape (B, h, w). Defaults to None.

    Returns:
        torch.Tensor: per patch error of shape (B, h, w)
    """
    errors = 0.85 * torch.mean(
        ssim(img0, img1, pad_reflection=False, gaussian_average=True, comp_mode=True),
        dim=1,
    ) + 0.15 * torch.mean(torch.abs(img0 - img1), dim=1)
    # checking if a mask is provided. If a mask is provided, it is returned along with the errors. Otherwise, only the errors are returned.
    # if mask is not None:
    #     return (
    #         errors,
    #         mask,
    #     )
    return errors  # (B, h, w)


def compute_normalized_l1(
    flow0: torch.Tensor, flow1: torch.Tensor) -> torch.Tensor:

    errors = (flow0 - flow1).abs() / (flow0.detach().norm(dim=1, keepdim=True) + 1e-4)

    return errors


# TODO: integrate the mask
def compute_edge_aware_smoothness(
    gt_img: torch.Tensor, input: torch.Tensor, mask: torch.Tensor | None = None, temperature: int = 1
) -> torch.Tensor:
    """Compute the edge aware smoothness loss of the depth prediction based on the gradient of the original image.

    Args:
        gt_img (torch.Tensor): ground truth images of shape (B, c, h, w)
        input (torch.Tensor): predicted tensor of shape (B, c, h, w)
        mask (torch.Tensor | None, optional): Not used yet. Defaults to None.

    Returns:
        torch.Tensor: per pixel edge aware smoothness loss of shape (B, h, w)
    """
    _, _, h, w = gt_img.shape

    # TODO: check whether interpolation is necessary
    # gt_img = F.interpolate(gt_img, (h, w))

    input_dx = torch.mean(
        torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:]), 1, keepdim=True
    )  # (B, 1, h, w-1)
    input_dy = torch.mean(
        torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]), 1, keepdim=True
    )  # (B, 1, h-1, w)

    i_dx = torch.mean(
        torch.abs(gt_img[:, :, :, :-1] - gt_img[:, :, :, 1:]), 1, keepdim=True
    )  # (B, 1, h, w-1)
    i_dy = torch.mean(
        torch.abs(gt_img[:, :, :-1, :] - gt_img[:, :, 1:, :]), 1, keepdim=True
    )  # (B, 1, h-1, w)

    input_dx *= torch.exp(-temperature * i_dx)  # (B, 1, h, w-1)
    input_dy *= torch.exp(-temperature * i_dy)  # (B, 1, h-1, w)

    errors = F.pad(input_dx, pad=(0, 1), mode="constant", value=0) + F.pad(
        input_dy, pad=(0, 0, 0, 1), mode="constant", value=0
    )  # (B, 1, h, w)
    return errors[:, 0, :, :]  # (B, h, w)


def compute_3d_smoothness(
    feature_sample: torch.Tensor, sigma_sample: torch.Tensor
) -> torch.Tensor:

    return torch.var(feature_sample, dim=2)


def compute_occupancy_error(
    teacher_field: torch.Tensor,
    student_field: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the distillation error between the teacher and student density.

    Args:
        teacher_density (torch.Tensor): teacher occpancy map of shape (B)
        student_density (torch.Tensor): student occupancy map of shape (B)
        mask (torch.Tensor | None, optional): Mask indicating bad occpancy values for student or teacher, e.g. invalid occupancies due to out of frustum. Defaults to None.

    Returns:
        torch.Tensor: distillation error of shape (B)
    """
    if mask is not None:
        teacher_field = teacher_field[mask]
        student_field = student_field[mask]

    return torch.nn.MSELoss(reduction="mean")(teacher_field, student_field)  # (1)


def depth_regularization(depth: torch.Tensor) -> torch.Tensor:
    """Compute the depth regularization loss.

    Args:
        depth (torch.Tensor): depth map of shape (B, 1, h, w)

    Returns:
        torch.Tensor: depth regularization loss of shape (B)
    """
    depth_grad_x = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    depth_grad_y = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    depth_reg_loss = (depth_grad_x**2).mean() + (depth_grad_y**2).mean()

    return depth_reg_loss


def alpha_regularization(
    alphas: torch.Tensor, invalids: torch.Tensor | None = None
) -> torch.Tensor:
    # TODO: make configurable
    alpha_reg_fraction = 1 / 8
    alpha_reg_reduction = "ray"
    """Compute the alpha regularization loss.

    Args:
        alphas (torch.Tensor): alpha map of shape (B, 1, h, w)
        invalids (torch.Tensor | None, optional): Mask indicating bad alpha values, e.g. invalid alpha due to out of frustum. Defaults to None.

    Returns:
        torch.Tensor: alpha regularization loss of shape (B)
    """
    n_smps = alphas.shape[-1]

    alpha_sum = alphas[..., :-1].sum(-1)
    min_cap = torch.ones_like(alpha_sum) * (n_smps * alpha_reg_fraction)

    if invalids is not None:
        alpha_sum = alpha_sum * (1 - invalids.squeeze(-1).to(torch.float32))
        min_cap = min_cap * (1 - invalids.squeeze(-1).to(torch.float32))

    match alpha_reg_reduction:
        case "ray":
            alpha_reg_loss = (alpha_sum - min_cap).clamp_min(0)
        case "slice":
            alpha_reg_loss = (alpha_sum.sum(dim=-1) - min_cap.sum(dim=-1)).clamp_min(
                0
            ) / alpha_sum.shape[-1]
        case _:
            raise ValueError(f"Invalid alpha_reg_reduction: {alpha_reg_reduction}")

    return alpha_reg_loss


def surfaceness_regularization(
    alphas: torch.Tensor, invalids: torch.Tensor | None = None
) -> torch.Tensor:
    p = -torch.log(torch.exp(-alphas.abs()) + torch.exp(-(1 - alphas).abs()))
    p = p.mean(-1)

    if invalids is not None:
        p = p * (1 - invalids.squeeze(-1).to(torch.float32))

    surfaceness_reg_loss = p.mean()
    return surfaceness_reg_loss


def depth_smoothness_regularization(depths: torch.Tensor) -> torch.Tensor:
    depth_smoothness_loss = ((depths[..., :-1, :] - depths[..., 1:, :]) ** 2).mean() + (
        (depths[..., :, :-1] - depths[..., :, 1:]) ** 2
    ).mean()

    return depth_smoothness_loss


def sdf_eikonal_regularization(sdf: torch.Tensor) -> torch.Tensor:
    grad_x = sdf[:, :1, :-1, :-1, 1:] - sdf[:, :1, :-1, :-1, :-1]
    grad_y = sdf[:, :1, :-1, 1:, :-1] - sdf[:, :1, :-1, :-1, :-1]
    grad_z = sdf[:, :1, 1:, :-1, :-1] - sdf[:, :1, :-1, :-1, :-1]
    grad = (torch.cat((grad_x, grad_y, grad_z), dim=1) ** 2).sum(dim=1) ** 0.5

    eikonal_loss = ((grad - 1) ** 2).mean(dim=(1, 2, 3))

    return eikonal_loss


def weight_entropy_regularization(
    weights: torch.Tensor, invalids: torch.Tensor | None = None
) -> torch.Tensor:
    ignore_last = False

    weights = weights.clone()

    if ignore_last:
        weights = weights[..., :-1]
        weights = weights / weights.sum(dim=-1, keepdim=True)

    H_max = math.log2(weights.shape[-1])

    # x log2 (x) -> 0 . Therefore, we can set log2 (x) to 0 if x is small enough.
    # This should ensure numerical stability.
    weights_too_small = weights < 2 ** (-16)
    weights[weights_too_small] = 2
    
    wlw = torch.log2(weights) * weights

    wlw[weights_too_small] = 0

    # This is the formula for the normalised entropy
    entropy = -wlw.sum(-1) / H_max
    return entropy


def max_alpha_regularization(alphas: torch.Tensor, invalids: torch.Tensor | None = None):
    alphas_max = alphas[..., :-1].max(dim=-1)[0]
    alphas_reg = (1 - alphas_max).clamp(0, 1).mean()
    return alphas_reg


def max_alpha_inputframe_regularization(alphas: torch.Tensor, ray_info, invalids: torch.Tensor | None = None):
    mask = ray_info[..., 0] == 0
    alphas_max = alphas.max(dim=-1)[0]
    alphas_reg = ((1 - alphas_max).clamp(0, 1) * mask.to(alphas_max.dtype)).mean()
    return alphas_reg


def epipolar_line_regularization(data, rgb_gt, scale):
    rgb = data["coarse"][scale]["rgb"]
    rgb_samps = data["coarse"][scale]["rgb_samps"]

    b, pc, h, w, n_samps, nv, c = rgb_samps.shape

    rgb_gt = data["rgb_gt"].unsqueeze(-2).expand(rgb.shape)

    alphas = data["coarse"][scale]["alphas"]

    # TODO


def density_grid_regularization(density_grid, threshold):
    density_grid = (density_grid.abs() - threshold).clamp_min(0)

    # Attempt to make it more numerically stable
    max_v = density_grid.max().clamp_min(1).detach()

    # print(max_v.item())

    error = (((density_grid / max_v)).mean() * max_v)

    error = torch.nan_to_num(error, 0, 0, 0)

    # Black magic to prevent error massages from anomaly detection when using AMP
    if torch.all(error == 0):
        error = error.detach()

    return error


def kl_prop(weights):
    entropy = normalized_entropy(weights.detach())

    kl_prop = entropy[..., 1:-1, 1:-1] * (entropy[..., 1:-1, 1:-1] - entropy[..., 2:, 1:-1]).clamp_min(0) * kl_div(weights[..., 2:, 1:-1, :].detach(), weights[..., 1:-1, 1:-1, :])
    kl_prop += entropy[..., 1:-1, 1:-1] * (entropy[..., 1:-1, 1:-1] - entropy[..., 0:-2, 1:-1]).clamp_min(0) * kl_div(weights[..., 0:-2, 1:-1, :].detach(), weights[..., 1:-1, 1:-1, :])
    kl_prop += entropy[..., 1:-1, 1:-1] * (entropy[..., 1:-1, 1:-1] - entropy[..., 1:-1, 2:]).clamp_min(0) * kl_div(weights[..., 1:-1, 2:, :].detach(), weights[..., 1:-1, 1:-1, :])
    kl_prop += entropy[..., 1:-1, 1:-1] * (entropy[..., 1:-1, 1:-1] - entropy[..., 1:-1, 0:-2]).clamp_min(0) * kl_div(weights[..., 1:-1, :-2, :].detach(), weights[..., 1:-1, 1:-1, :])

    return kl_prop.mean()


def alpha_consistency(alphas, invalids, consistency_policy):
    invalids = torch.all(invalids < .5, dim=-1)

    if consistency_policy == "max":
        target = torch.max(alphas, dim=-1, keepdim=True)[0].detach()
    elif consistency_policy == "min":
        target = torch.max(alphas, dim=-1, keepdim=True)[0].detach()
    elif consistency_policy == "median":
        target = torch.median(alphas, dim=-1, keepdim=True)[0].detach()
    elif consistency_policy == "mean":
        target = torch.mean(alphas, dim=-1, keepdim=True).detach()
    else:
        raise NotImplementedError

    diff = (alphas - target).abs().mean(dim=-1)

    invalids = invalids.to(diff.dtype)

    diff = (diff * invalids)

    return diff.mean()


def alpha_consistency_uncert(alphas, invalids, uncert):
    invalids = torch.all(invalids < .5, dim=-1)
    alphas = alphas.detach()
    nf = alphas.shape[-1]

    alphas_median = torch.median(alphas, dim=-1, keepdim=True)[0].detach()

    target = (alphas - alphas_median).abs().mean(dim=-1) * (nf / (nf-1))

    diff = (uncert[..., None] - target).abs()

    invalids = invalids.to(diff.dtype)

    diff = (diff * invalids)

    return diff.mean()


def entropy_based_smoothness(weights, depth, invalids=None):
    entropy = normalized_entropy(weights.detach())

    error_fn = lambda d0, d1: (d0 - d1.detach()).abs()

    if invalids is None:
        invalids = torch.zeros_like(depth)

    # up
    kl_prop_up = entropy[..., :-1, :] * (entropy[..., :-1, :] - entropy[..., 1:, :]).clamp_min(0) * error_fn(depth[..., :-1, :], depth[..., 1:, :]) * (1 - invalids[..., :-1, :])
    # down
    kl_prop_down = entropy[..., 1:, :] * (entropy[..., 1:, :] - entropy[..., :-1, :]).clamp_min(0) * error_fn(depth[..., 1:, :], depth[..., :-1, :]) * (1 - invalids[..., 1:, :])
    # left
    kl_prop_left = entropy[..., :, :-1] * (entropy[..., :, :-1] - entropy[..., :, 1:]).clamp_min(0) * error_fn(depth[..., :, :-1], depth[..., :, 1:]) * (1 - invalids[..., :, :-1])
    # right
    kl_prop_right = entropy[..., :, 1:] * (entropy[..., :, 1:] - entropy[..., :, :-1]).clamp_min(0) * error_fn(depth[..., :, 1:], depth[..., :, :-1]) * (1 - invalids[..., :, 1:])

    kl_prop = kl_prop_up.mean() + kl_prop_down.mean() + kl_prop_left.mean() + kl_prop_right.mean()

    return kl_prop.mean()


def flow_regularization(flow, gt_flow, invalids=None):
    flow_reg = (flow[..., 0, :] - gt_flow).abs().mean(dim=-1, keepdim=True)
    
    if invalids is not None:
        flow_reg = flow_reg * (1 - invalids)

    return flow_reg.mean()
