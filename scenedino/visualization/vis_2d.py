import logging
import math
from typing import Any, Callable
from dotdict import dotdict
import ignite.distributed as idist

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torchvision.utils import make_grid
from scenedino.models.bts import BTSNet
from torchvision.utils import flow_to_image


from scenedino.visualization.common import color_tensor

# TODO: configure logger somewhere else
logger = logging.getLogger("Visualization")


def render_profile(
    model: BTSNet,
    points: torch.Tensor,
    viewdirs: torch.Tensor | None = None,
    dim: int = 1,
):
    """Render a profile of the scene.

    Args:
        model (BTSNet): model
        points (torch.Tensor): points to render in world coordinates. Shape (res_x, res_y, res_z, 3)
        viewdirs (torch.Tensor | None, optional): view directions. Defaults to None.

    Returns:
        torch.Tensor: profile of the scene
    """

    res_x, res_y, res_z = points.shape[:-1]
    device = idist.device()
    points = points.to(device).view(1, -1, 3)
    batch_size = 65536
    split_points = torch.split(points, batch_size, dim=1)
    sigmas, invalid = [], []
    for pnts in split_points:
        _, invalid_, sigmas_, _, _ = model.forward(pnts, viewdirs=viewdirs)
        invalid.append(invalid_)
        sigmas.append(sigmas_)
    sigmas = torch.cat(sigmas, dim=0)
    invalid = torch.cat(invalid, dim=0)
    sigmas[invalid.bool()] = 1.0

    sigmas = sigmas.view(res_x, res_y, res_z)
    invalid = invalid.view(res_x, res_y, res_z)

    sigmas_sum = torch.cumsum(sigmas, dim=dim)
    profile = (sigmas_sum <= 1).float().sum(dim=dim) / sigmas.shape[dim]

    return profile


def get_profiles(data) -> torch.Tensor | None:
    # TODO: check for permutation
    if "profiles" in data:
        # profiles = data["profiles"]
        profiles = torch.stack([data["profiles"]], dim=0).transpose(-1, -2)
        return color_tensor(profiles, cmap="magma", norm=True).permute(0, 3, 1, 2)
    logger.warning(
        "No profiles found in model output. Not creating a profile visualization."
    )
    return None


def get_input_imgs(data) -> torch.Tensor | None:
    if "imgs" in data:
        return torch.stack(data["imgs"], dim=1).detach()[0] * 0.5 + 0.5
    logger.warning(
        "No images found in model output. Not creating a input image visualization."
    )
    return None


def get_reconstructed_imgs(data) -> torch.Tensor | None:
    if "rgb" in data["coarse"][0] and "imgs" in data:
        images = torch.stack(data["imgs"], dim=1).detach()[0]

        _, c, h, w = images.shape
        recon_imgs = data["coarse"][0]["rgb"].detach()[0][..., :3] * .5 + .5
        nv = recon_imgs.shape[0]
        recon_imgs = recon_imgs.view(nv, h, w, -1, c)

        # Aggregate recon_imgs by taking the mean
        return recon_imgs.mean(dim=-2).permute(0, 3, 1, 2)
    logger.warning(
        "No reconstructed images found in model output. Not creating a recontructed image visualization."
    )
    return None


def get_reconstruction_rmse(data) -> torch.Tensor | None:
    if "rgb" in data["coarse"][0] and "imgs" in data:
        images = torch.stack(data["imgs"], dim=1).detach()[0]
        recon_imgs = data["coarse"][0]["rgb"].detach()[0][..., :3]

        _, c, h, w = images.shape
        nv = recon_imgs.shape[0]

        images = images * 0.5 + 0.5

        recon_imgs = recon_imgs.view(nv, h, w, -1, c)
        # Aggregate recon_imgs by taking the mean
        recon_imgs = recon_imgs.mean(dim=-2).permute(0, 3, 1, 2)

        recon_mse = (((images - recon_imgs) ** 2) / 2).mean(dim=1).clamp(0, 1)
        return color_tensor(recon_mse, cmap="plasma").permute(0, 3, 1, 2)
    logger.warning(
        "No reconstructed images found in model output. Not creating a recontructed image visualization."
    )
    return None


def get_dino_cos_sim_downsampled(data) -> torch.Tensor | None:
    if "dino_features_downsampled" in data["coarse"][0] and "imgs" in data:
        dino_gt = data["dino_gt"].detach()
        recon_dino = data["coarse"][0]["dino_features_downsampled"].detach().squeeze(-2)

        cos_sim = torch.nn.CosineSimilarity(dim=-1)(dino_gt, recon_dino).squeeze(0)
        return color_tensor(cos_sim, cmap="plasma").permute(0, 3, 1, 2)
    logger.warning(
        "No downsampled dino cos-sim found in model output. Not creating a visualization."
    )
    return None


def get_dino_gt(data) -> torch.Tensor | None:
    if "vis_dino_gt" in data:
        vis_dino_gt = data["vis_dino_gt"][0].permute(0, -1, 1, 2) / 2 + 0.5
        return torch.clamp(vis_dino_gt, min=0, max=1)
    logger.warning(
        "No dino GT found in model output. Not creating a dino GT visualization."
    )
    return None


def get_reconstructed_dino(data) -> torch.Tensor | None:
    if "vis_dino_features" in data["coarse"][0]:
        vis_dino_features = data["coarse"][0]["vis_dino_features"][0, :, :, :, 0, :].permute(0, -1, 1, 2) / 2 + 0.5
        return torch.clamp(vis_dino_features, min=0, max=1)
    logger.warning(
        "No reconstructed dino features found in model output. Not creating a reconstructed dino visualization."
    )
    return None


def get_reconstructed_dino_downsampled(data) -> torch.Tensor | None:
    if "vis_dino_features_downsampled" in data["coarse"][0]:
        vis_dino_features = data["coarse"][0]["vis_dino_features_downsampled"][0, :, :, :, 0, :].permute(0, -1, 1, 2) / 2 + 0.5
        return torch.clamp(vis_dino_features, min=0, max=1)
    logger.warning(
        "No downsampled reconstructed dino features found in model output. Not creating a reconstructed dino visualization."
    )
    return None

def get_batch_dino_gt(data) -> torch.Tensor | None:
    if "vis_batch_dino_gt" in data:
        vis_batch_dino_gt = [v[0].permute(0, -1, 1, 2) / 2 + 0.5 for v in data["vis_batch_dino_gt"]]
        return [torch.clamp(v, min=0, max=1) for v in vis_batch_dino_gt]
    logger.warning(
        "No dino GT (batch vis) found in model output. Not creating a dino GT visualization."
    )
    return None

def get_batch_dino_artifacts(data) -> torch.Tensor | None:
    if "vis_batch_dino_artifacts" in data:
        vis_batch_dino_artifacts = [v[0].permute(0, -1, 1, 2) / 2 + 0.5 for v in data["vis_batch_dino_artifacts"]]
        return [torch.clamp(v, min=0, max=1) for v in vis_batch_dino_artifacts]
    logger.warning(
        "No dino artifacts (batch vis) found in model output. Not creating a dino GT visualization."
    )
    return None

def get_batch_dino_features_kmeans(data) -> torch.Tensor | None:
    if "vis_batch_dino_features_kmeans" in data["coarse"][0]:
        vis_batch_dino_features_kmeans = data["coarse"][0]["vis_batch_dino_features_kmeans"][0].permute(0, -1, 1, 2)
        return torch.clamp(vis_batch_dino_features_kmeans, min=0, max=1)
    logger.warning(
        "No dino kmeans segmentation (batch vis) found in model output. Not creating a visualization."
    )
    return None

def get_batch_dino_gt_kmeans(data) -> torch.Tensor | None:
    if "vis_batch_dino_gt_kmeans" in data:
        vis_batch_dino_gt_kmeans = data["vis_batch_dino_gt_kmeans"][0].permute(0, -1, 1, 2)
        return torch.clamp(vis_batch_dino_gt_kmeans, min=0, max=1)
    logger.warning(
        "No dino kmeans segmentation (batch vis) found in model output. Not creating a visualization."
    )
    return None

def get_segs_gt(data) -> torch.Tensor | None:
    if "segmentation" in data:
        vis_segs_gt = data["segmentation"]["visualization"]["target"].movedim(-1, -3)
        return torch.clamp(vis_segs_gt, min=0, max=1)
    logger.warning(
        "No Segmentation target (batch vis) found in model output. Not creating a visualization."
    )
    return None

def get_segs_pred(data) -> torch.Tensor | None:
    if "segmentation" in data:
        vis_segs_pred = torch.cat(
            [data["segmentation"]["visualization"][result_name][:, 0].squeeze(-2) 
             for result_name in data["segmentation"]["visualization"] if result_name != "target"], 
            dim=0
        ).movedim(-1, -3)
        
        return torch.clamp(vis_segs_pred, min=0, max=1)
    logger.warning(
        "No Segmentation (batch vis) found in model output. Not creating a visualization."
    )
    return None

def get_batch_reconstructed_dino(data) -> torch.Tensor | None:
    if "vis_batch_dino_features" in data["coarse"][0]:
        vis_batch_dino_features = [v[0, :, :, :, 0, :].permute(0, -1, 1, 2) / 2 + 0.5 for v in
                                   data["coarse"][0]["vis_batch_dino_features"]]
        return [torch.clamp(v, min=0, max=1) for v in vis_batch_dino_features]
    logger.warning(
        "No reconstructed dino features (batch vis) found in model output. Not creating a reconstructed dino visualization."
    )
    return None


def get_batch_reconstructed_dino_downsampled(data) -> torch.Tensor | None:
    if "vis_batch_dino_features_downsampled" in data["coarse"][0]:
        vis_batch_dino_features_downsampled = [v[0, :, :, :, 0, :].permute(0, -1, 1, 2) / 2 + 0.5 for v in
                                   data["coarse"][0]["vis_batch_dino_features_downsampled"]]
        return [torch.clamp(v, min=0, max=1) for v in vis_batch_dino_features_downsampled]
    logger.warning(
        "No downsampled reconstructed dino features (batch vis) found in model output. Not creating a reconstructed dino visualization."
    )
    return None


def get_dino_downsampling_weight(data) -> torch.Tensor | None:
    if "dino_features_weight_map" in data["coarse"][0]:
        vis_dino_weight = data["coarse"][0]["dino_features_weight_map"][0, :, :, :, 0, :].permute(0, -1, 1, 2)
        return vis_dino_weight / vis_dino_weight.max()
    logger.warning(
        "No vis_dino_weight found in model output. Not creating a reconstructed dino visualization."
    )
    return None


def get_dino_downsampling_salience(data) -> torch.Tensor | None:
    if "dino_features_salience_map" in data["coarse"][0]:
        vis_dino_salience = data["coarse"][0]["dino_features_salience_map"][0, :, :, :, 0, :].permute(0, -1, 1, 2)
        return (vis_dino_salience - vis_dino_salience.min()) / (vis_dino_salience.max() - vis_dino_salience.min())
    logger.warning(
        "No vis_dino_salience found in model output. Not creating a reconstructed dino visualization."
    )
    return None


def get_dino_downsampling_per_patch_weight(data) -> torch.Tensor | None:
    if "dino_features_per_patch_weight" in data["coarse"][0]:
        vis_dino_patch_weight = data["coarse"][0]["dino_features_per_patch_weight"]
        return (vis_dino_patch_weight - vis_dino_patch_weight.min()) / (vis_dino_patch_weight.max() - vis_dino_patch_weight.min())
    logger.warning(
        "No vis_dino_patch_weight found in model output. Not creating a reconstructed dino visualization."
    )
    return None


def get_depth(data) -> torch.Tensor | None:
    if "depth" in data["coarse"][0] and "imgs" in data:
        z_near = data["z_near"]
        z_far = data["z_far"]
        recon_depth = data["coarse"][0]["depth"].detach()[0]
        recon_depth = (1 / recon_depth - 1 / z_far) / (1 / z_near - 1 / z_far)
        return color_tensor(recon_depth.squeeze(1).clamp(0, 1), cmap="plasma").permute(
            0, 3, 1, 2
        )
    logger.warning(
        "No reconstructed depth found in model output. Not creating a depth visualization."
    )
    return None


def get_depth_profile(data) -> torch.Tensor | None:
    if "alphas" in data["coarse"][0] and "imgs" in data:
        images = torch.stack(data["imgs"], dim=1).detach()[0]
        _, _, h, w = images.shape
        depth_profile = data["coarse"][0]["alphas"].detach()[0]
        depth_profile = (
            depth_profile[:, [h // 4, h // 2, 3 * h // 4], :, :]
            .view(depth_profile.shape[0] * 3, w, -1)
            .permute(0, 2, 1)
        )
        depth_profile = depth_profile.clamp_min(0) / depth_profile.max()
        return color_tensor(depth_profile, cmap="plasma").permute(0, 3, 1, 2)
    logger.warning(
        "No alphas found in model output. Not creating a depth profile visualization."
    )
    return None


def get_invalids(data) -> torch.Tensor | None:
    if "invalid" in data["coarse"][0]:
        invalids = data["coarse"][0]["invalid"].detach()[0]
        invalids = invalids
        invalids = invalids.mean(-2).mean(-1)
        return color_tensor(invalids, cmap="plasma").permute(0, 3, 1, 2)
    logger.warning(
        "No invalids found in model output. Not creating a invalid visualization."
    )
    return None


def get_ray_entropy(data) -> torch.Tensor | None:
    if "alphas" in data["coarse"][0]:
        alphas = data["coarse"][0]["alphas"].detach()[0]
        alphas += 1e-5

        ray_density = alphas / alphas.sum(dim=-1, keepdim=True)
        ray_entropy = -(ray_density * torch.log(ray_density)).sum(-1) / (
            math.log2(alphas.shape[-1])
        )
        return color_tensor(ray_entropy, cmap="plasma").permute(0, 3, 1, 2)
    logger.warning(
        "No alphas found in model output. Not creating a ray entropy visualization."
    )
    return None


def get_ray_entropy_weights(data) -> torch.Tensor | None:
    if "weights" in data["coarse"][0]:
        weights = data["coarse"][0]["weights"].detach()[0]
        weights += 1e-5

        ray_density = weights / weights.sum(dim=-1, keepdim=True)
        ray_entropy = -(ray_density * torch.log(ray_density)).sum(-1) / (
            math.log2(weights.shape[-1])
        )
        return color_tensor(ray_entropy, cmap="plasma").permute(0, 3, 1, 2)
    logger.warning(
        "No alphas found in model output. Not creating a ray entropy for weights visualization."
    )
    return None


def get_alpha_sum(data) -> torch.Tensor | None:
    if "alphas" in data["coarse"][0]:
        alphas = data["coarse"][0]["alphas"].detach()[0]
        alphas += 1e-5

        alpha_sum = (alphas.sum(dim=-1) / alphas.shape[-1]).clamp(-1)
        return color_tensor(alpha_sum, cmap="plasma").permute(0, 3, 1, 2)
    logger.warning(
        "No alphas found in model output. Not creating a alpha sum visualization."
    )
    return None


def get_uncertainty(data) -> torch.Tensor | None:
    if data["rgb_gt"].shape[-1] >= 7:
        uncert = data["rgb_gt"][0][:, :, :, 6].detach()

        return color_tensor(uncert, cmap="plasma", norm=True).permute(0, 3, 1, 2)
    elif "extras" in data["coarse"][0]:
        uncert = data["coarse"][0]["extras"][0, :, :, :, 0].detach()

        return color_tensor(uncert, cmap="plasma", norm=True).permute(0, 3, 1, 2)
    logger.warning(
        "No uncertainty found in model output. Not creating a uncertainty visualization."
    )
    return None


def get_rendered_flow(data) -> torch.Tensor | None:
    if data["coarse"][0]["rgb"].shape[-1] >= 5:
        flow = data["coarse"][0]["rgb"][0][:, :, :, 0, 3:5].detach()

        images = torch.stack(data["imgs"], dim=1).detach()[0]

        _, c, h, w = images.shape
        nv = flow.shape[0]

        flow = flow.view(nv, h, w, 2)

        flow = torch.cat((flow[:, :, :, 0:1] / 2 * w , flow[:, :, :, 1:2] / 2 * h), dim=-1).permute(0, 3, 1, 2)

        flow_imgs = []
        for i in range(nv):
            flow_imgs.append(flow_to_image(flow[i].cpu().squeeze()).float() / 255)

        flow_imgs = torch.stack(flow_imgs, dim=0)
        return flow_imgs
    logger.warning(
        "No rendered flows found in model output. Not creating a rendered_flow visualization."
    )
    return None


def get_predicted_occlusions(data) -> torch.Tensor | None:
    if data["rgb_gt"].shape[-1] >= 6:
        occs = data["rgb_gt"][0][:, :, :, 5].detach()

        return color_tensor(occs, cmap="plasma", norm=True).permute(0, 3, 1, 2)
    logger.warning(
        "No predicted occlusions found in model output. Not creating a predicted occlusions visualization."
    )
    return None


def get_depth_direct(data) -> torch.Tensor | None:
    if "depths" in data:
        depths = 1 / data["depths"][0:2, 0].detach()

        return color_tensor(depths, cmap="plasma", norm=True).permute(0, 3, 1, 2)
    logger.warning(
        "No alphas found in model output. Not creating a alpha sum visualization."
    )
    return None


def get_occlusions(data) -> torch.Tensor | None:
    if "occs_fwd" in data and "occs_bwd" in data:
        occs_fwd = data["occs_fwd"][0:1, 0].detach()
        occs_bwd = data["occs_bwd"][0:1, 0].detach()

        occs = torch.cat((occs_fwd, occs_bwd), dim=-1)

        return color_tensor(occs, cmap="plasma", norm=True).permute(0, 3, 1, 2)
    logger.warning(
        "No alphas found in model output. Not creating a alpha sum visualization."
    )
    return None


def get_flow(data) -> torch.Tensor | None:
    if "flows_fwd" in data and "occs_bwd" in data:
        flows_fwd = data["flows_fwd"][0:1].detach()
        flows_bwd = data["flows_bwd"][0:1].detach()

        flows_fwd = flow_to_image(flows_fwd.cpu().squeeze())
        flows_bwd = flow_to_image(flows_bwd.cpu().squeeze())

        flows = torch.cat((flows_fwd, flows_bwd), dim=-1)

        return flows[None, :]
    logger.warning(
        "No alphas found in model output. Not creating a alpha sum visualization."
    )
    return None


def tb_visualize(model: BTSNet, dataset, config: dict[str, Any] | None = None):
    if config is None:
        vis_fns: dict[str, Callable[[Any], torch.Tensor | None]] = {
            "input_imgs": get_input_imgs,
            "reconstructed_imgs": get_reconstructed_imgs,
            "reconstruction_rmse": get_reconstruction_rmse,
            "get_dino_gt": get_dino_gt,
            "get_reconstructed_dino": get_reconstructed_dino,
            "get_reconstructed_dino_downsampled": get_reconstructed_dino_downsampled,
            "get_batch_dino_gt": get_batch_dino_gt,
            "get_batch_reconstructed_dino": get_batch_reconstructed_dino,
            "get_batch_reconstructed_dino_downsampled": get_batch_reconstructed_dino_downsampled,
            "get_dino_downsampling_weight": get_dino_downsampling_weight,
            "get_dino_cos_sim_downsampled": get_dino_cos_sim,
            "profiles": get_profiles,
            "depth": get_depth,
            "depth_profile": get_depth_profile,
            "alpha_sum": get_alpha_sum,
            "ray_entropy": get_ray_entropy,
            "ray_entropy_weights": get_ray_entropy_weights,
            "invalids": get_invalids,
            "rendered_flow": get_rendered_flow,
            "predicted_occlusions": get_predicted_occlusions,
            "uncertainty": get_uncertainty,
            "depth_direct": get_depth_direct,
            "occlusions": get_occlusions,
            "flow": get_flow,
        }
    else:
        # TODO: inform user about not found functions
        vis_fns = {
            name: globals()[f"get_{name}"]
            for name, _ in config.items()
            if [globals().get(f"get_{name}", None)]
        }

    def _visualize(engine: Engine, tb_logger: TensorboardLogger, step: int, tag: str):
        data = engine.state.output["output"]
        if "profiles" in vis_fns.keys():
            # TODO: choose between data["poses"][0][0] and model.grid_f_poses_w2c[0]
            points = dataset.get_points(model.grid_f_poses_w2c[0])
            # points = dataset.get_points(data["poses"][0][0])
            data["profiles"] = render_profile(model, points)

        writer = tb_logger.writer
        for name, vis_fn in vis_fns.items():
            output = vis_fn(data)
            if output is not None:
                if name == "profiles":
                    grid = make_grid(output, padding=0)
                elif isinstance(output, list):
                    nrow = len(output)
                    output = torch.stack(output, dim=1).flatten(0, 1)
                    grid = make_grid(output, nrow=nrow, padding=0)
                else:
                    grid = make_grid(output, nrow=int(math.sqrt(output.shape[0])), padding=0)
                writer.add_image(f"{tag}/{name}", grid.cpu(), global_step=step)

    return _visualize
