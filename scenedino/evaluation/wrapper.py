from typing import Callable
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import ignite.distributed as idst

from scenedino.common.geometry import distance_to_z
import scenedino.common.metrics as metrics


def create_depth_eval(
    model: nn.Module,
    scaling_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    | None = None,
):
    def _compute_depth_metrics(
        data,
        # TODO: maybe integrate model
        # model: nn.Module,
    ):
        return metrics.compute_depth_metrics(
            data["depths"][0], data["coarse"][0]["depth"][:, :1], scaling_function
        )

    return _compute_depth_metrics


def create_nvs_eval(model: nn.Module):
    lpips_fn = lpips.LPIPS().to(idst.device())

    def _compute_nvs_metrics(
        data,
        # model: nn.Module,
    ):
        return metrics.compute_nvs_metrics(data, lpips_fn)

    return _compute_nvs_metrics


def create_dino_eval(model: nn.Module):
    def _compute_dino_metrics(
        data,
    ):
        return metrics.compute_dino_metrics(data)

    return _compute_dino_metrics


def create_seg_eval(model: nn.Module, n_classes: int, gt_classes: int):
    def _compute_seg_metrics(
        data,
    ):
        return metrics.compute_seg_metrics(data, n_classes, gt_classes)  # Why is this necessary?

    return _compute_seg_metrics


def create_stego_eval(model: nn.Module):
    def _compute_stego_metrics(
        data,
    ):
        return metrics.compute_stego_metrics(data)  # Why is this necessary?

    return _compute_stego_metrics


# code for saving voxel grid
# def pack(uncompressed):
#     """convert a boolean array into a bitwise array."""
#     uncompressed_r = uncompressed.reshape(-1, 8)
#     compressed = uncompressed_r.dot(
#         1 << np.arange(uncompressed_r.shape[-1] - 1, -1, -1)
#     )
#     return compressed

# if self.save_bin_path:
#     # base_file = "/storage/user/hank/methods_test/semantic-kitti-api/bts_test/sequences/00/voxels"
#     outside_frustum = (
#         (
#             (cam_pts[:, 0] < -1.0)
#             | (cam_pts[:, 0] > 1.0)
#             | (cam_pts[:, 1] < -1.0)
#             | (cam_pts[:, 0] > 1.0)
#         )
#         .reshape(q_pts_shape)
#         .permute(1, 2, 0)
#         .detach()
#         .cpu()
#         .numpy()
#     )
#     is_occupied_numpy = (
#         is_occupied_pred.reshape(q_pts_shape)
#         .permute(1, 2, 0)
#         .detach()
#         .cpu()
#         .numpy()
#         .astype(np.float32)
#     )
#     is_occupied_numpy[outside_frustum] = 0.0
#     ## carving out the invisible regions out of view-frustum
#     # for i_ in range(
#     #     (is_occupied_numpy.shape[0]) // 2
#     # ):  ## left | right half of the space
#     #     for j_ in range(i_ + 1):
#     #         is_occupied_numpy[i_, j_] = 0

#     pack(np.flip(is_occupied_numpy, (0, 1, 2)).reshape(-1)).astype(
#         np.uint8
#     ).tofile(
#         # f"{base_file}/{self.counter:0>6}.bin"
#         f"{self.save_bin_path}/{self.counter:0>6}.bin"
#     )
#     # for idx_i, image in enumerate(images[0]):
#     #     torchvision.utils.save_image(
#     #         image, f"{self.save_bin_path}/{self.counter:0>6}_{idx_i}.png"
#     #     )


def project_into_cam(pts, proj, pose):
    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)
    cam_pts = (proj @ (torch.inverse(pose).squeeze()[:3, :] @ pts.T)).T
    cam_pts[:, :2] /= cam_pts[:, 2:3]
    dist = cam_pts[:, 2]
    return cam_pts, dist


def create_occ_eval(
    model: nn.Module,
    occ_threshold: float,
    query_batch_size: int,
):
    # TODO: deal with other models such as IBRnet

    def _compute_occ_metrics(
        data,
    ):
        projs = torch.stack(data["projs"], dim=1)
        images = torch.stack(data["imgs"], dim=1)
        _, _, _, h, w = images.shape
        poses = torch.stack(data["poses"], dim=1)
        device = poses.device
        # TODO: get occ points and occupation from dataset
        occ_pts = data["occ_pts"].permute(0, 2, 1, 3).contiguous()
        occ_pts = occ_pts.to(device).view(-1, 3)

        pred_depth = distance_to_z(data["coarse"]["depth"], projs[:1, :1])

        # is visible? Check whether point is closer than the computed pseudo depth
        cam_pts, dists = project_into_cam(occ_pts, projs[0, 0], poses[0, 0])
        pred_dist = F.grid_sample(
            pred_depth.view(1, 1, h, w),
            cam_pts[:, :2].view(1, 1, -1, 2),
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        ).view(-1)
        is_visible_pred = dists <= pred_dist

        depth_plus4meters = False
        if depth_plus4meters:
            mask = (dists >= pred_dist) & (dists < pred_dist + 4)
            densities = torch.zeros_like(occ_pts[..., 0])
            densities[mask] = 1.0
            is_occupied_pred = densities > occ_threshold
        else:
            # Query the density of the query points from the density field
            densities = []
            for i_from in range(0, len(occ_pts), query_batch_size):
                i_to = min(i_from + query_batch_size, len(occ_pts))
                q_pts_ = occ_pts[i_from:i_to]
                _, _, densities_, _ = model(
                    q_pts_.unsqueeze(0), only_density=True
                )  ## ! occupancy estimation
                densities.append(densities_.squeeze(0))
            densities = torch.cat(densities, dim=0).squeeze()
            is_occupied_pred = densities > occ_threshold

        is_occupied = data["is_occupied"]
        is_visible = data["is_visible"]

        return metrics.compute_occ_metrics(is_occupied_pred, is_occupied, is_visible)

    return _compute_occ_metrics


def make_eval_fn(
    model: nn.Module,
    conf,
):
    eval_type = conf["type"]
    eval_fn = globals().get(f"create_{eval_type}_eval", None)
    if eval_fn:
        if conf.get("args", None):
            return eval_fn(model, **conf["args"])
        else:
            return eval_fn(model)
    else:
        return None
