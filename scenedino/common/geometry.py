from math import sin, cos

import torch
from torch.cuda.amp import autocast


def transform_pts(pts: torch.Tensor, rel_pose: torch.Tensor) -> torch.Tensor:
    """Transform points by relative pose

    Args:
        pts (torch.Tensor): B, n_pts, 3
        rel_pose (torch.Tensor): B, 4, 4

    Returns:
        torch.Tensor: B, n_pts, 3
    """
    pts = torch.cat((pts, torch.ones_like(pts[..., :1])), dim=-1)
    return (pts @ rel_pose.transpose(-1, -2))[..., :3]


# TODO: unify
def distance_to_z(depths: torch.Tensor, projs: torch.Tensor):
    n, nv, h, w = depths.shape
    device = depths.device

    inv_K = torch.inverse(projs)

    grid_x = (
        torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).expand(-1, -1, h, -1)
    )
    grid_y = (
        torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).expand(-1, -1, -1, w)
    )
    img_points = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2).expand(
        n, nv, -1, -1, -1
    )
    cam_points = (inv_K @ img_points.view(n, nv, 3, -1)).view(n, nv, 3, h, w)
    factors = cam_points[:, :, 2, :, :] / torch.norm(cam_points, dim=2)

    return depths * factors


def z_to_distance(z: torch.Tensor, projs: torch.Tensor):
    n, nv, h, w = z.shape
    device = z.device

    inv_K = torch.inverse(projs)

    grid_x = (
        torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).expand(-1, -1, h, -1)
    )
    grid_y = (
        torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).expand(-1, -1, -1, w)
    )
    img_points = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2).expand(
        n, nv, -1, -1, -1
    )
    cam_points = (inv_K @ img_points.view(n, nv, 3, -1)).view(n, nv, 3, h, w)
    factors = cam_points[:, :, 2, :, :] / torch.norm(cam_points, dim=2)

    return z / factors


def azimuth_elevation_to_rotation(azimuth: float, elevation: float) -> torch.Tensor:
    rot_z = torch.tensor(
        [
            [cos(azimuth), -sin(azimuth), 0.0],
            [sin(azimuth), cos(azimuth), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot_x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(azimuth), -sin(azimuth)],
            [0.0, sin(azimuth), cos(azimuth)],
        ]
    )
    return rot_x @ rot_z


def estimate_frustum_overlap(proj_source: torch.Tensor, pose_source: torch.Tensor, proj_target: torch.Tensor, pose_target: torch.Tensor, dist_lim=50):
    device = proj_source.device
    dtype = proj_source.dtype

    # Check which camera has higher z value in target coordinate system
    with autocast(enabled=False):
        src2tgt = torch.inverse(pose_target) @ pose_source

    for i in range(len(src2tgt)):
        if src2tgt[i, 2, 3] < 0:
            print("SWAP", i)
            proj_ = proj_target[i].clone()
            pose_ = pose_target[i].clone()
            proj_target[i] = proj_source[i]
            pose_target[i] = pose_source[i]
            proj_source[i] = proj_
            pose_source[i] = pose_

    points = torch.tensor([[
        [-1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, -1, 1, 1],
        [-1, -1, 1, 1],
    ]], device=device, dtype=dtype)

    with autocast(enabled=False):
        K_src_inv = torch.inverse(proj_source)
        K_tgt_inv = torch.inverse(proj_target)

    _ = K_src_inv.new_zeros(K_src_inv.shape[0], 4, 4)
    _[:, 3, 3] = 1
    _[:, :3, :3] = K_src_inv
    K_src_inv = _

    _ = K_tgt_inv.new_zeros(K_tgt_inv.shape[0], 4, 4)
    _[:, 3, 3] = 1
    _[:, :3, :3] = K_tgt_inv
    K_tgt_inv = _

    points_src = K_src_inv @ points.permute(0, 2, 1)
    points_tgt = K_tgt_inv @ points.permute(0, 2, 1)

    normals_tgt = torch.cross(points_tgt[..., :3, :], torch.roll(points_tgt[..., :3, :], shifts=-1, dims=-2), dim=-2)
    normals_tgt = normals_tgt / torch.norm(normals_tgt, dim=-2, keepdim=True)

    with autocast(enabled=False):
        src2tgt = torch.inverse(pose_target) @ pose_source

    base = src2tgt[:, :3, 3, None]
    points_src_tgt = src2tgt @ points_src

    dirs = points_src_tgt[..., :3, :] - base
    # dirs = dirs / torch.norm(dirs, dim=-2) #dirs should have z length 1

    dists = - (base[..., None] * normals_tgt[..., None, :]).sum(dim=-3) / (dirs[..., None] * normals_tgt[..., None, :]).sum(dim=-3).clamp_min(1e-4)

    # print(dists)

    # Ignore all non-positive
    mask = (dists <= 0) | (dists > dist_lim)
    dists[mask] = dist_lim

    # print(dists)

    dists = torch.min(dists, dim=-1)[0]

    mean_dist = dists.mean(dim=-1)

    # print(mean_dist, (torch.max(points_src[..., 0], dim=-1)[0] - torch.min(points_src[..., 0], dim=-1)[0]), (torch.max(points_src[..., 1], dim=-1)[0] - torch.min(points_src[..., 1], dim=-1)[0]))

    volume_estimate = \
        1/3 * \
        (torch.max(points_src[..., 0], dim=-1)[0] - torch.min(points_src[..., 0], dim=-1)[0]) * mean_dist * \
        (torch.max(points_src[..., 1], dim=-1)[0] - torch.min(points_src[..., 1], dim=-1)[0]) * mean_dist * \
        mean_dist

    return volume_estimate


def estimate_frustum_overlap_2(proj_source: torch.Tensor, pose_source: torch.Tensor, proj_target: torch.Tensor, pose_target: torch.Tensor, z_range=(3, 40), res=(8, 8, 16)):
    device = proj_source.device
    dtype = proj_source.dtype

    with autocast(enabled=False):
        K_src_inv = torch.inverse(proj_source)

    n = proj_source.shape[0]
    w, h, d = res

    pixel_width = 2 / w
    pixel_height = 2 / h

    x = torch.linspace(-1 + .5 * pixel_width, 1 - .5 * pixel_width, w, dtype=dtype, device=device).view(1, 1, 1, w).expand(n, d, h, w)
    y = torch.linspace(-1 + .5 * pixel_height, 1 - .5 * pixel_height, h, dtype=dtype, device=device).view(1, 1, h, 1).expand(n, d, h, w)
    z = torch.ones_like(x)

    xyz = torch.stack((x, y, z), dim=-1)
    xyz = K_src_inv @ xyz.reshape(n, -1, 3).permute(0, 2, 1)
    xyz = xyz.reshape(n, 3, d, h, w)

    # xyz = xyz * (1 / torch.linspace(1 / z_range[0], 1 / z_range[1], d, dtype=dtype, device=device).view(1, 1, d, 1, 1).expand(n, 1, d, h, w))
    xyz = xyz * torch.linspace(z_range[0], z_range[1], d, dtype=dtype, device=device).view(1, 1, d, 1, 1).expand(n, 1, d, h, w)

    xyz = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)

    xyz = xyz.reshape(n, 4, -1)

    with autocast(enabled=False):
        src2tgt = torch.inverse(pose_target) @ pose_source

    xyz = src2tgt @ xyz

    # print(xyz)

    xyz = proj_target @ xyz[:, :3, :]

    xyz[:, :2] = xyz[:, :2] / xyz[:, 2:3, :]

    # print(xyz)

    valid = (xyz[:, 0].abs() < 1) & (xyz[:, 1].abs() < 1) & (xyz[:, 2].abs() > z_range[0])# & (xyz[:, 2].abs() < z_range[1])

    # print(valid)

    volume_estimate = valid.to(dtype).mean(-1)

    return volume_estimate


def compute_occlusions(flow0, flow1):
    n, _, h, w = flow0.shape
    device = flow0.device
    x = torch.linspace(-1, 1, w, device=device).view(1, 1, w).expand(1, h, w)
    y = torch.linspace(-1, 1, h, device=device).view(1, h, 1).expand(1, h, w)
    xy = torch.cat((x, y), dim=0).view(1, 2, h, w).expand(n, 2, h, w)
    flow0_r = torch.cat((flow0[:, 0:1, :, :] * 2 / w , flow0[:, 1:2, :, :] * 2 / h), dim=1)
    flow1_r = torch.cat((flow1[:, 0:1, :, :] * 2 / w , flow1[:, 1:2, :, :] * 2 / h), dim=1)

    xy_0 = xy + flow0_r
    xy_1 = xy + flow1_r

    xy_0 = xy_0.view(n, 2, -1)
    xy_1 = xy_1.view(n, 2, -1)

    ns = torch.arange(n, device=device, dtype=xy_0.dtype)
    nxy_0 = torch.cat((ns.view(n, 1, 1).expand(-1, 1, xy_0.shape[-1]), xy_0), dim=1)
    nxy_1 = torch.cat((ns.view(n, 1, 1).expand(-1, 1, xy_1.shape[-1]), xy_1), dim=1)

    mask0 = torch.zeros_like(flow0[:, :1, :, :])
    mask0[nxy_1[:, 0, :].long(), 0, ((nxy_1[:, 2, :] * .5 + .5) * h).round().long().clamp(0, h-1), ((nxy_1[:, 1, :] * .5 + .5) * w).round().long().clamp(0, w-1)] = 1

    mask1 = torch.zeros_like(flow1[:, :1, :, :])
    mask1[nxy_0[:, 0, :].long(), 0, ((nxy_0[:, 2, :] * .5 + .5) * h).round().long().clamp(0, h-1), ((nxy_0[:, 1, :] * .5 + .5) * w).round().long().clamp(0, w-1)] = 1

    return mask0, mask1