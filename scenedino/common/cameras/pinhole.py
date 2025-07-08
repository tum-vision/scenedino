import torch

EPS = 1e-3


def normalize_calib(K: torch.Tensor, img_sizes: torch.Tensor) -> torch.Tensor:
    """Normalize the calibration matrices for fisheye cameras based on the image size

    Args:
        calib (torch.Tensor): B, n_views, 3, 3
        img_sizes (torch.Tensor): B, n_views, 2

    Returns:
        torch.Tensor: B, n_views 7
    """

    K[..., :2, :] = K[..., :2, :] / img_sizes.unsqueeze(-1) * 2.0
    K[..., :2, 2] = K[..., :2, 2] - 1.0

    return K


def unnormalize_calib(K: torch.Tensor, img_sizes: torch.Tensor) -> torch.Tensor:
    """Unnormalize the calibration matrices for fisheye cameras based on the image size

    Args:
        calib (torch.Tensor): B, n_views, 3, 3
        img_sizes (torch.Tensor): B, n_views, 2

    Returns:
        torch.Tensor: B, n_views 7
    """

    K[..., :2, 2] = K[..., :2, 2] + 1.0
    K[..., :2, :] = K[..., :2, :] * img_sizes.unsqueeze(-1) / 2.0

    return K


def pts_into_camera(pts: torch.Tensor, poses_w2c: torch.Tensor) -> torch.Tensor:
    """Project points from world coordinates into camera coordinate

    Args:
        pts (torch.Tensor): B, n_pts, 3
        poses_w2c (torch.Tensor): B, n_view, 4, 4

    Returns:
        torch.Tensor: B, n_views, n_pts, 3
    """

    # Add a singleton dimension to the input point cloud to match grid_f_poses_w2c shape
    pts = pts.unsqueeze(1)  # [B, 1, n_pts, 3]
    ones = torch.ones_like(
        pts[..., :1]
    )  ## Create a tensor of ones to add a fourth dimension to the point cloud for homogeneous coordinates
    pts = torch.cat(
        (pts, ones), dim=-1
    )  ## Concatenate the tensor of ones with the point cloud to create homogeneous coordinates
    return (poses_w2c[:, :, :3, :]) @ pts.permute(0, 1, 3, 2)


def project_to_image(
    pts: torch.Tensor, Ks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project pts in camera coordinates into image coordinates.

    Args:
        pts (torch.Tensor): B, n_views, n_pts, 3
        Ks (torch.Tensor): B, n_views, 3, 3

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (B, n_views, n_pts, 2), (B, n_views, n_pts, 1)
    """
    pts = (Ks @ pts).permute(
        0, 1, 3, 2
    )  ## Apply the intrinsic camera parameters to the projected points to get pixel coordinates
    xy = pts[
        :, :, :, :2
    ]  ## Extract the x,y coordinates and depth value from the projected points
    z_ = pts[:, :, :, 2:3]

    xy = xy / z_.clamp_min(EPS)

    return xy, z_


def outside_frustum(
    xy: torch.Tensor,
    z: torch.Tensor,
    limits_x: tuple[float, float] | tuple[int, int] = (-1.0, 1.0),
    limits_y: tuple[float, float] | tuple[int, int] = (-1.0, 1.0),
    limit_z: float = EPS,
) -> torch.Tensor:
    """_summary_

    Args:
        xy (torch.Tensor): _description_
        z (torch.Tensor): _description_
        limits_x (tuple[float, float] | tuple[int, int], optional): _description_. Defaults to (-1.0, 1.0).
        limits_y (tuple[float, float] | tuple[int, int], optional): _description_. Defaults to (-1.0, 1.0).
        limit_z (float, optional): _description_. Defaults to EPS.

    Returns:
        torch.Tensor: _description_
    """
    return (
        (z <= limit_z)
        | (xy[..., :1] < limits_x[0])
        | (xy[..., :1] > limits_x[1])
        | (xy[..., 1:2] < limits_y[0])
        | (xy[..., 1:2] > limits_y[1])
    )
