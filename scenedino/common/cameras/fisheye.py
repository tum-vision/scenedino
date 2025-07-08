import torch


# def project_to_image(
#     pts: torch.Tensor, calib: dict[str, Any]
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """Project pts in camera coordinates into image coordinates.

#     Args:
#         pts (torch.Tensor): B, n_views, n_pts, 3
#         Ks (torch.Tensor): B, n_views, 3, 3

#     Returns:
#         tuple[torch.Tensor, torch.Tensor]: (B, n_views, n_pts, 2), (B, n_views, n_pts, 1)
#     """
#     pts = pts / torch.norm(pts, dim=-1, keepdim=True)
#     x = pts[:, 0]
#     y = pts[:, 1]
#     z = pts[:, 2]

#     xi_src = calib["mirror_parameters"]["xi"]
#     x = x / (z + xi_src)
#     y = y / (z + xi_src)

#     k1 = calib["distortion_parameters"]["k1"]
#     k2 = calib["distortion_parameters"]["k2"]

#     r = x * x + y * y
#     factor = 1 + k1 * r + k2 * r * r
#     x = x * factor
#     y = y * factor

#     gamma0 = calib["projection_parameters"]["gamma1"]
#     gamma1 = calib["projection_parameters"]["gamma2"]
#     u0 = calib["projection_parameters"]["u0"]
#     v0 = calib["projection_parameters"]["v0"]

#     x = x * gamma0 + u0
#     y = y * gamma1 + v0

#     return torch.stack([x, y], dim=-1), z
# TODO: lookup
EPS = 1.0e-6


def normalize_calib(calib: torch.Tensor, img_sizes: torch.Tensor) -> torch.Tensor:
    """Normalize the calibration matrices for fisheye cameras based on the image size

    Args:
        calib (torch.Tensor): B, n_views, 7, [xi, k1, k2, gamma1, gamma2, u0, v0]
        img_sizes (torch.Tensor): B, n_views, 2

    Returns:
        torch.Tensor: B, n_views 7
    """

    calib[..., 3:5] = calib[..., 3:5] / img_sizes * 2.0
    calib[..., 5:7] = calib[..., 5:7] / img_sizes * 2.0 - 1.0

    return calib


def unnormalize_calib(calib: torch.Tensor, img_sizes: torch.Tensor) -> torch.Tensor:
    """Unnormalize the calibration matrices for fisheye cameras based on the image size

    Args:
        calib (torch.Tensor): B, n_views, 7, [xi, k1, k2, gamma1, gamma2, u0, v0]
        img_sizes (torch.Tensor): B, n_views, 2

    Returns:
        torch.Tensor: B, n_views 7
    """

    calib[..., 3:5] = calib[..., 3:5] * img_sizes / 2.0
    calib[..., 5:7] = (calib[..., 5:7] + 1.0) * img_sizes / 2.0

    return calib


def project_to_image(
    pts: torch.Tensor, calib: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project pts in camera coordinates into image coordinates.

    Args:
        pts (torch.Tensor): B, n_views, n_pts, 3
        Ks (torch.Tensor): B, n_views, 7

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (B, n_views, n_pts, 2), (B, n_views, n_pts, 1)
    """
    pts = pts / torch.norm(pts, dim=-1, keepdim=True)
    xy = pts[..., 0:2]
    z = pts[..., 2:3]

    xi_src = calib[..., 0:1].unsqueeze(-2)  # B, n_views, 1, 1

    xy = xy / (z + xi_src)

    r = torch.sum(torch.square(xy), dim=-1)
    factor = 1 + calib[..., 1:2] * r + calib[..., 2:3] * torch.square(r)

    xy = xy * factor.unsqueeze(-1)

    xy = xy * calib[..., 3:5].unsqueeze(-2) + calib[..., 5:7].unsqueeze(-2)

    return xy, z


# TODO: define for fisheye
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
