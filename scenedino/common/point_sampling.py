from typing import Iterable
import torch

from scenedino.common.geometry import transform_pts


# TODO: cam_incl_adjust is not nesserarily needed in this function
def regular_grid(
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    z_range: tuple[int, int],
    x_res: int,
    y_res: int,
    z_res: int,
    cam_incl_adjust: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a regular grid of points.

    Args:
        x_range (tuple[int, int]): x range
        y_range (tuple[int, int]): y_range
        z_range (tuple[int, int]): z_range
        x_res (int): number of points in x direction
        y_res (int): number of points in y direction
        z_res (int): number of points in z direction
        cam_incl_adjust (torch.Tensor | None, optional): Opional rigid body transformation. Defaults to None.

    Returns:
        torch.Tensor: 3D grid of points of shape (y_res, z_res, x_res, 3)
    """
    x = (
        torch.linspace(x_range[0], x_range[1], x_res)
        .view(x_res, 1, 1)
        .expand(-1, y_res, z_res)
    )
    y = (
        torch.linspace(y_range[0], y_range[1], y_res)
        .view(1, y_res, 1)
        .expand(x_res, -1, z_res)
    )
    z = (
        torch.linspace(z_range[0], z_range[1], z_res)
        .view(1, 1, z_res)
        .expand(x_res, y_res, -1)
    )
    xyz = torch.stack((x, y, z), dim=-1)

    # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that.
    if cam_incl_adjust is not None:
        xyz = transform_pts(xyz.view(1, -1, 3), cam_incl_adjust[None])[0]
        xyz = xyz.view(x_res, y_res, z_res, 3)

    return xyz
