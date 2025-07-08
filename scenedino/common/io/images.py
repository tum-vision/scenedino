from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scenedino.common.cameras.fisheye import project_to_image

from scenedino.common.geometry import azimuth_elevation_to_rotation


class FisheyeToPinholeSampler:
    def __init__(self, K_target, image_size_target) -> None:
        self.K_target = K_target
        self.image_size_target = image_size_target
        x = (
            torch.linspace(-1, 1, image_size_target[1])
            .view(1, -1)
            .expand(image_size_target)
        )
        y = (
            torch.linspace(-1, 1, image_size_target[0])
            .view(-1, 1)
            .expand(image_size_target)
        )
        z = torch.ones_like(x)
        xyz = torch.stack((x, y, z), dim=-1).view(-1, 3)
        self.pts = (torch.inverse(torch.tensor(K_target)) @ xyz.T).T

    def sample_pinhole(
        self, img: torch.Tensor, calib, rotational_offset: torch.Tensor
    ) -> torch.Tensor:
        xyz_pinhole = (rotational_offset @ self.pts.T).T

        xy, _ = project_to_image(xyz_pinhole, calib)

        img = img.unsqueeze(0)
        resampled_img = F.grid_sample(
            img, xy.view(1, *self.image_size_target, 2), align_corners=True
        ).squeeze(0)
        return resampled_img


def load_image(path: Path) -> torch.Tensor:
    return torch.from_numpy(
        cv2.cvtColor(
            cv2.imread(str(path)),
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)
        / 255
    )


def load_fisheye_image_as_pinhole(
    path: Path,
    resampler: FisheyeToPinholeSampler,
    calib,
    azimuth: float,
    elevation: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    img = load_image(path)
    rotational_offset = azimuth_elevation_to_rotation(azimuth, elevation)
    return resampler.sample_pinhole(img, calib, rotational_offset), rotational_offset
