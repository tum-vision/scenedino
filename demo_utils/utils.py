import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import center_crop, resize

import os
from hydra import compose, initialize

from scenedino.models.bts import BTSNet
from scenedino.common.ray_sampler import ImageRaySampler
from scenedino.models import make_model
from scenedino.renderer.nerf import NeRFRenderer
from scenedino.training.trainer import BTSWrapper
from scenedino.datasets import make_datasets
from scenedino.common.array_operations import map_fn, unsqueezer


device = "cuda"


def load_modules(
    ckpt_path: str, 
    ckpt_name: str
) -> tuple[BTSNet, NeRFRenderer, ImageRaySampler, Dataset]:
    """
    Loads relevant modules with a SceneDINO checkpoint (*.pt) and corresponding config (training_config.yaml) file.

    Args:
        ckpt_path (str): Relative path to the directory containing the files.
        ckpt_name (str): File name of the checkpoint.

    Returns:
        net (BTSNet): The SceneDINO network.
        renderer (NeRFRenderer): Volume rendering module.
        ray_sampler (ImageRaySampler): Camera ray sampler for whole images.
        test_dataset (Dataset): Test set of the dataset trained on.
    """
    with initialize(version_base=None, config_path="../" + ckpt_path, job_name="demo_script"):
        config = compose(config_name="training_config")

    net = make_model(config["model"], config.get("downstream", None))

    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer.hard_alpha_cap = False
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    height, width = config["dataset"]["image_size"]
    ray_sampler = ImageRaySampler(z_near=3, z_far=80, width=width, height=height)

    model = BTSWrapper(renderer, ray_sampler, config["model"])
    cp = torch.load(ckpt_path + ckpt_name)
    # cp = cp["model"]  # Some older checkpoints have this structure
        
    model.load_state_dict(cp, strict=False)
    model = model.to(device)

    test_dataset = make_datasets(config["dataset"])[1]

    return net, renderer, ray_sampler, test_dataset


def load_sample_from_path(
    path: str,
    intrinsic: Tensor | None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Loads a test image from a provided path.

    Args:
        path (str): Image path.

    Returns:
        images (Tensor): RGB image normalized to [-1, 1].
        poses (Tensor): Camera pose (unit matrix).
        projs (Tensor): Camera matrix (unit matrix).
    """
    images = read_image(path)

    if not (images.size(1) == 192 and images.size(2) == 640):
        scale = max(192 / images.size(1), 640 / images.size(2))
        new_h, new_w = int(images.size(1) * scale), int(images.size(2) * scale)

        images_resized = resize(images, [new_h, new_w])
        images = center_crop(images_resized, (192, 640))
        print("WARNING: Custom image does not have correct dimensions! Taking center crop.")

    if images.dtype == torch.uint8:
        images = 2 * (images / 255) - 1
    elif images.dtype == torch.uint16:
        images = 2 * (images / (2**16 - 1)) - 1

    if images.size(0) == 4:
        images = images[:3]

    images = images.unsqueeze(0).unsqueeze(1)
    poses = torch.eye(4).unsqueeze(0).unsqueeze(1)

    if intrinsic:
        projs = intrinsic.unsqueeze(0).unsqueeze(1)
    else:
        projs = torch.Tensor([
            [0.7849,  0.0000, -0.0312],
            [0.0000,  2.9391,  0.2701],
            [0.0000,  0.0000,  1.0000]]).unsqueeze(0).unsqueeze(1)
        print("WARNING: Custom image has no provided intrinsics! Using KITTI-360 values.")

    return images.to(device), poses.to(device), projs.to(device)


def load_sample_from_dataset(
    idx: int, 
    dataset: Dataset
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Loads a data point from the provided dataset. In this demo, we just load the front view.

    Args:
        idx (int): Index in the dataset.
        dataset (Dataset): The dataset.

    Returns:
        images (Tensor): RGB image normalized to [-1, 1].
        poses (Tensor): Camera pose (since just front view, unit matrix).
        projs (Tensor): Camera matrix.
    """
    data = dataset[idx]

    data_batch = map_fn(map_fn(data, torch.tensor), unsqueezer)
    images = torch.stack(data_batch["imgs"], dim=1)
    poses = torch.stack(data_batch["poses"], dim=1)
    projs = torch.stack(data_batch["projs"], dim=1)

    poses = torch.inverse(poses[:, :1, :, :]) @ poses
 
    # Just front view
    images = images[:, :1]
    poses = poses[:, :1]
    projs = projs[:, :1]

    return images.to(device), poses.to(device), projs.to(device)


def inference_3d(
    net: BTSNet, 
    x_range: tuple[float, float], 
    y_range: tuple[float, float], 
    z_range: tuple[float, float], 
    resolution: float,
    prediction_mode: str = "stego_kmeans"
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Inference in a uniform 3D grid. All units are provided in meters.

    Args:
        net (BTSNet): The SceneDINO network.
        x_range (tuple[float, float]): Range along the X dimension.
        y_range (tuple[float, float]): Range along the Y dimension.
        z_range (tuple[float, float]): Range along the Z dimension, the viewing direction.
        resolution (float): Resolution of the grid.

    Returns:
        dino_full (Tensor): SceneDINO features [n_X, n_Y, n_Z, 768].
        sigma (Tensor): Volumentric density [n_X, n_Y, n_Z].
        seg (Tensor): Predicted semantic classes [n_X, n_Y, n_Z].
    """
    n_pts_x = int((x_range[1] - x_range[0]) / resolution) + 1
    n_pts_y = int((y_range[1] - y_range[0]) / resolution) + 1
    n_pts_z = int((z_range[1] - z_range[0]) / resolution) + 1

    x = torch.linspace(x_range[0], x_range[1], n_pts_x)
    y = torch.linspace(y_range[0], y_range[1], n_pts_y)
    z = torch.linspace(z_range[0], z_range[1], n_pts_z)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    xyz = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3).unsqueeze(0).to(device)

    dino_full, invalid, sigma, seg = net(xyz, predict_segmentation=True, prediction_mode=prediction_mode)

    dino_full = dino_full.reshape(n_pts_x, n_pts_y, n_pts_z, -1)
    sigma = sigma.reshape(n_pts_x, n_pts_y, n_pts_z)

    if seg is not None:
        seg = seg.reshape(n_pts_x, n_pts_y, n_pts_z, -1).argmax(-1)

    return xyz, dino_full, sigma, seg


def get_fov_mask(proj_matrix, xyz):
    proj_xyz = xyz @ proj_matrix.T
    proj_xyz = proj_xyz / proj_xyz[..., 2:3]

    fov_mask = (proj_xyz[..., 0] > -0.99) & (proj_xyz[..., 0] < 0.99) & (proj_xyz[..., 1] > -0.99) & (proj_xyz[..., 1] < 0.99)

    return fov_mask

    

def inference_rendered_2d(
    net: BTSNet, 
    poses: Tensor, 
    projs: Tensor, 
    ray_sampler: ImageRaySampler, 
    renderer: NeRFRenderer,
    prediction_mode: str = "stego_kmeans"
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Inference in 3D, rendered back into a 2D image, based on a provided camera pose and matrix.

    Args:
        net (BTSNet): The SceneDINO network.
        poses (Tensor): Camera pose.
        projs (Tensor): Camera matrix.
        ray_sampler (ImageRaySampler): Camera ray sampler for whole images.
        renderer (NeRFRenderer): Volume rendering module.

    Returns:
        dino_full (Tensor): SceneDINO features [H, W, 768].
        depth (Tensor): Ray termination depth [H, W].
        seg (Tensor): Predicted semantic classes [H, W].
    """
    all_rays, _ = ray_sampler.sample(None, poses[:, :], projs[:, :])
    render_dict = renderer(all_rays, want_weights=True, want_alphas=True)
    render_dict = ray_sampler.reconstruct(render_dict)

    depth = render_dict["coarse"]["depth"].squeeze()

    dino_distilled = render_dict["coarse"]["dino_features"].squeeze()
    dino_full = net.encoder.expand_dim(dino_distilled)

    if net.downstream_head is not None:
        seg = net.downstream_head(dino_full, mode=prediction_mode)
    else:
        seg = None

    return dino_full, depth, seg
    