import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.utils import save_image

from hydra import compose, initialize

from bts.models.bts import BTSNet
from bts.common.ray_sampler import ImageRaySampler
from bts.models import make_model
from bts.renderer.nerf import NeRFRenderer
from bts.training.trainer import BTSWrapper
from bts.datasets import make_datasets
from bts.common.array_operations import map_fn, unsqueezer


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
    with initialize(version_base=None, config_path=ckpt_path, job_name="demo_script"):
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


def load_test_image(
    idx: int, 
    dataset: Dataset
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Loads a data point from the provided dataset. In this demo, we just load the front view.

    Args:
        idx (int): Index in the dataset.
        dataset (Dataset): The dataset.

    Returns:
        images (Tensor): RGB image.
        poses (Tensor): Camera pose (since just front view, unit matrix).
        projs (Tensor): Camera matrix.
    """
    data = dataset[idx]

    data_batch = map_fn(map_fn(data, torch.tensor), unsqueezer)
    images = torch.stack(data_batch["imgs"], dim=1).to(device)
    poses = torch.stack(data_batch["poses"], dim=1).to(device)
    projs = torch.stack(data_batch["projs"], dim=1).to(device)

    poses = torch.inverse(poses[:, :1, :, :]) @ poses
 
    # Just front view
    images = images[:, :1]
    poses = poses[:, :1]
    projs = projs[:, :1]

    return images, poses, projs


def inference_3d(
    net: BTSNet, 
    x_range: tuple[float, float], 
    y_range: tuple[float, float], 
    z_range: tuple[float, float], 
    resolution: float
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

    dino_full, invalid, sigma, seg = net(xyz, predict_segmentation=True)

    dino_full = dino_full.reshape(n_pts_x, n_pts_y, n_pts_z, -1)
    sigma = sigma.reshape(n_pts_x, n_pts_y, n_pts_z)

    if seg is not None:
        seg = seg.reshape(n_pts_x, n_pts_y, n_pts_z, -1).argmax(-1)

    return dino_full, sigma, seg


def inference_rendered_2d(
    net: BTSNet, 
    poses: Tensor, 
    projs: Tensor, 
    ray_sampler: ImageRaySampler, 
    renderer: NeRFRenderer
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
        seg = net.downstream_head(dino_full, mode="stego_kmeans")
    else:
        seg = None

    return dino_full, depth, seg


def main():
    # Load model, ray sampler, datasets
    ckpt_path = "out/scenedino-pretrained/seg-best-dino/"
    ckpt_name = "checkpoint.pt"
    net, renderer, ray_sampler, test_dataset = load_modules(ckpt_path, ckpt_name)

    # Load image from test dataset
    idx = 0
    images, poses, projs = load_test_image(0, test_dataset)

    # Encode input image
    net.encode(images, projs, poses, ids_encoder=[0])
    net.set_scale(0)

    # Inference rendered back into 2D (input view, can also be any other though)
    dino_full_2d, depth_2d, seg_2d = inference_rendered_2d(net, poses, projs, ray_sampler, renderer)
    print("\n--- 2D Inference ---")
    print("Rendered Features: ", dino_full_2d.size())
    print("Depth:             ", depth_2d.size())
    if seg_2d is not None:
        print("Segmentation:      ", seg_2d.size())
    else:
        print("-> No SSC head linked.")

    # Fit PCA for visualization
    net.encoder.fit_visualization(dino_full_2d.flatten(0, -2))

    # Save images + rendered features
    save_image(images.squeeze()/2 + 0.5, "out/demo-out/input_image.png")
    dino_pca_2d = net.encoder.transform_visualization(dino_full_2d, from_dim=0).permute(2,0,1)
    save_image(dino_pca_2d / dino_pca_2d.max(), "out/demo-out/feat_pca_00_02.png")
    dino_pca_2d = net.encoder.transform_visualization(dino_full_2d, from_dim=3).permute(2,0,1)
    save_image(dino_pca_2d / dino_pca_2d.max(), "out/demo-out/feat_pca_03_05.png")
    dino_pca_2d = net.encoder.transform_visualization(dino_full_2d, from_dim=6).permute(2,0,1)
    save_image(dino_pca_2d / dino_pca_2d.max(), "out/demo-out/feat_pca_06_08.png")

    # Inference in 3D grid
    x_range = (-10, 10)
    y_range = (-5, 5)
    z_range = (0, 20)
    resolution = 0.2
    dino_full_3d, sigma_3d, seg_3d = inference_3d(net, x_range, y_range, z_range, resolution)
    print("\n--- 3D Inference ---")
    print("Features:     ", dino_full_3d.size())
    print("Vol. Density: ", sigma_3d.size())
    if seg_3d is not None:
        print("SSC:          ", seg_3d.size())
    else:
        print("-> No SSC head linked.")

    # Not optimal, fitting PCA on surfaces would be better
    # net.encoder.fit_visualization(dino_full_3d.flatten(0, -2))  
    dino_pca_3d = net.encoder.transform_visualization(dino_full_3d, from_dim=0)


if __name__ == "__main__":
    main()
