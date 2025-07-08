import os
import tempfile
import sys
import yaml

import numpy as np
import open3d as o3d

import gradio as gr
import torch

sys.path.append("./sscbench")
from sscbench.gen_voxelgrid_npy import save_as_voxel_ply, classes_to_colors
from download_checkpoint import download_scenedino_checkpoint

from demo_utils.utils import (load_modules, 
                              load_sample_from_path, 
                              load_sample_from_dataset,
                              get_fov_mask,
                              inference_3d, 
                              inference_rendered_2d)


# Load checkpoints from Hugging Face
download_scenedino_checkpoint("ssc-kitti-360-dino")
download_scenedino_checkpoint("ssc-kitti-360-dinov2")


# Load model, ray sampler, datasets
ckpt_path = "out/scenedino-pretrained/seg-best-dino/"
ckpt_name = "checkpoint.pt"
net_v1, renderer_v1, ray_sampler_v1, test_dataset = load_modules(ckpt_path, ckpt_name)
renderer_v1.eval()

ckpt_path = "out/scenedino-pretrained/seg-best-dinov2/"
ckpt_name = "checkpoint.pt"
net_v2, renderer_v2, ray_sampler_v2, _ = load_modules(ckpt_path, ckpt_name)
renderer_v2.eval()


def convert_voxels(arr, map_dict):
    f = np.vectorize(map_dict.__getitem__)
    return f(arr)

with open("sscbench/label_maps.yaml", "r") as f:
    label_maps = yaml.safe_load(f)


def demo_run(image: str, 
             backbone: str,
             mode: str,
             sigma_threshold: float, 
             resolution: float, 
             x_range: int, 
             y_range: int, 
             z_range: int):

    if backbone == "DINO (ViT-B)":
        net, renderer, ray_sampler = net_v1, renderer_v1, ray_sampler_v1
    elif backbone == "DINOv2 (ViT-B)":
        net, renderer, ray_sampler = net_v2, renderer_v2, ray_sampler_v2

    prediction_mode = "stego_kmeans"
    if mode == "Feature PCA 1-3":
        segmentation = False
        rgb_from_pca_dim = 0
    elif mode == "Feature PCA 4-6":
        segmentation = False
        rgb_from_pca_dim = 3
    elif mode == "Feature PCA 7-9":
        segmentation = False
        rgb_from_pca_dim = 6
    elif mode == "SSC (unsup.)":
        segmentation = True
    elif mode == "SSC (linear)":
        segmentation = True
        prediction_mode = "direct_linear"

    # Necessary when reading from examples? cast from str
    sigma_threshold, resolution = float(sigma_threshold), float(resolution)
    x_range, y_range, z_range = int(x_range), int(y_range), int(z_range)

    # Too many voxels
    max_voxel_count = 5000000
    voxel_count = (x_range//resolution + 1) * (y_range//resolution + 1) * (z_range//resolution + 1)
    if voxel_count > max_voxel_count:
        raise gr.Error(f"Too many voxels ({int(voxel_count) / 1_000_000:.1f}M > {max_voxel_count / 1_000_000:.1f}M).\n" +
                        "Reduce voxel resolution or range.", duration=5)

    with torch.no_grad():
        images, poses, projs = load_sample_from_path(image, intrinsic=None)

        net.encode(images, projs, poses, ids_encoder=[0])
        net.set_scale(0)

        # 2D Features output
        dino_full_2d, depth_2d, seg_2d = inference_rendered_2d(net, poses, projs, ray_sampler, renderer, prediction_mode)
        net.encoder.fit_visualization(dino_full_2d.flatten(0, -2))

        if segmentation:
            output_2d = convert_voxels(seg_2d.detach().cpu(), label_maps["cityscapes_to_label"])
            output_2d = classes_to_colors[output_2d].cpu().detach().numpy()
        else:
            output_2d = net.encoder.transform_visualization(dino_full_2d, from_dim=rgb_from_pca_dim)
            output_2d -= output_2d.min()
            output_2d /= output_2d.max()
            output_2d = output_2d.cpu().detach().numpy()

        # Chunking
        max_chunk_size = 100000
        z_layers_per_chunk = max_chunk_size // ((x_range//resolution + 1) * (y_range//resolution + 1))

        # 3D Features output
        x_range = (-x_range/2, x_range)
        y_range = (-y_range/2, y_range)
        z_range = (0, z_range)

        is_occupied, output_3d, fov_mask = [], [], []
        current_z = 0

        while current_z <= z_range[1]:
            z_range_chunk = (current_z, min(current_z + z_layers_per_chunk*resolution, z_range[1]))
            current_z += (z_layers_per_chunk+1) * resolution

            xyz_chunk, dino_full_3d_chunk, sigma_3d_chunk, seg_3d_chunk = inference_3d(net, x_range, y_range, z_range_chunk, resolution, prediction_mode)
            fov_mask_chunk = get_fov_mask(projs[0, 0], xyz_chunk)

            is_occupied_chunk = sigma_3d_chunk > sigma_threshold

            if segmentation:
                output_3d_chunk = seg_3d_chunk
            else:
                output_3d_chunk = net.encoder.transform_visualization(dino_full_3d_chunk, from_dim=rgb_from_pca_dim)
                output_3d_chunk -= output_3d_chunk.min()
                output_3d_chunk /= output_3d_chunk.max()

                output_3d_chunk = torch.clamp(output_3d_chunk*1.2 - 0.1, 0.0, 1.0)
                output_3d_chunk = (255*output_3d_chunk).int()

            fov_mask_chunk = fov_mask_chunk.reshape(is_occupied_chunk.shape)

            is_occupied.append(is_occupied_chunk)
            output_3d.append(output_3d_chunk)
            fov_mask.append(fov_mask_chunk)

        is_occupied = torch.cat(is_occupied, dim=2)
        output_3d = torch.cat(output_3d, dim=2)
        fov_mask = torch.cat(fov_mask, dim=2)

    temp_dir = tempfile.gettempdir()
    ply_path = os.path.join(temp_dir, "output.ply")

    if segmentation:
        # mapped to "unlabeled"
        is_occupied[output_3d == 10] = 0
        is_occupied[output_3d == 12] = 0

        save_as_voxel_ply(ply_path, 
                          is_occupied.detach().cpu(), 
                          voxel_size=resolution, 
                          size=is_occupied.size(), 
                          classes=torch.Tensor(
                              convert_voxels(
                                  output_3d.detach().cpu(), 
                                  label_maps["cityscapes_to_label"])),
                          fov_mask=fov_mask)
    else:
        save_as_voxel_ply(ply_path, 
                          is_occupied.detach().cpu(), 
                          voxel_size=resolution, 
                          size=is_occupied.size(), 
                          colors=output_3d.detach().cpu(),
                          fov_mask=fov_mask)

    mesh = o3d.io.read_triangle_mesh(ply_path)
    glb_path = os.path.join(temp_dir, "output.glb")
    o3d.io.write_triangle_mesh(glb_path, mesh, write_ascii=True)

    del dino_full_2d, depth_2d, seg_2d
    del dino_full_3d_chunk, sigma_3d_chunk, seg_3d_chunk, is_occupied_chunk
    del is_occupied, output_3d, fov_mask

    torch.cuda.empty_cache()

    return output_2d, glb_path


markdown_description = """
    # SceneDINO Demo
    [Paper](https://arxiv.org/abs/xxxx.xxxxx) | [Code](https://github.com/tum-vision/scenedino) | [Project Page](https://visinf.github.io/scenedino/)

    Upload a single image to infer 3D geometry and semantics with **SceneDINO**. You can find some example images below.
    <span style="color:orange">⚠️ NOTE: We assume the intrinsic camera matrix of KITTI-360, images are cropped and rescaled to 192x640. Further note our demo's voxel limit of 5M. </span>
    """

demo = gr.Interface(
    demo_run,
    inputs=[
        gr.Image(label="Input image", type="filepath"),
        gr.Radio(label="Backbone", choices=["DINO (ViT-B)", "DINOv2 (ViT-B)"]),
        gr.Radio(label="Mode", choices=["Feature PCA 1-3", "Feature PCA 4-6", "Feature PCA 7-9", "SSC (unsup.)", "SSC (linear)"]),
        gr.Slider(label="Density threshold", minimum=0, maximum=1, step=0.05, value=0.2),
        gr.Slider(label="Resolution [m]", minimum=0.05, maximum=0.5, step=0.05, value=0.2),
        gr.Slider(label="X Range [m]", minimum=1, maximum=50, step=1, value=10),
        gr.Slider(label="Y Range [m]", minimum=1, maximum=50, step=1, value=10),
        gr.Slider(label="Z Range [m]", minimum=1, maximum=100, step=1, value=20),
    ], 
    outputs=[
        gr.Image(label="Rendered 2D Visualization"),
        gr.Model3D(label="Voxel Surface 3D Visualization",
                   zoom_speed=0.5, pan_speed=0.5, 
                   clear_color=[0.0, 0.0, 0.0, 0.0], 
                   camera_position=[-90, 80, None], 
                   display_mode="solid"),
    ],
    title="",
    examples="demo_utils/examples",
    description=markdown_description,
)

demo.launch()
