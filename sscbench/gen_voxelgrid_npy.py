import os
import os
import sys
from pathlib import Path

sys.path.append(".")

from point_utils import get_fov_mask

import cv2
import hydra as hydra
import torch.nn.functional as F
import yaml
from matplotlib import pyplot as plt
from omegaconf import open_dict
from torch import nn
from tqdm import tqdm
import glob


import numpy as np
import torch
from plyfile import PlyData, PlyElement


os.system("nvidia-smi")

in_path = Path("<PATH-IN>")
TARGET_PATH = Path("<PATH-TARGET>")
# out_path = Path("media/voxel/npy/")
out_path = Path("<PATH-OUT>")
# out_path = Path("/storage/slurm/hayler/bts/voxel_outputs/sscnet")
# out_path.mkdir(exist_ok=True, parents=True)

fov_mask = get_fov_mask()

X_RANGE = (25.6, -25.6)
Y_RANGE = (51.2, 0)
Z_RANGE = (0, 6.4)
#
# gpu_id = 1
#
# device = f'cpu'
# if gpu_id is not None:
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# if torch.cuda.is_available():
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True

# classes_to_colors = torch.tensor(
#     [
#         [255, 255, 255],
#         [100, 150, 245],  # 1
#         [255, 0, 0],
#         [255, 0, 255],
#         [255, 150, 255],
#         [75, 0, 75],
#         [175, 0, 75],  # 6
#         [255, 200, 0],
#         [150, 150, 150],
#         [30, 60, 150],
#         [80, 30, 180],
#         [8, 97, 0],  # 11
#         [184, 56, 2],
#         [255, 143, 46],
#         [112, 255, 50],
#         [194, 0, 0],
#         [135, 60, 0],
#         [150, 240, 80],
#         [255, 240, 150],
#         [255, 0, 0],
#     ]
# )

classes_to_colors = torch.tensor(
    [
        [70, 130, 180],  # sky for unlabeled (?)
        [0, 0, 142],  # 1
        [119, 11, 32],
        [0,  0, 230],
        [0,  0, 70],
        [0, 60,100],
        [220, 20, 60],  # 6
        [128, 64,128],
        [244, 35,232],
        [90, 90, 90],
        [190,153,153],
        [107,142, 35],  # 11
        [152,251,152],
        [153,153,153],
        [220,220,  0],
        [250,170, 30],
        [135, 60, 0],
        [150, 240, 80],
        [255, 240, 150],
        [255, 0, 0],
    ]
)

with open("sscbench/label_maps.yaml", "r") as f:
    label_maps = yaml.safe_load(f)

device = "cpu"

r, c, = 0, 0
n_rows, n_cols = 3, 3

def plot(img, fig, axs, i=None):
    global r, c
    if r == 0 and c == 0:
        plt.show()
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2))
    axs[r][c].imshow(img, interpolation="none")
    if i is not None:
        axs[r][c].title.set_text(f"{i}")
    c += 1
    r += c // n_cols
    c %= n_cols
    r %= n_rows
    return fig, axs


def save_plot(img, file_name=None, grey=False, mask=None):
    if mask is not None:
        if mask.shape[-1] != img.shape[-1]:
            mask = np.broadcast_to(np.expand_dims(mask, -1), img.shape)
        img = np.array(img)
        img[~mask] = 0
    if dry_run:
        plt.imshow(img)
        plt.title(file_name)
        plt.show()
    else:
        cv2.imwrite(file_name, cv2.cvtColor((img * 255).clip(max=255).astype(np.uint8), cv2.COLOR_RGB2BGR) if not grey else (img * 255).clip(max=255).astype(np.uint8))

faces = [[0, 1, 2, 3], [0, 3, 7, 4], [2, 6, 7, 3], [5, 6, 2, 1], [4, 5, 1, 0], [7, 6, 5, 4]]
faces_t = torch.tensor(faces, device=device)



def build_voxel(i, j, k, x_res, y_res, z_res, xyz, offset):
    ids = [[i+1, j+1, k], [i+1, j, k],
           [i, j, k], [i, j+1, k],
           [i+1, j+1, k+1], [i+1, j, k+1],
           [i, j, k+1], [i, j+1, k+1]]

    faces_off = [[v+offset for v in f] for f in faces]

    ids_flat = list(map(lambda ijk: ijk[0]*y_res*z_res + ijk[1]*z_res + ijk[2], ids))

    verts = xyz[:, ids_flat].cpu().numpy().T

    colors = np.tile(np.array(plt.cm.get_cmap("magma")(1 - (verts[..., 1].mean().item() - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]))[:3]).reshape((1, 3)), ((len(faces_off), 1)))
    colors = (colors * 255).astype(np.uint8)

    return verts, faces_off, colors


ids_offset = torch.tensor(
        [[1, 1, 0], [1, 0, 0],
        [0, 0, 0], [0, 1, 0],
        [1, 1, 1], [1, 0, 1],
        [0, 0, 1], [0, 1, 1]],
    dtype=torch.int32,
    device=device
) # (8, 3)


def remove_invisible(volume):
    kernel = torch.tensor([[[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                           [[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]],
                           [[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]]], dtype=torch.float32, device=volume.device).view(1, 1, 3, 3, 3)

    neighbors = F.conv3d(volume.to(torch.float32).view(1, 1, *volume.shape), kernel, stride=1, padding=1)[0, 0, :, :, :]
    is_hidden = neighbors >= 6
    volume = volume & (~is_hidden)
    return volume


def check_neighbors(volume):
    kernel = torch.zeros((6, 3, 3, 3), device=volume.device, dtype=torch.float32)
    kernel[0, 1, 1, 0] = 1
    kernel[1, 1, 2, 1] = 1
    kernel[2, 0, 1, 1] = 1
    kernel[3, 1, 0, 1] = 1
    kernel[4, 2, 1, 1] = 1
    kernel[5, 1, 1, 2] = 1

    kernel = kernel.unsqueeze(1)

    neighbors = F.conv3d(volume.to(torch.float32).view(1, 1, *volume.shape), kernel, stride=1, padding=1)[0, :, :, :, :]
    neighbors = neighbors >= 1
    return neighbors


def build_voxels(ijks, x_res, y_res, z_res, xyz, neighbors=None, colors=None, classes=None):
    # ijks (N, 3)

    ids = ijks.view(-1, 1, 3) + ids_offset.view(1, -1, 3)

    ids_flat = ids[..., 0] * y_res * z_res + ids[..., 1] * z_res + ids[..., 2]

    verts = xyz[:, ids_flat.reshape(-1)]

    faces_off = torch.arange(0, ijks.shape[0] * 8, 8, device=device)
    faces_off = faces_off.view(-1, 1, 1) + faces_t.view(-1, 6, 4)

    if classes is not None:
        index_classes = classes[ijks[:, 0], ijks[:, 1], ijks[:, 2]].to(int)

        colors = classes_to_colors[index_classes].view(-1, 1, 3).expand(-1, 8, -1)
    elif colors is None:
        z_steps = (1 - (torch.linspace(0, 1 - 1 / z_res, z_res) + 1 / (2 * z_res))).tolist()
        cmap = plt.cm.get_cmap("magma")
        z_to_color = (torch.tensor(list(map(cmap, z_steps)), device=device)[:, :3] * 255).to(torch.uint8)

        colors = z_to_color[ijks[:, 2], :].view(-1, 1, 3).expand(-1, 8, -1)
    else:
        colors = colors[ijks[:, 0], ijks[:, 1], ijks[:, 2]].view(-1, 1, 3).expand(-1, 8, -1)

    if neighbors is not None:
        faces_off = faces_off.reshape(-1, 4)[~neighbors.reshape(-1), :]

    return verts.cpu().numpy().T, faces_off.reshape(-1, 4).cpu().numpy(), colors.reshape(-1, 3).cpu().numpy()

def get_pts(x_range, y_range, z_range, x_res, y_res, z_res):
    x = torch.linspace(x_range[0], x_range[1], x_res).view(x_res, 1, 1).expand(-1, y_res, z_res)
    y = torch.linspace(y_range[0], y_range[1], y_res).view(1, y_res, 1).expand(x_res, -1, z_res)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, 1, z_res).expand(x_res, y_res, -1)
    xyz = torch.stack((x, y, z), dim=-1)                                            # (x, y, z)

    # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that tan(5°) = 0.0874886635
    return xyz


def save_as_voxel_ply(path, is_occupied, voxel_size=0.2, size=(256, 256, 32), classes=None, colors=None, fov_mask=None):
    is_occupied = remove_invisible(is_occupied)
    if fov_mask is not None:
        is_occupied &= fov_mask.to(is_occupied.device)

    is_occupied[0] = 0
    is_occupied[-1] = 0
    is_occupied[:, 0] = 0
    is_occupied[:, -1] = 0
    is_occupied[:, :, 0] = 0
    is_occupied[:, :, -1] = 0

    res = (size[0] + 1, size[1] + 1, size[2] + 1)
    x_range = (size[0] * voxel_size * .5, -size[0] * voxel_size * .5)
    y_range = (size[1] * voxel_size, 0)
    z_range = (0, size[2] * voxel_size)

    neighbors = check_neighbors(is_occupied)
    neighbors = neighbors.view(6, -1)[:, is_occupied.reshape(-1)].T

    q_pts = get_pts(x_range, y_range, z_range, *res)
    q_pts = q_pts.to(device).reshape(1, -1, 3)
    verts, faces, colors = build_voxels(is_occupied.nonzero(), *res, q_pts.squeeze(0).T, neighbors, classes=classes, colors=colors)

    verts = list(map(tuple, verts))
    colors = list(map(tuple, colors))
    verts_colors = [v + c for v, c in zip(verts, colors)]
    verts_data = np.array(verts_colors, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    face_data = np.array(faces, dtype='i4')
    ply_faces = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (4,))])
    ply_faces['vertex_indices'] = face_data

    verts_el = PlyElement.describe(verts_data, "vertex")
    faces_el = PlyElement.describe(ply_faces, "face")
    PlyData([verts_el, faces_el]).write(str(path))

def convert_voxels(arr, map_dict):
    f = np.vectorize(map_dict.__getitem__)
    return f(arr)

def main():
    print('Loading file')

    is_occupied = torch.tensor(np.load(in_path), dtype=torch.bool, device=device) > 0
    save_as_voxel_ply(out_path / f"{in_path.stem}.ply", is_occupied)

def safe_filepath_segementation(path, suffix="", gt=None, sizes=None, use_fov_mask=False):
    path = Path(path)
    segmentations = convert_voxels(np.load(path).astype(int), label_maps["sscbench_to_label"])
    if gt is not None:
        segmentations[gt == 255] = 0
    segmentations[segmentations == 255] = 0
    is_occupied = torch.tensor(segmentations > 0)

    if use_fov_mask:
        is_occupied[~fov_mask] = 0

    if sizes:
        for size in sizes:
            if suffix != "":
                fp = out_path / str(int(size)) / f"{id:06d}.ply"
            else:
                fp = out_path / str(int(size)) / f"{path.stem}.ply"
            num_voxels = int(size // 0.2)
            save_as_voxel_ply(fp,
                              is_occupied[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2), :],
                              classes=torch.tensor(
                                  segmentations[: num_voxels, (128 - num_voxels // 2): (128 + num_voxels // 2), :]))
    else:
        if suffix != "":
            save_as_voxel_ply(out_path / f"{path.stem}_{suffix}.ply", is_occupied, classes=torch.tensor(segmentations))
        else:
            save_as_voxel_ply(out_path / f"{path.stem}.ply", is_occupied, classes=torch.tensor(segmentations))


def safe_folder_segmentation(path:str, suffix="", ids=None, sizes=None, use_fov_mask=False):
    if sizes:
        for size in sizes:
            if not os.path.exists(out_path / str(int(size))):
                os.makedirs(out_path / str(int(size)))

    for file in tqdm(sorted(glob.glob(path + "/*"))):
        frameId = int(file.split('/')[-1].split(".")[0])

        gt = np.load(TARGET_PATH / f"{frameId:06d}_1_1.npy")

        if ids and frameId not in ids:
            continue
        safe_filepath_segementation(file, suffix, gt=gt, sizes=sizes, use_fov_mask=use_fov_mask)

def main_segmentation():
    segmentations = convert_voxels(np.load(in_path).astype(int), label_maps["sscbench_to_label"])
    segmentations[segmentations == 255] = 0
    is_occupied = torch.tensor(segmentations > 0)
    save_as_voxel_ply(out_path / f"{in_path.stem}.ply", is_occupied, classes=torch.tensor(segmentations))

if __name__ == '__main__':
    # safe_folder_segmentation("/storage/slurm/hayler/sscbench/3data/monoscene/paper", "monoscene")
    safe_folder_segmentation("/storage/slurm/hayler/sscbench/outputs/lmscnet", suffix="", sizes=[12.8, 25.6, 51.2], use_fov_mask=True)
    # safe_folder_segmentation("/storage/slurm/hayler/sscbench/outputs/sscnet", suffix="", sizes=[12.8, 25.6, 51.2], use_fov_mask=True)
    # safe_folder_segmentation("/storage/slurm/hayler/sscbench/outputs/sscnet", suffix="sscnet",
    #                          ids=[125, 5475, 6670, 6775, 7860, 8000])
