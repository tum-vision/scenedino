import argparse
import sys

from omegaconf import open_dict

import matplotlib.pyplot as plt

sys.path.append(".")

from gen_voxelgrid_npy import save_as_voxel_ply

import logging

from pathlib import Path
import subprocess
import yaml

import cv2
import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from hydra import compose, initialize

import matplotlib.pyplot as plt

# from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset

from fusion import TSDFVolume, rigid_transform

from sscbench_dataset import SSCBenchDataset
from pathlib import Path

RELOAD_DATASET = True
DATASET_LENGTH = 100
FULL_EVAL = True
SAMPLE_EVERY = None
SAMPLE_OFFSET = 2
# SAMPLE_RANGE = list(range(1000, 1600))
SAMPLE_RANGE = None

import time


SIZE = 51.2 # Can be: 51.2, 25.6, 12.8
SIZES = (12.8, 25.6, 51.2)
VOXEL_SIZE = 0.1 # Needs: 0.2 % VOXEL_SIZE == 0

USE_CUSTOM_CUTOFFS = False
SIGMA_CUTOFF = 0.25

USE_ALPHA_WEIGHTING = True
USE_MAXPOOLING = False

GENERATE_PLY_FILES = True
PLY_ONLY_FOV = True
# PLY_IDS = [2235, 2495, 2385, 3385, 4360, 6035, 8575, 9010, 11260] # 10:40
# PLY_IDS = [2495, 6035, 8575, 9010, 11260] # 10
#PLY_IDS = [125, 5475, 6035, 6670, 6775, 7860, 8000]
# PLY_IDS = list(range(1000, 1600))
PLY_IDS = None
# PLY_PATH = Path("/usr/stud/hayler/dev/BehindTheScenes/scripts/benchmarks/sscbench/ply10_fov")
PLY_PATH = Path("<PATH-TO-PLY-OUTPUT>")
PLY_SIZES = [12.8, 25.6, 51.2]

if GENERATE_PLY_FILES:
    assert (not USE_MAXPOOLING) and VOXEL_SIZE == 0.1

    # make the necessary dirs
    for size in PLY_SIZES:
        if not os.path.exists(PLY_PATH / str(int(size))):
            os.makedirs(PLY_PATH / str(int(size)))


# Setup of CUDA device and logging

os.system("nvidia-smi")

device = f'cuda:0'

# DO NOT TOUCH OR YOU WILL BREAK RUNS (should be None)
gpu_id = None

if gpu_id is not None:
    print("GPU ID: " + str(gpu_id))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.INFO)

times = []


def downsample_and_predict(data, net, pts, factor):
    pts = pts.reshape(256*factor, 256*factor, 32*factor, 3)

    sigmas = torch.zeros(256, 256, 32).numpy()
    segs = torch.zeros(256, 256, 32).numpy()

    chunk_size_x = chunk_size_y = 256
    chunk_size_z = 32

    n_chunks_x = int(256*factor / chunk_size_x)
    n_chunks_y = int(256*factor / chunk_size_y)
    n_chunks_z = int(32*factor / chunk_size_z)


    b_x = chunk_size_x // factor # size of the mini blocks
    b_y = chunk_size_y // factor
    b_z = chunk_size_z // factor


    for i in range(n_chunks_x):
        for j in range(n_chunks_y):
            for k in range(n_chunks_z):
                pts_block = pts[i * chunk_size_x:(i + 1) * chunk_size_x, j * chunk_size_y:(j + 1) * chunk_size_y, k * chunk_size_z:(k + 1) * chunk_size_z]
                sigmas_block, segs_block = predict_grid(data, net, pts_block)
                sigmas_block = sigmas_block.reshape(chunk_size_x, chunk_size_y, chunk_size_z)
                segs_block = segs_block.reshape(chunk_size_x, chunk_size_y, chunk_size_z, 19)

                if USE_ALPHA_WEIGHTING:
                    alphas = 1 - torch.exp(- VOXEL_SIZE * sigmas_block)
                    segs_block = (alphas.unsqueeze(-1) * segs_block).unsqueeze(0)
                else:
                    segs_block = (sigmas_block.unsqueeze(-1) * segs_block).unsqueeze(0)

                segs_pool_list = [F.avg_pool3d(segs_block[..., i], kernel_size=factor, stride=factor, padding=0) for i in
                                  range(segs_block.shape[-1])]
                segs_pool = torch.stack(segs_pool_list, dim=-1).unsqueeze(0)
                segs_pool = torch.argmax(segs_pool, dim=-1).detach().cpu().numpy()

                # pool the observations
                sigmas_block = F.max_pool3d(sigmas_block.unsqueeze(0), kernel_size=factor, stride=factor, padding=0).squeeze(0).detach().cpu().numpy()

                sigmas[i * b_x:(i + 1) * b_x, j * b_y: (j + 1) * b_y, b_z * k:b_z * (k + 1)] = sigmas_block
                segs[i * b_x:(i + 1) * b_x, j * b_y: (j + 1) * b_y, b_z * k:b_z * (k + 1)] = segs_pool

                torch.cuda.empty_cache()

    if USE_MAXPOOLING:
        sigmas = F.max_pool3d(torch.tensor(sigmas).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).numpy()

    return sigmas, segs

def use_custom_maxpool(_sigmas):
    sigmas = torch.zeros(258, 258, 34)
    sigmas[1:257, 1:257, 1:33] = torch.tensor(_sigmas)
    sigmas_pooled = torch.zeros(256, 256, 32)

    for i in range(256):
        for j in range(256):
            for k in range(32):
                sigmas_pooled[i, j, k] = max(sigmas[i+1, j+1, k+1],
                                             sigmas[i, j+1, k+1], sigmas[i+1, j, k+1],sigmas[i+1, j+1, k],
                                             sigmas[i+2, j+1, k+1], sigmas[i+1, j+2, k+1],sigmas[i+1, j+1, k+2])
    return sigmas_pooled

def plot_images(images_dict):
    """The images dict should include six images and six corresponding ids"""
    images = images_dict["images"]
    ids = images_dict["ids"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 6))

    axes = axes.flatten()

    for i, img in enumerate(images):
        axes[i].imshow(images[i])
        axes[i].axis("off")
        axes[i].set_title(f"FrameId: {ids[i]}")

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()

def plot_image_at_frame_id(dataset, frame_id):

    for i in range(len(dataset)):
        sequence, id, is_right = dataset._datapoints[i]
        if id == frame_id:
            data = dataset[i]
            plt.figure(figsize=(10, 4))
            plt.imshow(((data["imgs"][0] + 1) / 2).permute(1, 2, 0))
            plt.gca().set_axis_off()
            plt.show()
            return



def identify_additional_invalids(target):
    # Note: The Numpy implementation is a bit faster (about 0.1 seconds per iteration)

    _t = np.concatenate([np.zeros([256, 256, 1]), target], axis=2)
    invalids = np.cumsum(np.logical_and(_t != 255, _t != 0), axis=2)[:, :, :32] == 0
    # _t = torch.cat([torch.zeros([256, 256, 1], device=device, dtype=torch.int32), torch.tensor(target, dtype=torch.int32).to(device)], dim=2)
    # invalids = torch.cumsum((_t != 255) & (_t != 0), axis=2)[:,:, :32] == 0
    # height cut-off (z > 6 ==> no invalid)
    invalids[: , :, 7:] = 0
    # only empty voxels matter
    invalids[target != 0] = 0

    # return invalids.cpu().numpy()
    return invalids


def generate_point_grid(cam_E, vox_origin, voxel_size, scene_size, cam_k, img_W=1408, img_H=376):
        """
        compute the 2D projection of voxels centroids

        Parameters:
        ----------
        cam_E: 4x4
           =camera pose in case of NYUv2 dataset
           =Transformation from camera to lidar coordinate in case of SemKITTI
        cam_k: 3x3
            camera intrinsics
        vox_origin: (3,)
            world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
        img_W: int
            image width
        img_H: int
            image height
        scene_size: (3,)
            scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2

        Returns
        -------
        projected_pix: (N, 2)
            Projected 2D positions of voxels
        fov_mask: (N,)
            Voxels mask indice voxels inside image's FOV
        pix_z: (N,)
            Voxels'distance to the sensor in meter
        """
        # Compute the x, y, z bounding of the scene in meter
        vol_bnds = np.zeros((3, 2))
        vol_bnds[:, 0] = vox_origin
        vol_bnds[:, 1] = vox_origin + np.array(scene_size)

        # Compute the voxels centroids in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)
        xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
        )
        vox_coords = np.concatenate([
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            zv.reshape(1, -1)
        ], axis=0).astype(int).T

        # Project voxels'centroid from lidar coordinates to camera coordinates
        cam_pts = TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
        cam_pts = rigid_transform(cam_pts, cam_E)

        # Project camera coordinates to pixel positions
        projected_pix = TSDFVolume.cam2pix(cam_pts, cam_k)
        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

        # Eliminate pixels outside view frustum
        pix_z = cam_pts[:, 2]
        fov_mask = np.logical_and(pix_x >= 0,
                                  np.logical_and(pix_x < img_W,
                                                 np.logical_and(pix_y >= 0,
                                                                np.logical_and(pix_y < img_H,
                                                                               pix_z > 0))))

        return cam_pts, fov_mask


def predict_grid(data_batch, net, points):
    images = torch.stack(data_batch["imgs"], dim=0).unsqueeze(0).to(device).float()
    poses = torch.tensor(np.stack(data_batch["poses"], 0)).unsqueeze(0).to(device).float()
    projs = torch.tensor(np.stack(data_batch["projs"], 0)).unsqueeze(0).to(device).float()

    poses = torch.inverse(poses[:, :1]) @ poses

    n, nv, c, h, w = images.shape

    net.compute_grid_transforms(projs, poses)
    net.encode(images, projs, poses, ids_encoder=[0], ids_render=[0])

    net.set_scale(0)

    # q_pts = get_pts(X_RANGE, Y_RANGE, Z_RANGE, p_res[1], p_res_y, p_res[0])
    # q_pts = q_pts.to(device).reshape(1, -1, 3)
    # # _, invalid, sigmas = net.forward(q_pts)
    #
    points = points.reshape(1, -1, 3)
    _, invalid, sigmas, segs = net.forward(points, predict_segmentation=True)

    return sigmas, segs


def convert_voxels(arr, map_dict):
    f = np.vectorize(map_dict.__getitem__)
    return f(arr)


def compute_occupancy_numbers_segmentation(y_pred, y_true, fov_mask, labels):
    label_ids = list(labels.keys())[1:]
    mask = y_true != 255
    mask = np.logical_and(mask, fov_mask)
    mask = mask.flatten()

    y_pred = y_pred.flatten()[mask]
    y_true = y_true.flatten()[mask]

    tp = np.zeros(len(label_ids))
    fp = np.zeros(len(label_ids))
    fn = np.zeros(len(label_ids))
    tn = np.zeros(len(label_ids))

    for label_id in label_ids:
        tp[label_id - 1] = np.sum(np.logical_and(y_true == label_id, y_pred == label_id))
        fp[label_id - 1] = np.sum(np.logical_and(y_true != label_id, y_pred == label_id))
        fn[label_id - 1] = np.sum(np.logical_and(y_true == label_id, y_pred != label_id))
        tn[label_id - 1] = np.sum(np.logical_and(y_true != label_id, y_pred != label_id))

    return tp, fp, tn, fn


def compute_occupancy_numbers(y_pred, y_true, fov_mask):
    mask = y_true != 255
    mask = np.logical_and(mask, fov_mask)
    mask = mask.flatten()

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    occ_true = y_true[mask] > 0
    occ_pred = y_pred[mask] > 0

    tp = np.sum(np.logical_and(occ_true == 1, occ_pred == 1))
    fp = np.sum(np.logical_and(occ_true == 0, occ_pred == 1))
    fn = np.sum(np.logical_and(occ_true == 1, occ_pred == 0))
    tn = np.sum(np.logical_and(occ_true == 0, occ_pred == 0))

    return tp, fp, tn, fn


def read_calib():
    """
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    P = np.array(
        [
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]
    ).reshape(3, 4)

    cam2velo = np.array(
        [
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
        ]
    ).reshape(3, 4)
    C2V = np.concatenate(
        [cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0
    )
    # print("C2V: ", C2V)
    V2C = np.linalg.inv(C2V)
    # print("V2C: ", V2C)
    V2C = V2C[:3, :]
    # print("V2C: ", V2C)

    # reshape matrices
    calib_out = {}
    # 3x4 projection matrix for left camera
    calib_out["P2"] = P
    calib_out["Tr"] = np.identity(4)  # 4x4 matrix
    calib_out["Tr"][:3, :4] = V2C
    return calib_out


def get_cam_k():
    cam_k = np.array(
        [
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]
    ).reshape(3, 4)
    return cam_k[:3, :3]

