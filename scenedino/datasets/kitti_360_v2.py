import os
import random
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional
from dotdict import dotdict
import yaml

import cv2
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from scenedino.common.geometry import estimate_frustum_overlap_2
from scenedino.common.point_sampling import regular_grid

from scenedino.datasets.old_kitti_360 import FisheyeToPinholeSampler, OldKITTI360Dataset
from datasets.kitti_360.annotation import KITTI360Bbox3D
from scenedino.common.augmentation import get_color_aug_fn


def three_to_four(matrix):
    new_matrix = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
    dims = len(matrix.shape)
    new_matrix = new_matrix.view(*([1] * (dims-2)), 4, 4)
    new_matrix[..., :3, :3] = matrix
    return new_matrix


class FrameSamplingStrategy:
    """Strategy that determines how frames should be sampled around a given base frame.
    """    
    def sample(self, index, nbr_samples):
        raise NotImplementedError
    

class OverlapFrameSamplingStrategy(FrameSamplingStrategy):
    def __init__(self, n_frames, cams=("00", "01"), **kwargs) -> None:
        """Strategy to sample consecutive frames from the same cam.

        Args:
            n_frames (int): How many frames should be sampled around the given frame
            dilation (int, optional): By how much the returned frames should be separated. Defaults to 1.
        """        
        super().__init__()

        self.n_frames = n_frames
        self.cams = cams

        self.max_samples = kwargs.get("max_samples", 128)
        self.min_ratio = kwargs.get("min_ratio", .4)
        self.max_steps = kwargs.get("max_steps", 5)

        self.ranges_00 = kwargs.get("ranges_00", {
            "00": (-10, 20),
            # "01": (10, 45),
            "02": (10, 50),
            "03": (10, 50),
        })

        self.ranges_01 = kwargs.get("ranges_01", {
            # "00": self.ranges_00["01"],
            "01": self.ranges_00["00"],
            "02": self.ranges_00["02"],
            "03": self.ranges_00["03"],
        })

        self.ranges = {
            "00": self.ranges_00,
            "01": self.ranges_01,
        }
    
    def sample(self, index, nbr_samples, poses, calibs):
        poses = torch.tensor(poses, dtype=torch.float32)

        ids = []

        p_cam = random.random()
        if p_cam < .5:
            base_cam = "00"
        else:
            base_cam = "01"

        all_ranges = self.ranges[base_cam]

        # print(index, nbr_samples, len(poses), poses.shape, poses[index:index+1, :, :])

        encoder_frame = (base_cam, index)
        encoder_proj = torch.tensor(calibs["K_perspective"] if base_cam in ("00", "01") else calibs["K_fisheye"], dtype=torch.float32)[None, :, :]
        encoder_pose = poses[index:index+1, :, :] @ calibs["T_cam_to_pose"][base_cam] # [None, :, :]

        off = 1 if random.random() > .5 else -1
        target_frame = (base_cam, encoder_frame[1] + off)

        ids += [encoder_frame, target_frame]

        for i in range(self.max_samples):
            if len(ids) >= self.n_frames:
                break

            c = random.choice(list(all_ranges.keys()))
            index_offset = random.choice(range(all_ranges[c][0], all_ranges[c][1]))
            off = 1 if random.random() >= .5 else -1

            base_frame = (c, max(min(index + index_offset, nbr_samples-1), 0))
            target_frame = (c, max(min(base_frame[1] + off, nbr_samples-1), 0))

            proj = torch.tensor(calibs["K_perspective"] if c in ("00", "01") else calibs["K_fisheye"], dtype=torch.float32)[None, :, :]
            extr = poses[base_frame[1]:base_frame[1]+1, :, :] @ calibs["T_cam_to_pose"][c] #[None, :, :]

            # print(proj, extr, encoder_proj, encoder_pose)
            # print(proj.shape, extr.shape, encoder_proj.shape, encoder_pose.shape)

            overlap = estimate_frustum_overlap_2(proj, extr, encoder_proj, encoder_pose)

            overlap = overlap.item()

            # print(overlap)

            # p_keep = (overlap - self.min_ratio) / (1 - self.min_ratio)

            # if p_keep < random.random() and (self.max_samples - i) * 2 > (self.n_frames - len(ids)):
            #     print("Skip frame:", base_frame)
            #     continue

            if overlap < self.min_ratio and (self.max_samples - i) * 2 > (self.n_frames - len(ids)):
                continue

            ids += [base_frame, target_frame]

        ids = [(cam, max(min(id, nbr_samples-1), 0)) for cam, id in ids]

        # print(ids)

        return ids


class KITTI360DatasetV2(OldKITTI360Dataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._resamplers = {
            "00": None,
            "01": None,
            "02": self._resampler_02,
            "03": self._resampler_03,
        }

        self.frame_sampling_strategy = OverlapFrameSamplingStrategy(n_frames=self.frame_count)

    def load_images(self, seq, img_ids):
        imgs = []

        for cam, id, img_id in img_ids:
            path = os.path.join(
                self.data_path,
                "data_2d_raw",
                seq,
                f"image_{cam}",
                self._perspective_folder if cam in ("00", "01") else self._fisheye_folder,
                f"{img_id:010d}.png",
            )

            # print(path, os.path.exists(path))

            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB,).astype(np.float32) / 255

            imgs.append(img)

        return imgs

    def process_img(
        self,
        img: np.array,
        color_aug_fn=None,
        resampler: FisheyeToPinholeSampler = None,
    ):
        if resampler is not None and not self.is_preprocessed:
            img = torch.tensor(img).permute(2, 0, 1)
            img = resampler.resample(img)
        else:
            if self.target_image_size:
                img = cv2.resize(
                    img,
                    (self.target_image_size[1], self.target_image_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img)

        if color_aug_fn is not None:
            img = color_aug_fn(img)

        img = img * 2 - 1
        return img

    def load_depth(self, seq, img_id, cam):
        assert cam in ("00", "01")

        points = np.fromfile(
            os.path.join(
                self.data_path,
                "data_3d_raw",
                seq,
                "velodyne_points",
                "data",
                f"{img_id:010d}.bin",
            ),
            dtype=np.float32,
        ).reshape(-1, 4)
        points[:, 3] = 1.0

        T_velo_to_cam = self._calibs["T_velo_to_cam"][cam]
        K = self._calibs["K_perspective"]

        # project the points to the camera
        velo_pts_im = np.dot(K @ T_velo_to_cam[:3, :], points.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., None]

        # the projection is normalized to [-1, 1] -> transform to [0, height-1] x [0, width-1]
        velo_pts_im[:, 0] = np.round(
            (velo_pts_im[:, 0] * 0.5 + 0.5) * self.target_image_size[1]
        )
        velo_pts_im[:, 1] = np.round(
            (velo_pts_im[:, 1] * 0.5 + 0.5) * self.target_image_size[0]
        )

        # check if in bounds
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = (
            val_inds
            & (velo_pts_im[:, 0] < self.target_image_size[1])
            & (velo_pts_im[:, 1] < self.target_image_size[0])
        )
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros(self.target_image_size)
        depth[
            velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)
        ] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = (
            velo_pts_im[:, 1] * (self.target_image_size[1] - 1) + velo_pts_im[:, 0] - 1
        )
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
        depth[depth < 0] = 0

        return depth[None, :, :]

    def __getitem__(self, index: int):
        _start_time = time.time()

        if index >= self.length:
            raise IndexError()

        if self._skip != 0:
            index += self._skip

        sequence, id, is_right = self._datapoints[index]
        seq_len = self._img_ids[sequence].shape[0]

        samples = self.frame_sampling_strategy.sample(id, seq_len, self._poses[sequence], self._calibs)

        samples = [(cam, id, self.get_img_id_from_id(sequence, id)) for cam, id in samples]

        if self.color_aug:
            color_aug_fn = get_color_aug_fn(
                ColorJitter.get_params(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.1, 0.1),
                )
            )
        else:
            color_aug_fn = None

        _start_time_loading = time.time()
        imgs = self.load_images(sequence, samples)
        _loading_time = np.array(time.time() - _start_time_loading)

        _start_time_processing = time.time()
        imgs = [self.process_img(img, color_aug_fn=color_aug_fn, resampler=self._resamplers[cam]) for ((cam, id, img_id), img) in zip(samples, imgs)]
        _processing_time = np.array(time.time() - _start_time_processing)

        # These poses are camera to world !!
        poses = [self._poses[sequence][id, :, :] @ self._calibs["T_cam_to_pose"][cam] for cam, id, img_id in samples]
        
        projs = [self._calibs["K_perspective"] if cam in ("00", "01") else self._calibs["K_fisheye"] for cam, id, img_id in samples]

        ids = [id for cam, id, img_id in samples]

        if self.return_depth:
            depths = [self.load_depth(sequence, samples[0][2], samples[0][1])]
        else:
            depths = []

        if self.return_3d_bboxes:
            bboxes_3d = [self.get_3d_bboxes(sequence, samples[0][2], poses[0], projs[0])]
        else:
            bboxes_3d = []

        if self.return_segmentation:
            segs = [self.load_segmentation(sequence, samples[0][2])]
        else:
            segs = []

        _proc_time = np.array(time.time() - _start_time)

        data = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "depths": depths,
            "ts": ids,
            "3d_bboxes": bboxes_3d,
            "segs": segs,
            "t__get_item__": np.array([_proc_time]),
            "index": np.array([index]),
        }

        return data

    def __len__(self) -> int:
        return self.length
