import os
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from datasets.kitti_360.annotation import KITTI360Bbox3D
from datasets.kitti_360.labels import labels

from augmentation import get_color_aug_fn


name2label = {label.name: label for label in labels}
id2ProposedId = {label.id: label.trainId for label in labels}

PropsedId2TrainId = dict(enumerate(list(set(id2ProposedId.values()))))
PropsedId2TrainId = {v : k for k, v in PropsedId2TrainId.items()}
id2TrainId = {k : PropsedId2TrainId[v] for k, v in id2ProposedId.items()}


class FisheyeToPinholeSampler:
    def __init__(self, K_target, target_image_size, calibs, rotation=None):
        self._compute_transform(K_target, target_image_size, calibs, rotation)

    def _compute_transform(self, K_target, target_image_size, calibs, rotation=None):
        x = torch.linspace(-1, 1, target_image_size[1]).view(1, -1).expand(target_image_size)
        y = torch.linspace(-1, 1, target_image_size[0]).view(-1, 1).expand(target_image_size)
        z = torch.ones_like(x)
        xyz = torch.stack((x, y, z), dim=-1).view(-1, 3)

        # Unproject
        xyz = (torch.inverse(torch.tensor(K_target)) @ xyz.T).T

        if rotation is not None:
            xyz = (torch.tensor(rotation) @ xyz.T).T

        # Backproject into fisheye
        xyz = xyz / torch.norm(xyz, dim=-1, keepdim=True)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        xi_src = calibs["mirror_parameters"]["xi"]
        x = x / (z + xi_src)
        y = y / (z + xi_src)

        k1 = calibs["distortion_parameters"]["k1"]
        k2 = calibs["distortion_parameters"]["k2"]

        r = x*x + y*y
        factor = (1 + k1 * r + k2 * r * r)
        x = x * factor
        y = y * factor

        gamma0 = calibs["projection_parameters"]["gamma1"]
        gamma1 = calibs["projection_parameters"]["gamma2"]
        u0 = calibs["projection_parameters"]["u0"]
        v0 = calibs["projection_parameters"]["v0"]

        x = x * gamma0 + u0
        y = y * gamma1 + v0

        xy = torch.stack((x, y), dim=-1).view(1, *target_image_size, 2)
        self.sample_pts = xy

    def resample(self, img):
        img = img.unsqueeze(0)
        resampled_img = F.grid_sample(img, self.sample_pts, align_corners=True).squeeze(0)
        return resampled_img


class SSCBenchDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 voxel_gt_path: str,
                 sequences: Optional[tuple],
                 target_image_size=(192, 640),
                 return_stereo=False,
                 return_depth=False,
                 data_segmentation_path=None,
                 frame_count=2,
                 keyframe_offset=0,
                 dilation=1,
                 eigen_depth=True,
                 color_aug=False,
                 load_kitti_360_segmentation_gt=False,
                 load_all=False,
                 load_fisheye=False,
                 fisheye_offset=0,
                 ):
        self.data_path = Path(data_path)
        self.voxel_gt_path = Path(voxel_gt_path)
        self.data_segmentation_path = data_segmentation_path
        self.pose_path = os.path.join("<PATH-KITTI-360-DATA-POSES>")
        self.target_image_size = target_image_size
        self.return_stereo = return_stereo
        self.return_depth = return_depth

        self.frame_count = frame_count
        self.dilation = dilation
        self.keyframe_offset = keyframe_offset
        self.eigen_depth = eigen_depth
        self.color_aug = color_aug
        self.load_kitti_360_segmentation_gt = load_kitti_360_segmentation_gt
        self.load_all = load_all
        self.load_fisheye = load_fisheye
        self.fisheye_offset = fisheye_offset

        if sequences is None:
            self._sequences = self._get_sequences(self.data_path)
        else:
            self._sequences = [f"2013_05_28_drive_00{s:02d}_sync" for s in sequences]

        self._calibs = self._load_calibs(self.data_path)
        self._left_offset = ((self.frame_count - 1) // 2 + self.keyframe_offset) * self.dilation

        self._img_ids, self._poses = self._load_poses(self.pose_path, self._sequences)
        self._perspective_folder = "data_rect"
        self._segmentation_perspective_folder = "data_192x640"
        self._segmentation_fisheye_folder = "data_192x640_0x-15"

        if self.load_all:
            self._datapoints = self._load_all_datapoints(self.data_path, self._sequences)
        else:
            self._datapoints = self._load_datapoints(self.voxel_gt_path, self._sequences)

        self._skip = 0
        self.length = len(self._datapoints)

    @staticmethod
    def _get_sequences(data_path):
        all_sequences = []

        seqs_path = Path(data_path) / "data_2d_raw"
        for seq in seqs_path.iterdir():
            if not seq.is_dir():
                continue
            all_sequences.append(seq.name)

        return all_sequences

    @staticmethod
    def _load_calibs(data_path, fisheye_rotation=(0, 0)):
        data_path = Path(data_path)

        calib_folder = data_path / "calibration"
        cam_to_pose_file = calib_folder / "calib_cam_to_pose.txt"
        cam_to_velo_file = calib_folder / "calib_cam_to_velo.txt"
        intrinsics_file = calib_folder / "perspective.txt"
        fisheye_02_file = calib_folder / "image_02.yaml"
        fisheye_03_file = calib_folder / "image_03.yaml"

        cam_to_pose_data = {}
        with open(cam_to_pose_file, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    cam_to_pose_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                except ValueError:
                    pass

        cam_to_velo_data = None
        with open(cam_to_velo_file, 'r') as f:
            line = f.readline()
            try:
                cam_to_velo_data = np.array([float(x) for x in line.split()], dtype=np.float32)
            except ValueError:
                pass

        intrinsics_data = {}
        with open(intrinsics_file, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                try:
                    intrinsics_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                except ValueError:
                    pass

        with open(fisheye_02_file, 'r') as f:
            f.readline() # Skips first line that defines the YAML version
            fisheye_02_data = yaml.safe_load(f)

        with open(fisheye_03_file, 'r') as f:
            f.readline() # Skips first line that defines the YAML version
            fisheye_03_data = yaml.safe_load(f)

        im_size_rect = (int(intrinsics_data["S_rect_00"][1]), int(intrinsics_data["S_rect_00"][0]))
        im_size_fish = (fisheye_02_data["image_height"], fisheye_02_data["image_width"])

        # Projection matrices
        # We use these projection matrices also when resampling the fisheye cameras.
        # This makes downstream processing easier, but it could be done differently.
        P_rect_00 = np.reshape(intrinsics_data['P_rect_00'], (3, 4))
        P_rect_01 = np.reshape(intrinsics_data['P_rect_01'], (3, 4))

        # Rotation matrices from raw to rectified -> Needs to be inverted later
        R_rect_00 = np.eye(4, dtype=np.float32)
        R_rect_01 = np.eye(4, dtype=np.float32)
        R_rect_00[:3, :3] = np.reshape(intrinsics_data['R_rect_00'], (3, 3))
        R_rect_01[:3, :3] = np.reshape(intrinsics_data['R_rect_01'], (3, 3))

        # Rotation matrices from resampled fisheye to raw fisheye
        fisheye_rotation = np.array(fisheye_rotation).reshape((1, 2))
        R_02 = np.eye(4, dtype=np.float32)
        R_03 = np.eye(4, dtype=np.float32)
        R_02[:3, :3] = Rotation.from_euler("xy", fisheye_rotation[:, [1, 0]], degrees=True).as_matrix().astype(np.float32)
        R_03[:3, :3] = Rotation.from_euler("xy", fisheye_rotation[:, [1, 0]] * np.array([[1, -1]]), degrees=True).as_matrix().astype(np.float32)

        # Load cam to pose transforms
        T_00_to_pose = np.eye(4, dtype=np.float32)
        T_01_to_pose = np.eye(4, dtype=np.float32)
        T_02_to_pose = np.eye(4, dtype=np.float32)
        T_03_to_pose = np.eye(4, dtype=np.float32)
        T_00_to_velo = np.eye(4, dtype=np.float32)

        T_00_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_00"], (3, 4))
        T_01_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_01"], (3, 4))
        T_02_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_02"], (3, 4))
        T_03_to_pose[:3, :] = np.reshape(cam_to_pose_data["image_03"], (3, 4))
        T_00_to_velo[:3, :] = np.reshape(cam_to_velo_data, (3, 4))

        # Compute cam to pose transforms for rectified perspective cameras
        T_rect_00_to_pose = T_00_to_pose @ np.linalg.inv(R_rect_00)
        T_rect_01_to_pose = T_01_to_pose @ np.linalg.inv(R_rect_01)

        # Compute cam to pose transform for fisheye cameras
        T_02_to_pose = T_02_to_pose @ R_02
        T_03_to_pose = T_03_to_pose @ R_03

        # Compute velo to cameras and velo to pose transforms
        T_velo_to_rect_00 = R_rect_00 @ np.linalg.inv(T_00_to_velo)
        T_velo_to_pose = T_rect_00_to_pose @ T_velo_to_rect_00
        T_velo_to_rect_01 = np.linalg.inv(T_rect_01_to_pose) @ T_velo_to_pose

        # Calibration matrix is the same for both perspective cameras
        K = P_rect_00[:3, :3]

        # Normalize calibration
        f_x = K[0, 0] / im_size_rect[1]
        f_y = K[1, 1] / im_size_rect[0]
        c_x = K[0, 2] / im_size_rect[1]
        c_y = K[1, 2] / im_size_rect[0]

        # Change to image coordinates [-1, 1]
        K[0, 0] = f_x * 2.
        K[1, 1] = f_y * 2.
        K[0, 2] = c_x * 2. - 1
        K[1, 2] = c_y * 2. - 1

        # Convert fisheye calibration to [-1, 1] image dimensions
        fisheye_02_data["projection_parameters"]["gamma1"] = (fisheye_02_data["projection_parameters"]["gamma1"] / im_size_fish[1]) * 2.
        fisheye_02_data["projection_parameters"]["gamma2"] = (fisheye_02_data["projection_parameters"]["gamma2"] / im_size_fish[0]) * 2.
        fisheye_02_data["projection_parameters"]["u0"] = (fisheye_02_data["projection_parameters"]["u0"] / im_size_fish[1]) * 2. - 1.
        fisheye_02_data["projection_parameters"]["v0"] = (fisheye_02_data["projection_parameters"]["v0"] / im_size_fish[0]) * 2. - 1.

        fisheye_03_data["projection_parameters"]["gamma1"] = (fisheye_03_data["projection_parameters"]["gamma1"] / im_size_fish[1]) * 2.
        fisheye_03_data["projection_parameters"]["gamma2"] = (fisheye_03_data["projection_parameters"]["gamma2"] / im_size_fish[0]) * 2.
        fisheye_03_data["projection_parameters"]["u0"] = (fisheye_03_data["projection_parameters"]["u0"] / im_size_fish[1]) * 2. - 1.
        fisheye_03_data["projection_parameters"]["v0"] = (fisheye_03_data["projection_parameters"]["v0"] / im_size_fish[0]) * 2. - 1.

        # Use same camera calibration as perspective cameras for resampling
        # K_fisheye = np.eye(3, dtype=np.float32)
        # K_fisheye[0, 0] = 2
        # K_fisheye[1, 1] = 2

        K_fisheye = K

        calibs = {
            "K_perspective": K,
            "K_fisheye": K_fisheye,
            "T_cam_to_pose": {
                "00": T_rect_00_to_pose,
                "01": T_rect_01_to_pose,
                "02": T_02_to_pose,
                "03": T_03_to_pose,
            },
            "T_velo_to_cam": {
                "00": T_velo_to_rect_00,
                "01": T_velo_to_rect_01,
            },
            "T_velo_to_pose": T_velo_to_pose,
            "fisheye": {
                "calib_02": fisheye_02_data,
                "calib_03": fisheye_03_data,
                "R_02": R_02[:3, :3],
                "R_03": R_03[:3, :3]
            },
            "im_size": im_size_rect
        }

        return calibs

    @staticmethod
    def _load_poses(pose_path, sequences):
        ids = {}
        poses = {}

        for seq in sequences:
            pose_file = Path(pose_path) / seq / f"poses.txt"

            try:
                pose_data = np.loadtxt(pose_file)
            except FileNotFoundError:
                print(f'Ground truth poses are not avaialble for sequence {seq}, {pose_file}.')

            ids_seq = pose_data[:, 0].astype(int)
            poses_seq = pose_data[:, 1:].astype(np.float32).reshape((-1, 3, 4))
            poses_seq = np.concatenate((poses_seq, np.zeros_like(poses_seq[:, :1, :])), axis=1)
            poses_seq[:, 3, 3] = 1

            ids[seq] = ids_seq
            poses[seq] = poses_seq
        return ids, poses

    @staticmethod
    def _get_resamplers(calibs, K_target, target_image_size):
        resampler_02 = FisheyeToPinholeSampler(K_target, target_image_size, calibs["fisheye"]["calib_02"], calibs["fisheye"]["R_02"])
        resampler_03 = FisheyeToPinholeSampler(K_target, target_image_size, calibs["fisheye"]["calib_03"], calibs["fisheye"]["R_03"])

        return resampler_02, resampler_03

    @staticmethod
    def _load_datapoints(voxel_gt_path, sequences):
        datapoints = []
        for seq in sorted(sequences):
            ids = [int(file.name[:6]) for file in sorted((voxel_gt_path / seq).glob("*_1_1.npy"))]
            datapoints_seq = [(seq, id, False) for id in ids]
            datapoints.extend(datapoints_seq)
        return datapoints

    @staticmethod
    def _load_all_datapoints(voxel_gt_path, sequences):
        datapoints = []
        for seq in sorted(sequences):
            ids = [int(file.name[:6]) for file in sorted((voxel_gt_path / 'data_2d_raw' / seq / 'image_00' / 'data_rect').glob("*.png"))]
            datapoints_seq = [(seq, id, False) for id in ids]
            datapoints.extend(datapoints_seq)
        return datapoints

    def load_images(self, seq, img_ids):
        imgs_p_left = []

        for id in img_ids:
            # id = self._img_ids[seq][id]
            img_perspective = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, "data_2d_raw", seq, "image_00", self._perspective_folder, f"{id:06d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

            imgs_p_left += [img_perspective]

        return imgs_p_left
    
    def load_fisheye_images(self, seq, img_ids):
        imgs_f_left, imgs_f_right = [], []

        for id in img_ids:
            #img_fisheye = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, "data_2d_raw", seq, "image_02", self._segmentation_fisheye_folder, f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            #img_fisheye = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, "data_2d_raw", seq, "image_03", self._segmentation_fisheye_folder, f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

            id = self._img_ids[seq][id]
            
            img_fisheye_left = cv2.cvtColor(cv2.imread(os.path.join("<PATH-KITTI-360-DATA-POSES>", "data_2d_raw", seq, "image_02", self._segmentation_fisheye_folder, f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            img_fisheye_right = cv2.cvtColor(cv2.imread(os.path.join("<PATH-KITTI-360-DATA-POSES>", "data_2d_raw", seq, "image_03", self._segmentation_fisheye_folder, f"{id:010d}.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

            imgs_f_left += [img_fisheye_left]
            imgs_f_right += [img_fisheye_right]

        return imgs_f_left, imgs_f_right

    def load_voxel_gt(self, sequence, img_ids):
        voxel_gt = []

        for id in img_ids:
            target_1_path = os.path.join(self.voxel_gt_path, sequence, f"{id:06d}" + "_1_1.npy")

            if not self.load_all or os.path.isfile(target_1_path):
                voxel_gt.append(np.load(target_1_path))
            else:
                voxel_gt.append(None)

        return voxel_gt

    def process_img(self, img: np.array, color_aug_fn=None, resampler:FisheyeToPinholeSampler=None):
        if resampler is not None and not self.is_preprocessed:
            img = torch.tensor(img).permute(2, 0, 1)
            img = resampler.resample(img)
        else:
            if self.target_image_size:
                img = cv2.resize(img, (self.target_image_size[1], self.target_image_size[0]), interpolation=cv2.INTER_LINEAR)
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img)

        if color_aug_fn is not None:
            img = color_aug_fn(img)

        img = img * 2 - 1
        return img

    def load_depth(self, seq, img_id, is_right):
        points = np.fromfile(os.path.join(self.data_path, "data_3d_raw", seq, "velodyne_points", "data", f"{img_id:010d}.bin"), dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0

        T_velo_to_cam = self._calibs["T_velo_to_cam"]["00" if not is_right else "01"]
        K = self._calibs["K_perspective"]

        # project the points to the camera
        velo_pts_im = np.dot(K @ T_velo_to_cam[:3, :], points.T).T
        velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., None]

        # the projection is normalized to [-1, 1] -> transform to [0, height-1] x [0, width-1]
        velo_pts_im[:, 0] = np.round((velo_pts_im[:, 0] * .5 + .5) * self.target_image_size[1])
        velo_pts_im[:, 1] = np.round((velo_pts_im[:, 1] * .5 + .5) * self.target_image_size[0])

        # check if in bounds
        val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
        val_inds = val_inds & (velo_pts_im[:, 0] < self.target_image_size[1]) & (velo_pts_im[:, 1] < self.target_image_size[0])
        velo_pts_im = velo_pts_im[val_inds, :]

        # project to image
        depth = np.zeros(self.target_image_size)
        depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]

        # find the duplicate points and choose the closest depth
        inds = velo_pts_im[:, 1] * (self.target_image_size[1] - 1) + velo_pts_im[:, 0] - 1
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

        load_left = (not is_right) or self.return_stereo
        load_right = is_right or self.return_stereo

        ids = [id]
        ids_fish = [id + self.fisheye_offset]

        if self.color_aug:
            color_aug_fn = get_color_aug_fn(ColorJitter.get_params(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)))
        else:
            color_aug_fn = None

        _start_time_loading = time.time()
        imgs_p_left = self.load_images(sequence, ids)
        imgs_f_left, imgs_f_right = self.load_fisheye_images(sequence, ids_fish)
        voxel_gt = self.load_voxel_gt(sequence, ids)
        _loading_time = np.array(time.time() - _start_time_loading)

        _start_time_processing = time.time()
        imgs_p_left = [self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_p_left]
        imgs_f_left = [self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_f_left]
        imgs_f_right = [self.process_img(img, color_aug_fn=color_aug_fn) for img in imgs_f_right]
        _processing_time = np.array(time.time() - _start_time_processing)

        # These poses are not camera to world !!
        poses_p_left = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["00"] for i in ids] if load_left else []
        poses_f_left = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["02"] for i in ids_fish]
        poses_f_right = [self._poses[sequence][i, :, :] @ self._calibs["T_cam_to_pose"]["03"] for i in ids_fish]

        projs_p_left = [self._calibs["K_perspective"] for _ in ids] if load_left else []
        projs_f_left = [self._calibs["K_fisheye"] for _ in ids_fish]
        projs_f_right = [self._calibs["K_fisheye"] for _ in ids_fish]

        imgs = imgs_p_left
        projs = projs_p_left
        poses = poses_p_left

        if self.load_fisheye:
            imgs += imgs_f_left + imgs_f_right
            projs += projs_f_left + projs_f_right
            poses += poses_f_left + poses_f_right

        _proc_time = np.array(time.time() - _start_time)

        # print(_loading_time, _processing_time, _proc_time)

        data = {
            "imgs": imgs,
            "projs": projs,
            "voxel_gt": voxel_gt,
            "poses": poses,
            "t__get_item__": np.array([_proc_time]),
            "index": np.array([index])
        }

        return data

    def __len__(self) -> int:
        return self.length
