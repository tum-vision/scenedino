from dataclasses import dataclass
import os
from random import shuffle
from typing import Any, Dict, List, Tuple
import yaml
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scenedino.common.io.images import FisheyeToPinholeSampler

from .base_dataset import BaseDataset


class KITTI360Dataset(BaseDataset):
    def __init__(
        self,
        data_path: Path,
        pose_path: Path,
        split_path: Path | None,
        target_image_size: Tuple[int, int] = (192, 640),
        return_stereo: bool = False,
        return_fisheye: bool = True,
        frame_count: int = 2,
        return_depth: bool = False,
        return_segmentation: bool = False,
        return_occupancy: bool = False,
        keyframe_offset: int = 0,
        dilation: int = 1,
        fisheye_rotation: int = 0,
        fisheye_offsets: List[int] = [10],
        stereo_offsets: List[int] = [1],
        is_preprocessed: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.pose_path = pose_path
        self.split_path = split_path
        self.target_image_size = target_image_size
        self.return_stereo = return_stereo
        self.return_fisheye = return_fisheye
        self.return_depth = return_depth
        self.return_occupancy = return_occupancy
        self.return_segmentation = return_segmentation
        self.frame_count = frame_count
        self.dilation = dilation

        self.fisheye_offsets = fisheye_offsets
        self.stereo_offsets = stereo_offsets

        self.keyframe_offset = keyframe_offset
        self._is_preprocessed = is_preprocessed

        # if isinstance(self.fisheye_rotation, float) or isinstance(
        #     self.fisheye_rotation, int
        # ):
        #     self.fisheye_rotation = (0, self.fisheye_rotation)
        # self.fisheye_rotation = tuple(self.fisheye_rotation)

        self._perspective_folder = (
            "data_rect"
            if not self._is_preprocessed
            else f"data_{self.target_image_size[0]}x{self.target_image_size[1]}"
        )
        self._calibs = self._load_calibs(self.data_path)

        self._timestamps, self._sequences = self._load_sequences(
            self.data_path, self.pose_path
        )

        self._datapoints = self._load_split(
            self.split_path, self._timestamps, self._sequences
        )

        self._resampler = FisheyeToPinholeSampler(
            self._calibs["K_00"], self.target_image_size
        )

        self.length = len(self._datapoints)

    @dataclass
    class Datapoint:
        sequence: str
        id: int
        pose: np.ndarray
        is_split: bool

    @staticmethod
    def _get_sequences(data_path: Path):
        all_sequences = []

        seqs_path = data_path / "data_2d_raw"
        for seq in seqs_path.iterdir():
            if not seq.is_dir():
                continue
            all_sequences.append(seq.name)

        return all_sequences

    @staticmethod
    def _load_calibs(data_path: Path):
        calib_folder = data_path / "calibration"
        cam_to_pose_file = calib_folder / "calib_cam_to_pose.txt"
        cam_to_velo_file = calib_folder / "calib_cam_to_velo.txt"
        intrinsics_file = calib_folder / "perspective.txt"
        fisheye_02_file = calib_folder / "image_02.yaml"
        fisheye_03_file = calib_folder / "image_03.yaml"

        cam_to_pose_data = {}
        with open(cam_to_pose_file, "r") as f:
            for line in f.readlines():
                key, value = line.split(":", 1)
                try:
                    cam_to_pose_data[key] = np.array(
                        [float(x) for x in value.split()], dtype=np.float32
                    )
                except ValueError:
                    pass

        cam_to_velo_data = None
        with open(cam_to_velo_file, "r") as f:
            line = f.readline()
            try:
                cam_to_velo_data = np.array(
                    [float(x) for x in line.split()], dtype=np.float32
                )
            except ValueError:
                pass

        intrinsics_data = {}
        with open(intrinsics_file, "r") as f:
            for line in f.readlines():
                key, value = line.split(":", 1)
                try:
                    intrinsics_data[key] = np.array(
                        [float(x) for x in value.split()], dtype=np.float32
                    )
                except ValueError:
                    pass

        with open(fisheye_02_file, "r") as f:
            f.readline()  # Skips first line that defines the YAML version
            fisheye_02_data = yaml.safe_load(f)

        with open(fisheye_03_file, "r") as f:
            f.readline()  # Skips first line that defines the YAML version
            fisheye_03_data = yaml.safe_load(f)

        im_size_rect = (
            int(intrinsics_data["S_rect_00"][1]),
            int(intrinsics_data["S_rect_00"][0]),
        )
        im_size_fish = (fisheye_02_data["image_height"], fisheye_02_data["image_width"])

        # Projection matrices
        # We use these projection matrices also when resampling the fisheye cameras.
        # This makes downstream processing easier, but it could be done differently.
        proj_rect_00 = np.reshape(intrinsics_data["P_rect_00"], (3, 4))
        proj_rect_01 = np.reshape(intrinsics_data["P_rect_01"], (3, 4))

        # Rotation matrices from raw to rectified -> Needs to be inverted later
        rotation_rect_00 = np.eye(4, dtype=np.float32)
        rotation_rect_01 = np.eye(4, dtype=np.float32)
        rotation_rect_00[:3, :3] = np.reshape(intrinsics_data["R_rect_00"], (3, 3))
        rotation_rect_01[:3, :3] = np.reshape(intrinsics_data["R_rect_01"], (3, 3))

        # Rotation matrices from resampled fisheye to raw fisheye
        # TODO: this is dummy
        fisheye_rotation = [0, 0]
        fisheye_rotation = np.array(fisheye_rotation).reshape((1, 2))
        R_02 = np.eye(4, dtype=np.float32)
        R_03 = np.eye(4, dtype=np.float32)
        R_02[:3, :3] = (
            Rotation.from_euler("xy", fisheye_rotation[:, [1, 0]], degrees=True)
            .as_matrix()
            .astype(np.float32)
        )
        R_03[:3, :3] = (
            Rotation.from_euler(
                "xy", fisheye_rotation[:, [1, 0]] * np.array([[1, -1]]), degrees=True
            )
            .as_matrix()
            .astype(np.float32)
        )

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
        T_rect_00_to_pose = T_00_to_pose @ np.linalg.inv(rotation_rect_00)
        T_rect_01_to_pose = T_01_to_pose @ np.linalg.inv(rotation_rect_01)

        # Compute cam to pose transform for fisheye cameras
        T_02_to_pose = T_02_to_pose @ R_02
        T_03_to_pose = T_03_to_pose @ R_03

        # Compute velo to cameras and velo to pose transforms
        T_velo_to_rect_00 = rotation_rect_00 @ np.linalg.inv(T_00_to_velo)
        T_velo_to_pose = T_rect_00_to_pose @ T_velo_to_rect_00
        T_velo_to_rect_01 = np.linalg.inv(T_rect_01_to_pose) @ T_velo_to_pose

        # TODO: possibly normalize image coordinates

        calibs = {
            "K_00": proj_rect_00[:3, :3],
            "K_01": proj_rect_01[:3, :3],
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
                "R_03": R_03[:3, :3],
            },
            "im_size": im_size_rect,
        }

        return calibs

    def _load_sequences(
        self, data_path: Path, pose_path: Path
    ) -> tuple[dict[str, list[Datapoint]], list[str]]:
        sequences = self._get_sequences(data_path)
        timestamps = {"pinhole": [], "fisheye": []}
        for seq in sequences:
            try:
                pose_data = np.loadtxt(pose_path / seq / f"poses.txt")
            except FileNotFoundError:
                print(f"Ground truth poses are not avaialble for sequence {seq}.")
                continue
            ids_seq = pose_data[:, 0].astype(int)
            poses_seq = pose_data[:, 1:].astype(np.float32).reshape((-1, 3, 4))
            poses_seq = np.concatenate(
                (poses_seq, np.zeros_like(poses_seq[:, :1, :])), axis=1
            )
            poses_seq[:, 3, 3] = 1

            for id, pose in zip(ids_seq, poses_seq):
                file_name = f"{id:010d}.png"

                datapoint = self.Datapoint(
                    sequence=seq, id=id, pose=pose, is_split=False
                )
                timestamps["pinhole"].append(datapoint)

                if self.return_fisheye:
                    datapoint = self.Datapoint(
                        sequence=seq, id=id, pose=pose, is_split=False
                    )
                    timestamps["fisheye"].append(datapoint)

        return timestamps, sequences

    def _load_split(
        self,
        split_path: Path,
        timestamps: Dict[str, List[Datapoint]],
        sequences: List[str],
    ):
        timestamp_idx = {seq: {} for seq in sequences}
        for idx, timestamp in enumerate(timestamps["pinhole"]):
            timestamp_idx[timestamp.sequence][timestamp.id] = idx

        with open(split_path, "r") as f:
            lines = f.readlines()

        def split_line(line):
            segments = line.split(" ")
            seq = segments[0]
            id = int(segments[1])

            return seq, id

        whole_split = list(map(split_line, lines))
        whole_split.sort()

        cut_split = []
        for idx, (seq, id) in enumerate(whole_split[: -self.frame_count]):
            keep = True
            for offset in range(1, self.frame_count):
                if whole_split[idx + offset][0] != seq:
                    keep = False
                    break
                if whole_split[idx + offset][1] != id + offset:
                    keep = False
                    break
            t_idx = timestamp_idx[seq][id]
            timestamps["pinhole"][t_idx].is_split = True
            if self.return_fisheye:
                timestamps["fisheye"][t_idx].is_split = True

            if keep:
                cut_split.append((seq, id, timestamp_idx[seq][id]))

        return cut_split

    @staticmethod
    def _load_poses(pose_path: Path, sequences: List[str]):
        ids = {}
        poses = {}

        for seq in sequences:
            pose_file = pose_path / seq / f"poses.txt"

            try:
                pose_data = np.loadtxt(pose_file)
            except FileNotFoundError:
                print(f"Ground truth poses are not avaialble for sequence {seq}.")
                continue

            ids_seq = pose_data[:, 0].astype(int)
            poses_seq = pose_data[:, 1:].astype(np.float32).reshape((-1, 3, 4))
            poses_seq = np.concatenate(
                (poses_seq, np.zeros_like(poses_seq[:, :1, :])), axis=1
            )
            poses_seq[:, 3, 3] = 1

            ids[seq] = ids_seq
            poses[seq] = poses_seq
        return ids, poses

    def get_img_id_from_id(self, sequence, id):
        return self._img_ids[sequence][id]

    def _get_img_indices(self, index) -> Dict[str, List[Path]]:
        sequence, id, is_right = self._datapoints[index]
        seq_len = self._img_ids[sequence].shape[0]

        # TODO: reorganize the splits
        load_left, load_right = (
            not is_right
        ) or self.return_stereo, is_right or self.return_stereo

        shuffle(self.stereo_offsets)
        ## randomly sample fisheye in the time steps where it can see the occlusion with the stereo
        stereo_offsets = sorted(self.stereo_offsets[: self.frame_count - 1])

        ids = [id] + [
            max(min(id + offset * self.dilation, seq_len - 1), 0)
            for offset in stereo_offsets
        ]

        img_ids = [self.get_img_id_from_id(sequence, id) for id in ids]

        pinhole_paths: List[Path] = []
        for idx in img_ids:
            if load_left:
                pinhole_paths.append(
                    self.data_path
                    / "data_2d_raw"
                    / sequence
                    / "image_00"
                    / self.perspective_folder
                    / f"{idx:010d}.png"
                )
            if load_right:
                pinhole_paths.append(
                    self.data_path
                    / "data_2d_raw"
                    / sequence
                    / "image_01"
                    / self.perspective_folder
                    / f"{idx:010d}.png"
                )

        fisheye_paths: List[Path] = []
        if self.return_fisheye:
            shuffle(self.fisheye_offsets)
            fisheye_offsets = sorted(self.fisheye_offsets[: self.frame_count])
            ids_fish = [
                max(min(id + fisheye_offsets * self.dilation, seq_len - 1), 0)
                for offset in fisheye_offsets
            ]
            img_ids_fish = [self.get_img_id_from_id(sequence, id) for id in ids_fish]

            for idx in img_ids_fish:
                if load_left:
                    pinhole_paths.append(
                        self.data_path
                        / "data_2d_raw"
                        / sequence
                        / "image_02"
                        / self._fisheye_folder
                        / f"{idx:010d}.png"
                    )
                if load_right:
                    pinhole_paths.append(
                        self.data_path
                        / "data_2d_raw"
                        / sequence
                        / "image_03"
                        / self._fisheye_folder
                        / f"{idx:010d}.png"
                    )

        if self._is_preprocessed:
            pinhole_paths.extend(fisheye_paths)

        if self.return_fisheye and not self._is_preprocessed:
            return {"pinhole": pinhole_paths, "fisheye": fisheye_paths}
        else:
            return {"pinhole": pinhole_paths}

    def load_images(self, seq, img_ids, load_left, load_right, img_ids_fish=None):
        imgs_p_left = []
        imgs_f_left = []
        imgs_p_right = []
        imgs_f_right = []

        if img_ids_fish is None:
            img_ids_fish = img_ids

        for id in img_ids:
            if load_left:
                img_perspective = (
                    cv2.cvtColor(
                        cv2.imread(
                            os.path.join(
                                self.data_path,
                                "data_2d_raw",
                                seq,
                                "image_00",
                                self._perspective_folder,
                                f"{id:010d}.png",
                            )
                        ),
                        cv2.COLOR_BGR2RGB,
                    ).astype(np.float32)
                    / 255
                )
                imgs_p_left += [img_perspective]

            if load_right:
                img_perspective = (
                    cv2.cvtColor(
                        cv2.imread(
                            os.path.join(
                                self.data_path,
                                "data_2d_raw",
                                seq,
                                "image_01",
                                self._perspective_folder,
                                f"{id:010d}.png",
                            )
                        ),
                        cv2.COLOR_BGR2RGB,
                    ).astype(np.float32)
                    / 255
                )
                imgs_p_right += [img_perspective]

        for id in img_ids_fish:
            if load_left:
                img_fisheye = (
                    cv2.cvtColor(
                        cv2.imread(
                            os.path.join(
                                self.data_path,
                                "data_2d_raw",
                                seq,
                                "image_02",
                                self._fisheye_folder,
                                f"{id:010d}.png",
                            )
                        ),
                        cv2.COLOR_BGR2RGB,
                    ).astype(np.float32)
                    / 255
                )
                imgs_f_left += [img_fisheye]
            if load_right:
                img_fisheye = (
                    cv2.cvtColor(
                        cv2.imread(
                            os.path.join(
                                self.data_path,
                                "data_2d_raw",
                                seq,
                                "image_03",
                                self._fisheye_folder,
                                f"{id:010d}.png",
                            )
                        ),
                        cv2.COLOR_BGR2RGB,
                    ).astype(np.float32)
                    / 255
                )
                imgs_f_right += [img_fisheye]

        return imgs_p_left, imgs_f_left, imgs_p_right, imgs_f_right

    def process_img(
        self,
        img: np.ndarray,
        color_aug_fn=None,
        resampler: FisheyeToPinholeSampler | None = None,
    ):
        if resampler is not None and not self._is_preprocessed:
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

    def load_depth(self, seq, img_id, is_right):
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

        T_velo_to_cam = self._calibs["T_velo_to_cam"]["00" if not is_right else "01"]
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

    def _process_image(
        self,
        img: np.ndarray,
        proj: np.ndarray,
        pose: np.ndarray,
        depth: np.ndarray | None,
        camera_type: str,
        aug_fn: dict[str, Any],
    ):
        return {
            "model": camera_type,
            "imgs": img,
            "proj": proj,
            "pose": pose,
            "depth": depth,
        }

    def _create_aug_fn(self) -> dict[str, Any]:
        return {}

    def __len__(self) -> int:
        return self.length
