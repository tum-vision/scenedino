import os
import pickle
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .re10k_util import get_target_size_and_crop, process_flow, process_img, process_proj


class RealEstate10kDataset(Dataset):
    NAME = "Re10K"

    def __init__(
            self,
            data_path: str, 
            split_path: Optional[str],
            image_size: Optional[tuple] = None,
            frame_count: int = 4,
            keyframe_offset: int = 0,
            dilation: int = 3,
            return_depth: bool = False,
            full_size_depth: bool = False,
            return_flow: bool = False,
            preprocessed_path: Optional[str] = None,
            index_selector=None,
            sequence_sampler=None,
        ):

        self.data_path = os.path.dirname(data_path)
        self.split = os.path.basename(data_path).split(".")[0]
        self.split_path = split_path
        self.image_size = image_size

        self.return_depth = return_depth
        self.full_size_depth = full_size_depth
        self.return_flow = return_flow
        self.preprocessed_path = preprocessed_path

        self.frame_count = frame_count
        self.keyframe_offset = keyframe_offset
        self.dilation = dilation

        self._left_offset = (
            (self.frame_count - 1) // 2 + self.keyframe_offset
        ) * self.dilation

        self._seq_data = self._get_sequences(data_path, self.data_path, self.split, has_split=split_path is not None)
        self._seq_keys = list(self._seq_data.keys())

        if self.split_path is not None:
            self._datapoints = self._load_split(self.split_path)
        else:
            self._left_offset = 0
            self._datapoints = self._full_split(self._seq_data, self._left_offset, (self.frame_count - 1) * dilation, sequence_sampler)

        self.index_selector = index_selector

        self.length = len(self._datapoints)

        self._skip = 0

    @staticmethod
    def _get_sequences(data_path: str, data_root: str, split: str, has_split: bool = False):
        with open(data_path, "rb") as f:
            seq_data = pickle.load(f)

        seq_data = {k: v for k, v in seq_data.items() if os.path.exists(os.path.join(data_root, "frames_720", split, k))}

        if not has_split:
            for k in seq_data.keys():
                seq_data[k]["timestamps"] = seq_data[k]["timestamps"][::10]
                seq_data[k]["poses"] = seq_data[k]["poses"][::10]
                seq_data[k]["intrinsics"] = seq_data[k]["intrinsics"][::10]

        return seq_data

    @staticmethod
    def _full_split(seq_data, left_offset: int = 0, sub_seq_len: int = 2, sequence_sampler=None):
        datapoints = []
        for k in seq_data.keys():
            seq_len = len(seq_data[k]["timestamps"])
            if sequence_sampler is not None:
                datapoints.extend(sequence_sampler(k, seq_len, left_offset, sub_seq_len))
            else:
                if seq_len < sub_seq_len:
                    continue
                for i in range(seq_len - 1): # -1 because we need at least two frames
                    datapoints.append((k, i))
        return datapoints


    def _get_id_from_timestamp(self, seq, timestamp):
        data = self._seq_data[seq]
        id = int(np.where(((data["timestamps"] / 1000).astype(np.int64)==int(timestamp)) | ((data["timestamps"]).astype(np.int64)==int(timestamp)))[0])
        return id

    def _load_split(self, split_path: str):
        def get_key_id(s):
            parts = s.split(" ")
            key = parts[0]
            t0 = parts[1]
            t1 = parts[2]
            id0 = self._get_id_from_timestamp(key, t0)
            id1 = self._get_id_from_timestamp(key, t1)
            return key, (id0, id1)

        with open(split_path, "r") as f:
            lines = f.readlines()
        datapoints = list(map(get_key_id, lines))
        return datapoints
    
    def __len__(self) -> int:
        return self.length

    def load_images(self, seq: str, ids: list):
        imgs = []

        for id in ids:
            timestamp = int(self._seq_data[seq]["timestamps"][id] / 1000)
            img = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, "frames_720", self.split, seq, f"{timestamp}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            imgs += [img]

        return imgs

    @staticmethod
    def process_pose(pose):
        pose = np.concatenate((pose.astype(np.float32), np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0)
        pose = np.linalg.inv(pose)
        return pose

    @staticmethod
    def scale_projs(proj, original_size):
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = proj[0] * original_size[1]
        K[1, 1] = proj[1] * original_size[0]
        K[0, 2] = proj[2] * original_size[1]
        K[1, 2] = proj[3] * original_size[0]
        return K

    def _index_to_seq_ids(self, index):
        if index >= self.length:
            raise IndexError()

        sequence, id = self._datapoints[index]
        seq_len = len(self._seq_data[sequence]["timestamps"])

        if type(id) != int:
            ids = id
        else:
            if self.index_selector is not None:
                ids = self.index_selector(id, self.frame_count, self.dilation, self._left_offset)
            else:
                ids = [id] + [i
                    for i in range(
                        id - self._left_offset,
                        id - self._left_offset + self.frame_count * self.dilation,
                        self.dilation,
                    )
                    if i != id
                ]

        ids = [max(min(i, seq_len - 1), 0) for i in ids]

        return sequence, ids

    def __getitem__(self, index: int):
        sequence, ids = self._index_to_seq_ids(index)

        imgs = self.load_images(sequence, ids)

        original_size = imgs[0].shape[:2]

        target_size, crop = get_target_size_and_crop(self.image_size, original_size)

        if self.return_flow:
            raise ValueError("Flow not implemented.")  # flows_fwd, flows_bwd = self.load_flows(sequence, ids)
        else:
            flows_fwd = None
            flows_bwd = None

        imgs = [process_img(img, target_size, crop) * 2.0 - 1.0 for img in imgs]

        if self.return_flow:
            flows_fwd = np.stack([process_flow(flow, target_size, crop) for flow in flows_fwd])
            flows_bwd = np.stack([process_flow(flow, target_size, crop) for flow in flows_bwd])

        # These poses are camera to world !!
        poses = [self.process_pose(self._seq_data[sequence]["poses"][i, :, :]) for i in ids]
        projs = [process_proj(self.scale_projs(self._seq_data[sequence]["intrinsics"][i, :], original_size), original_size, target_size, crop) for i in ids]

        depth = np.ones_like(imgs[0][:1, :, :])

        # print(projs[0])
        # print(poses[0])

        data = {
            "imgs": imgs,
            "projs": projs,
            "poses": poses,
            "ids": np.array(ids, dtype=np.int64),
            "index": np.array([index]),
        }

        if self.return_depth:
            data["depths"] = depth[None, ...]
        
        if self.return_flow:
            data["flows_fwd"] = flows_fwd
            data["flows_bwd"] = flows_bwd

        return data

    def get_img_paths(self, index):
        sequence, ids = self._index_to_seq_ids(index)

        img_paths = [
            os.path.join(self.data_path, "frames_720", self.split, sequence, f"{self._seq_data[sequence]['timestamps'][id]}.jpg")
            for id in ids
        ]

        return img_paths
    
    def get_sequence(self, index: int):
        sequence, _ = self._index_to_seq_ids(index)
        return sequence
