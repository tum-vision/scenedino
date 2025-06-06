from abc import abstractmethod
import time
from typing import Any
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    # @abstractmethod
    # def _get_img_indices(self, index) -> dict[str, list[Any]]:
    #     pass

    # @abstractmethod
    # def _load_image(self, unique_id: Any) -> np.ndarray:
    #     pass

    # @abstractmethod
    # def _load_depth_map(self, unique_id: Any) -> np.ndarray | None:
    #     pass

    # @abstractmethod
    # def _get_pose(self, unique_id: Any) -> np.ndarray:
    #     pass

    # @abstractmethod
    # def _get_calib(self, unique_id: Any) -> np.ndarray:
    #     pass

    # @abstractmethod
    # def _load_occ(self, idx) -> np.ndarray | None:
    #     pass

    # TODO: Check if needs to return the values
    @staticmethod
    @abstractmethod
    def _process_image(
        img: np.ndarray,
        proj: np.ndarray,
        pose: np.ndarray,
        depth: np.ndarray | None,
        camera_type: str,
        aug_fn: dict[str, Any],
    ):
        pass

    @abstractmethod
    def _create_aug_fn(self) -> dict[str, Any]:
        pass

    def __getitem__(self, index) -> dict[str, Any]:
        _start_time = time.time()

        img_paths = self._get_img_indices(index)
        occ = self._load_occ(index)

        aug_fn = self._create_aug_fn()

        frames = []
        for camera_type, unique_id in img_paths.items():
            img = self._load_image(unique_id)
            proj = self._get_calib(unique_id)
            pose = self._get_pose(unique_id)
            depth = self._load_depth_map(unique_id)

            self._process_image(img, proj, pose, depth, camera_type, aug_fn)

            frames.append(
                {
                    "model": camera_type,
                    "imgs": img,
                    "proj": proj,
                    "pose": pose,
                    "depth": depth,
                }
            )
        _proc_time = np.array(time.time() - _start_time)

        return {
            "frames": frames,
            "occ": occ,
            "__t_get_item__": np.array([_proc_time]),
        }
