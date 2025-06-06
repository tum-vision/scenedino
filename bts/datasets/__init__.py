import os
from pathlib import Path

from bts.datasets.kitti_360_v2 import KITTI360DatasetV2
from datasets.cityscapes.cityscapes_dataset import CityscapesSeg
from datasets.bdd.bdd_dataset import BDDSeg

from .base_dataset import BaseDataset
from .kitti_360 import KITTI360Dataset
from .old_kitti_360 import OldKITTI360Dataset
from .re10k_dataset import RealEstate10kDataset

import torch


# TODO: make more generic -> no more two function
def make_datasets(config) -> tuple[BaseDataset, BaseDataset]:
    dataset_type = config.get("type", "KITTI_360")
    match dataset_type:
        case "KITTI_360":
            if config.get("split_path", None) is None:
                train_split_path = None
                test_split_path = None
            else:
                train_split_path = Path(config["split_path"]) / "train_files.txt"
                test_split_path = Path(config["split_path"]) / "val_files.txt"
            train_dataset = KITTI360Dataset(
                data_path=Path(config["data_path"]),
                pose_path=Path(config["pose_path"]),
                split_path=train_split_path,
                target_image_size=tuple(config.get("image_size", (192, 640))),
                return_stereo=config.get("data_stereo", True),
                return_fisheye=config.get("data_fisheye", True),
                frame_count=config.get("data_fc", 3),
                return_depth=False,
                return_segmentation=config.get("data_segmentation", False),
                return_occupancy=False,
                keyframe_offset=config.get("keyframe_offset", 0),
                dilation=config.get("dilation", 1),
                fisheye_rotation=config.get("fisheye_rotation", 0),
                fisheye_offsets=config.get("fisheye_offset", [10]),
                stereo_offsets=config.get("stereo_offset", [1]),
                # color_aug=config.get("color_aug", False),
                is_preprocessed=config.get("is_preprocessed", False),
            )
            test_dataset = KITTI360Dataset(
                data_path=Path(config["data_path"]),
                pose_path=Path(config["pose_path"]),
                split_path=test_split_path,
                target_image_size=tuple(config.get("image_size", (192, 640))),
                return_stereo=config.get("data_stereo", True),
                return_fisheye=config.get("data_fisheye", True),
                frame_count=config.get("data_fc", 3),
                return_depth=True,
                return_segmentation=config.get("data_segmentation", False),
                return_occupancy=config.get("occupancy", False),
                keyframe_offset=config.get("keyframe_offset", 0),
                dilation=config.get("dilation", 1),
                fisheye_rotation=config.get("fisheye_rotation", 0),
                fisheye_offsets=[10],
                stereo_offsets=[1],
                is_preprocessed=config.get("is_preprocessed", False),
            )
            return train_dataset, test_dataset
        
        case "old_KITTI_360":
            if config.get("split_path", None) is None:
                train_split_path = None
                test_split_path = None
            else:
                train_split_path = Path(config["split_path"]) / "train_files.txt"
                test_split_path = Path(config["split_path"]) / "test_files.txt"
            train_dataset = OldKITTI360Dataset(
                data_path=Path(config["data_path"]),
                pose_path=Path(config["pose_path"]),
                split_path=train_split_path,
                target_image_size=tuple(config.get("image_size", (192, 640))),
                return_stereo=config.get("data_stereo", True),
                return_fisheye=config.get("data_fisheye", True),
                frame_count=config.get("data_fc", 3),
                return_depth=False,
                return_segmentation=config.get("data_segmentation", False),
                keyframe_offset=config.get("keyframe_offset", 0),
                dilation=config.get("dilation", 1),
                fisheye_rotation=config.get("fisheye_rotation", 0),
                fisheye_offset=config.get("fisheye_offset", [10]),
                # stereo_offsets=config.get("stereo_offset", [1]),
                color_aug=config.get("color_aug", False),
                is_preprocessed=config.get("is_preprocessed", False),
            )
            test_dataset = OldKITTI360Dataset(
                data_path=Path(config["data_path"]),
                pose_path=Path(config["pose_path"]),
                split_path=test_split_path,
                target_image_size=tuple(config.get("image_size", (192, 640))),
                return_stereo=config.get("data_stereo", True),
                return_fisheye=config.get("data_fisheye", True),
                frame_count=config.get("data_fc", 3),
                return_depth=True,
                return_segmentation=config.get("data_segmentation", False),
                keyframe_offset=config.get("keyframe_offset", 0),
                dilation=config.get("dilation", 1),
                fisheye_rotation=config.get("fisheye_rotation", 0),
                fisheye_offset=[10],
                # stereo_offsets=[1],
                is_preprocessed=config.get("is_preprocessed", False),
            )
            return train_dataset, test_dataset
                
        case "KITTI_360_v2":
            if config.get("split_path", None) is None:
                train_split_path = None
                test_split_path = None
            else:
                train_split_path = Path(config["split_path"]) / "train_files.txt"
                test_split_path = Path(config["split_path"]) / "val_files.txt"
            train_dataset = KITTI360DatasetV2(
                data_path=Path(config["data_path"]),
                pose_path=Path(config["pose_path"]),
                split_path=train_split_path,
                target_image_size=tuple(config.get("image_size", (192, 640))),
                return_stereo=config.get("data_stereo", True),
                return_fisheye=config.get("data_fisheye", True),
                frame_count=config.get("data_fc", 3),
                return_depth=False,
                return_segmentation=config.get("data_segmentation", False),
                keyframe_offset=config.get("keyframe_offset", 0),
                dilation=config.get("dilation", 1),
                fisheye_rotation=config.get("fisheye_rotation", 0),
                fisheye_offset=config.get("fisheye_offset", [10]),
                color_aug=config.get("color_aug", False),
                is_preprocessed=config.get("is_preprocessed", False),
            )
            test_dataset = KITTI360DatasetV2(
                data_path=Path(config["data_path"]),
                pose_path=Path(config["pose_path"]),
                split_path=test_split_path,
                target_image_size=tuple(config.get("image_size", (192, 640))),
                return_stereo=config.get("data_stereo", True),
                return_fisheye=config.get("data_fisheye", True),
                frame_count=config.get("data_fc", 3),
                return_depth=True,
                return_segmentation=config.get("data_segmentation", False),
                keyframe_offset=config.get("keyframe_offset", 0),
                dilation=config.get("dilation", 1),
                fisheye_rotation=config.get("fisheye_rotation", 0),
                fisheye_offset=[10],
                is_preprocessed=config.get("is_preprocessed", False),
            )
            return train_dataset, test_dataset

        case "Cityscapes_seg":
            train_dataset = CityscapesSeg(
                root=config["data_path"],
                image_set="train",
            )
            test_dataset = CityscapesSeg(
                root=config["data_path"],
                image_set="val",
            )
            return train_dataset, test_dataset

        case "RealEstate10K":
            if config.get("split_path", None) is None:
                train_split_path = None
                test_split_path = None
            else:
                train_split_path = Path(config["split_path"]) / "train_files.txt"
                test_split_path = Path(config["split_path"]) / "val_files.txt"

            train_dataset = RealEstate10kDataset(
                data_path=config["data_path"],
                split_path=train_split_path,
                image_size=config["image_size"],
            )
            test_dataset = RealEstate10kDataset(
                data_path=config["data_path"],
                split_path=test_split_path,
                image_size=config["image_size"],
            )
            return train_dataset, test_dataset

        case "BDD_seg":
            train_dataset = BDDSeg(
                root=config["data_path"],
                image_set="train",
            )
            test_dataset = BDDSeg(
                root=config["data_path"],
                image_set="val",
            )
            return train_dataset, test_dataset
        
        case _:
            raise NotImplementedError(f"Unsupported dataset type: {type}")


def make_test_dataset(config):
    dataset_type = config.get("type", "KITTI_Raw")
    match dataset_type:
        case "KITTI_360":
            test_dataset = OldKITTI360Dataset(
                data_path=config["data_path"],
                pose_path=config["pose_path"],
                split_path=os.path.join(
                    config.get("split_path", None), "test_files.txt"
                ),
                target_image_size=tuple(config.get("image_size", (192, 640))),
                frame_count=config.get("data_fc", 1),
                return_stereo=config.get("data_stereo", False),
                return_fisheye=config.get("data_fisheye", False),
                return_3d_bboxes=config.get("data_3d_bboxes", False),
                return_segmentation=config.get("data_segmentation", False),
                keyframe_offset=0,
                fisheye_rotation=config.get("fisheye_rotation", 0),
                fisheye_offset=config.get("fisheye_offset", 1),
                dilation=config.get("dilation", 1),
                is_preprocessed=config.get("is_preprocessed", False),
            )
            return test_dataset
        case "old_KITTI_360":
            test_dataset = OldKITTI360Dataset(
                data_path=Path(config["data_path"]),
                pose_path=Path(config["pose_path"]),
                split_path=os.path.join(
                    config.get("split_path", None), "test_files.txt"
                ),
                target_image_size=tuple(config.get("image_size", (192, 640))),
                return_stereo=config.get("data_stereo", True),
                return_fisheye=config.get("data_fisheye", True),
                frame_count=config.get("data_fc", 3),
                return_depth=True,
                return_segmentation=config.get("data_segmentation", False),
                keyframe_offset=config.get("keyframe_offset", 0),
                dilation=config.get("dilation", 1),
                fisheye_rotation=config.get("fisheye_rotation", 0),
                fisheye_offset=config.get("fisheye_offset", 1),
                # stereo_offsets=[1],
                is_preprocessed=config.get("is_preprocessed", False),
            )
            return test_dataset
        case "Cityscapes_seg":
            test_dataset = CityscapesSeg(
                root=config["data_path"],
                image_set="val",
            )
            return test_dataset
        case "RealEstate10K":
            test_dataset = RealEstate10kDataset(
                data_path=config["data_path"],
                image_size=config["image_size"],
            )
            return test_dataset
        case "BDD_seg":
            test_dataset = BDDSeg(
                root=config["data_path"],
                image_set="val",
            )
            return test_dataset
        case _:
            raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")
