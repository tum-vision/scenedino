from random import shuffle
import random
from typing import Callable, Optional
import numpy as np

import torch


EncoderSamplingStrategy = Callable[[int], list[int]]
LossSamplingStrategy = Callable[[int], tuple[list[int], list[int], Optional[list[list[bool]]]]]


# ============================================ ENCODING SAMPLING STRATEGIES ============================================
def default_encoder_sampler() -> EncoderSamplingStrategy:
    def _sampling_strategy(num_frames: int) -> list[int]:
        return [0]

    return _sampling_strategy


def kitti_360_full_encoder_sampler(
    num_encoder_frames: int, always_use_base_frame: bool = True
) -> EncoderSamplingStrategy:
    def _sampling_strategy(num_frames: int) -> list[int]:
        if always_use_base_frame:
            encoder_perm = (torch.randperm(num_frames - 1) + 1)[
                : num_encoder_frames - 1
            ].tolist()
            ids_encoder = [0]
            ids_encoder.extend(encoder_perm)
        else:
            ids_encoder = (torch.randperm(num_frames - 1) + 1)[
                :num_encoder_frames
            ].tolist()
        return ids_encoder

    return _sampling_strategy


def kitti_360_stereo_encoder_sampler(
    num_encoder_frames: int, num_stereo_frames: int, always_use_base_frame: bool = True
) -> EncoderSamplingStrategy:
    def _sampling_strategy(num_frames: int) -> list[int]:
        num_frames = min(num_frames, num_stereo_frames)

        if always_use_base_frame:
            encoder_perm = (torch.randperm(num_frames - 1) + 1)[
                : num_encoder_frames - 1
            ].tolist()
            ids_encoder = [0]
            ids_encoder.extend(encoder_perm)
        else:
            ids_encoder = (torch.randperm(num_frames - 1) + 1)[
                :num_encoder_frames
            ].tolist()
        return ids_encoder

    return _sampling_strategy


def get_encoder_sampling(config) -> EncoderSamplingStrategy:
    strategy = config.get("name", None)
    match strategy:
        case "kitti_360_full":
            return kitti_360_full_encoder_sampler(**config["args"])
        case "kitti_360_stereo":
            return kitti_360_stereo_encoder_sampler(**config["args"])
        case _:
            return default_encoder_sampler()


# =============================================== LOSS SAMPLING STRATEGIES =============================================
def single_view_loss_sampler(
    shuffle_frames: bool = False, all_frames: bool = False
) -> LossSamplingStrategy:
    if all_frames:
        starting_frame = 0
    else:
        starting_frame = 1

    def _sampling_strategy(num_frames: int) -> tuple[list[int], list[int]]:
        frames = [id for id in range(num_frames)]
        if shuffle_frames:
            shuffle(frames)
        return frames[0:1], frames[starting_frame:], None

    return _sampling_strategy


def single_view_renderer_sampler(
    shuffle_frames: bool = False, all_frames: bool = False
) -> LossSamplingStrategy:
    def _sampling_strategy(num_frames: int) -> tuple[list[int], list[int]]:
        frames = [id for id in range(num_frames)]
        if shuffle_frames:
            shuffle(frames)
        if all_frames:
            return frames, frames[0:1], None
        else:
            return frames[0:-1], frames[0:1], None

    return _sampling_strategy


def stereo_view_loss_sampler(shuffle_frames: bool = False) -> LossSamplingStrategy:
    def _sampling_strategy(num_frames: int) -> tuple[list[int], list[int]]:
        all_frames = [id for id in range(num_frames)]
        if shuffle_frames:
            shuffle(all_frames)
        if all_frames[0] < num_frames // 2:
            ids_loss = list(range(num_frames // 2))
            ids_renderer = list(range(num_frames // 2, num_frames))
        else:
            ids_renderer = list(range(num_frames // 2))
            ids_loss = list(range(num_frames // 2, num_frames))

        return ids_loss, ids_renderer, None

    return _sampling_strategy


def kitti_360_loss_sampler() -> LossSamplingStrategy:
    def _sampling_strategy(num_frames: int) -> tuple[list[int], list[int]]:
        ids_loss: list[int] = []
        ids_renderer: list[int] = []
        for cam_pair_base_id in range(0, num_frames, 2):
            if random.randint(0, 2):
                ids_loss.append(cam_pair_base_id)
                ids_renderer.append(cam_pair_base_id + 1)
            else:
                ids_loss.append(cam_pair_base_id + 1)
                ids_renderer.append(cam_pair_base_id)

        return ids_loss, ids_renderer, None

    return _sampling_strategy


def kitti_360_loss_sampler() -> LossSamplingStrategy:
    def _sampling_strategy(num_frames: int) -> tuple[list[int], list[int]]:
        ids_loss: list[int] = []
        ids_renderer: list[int] = []
        for cam_pair_base_id in range(0, num_frames, 2):
            if random.randint(0, 2):
                ids_loss.append(cam_pair_base_id)
                ids_renderer.append(cam_pair_base_id + 1)
            else:
                ids_loss.append(cam_pair_base_id + 1)
                ids_renderer.append(cam_pair_base_id)

        return ids_loss, ids_renderer, None

    return _sampling_strategy


def kitti_360_with_mapping_loss_sampler() -> LossSamplingStrategy:
    def _sampling_strategy(num_frames: int) -> tuple[list[int], list[int]]:
        ids_loss: list[int] = []
        ids_renderer: list[int] = []
        mapping = []
        for cam_pair_base_id in range(0, num_frames, 2):
            if random.randint(0, 2):
                ids_loss.append(cam_pair_base_id)
                ids_renderer.append(cam_pair_base_id + 1)
                mapping.append([len(ids_renderer) - 1])
            else:
                ids_loss.append(cam_pair_base_id + 1)
                ids_renderer.append(cam_pair_base_id)
                mapping.append([len(ids_renderer) - 1])

        mapping = np.array(mapping, dtype=np.int64)

        return ids_loss, ids_renderer, mapping

    return _sampling_strategy


def waymo_with_mapping_loss_sampler() -> LossSamplingStrategy:
    def _sampling_strategy(num_frames: int) -> tuple[list[int], list[int]]:
        ids_loss: list[int] = []
        ids_renderer: list[int] = []
        mapping = []
        for cam_pair_base_id in range(0, num_frames, 2):
            if random.randint(0, 2):
                ids_loss.append(cam_pair_base_id)
                ids_renderer.append(cam_pair_base_id + 1)
                mapping.extend([[len(ids_renderer) - 1], [len(ids_renderer) - 1]])
            else:
                ids_loss.append(cam_pair_base_id + 1)
                ids_renderer.append(cam_pair_base_id)
                mapping.extend([[len(ids_renderer) - 1], [len(ids_renderer) - 1]])

        mapping = np.array(mapping, dtype=np.int64)

        return ids_loss, ids_renderer, mapping

    return _sampling_strategy


def alternate_loss_sampler() -> LossSamplingStrategy:
    def _sampling_strategy(num_frames: int) -> tuple[list[int], list[int]]:
        frames = [id for id in range(num_frames)]
        if random.randint(0, 2):
            return list(range(0, num_frames, 2)), list(range(1, num_frames, 2)), None
        else:
            return list(range(1, num_frames, 2)), list(range(0, num_frames, 2)), None

    return _sampling_strategy


def get_loss_renderer_sampling(config) -> EncoderSamplingStrategy:
    strategy = config.get("name", None)
    match strategy:
        case "single_loss":
            return single_view_loss_sampler(**config.get("args", {}))
        case "single_renderer":
            return single_view_renderer_sampler(**config.get("args", {}))
        case "stereo_loss":
            return stereo_view_loss_sampler(**config.get("args", {}))
        case "kitti_360":
            return kitti_360_loss_sampler()
        case "kitti_360_with_mapping":
            return kitti_360_with_mapping_loss_sampler()
        case "waymo_with_mapping":
            return waymo_with_mapping_loss_sampler()
        case "alternate":
            return alternate_loss_sampler()
        case _:
            return single_view_loss_sampler(False)


# old sampling strategies

# if self.training:
#     frame_perm = torch.randperm(v)
# else:
#     frame_perm = torch.arange(v)  ## eval

# if self.enc_style == "random":  ## encoded views
#     encoder_perm = (torch.randperm(v - 1) + 1)[
#         : self.nv_ - 1
#     ].tolist()  ## nv-1 for mono [0] idx
#     ids_encoder = [0]  ## always starts sampling from mono cam
#     ids_encoder.extend(encoder_perm)  ## add more cam_views randomly incl. fe
# elif self.enc_style == "default":
#     ids_encoder = [
#         v_ for v_ in range(self.nv_)
#     ]  ## iterating view(v_) over num_views(nv_)
# elif self.enc_style == "stereo":
#     if self.training:
#         # if v < 8:   raise RuntimeError(f"__number of views should be more than 4 when excluding fisheye views")
#         # if v < 8:   raise RuntimeError(f"__number of views should be more than 4 when excluding fisheye views")
#         encoder_perm = (torch.randperm(v - (1 + 4)) + 1)[
#             : self.nv_ - 1
#         ].tolist()
#         ids_encoder = [0]
#         ids_encoder.extend(encoder_perm)
#     else:
#         ids_encoder = [0]
# else:
#     raise NotImplementedError(f"__unrecognized enc_style: {self.enc_style}")
# ## default: ids_encoder = [0,1,2,3] <=> front stereo for 1st + 2nd time stamps

# if (
#     not self.training and self.ids_enc_viz_eval
# ):  ## when eval in viz to be standardized with test:  it's eval from line 354, base_trainer.py
#     ids_encoder = self.ids_enc_viz_eval  ## fixed during eval

# ids_render = torch.sort(
#     frame_perm[[i for i in self.frames_render if i < v]]
# ).values  ## ?    ### tensor([0, 4])

# combine_ids = None

# if self.training:
#     if self.frame_sample_mode == "only":
#         ids_loss = [0]
#         ids_render = ids_render[ids_render != 0]

#     elif self.frame_sample_mode == "not":
#         frame_perm = torch.randperm(v - 1) + 1
#         ids_loss = torch.sort(
#             frame_perm[[i for i in self.frames_render if i < v - 1]]
#         ).values
#         ids_render = [i for i in range(v) if i not in ids_loss]

#     elif self.frame_sample_mode == "stereo":
#         if frame_perm[0] < v // 2:
#             ids_loss = list(range(v // 2))
#             ids_render = list(range(v // 2, v))
#         else:
#             ids_loss = list(range(v // 2, v))
#             ids_render = list(range(v // 2))

#     elif self.frame_sample_mode == "mono":
#         split_i = v // 2
#         if frame_perm[0] < v // 2:
#             ids_loss = list(range(0, split_i, 2)) + list(
#                 range(split_i + 1, v, 2)
#             )
#             ids_render = list(range(1, split_i, 2)) + list(range(split_i, v, 2))
#         else:
#             ids_loss = list(range(1, split_i, 2)) + list(range(split_i, v, 2))
#             ids_render = list(range(0, split_i, 2)) + list(
#                 range(split_i + 1, v, 2)
#             )

#     elif self.frame_sample_mode == "kitti360-mono":
#         steps = v // 4
#         start_from = 0 if frame_perm[0] < v // 2 else 1

#         ids_loss, ids_render = [], []

#         for cam in range(
#             4
#         ):  ## stereo cam sampled for each time     ## ! c.f. paper: N_{render}, N_{loss}
#             ids_loss += [cam * steps + i for i in range(start_from, steps, 2)]
#             ids_render += [
#                 cam * steps + i for i in range(1 - start_from, steps, 2)
#             ]
#             start_from = 1 - start_from

#         if self.enc_style == "test":
#             ids_encoder = ids_loss[: self.nv_]

#     elif self.frame_sample_mode.startswith("waymo"):
#         num_views = int(self.frame_sample_mode.split("-")[-1])
#         steps = v // num_views
#         split = steps // 2

#         # Predict features from half-left, center, half-right
#         ids_encoder = [0, steps, steps * 2]

#         # Combine all frames half-left, center, half-right for efficiency reasons
#         combine_ids = [(i, steps + i, steps * 2 + i) for i in range(steps)]

#         if self.training:
#             step_perm = torch.randperm(steps)
#         else:
#             step_perm = torch.arange(steps)  ## eval
#         step_perm = step_perm.tolist()

#         ids_loss = sum(
#             [
#                 [i + j * steps for j in range(num_views)]
#                 for i in step_perm[:split]
#             ],
#             [],
#         )
#         ids_render = sum(
#             [
#                 [i + j * steps for j in range(num_views)]
#                 for i in step_perm[split:]
#             ],
#             [],
#         )

#     elif self.frame_sample_mode == "default":
#         ids_loss = frame_perm[
#             [i for i in range(v) if frame_perm[i] not in ids_render]
#         ]
#     else:
#         raise NotImplementedError

# else:  ## eval (!= self.training)
#     ids_loss = torch.arange(v)
#     ids_render = [0]

#     if self.frame_sample_mode.startswith("waymo"):
#         num_views = int(self.frame_sample_mode.split("-")[-1])
#         steps = v // num_views
#         split = steps // 2
#         # Predict features from half-left, center, half-right
#         ids_encoder = [0, steps, steps * 2]
#         ids_render = [0, steps, steps * 2]
#         combine_ids = [(i, steps + i, steps * 2 + i) for i in range(steps)]
