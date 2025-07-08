from typing import Iterable

import cv2
import numpy as np


def get_target_size_and_crop(target_size, image_size):
    if target_size is None:
        return None, None
    elif not isinstance(target_size, Iterable): # Resize the shorter side to target size and center crop to get a square
        if image_size[0] <= image_size[1]:
            aspect_ratio = image_size[0] / image_size[1]
            resize_to = (target_size, int(round(target_size / aspect_ratio)))

            if resize_to[1] > target_size:
                crop = ((resize_to[1] - target_size) // 2, 0, target_size, target_size)
            else:
                crop = None

        else:
            aspect_ratio = image_size[1] / image_size[0]
            resize_to = (int(round(target_size / aspect_ratio)), target_size)
            if resize_to[0] > target_size:
                crop = (0, (resize_to[0] - target_size) // 2, target_size, target_size)
            else:
                crop = None
    elif isinstance(target_size, Iterable) and len(target_size) == 2: # Resize to target size and center crop
        target_aspect_ratio = target_size[1] / target_size[0]
        image_aspect_ratio = image_size[1] / image_size[0]

        if target_aspect_ratio > image_aspect_ratio:
            resize_to = (int(round(target_size[1] / image_aspect_ratio)), target_size[1])
            crop = (0, (resize_to[0] - target_size[0]) // 2, target_size[1], target_size[0])
        else:
            resize_to = (target_size[0], int(round(target_size[0] * image_aspect_ratio)))
            crop = ((resize_to[1] - target_size[1]) // 2, 0, target_size[1], target_size[0])

        # print(f"Resizing to {resize_to} and cropping to {crop}")

    return resize_to, crop


def process_img(
    img: np.array, target_size=None, crop=None
):
    if target_size is not None and (target_size[0] != img.shape[0] or target_size[1] != img.shape[1]):
        img = cv2.resize(
            img,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    
    if crop is not None:
        img = img[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2], :]

    img = np.transpose(img, (2, 0, 1))

    return img


def process_depth(depth: np.array, target_size=None, crop=None):
    if target_size is not None and (target_size[0] != depth.shape[0] or target_size[1] != depth.shape[1]):
        depth = cv2.resize(
            depth,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = depth[..., None]
  
    if crop is not None:
        depth = depth[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2], :]

    depth = np.transpose(depth, (2, 0, 1))

    return depth


def process_flow(flow: np.array, target_size=None, crop=None):
    flow = flow.transpose(1, 2, 0)

    if target_size is not None and (target_size[0] != flow.shape[0] or target_size[1] != flow.shape[1]):
        x_scale = target_size[1] / flow.shape[1]
        y_scale = target_size[0] / flow.shape[0]

        flow = cv2.resize(
            flow,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        flow = flow * np.array([x_scale, y_scale], dtype=np.float32).reshape(1, 1, 2)
    
    if crop is not None:
        flow = flow[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2], :]

    flow = np.transpose(flow, (2, 0, 1))

    return flow


def process_proj(proj: np.array, original_size, target_size=None, crop=None):
    proj = proj.copy()

    proj[0, :] *= 2.0 / original_size[1]
    proj[1, :] *= 2.0 / original_size[0]

    proj[0, 2] = proj[0, 2] - 1.0
    proj[1, 2] = proj[1, 2] - 1.0

    #if crop is not None:
    #    raise NotImplementedError()

    #if target_size is not None and (target_size[0] != original_size[0] or target_size[1] != original_size[1]):
    #    x_scale = target_size[1] / original_size[1]
    #    y_scale = target_size[0] / original_size[0]

    #    proj[0, :] *= x_scale
    #    proj[1, :] *= y_scale

    #if crop is not None:
    #    proj[0, 2] -= crop[0]
    #    proj[1, 2] -= crop[1]

    return proj


def flow_selector_2(ids):
    return ((ids[0], True),), ((ids[1], False),)


def flow_selector_3(ids):
    return ((ids[0], False), (ids[0], True)), ((ids[1], True), (ids[2], False))


def flow_selector_seq(ids):
    fwd = [(i, True) for i in ids[:-1]]
    bwd = [(i, False) for i in ids[1:]]
    return fwd, bwd


def get_flow_selector(frame_count, is_sequential=False):
    if not is_sequential:
        if frame_count == 2:
            flow_selector = flow_selector_2
        elif frame_count == 3:
            flow_selector = flow_selector_3
        else:
            raise ValueError(f"Unknown frame count: {frame_count}")
    else:
        flow_selector = flow_selector_seq
    return flow_selector


def index_selector_pair(id, frame_count, dilation, left_offset):
    ids = [i
        for i in range(
            id - left_offset,
            id - left_offset + frame_count * dilation,
            dilation,
        )
        if i != id
    ]
    return ids


def index_selector_seq(id, frame_count, dilation, left_offset):
    ids = [id + i * dilation - left_offset for i in range(frame_count)]
    return ids


def get_index_selector(is_sequential=False):
    if not is_sequential:
        index_selector = index_selector_pair
    else:
        index_selector = index_selector_seq
    return index_selector


def sequence_sampler_crop(seq, seq_len, left_offset, sub_seq_len):
    if seq_len < sub_seq_len:
        return []

    datapoints = [(seq, i + left_offset) for i in range(seq_len - sub_seq_len)]
    return datapoints


def sequence_sampler_full(seq, seq_len, left_offset, sub_seq_len):
    datapoints = [(seq, i) for i in range(seq_len - 1)]
    return datapoints


def get_sequence_sampler(crop=True):
    if crop:
        sequence_sampler = sequence_sampler_crop
    else:
        sequence_sampler = sequence_sampler_full
    return sequence_sampler

def get_ids_for_sequence(dataset, sequence):
    seq_id_datapoints = [(i, dp) for i, dp in enumerate(dataset._datapoints) if dp[0] == sequence]
    seq_ids = [i for i, _ in seq_id_datapoints]
    seq_datapoints = [dp for _, dp in seq_id_datapoints]
    return seq_ids, seq_datapoints
