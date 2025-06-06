import math
import random
from typing import Tuple, Optional

import kornia
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.transforms as tf

from bts.models.backbones.dino.decoder import NoDecoder

import logging
logger = logging.getLogger("training")


class MultiScaleCropGT_kornia(nn.Module):
    """This class implements multi-scale-crop augmentation for DINO features."""
    def __init__(
            self,
            gt_encoder: nn.Module,
            num_views: int = 8,
            image_size: Tuple[int, int] = (192, 640),
            feature_stride: int = 16,
    ) -> None:
        """Constructor method.

        Args:
            num_views (int): Number of view per image. Default 8.
            augmentations (Tuple[AugmentationBase2D, ...]): Geometric augmentations to be applied.
            feature_stride (int): Stride of the features. Default 16.
        """
        # Call super constructor
        super(MultiScaleCropGT_kornia, self).__init__()

        # GT encoder
        self.gt_encoder = gt_encoder

        # Save parameters
        self.augmentations_per_sample: int = num_views
        self.feature_stride: int = feature_stride
        # Init augmentations
        
        image_ratio = image_size[0] / image_size[1]
        augmentations = (
            kornia.augmentation.RandomHorizontalFlip(p=0.5),
            #kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            kornia.augmentation.RandomResizedCrop(
                scale=(0.5, 1.0), size=tuple(image_size), ratio=(image_ratio/1.2, image_ratio*1.2), p=1.0
                # Here you need to set your resolution
            ),
        )
        self.augmentations: nn.Module = kornia.augmentation.VideoSequential(*augmentations, same_on_frame=True)

    @staticmethod
    def _affine_transform_valid_pixels(transform: Tensor, mask: Tensor) -> Tensor:
        """Applies affine transform to a mask of ones to estimate valid pixels.

        Args:
            transform (Tensor): Affine transform of the shape [B, 3, 3]
            mask (Tensor): Mask of the shape [B, 1, H, W].

        Returns:
            valid_pixels (Tensor): Mask of valid pixels of the shape [B, 1, H, W].
        """
        # Get shape
        H, W = mask.shape[2:]  # type: int, int
        # Resample mask map
        valid_pixels: Tensor = kornia.geometry.warp_perspective(
            mask,
            transform,
            (H, W),
            mode="nearest",
        )
        # Threshold mask
        valid_pixels = torch.where(  # type: ignore
            valid_pixels > 0.999, torch.ones_like(valid_pixels), torch.zeros_like(valid_pixels)
        )
        return valid_pixels

    def _accumulate_predictions(self, features: Tensor, transforms: Tensor) -> Tensor:
        """Accumulates features over multiple predictions.

        Args:
            features (Tensor): Feature predictions of the shape [B, num_views, H, W].
            transforms (Tensor): Affine transformations of the shape [B, num_views, 3, 3].

        Returns:
            optical_flow_predictions_accumulated (Tensor): Accumulated optical flow of the shape [B, 2, H, W].
        """
        # Get shape
        B, N, C, H, W = features.shape  # type: int, int, int, int, int
        # Get base and augmented views
        features_base = features[:, -2:]
        features_augmented = features[:, :-2]
        # Combine batch dimension and view dimension
        features_augmented = features_augmented.flatten(0, 1)
        transforms = transforms.flatten(0, 1)
        # Rescale transformation
        transforms[:, 0, -1] = transforms[:, 0, -1] #/ float(self.feature_stride)
        transforms[:, 1, -1] = transforms[:, 1, -1] #/ float(self.feature_stride)
        # Invert transformations
        transforms_inv: Tensor = torch.inverse(transforms)
        # Resample optical flow map
        features_resampled: Tensor = kornia.geometry.warp_perspective(
            features_augmented,
            transforms_inv,
            (H, W),
            mode="bilinear",
        )
        # Separate batch and view dimension again
        features_resampled = features_resampled.reshape(B, -1, C, H, W)
        # Add base views
        features_resampled = torch.cat((features_resampled, features_base), dim=1)
        # Reverse flip
        features_resampled[:, -2] = features_resampled[:, -2].flip(dims=(-1,))
        # Compute valid pixels
        mask: Tensor = torch.ones(
            B, N - 2, 1, H, W, dtype=features_resampled.dtype, device=features_resampled.device
        )
        mask = mask.flatten(0, 1)
        valid_pixels: Tensor = self._affine_transform_valid_pixels(transforms_inv, mask)
        valid_pixels = valid_pixels.reshape(B, N - 2, 1, H, W)
        valid_pixels = F.pad(valid_pixels, (0, 0, 0, 0, 0, 0, 0, 2), value=1)
        # Set invalid flow vectors to zero
        features_resampled[valid_pixels.repeat(1, 1, C, 1, 1) == 0.0] = torch.nan
        # Average optical flow over different views given the sum valid pixels for the specific pixel

        # logger.info(features_resampled.shape)
        return features_resampled.nanmean(dim=1)

    def _get_augmentations(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass generates different augmentations of the input images.

        Args:
            images (Tensor): Images of the shape [B, 3, H, W]

        Returns:
            images_augmented (Tensor): Augmented images of the shape [B, N, H, W].
            transforms (Tensor): Transformations of the shape [B, N, 3, 3].
        """
        # Add dummy dimension shape is [B, num_views, 3, H, W]
        images = images[:, None]
        # Init tensor to store transformations
        transformations: Tensor = torch.empty(
            images.shape[0], self.augmentations_per_sample - 2, 3, 3, dtype=torch.float32, device=images.device
        )
        # Init tensor to store augmented images
        images_augmented: Tensor = torch.empty_like(images)
        images_augmented = images_augmented[:, None].repeat_interleave(self.augmentations_per_sample, dim=1)
        # Save original and flipped images
        images_augmented[:, -1] = images.clone()
        images_augmented[:, -2] = images.clone().flip(dims=(-1,))
        # Apply geometric augmentations
        for index in range(images.shape[0]):
            images_repeated: Tensor = images[index][None].repeat_interleave(self.augmentations_per_sample - 2, dim=0)
            images_augmented[index, :-2] = self.augmentations(images_repeated)
            transformations[index] = self.augmentations.get_transformation_matrix(
                images_repeated, self.augmentations._params
            )
        return images_augmented[:, :, 0], transformations

    def forward_chunk(self, images):
        batch_size, _, h, w = images.shape

        # Perform augmentation
        images_aug, transformations = self._get_augmentations(images)
        # Get representations
        features = self.gt_encoder(images_aug.flatten(0, 1))[-1]
        features = F.interpolate(features, size=(h, w), mode="bilinear")
        # features = features.repeat_interleave(self.feature_stride, -1).repeat_interleave(self.feature_stride, -2)

        _, dino_dim, _, _ = features.shape
        features = features.view(batch_size, -1, dino_dim, h, w)

        chunks = torch.chunk(features, chunks=4, dim=2)  # Split into 4 parts along dim=3
        chunks = [self._accumulate_predictions(chunk, transformations) for chunk in chunks]
        features_accumulated = torch.cat(chunks, dim=1)

        # features_accumulated = self._accumulate_predictions(features, transformations)
        return features_accumulated / torch.linalg.norm(features_accumulated, dim=1, keepdim=True)

    def forward(self, images):
        max_chunk = 16
        aug_no_images = images.shape[0] * self.augmentations_per_sample

        if aug_no_images > max_chunk:
            no_chunks = aug_no_images // max_chunk
            images = torch.chunk(images, no_chunks)
            features = [self.forward_chunk(image) for image in images]
            features = torch.cat(features, dim=0)
            return [features]
        else:
            return [self.forward_chunk(images)]


class InterpolatedGT(nn.Module):
    def __init__(self, arch: str, gt_encoder: nn.Module, image_size: Tuple[int, int]):
        super().__init__()
        self.upsampler = NoDecoder(image_size, arch, normalize_features=False)
        self.gt_encoder = gt_encoder

    def forward(self, x):
        gt_patches = self.gt_encoder(x)
        return self.upsampler(gt_patches)


def _get_affine(params, crop_size, batch_size):
    # construct affine operator
    affine = torch.zeros(batch_size, 2, 3)

    aspect_ratio = float(crop_size[0]) / float(crop_size[1])
    for i, (dy, dx, alpha, scale, flip) in enumerate(params):
        # R inverse
        sin = math.sin(alpha * math.pi / 180.)
        cos = math.cos(alpha * math.pi / 180.)

        # inverse, note how flipping is incorporated
        affine[i, 0, 0], affine[i, 0, 1] = flip * cos, sin * aspect_ratio
        affine[i, 1, 0], affine[i, 1, 1] = -sin / aspect_ratio, cos

        # T inverse Rinv * t == R^T * t
        affine[i, 0, 2] = -1. * (cos * dx + sin * dy)
        affine[i, 1, 2] = -1. * (-sin * dx + cos * dy)

        # T
        affine[i, 0, 2] /= float(crop_size[1] // 2)
        affine[i, 1, 2] /= float(crop_size[0] // 2)

        # scaling
        affine[i] *= scale

    return affine


class MultiScaleCropGT(nn.Module):
    def __init__(self,
                 gt_encoder: nn.Module,
                 num_views: int,
                 scale_from: float = 0.4,
                 grid_sample_batch: Optional[int] = 96):

        super().__init__()
        self.gt_encoder = gt_encoder
        self.num_views = num_views
        self.augmentation = MaskRandScaleCrop(scale_from)
        self.grid_sample_batch = grid_sample_batch

    def forward(self, x):
        result = None
        count = 0
        batch_size, _, h, w = x.shape

        for i in range(self.num_views):
            if i > 0:
                x, params = self.augmentation(x)
            else:
                params = [[0., 0., 0., 1., 1.] for _ in range(x.shape[0])]
            gt_patches = self.gt_encoder(x)[-1]
            affine = _get_affine(params, (h, w), batch_size).cuda()
            affine_grid_gt = F.affine_grid(affine, x.size(), align_corners=False)
            if self.grid_sample_batch:
                d = gt_patches.shape[1]
                assert d % self.grid_sample_batch == 0
                for idx in range(0, d, self.grid_sample_batch):
                    gt_aligned_batch = F.grid_sample(gt_patches[:, idx:idx+self.grid_sample_batch], affine_grid_gt,
                                                     mode="bilinear", align_corners=False)
                    if result is None:
                        result = torch.zeros(batch_size, d, h, w, device="cuda")
                    result[:, idx:idx+self.grid_sample_batch] += gt_aligned_batch
            else:
                gt_aligned = F.grid_sample(gt_patches, affine_grid_gt, mode="bilinear", align_corners=False)
                if result is None:
                    result = 0
                result += gt_aligned

            within_bounds_x = (affine_grid_gt[..., 0] >= -1) & (affine_grid_gt[..., 0] <= 1)
            within_bounds_y = (affine_grid_gt[..., 1] >= -1) & (affine_grid_gt[..., 1] <= 1)
            not_padded_mask = within_bounds_x & within_bounds_y
            count += not_padded_mask.unsqueeze(1)

        count[count == 0] = 1
        return [result.div_(count)]


class MaskRandScaleCrop(object):
    def __init__(self, scale_from):
        self.scale_from = scale_from

    def get_params(self, h, w):
        new_scale = random.uniform(self.scale_from, 1)
        new_h = int(new_scale * h)
        new_w = int(new_scale * w)
        i = random.randint(0, h - new_h)
        j = random.randint(0, w - new_w)
        flip = 1 if random.random() > 0.5 else -1
        return i, j, new_h, new_w, new_scale, flip

    def __call__(self, images, affine=None):
        if affine is None:
            affine = [[0., 0., 0., 1., 1.] for _ in range(len(images))]
        _, H, W = images[0].shape
        i2 = H / 2
        j2 = W / 2
        for k, image in enumerate(images):
            ii, jj, h, w, s, flip = self.get_params(H, W)
            if s == 1.:
                continue  # no change in scale
            # displacement of the centre
            dy = ii + h / 2 - i2
            dx = jj + w / 2 - j2
            affine[k][0] = dy
            affine[k][1] = dx
            affine[k][3] = 1 / s
            # affine[k][4] = flip

            assert ii >= 0 and jj >= 0

            image_crop = tf.functional.crop(image, ii, jj, h, w)
            images[k] = tf.functional.resize(image_crop, (H, W), tf.InterpolationMode.BILINEAR)

        return images, affine
