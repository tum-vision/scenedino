import numpy as np
from typing import Optional, Tuple, List

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from .vit import dino_small8, dino_base8, dino_small, dino_base, dinov2_small, dinov2_base, dino_reg_small, dino_reg_base

from .decoder import NoDecoder, SimpleFeaturePyramidDecoder
from .dpt_head import DPTHead
from .downsampler import PatchSalienceDownsampler, BilinearDownsampler
from .upsampler import InterpolatedGT, MultiScaleCropGT, MultiScaleCropGT_kornia
from .dim_reduction import OrthogonalLinearDimReduction, MlpDimReduction, NoDimReduction
from .visualization import VisualizationModule


def build_encoder(backbone: str, image_size: Tuple[int, int], intermediate_features: List[int], key_features: bool, version: str):
    match backbone:
        case "vit-s" | "vit-b" | "fit3d-s":
            return DINOv2Encoder(backbone, 
                                 image_size,
                                 intermediate_features=intermediate_features,
                                 key_features=key_features,
                                 version=version)
        case _:
            raise NotImplementedError


def build_decoder(decoder_arch: str, patch_size: int, image_size: Tuple[int, int], latent_size: int, num_ch_enc: List[int], decoder_out_dim: int):
    match decoder_arch:
        case "nearest" | "bilinear" | "bicubic":
            return NoDecoder(image_size,
                             interpolation=decoder_arch,
                             normalize_features=True)
        case "spf":
            # TODO: SPF with patch size 8 is not implemented yet
            num_ch_dec = np.array([128, 128, 256, 256, 512])
            scales = range(4)
            return SimpleFeaturePyramidDecoder(latent_size=latent_size,
                                               num_ch_enc=num_ch_enc,
                                               num_ch_dec=num_ch_dec,
                                               d_out=decoder_out_dim,
                                               scales=scales,
                                               use_skips=True,
                                               device="cuda")
        case "dpt":
            return DPTHead(embed_dims=latent_size,
                           post_process_channels=num_ch_enc,
                           readout_type="ignore",
                           patch_size=patch_size,
                           d_out=decoder_out_dim,
                           expand_channels=False)
        case _:
            raise NotImplementedError


def build_downsampler(arch: str, dim: int, patch_size: int):
    match arch:
        case "featup":
            return PatchSalienceDownsampler(dim, patch_size=patch_size, normalize_features=True)
        case "bilinear":
            return BilinearDownsampler(patch_size=patch_size)
        case _:
            raise NotImplementedError


def build_gt_upsampling_wrapper(arch: str, gt_encoder: nn.Module, image_size: Tuple[int, int]):
    match arch:
        case "nearest" | "bilinear" | "bicubic":
            return InterpolatedGT(arch, gt_encoder, image_size)
        case "multiscale-crop":
            return MultiScaleCropGT_kornia(gt_encoder, num_views=4, image_size=image_size)
        case _:
            raise NotImplementedError


def build_dim_reduction(arch: str, full_channels: int, reduced_channels: int):
    match arch:
        case "none":
            return NoDimReduction(full_channels, reduced_channels)
        case "mlp":
            return MlpDimReduction(full_channels, reduced_channels, latent_channels=128)
        case "orthogonal-linear":
            return OrthogonalLinearDimReduction(full_channels, reduced_channels)
        case _:
            raise NotImplementedError


class DINOv2Module(nn.Module):
    def __init__(self,
                 mode: str,                                 # downsample-prediction, upsample-gt
                 decoder_arch: str,                         # nearest, bilinear, sfp, dpt
                 upsampler_arch: Optional[str],             # nearest, bilinear, multiscale-crop
                 downsampler_arch: Optional[str],           # sample-center, featup
                 encoder_arch: str,                         # vit-s, vit-b
                 encoder_freeze: bool,
                 flip_avg_gt: bool,
                 dim_reduction_arch: str,                   # orthogonal-linear, mlp
                 num_ch_enc: np.array,
                 intermediate_features: List[int],
                 decoder_out_dim: int,
                 dino_pca_dim: int,
                 image_size: Tuple[int, int],
                 key_features: bool,
                 dino_version: str,                         # v1, v2, reg, fit3d
                 separate_gt_version: Optional[str],        # v1, v2, reg, fit3d, None (reuses encoder)
                 ):

        super().__init__()

        self.encoder = build_encoder(encoder_arch, image_size, intermediate_features, key_features, dino_version)
        self.flip_avg_gt = flip_avg_gt

        if encoder_freeze or separate_gt_version is None:
            self.encoder_frozen = True
            for p in self.encoder.parameters(True):
                p.requires_grad = False
        else:
            self.encoder_frozen = False

        self.decoder = build_decoder(decoder_arch,
                                     self.encoder.patch_size,
                                     image_size,
                                     self.encoder.latent_size,
                                     num_ch_enc,
                                     decoder_out_dim)

        if separate_gt_version is None:
            self.gt_encoder = self.encoder
        else:
            self.gt_encoder = build_encoder(encoder_arch, image_size, [], key_features, separate_gt_version)
            for p in self.gt_encoder.parameters(True):
                p.requires_grad = False

        # General way of creating loss
        if mode == "downsample-prediction":
            assert upsampler_arch is None
            self.downsampler = build_downsampler(downsampler_arch, self.gt_encoder.latent_size, self.gt_encoder.patch_size)
            self.gt_wrapper = None

        elif mode == "upsample-gt":
            assert downsampler_arch is None
            self.downsampler = None
            self.gt_wrapper = build_gt_upsampling_wrapper(upsampler_arch, self.gt_encoder, image_size)

        else:
            raise NotImplementedError

        self.extra_outs = 0
        self.latent_size = decoder_out_dim

        self.dino_pca_dim = dino_pca_dim
        self.dim_reduction = build_dim_reduction(dim_reduction_arch, self.encoder.latent_size, dino_pca_dim)
        self.visualization = VisualizationModule(self.encoder.latent_size)

    def forward(self, x, ground_truth=False):
        if ground_truth:
            with torch.no_grad():
                if self.gt_wrapper is not None:
                    gt_0 = self.gt_wrapper(x)
                    if self.flip_avg_gt:
                        gt_flipped = self.gt_wrapper(x.flip([-1]))
                        gt_avg = [F.normalize(gt_flipped[i].flip([-1]) + gt_0[i], dim=1) for i in range(len(gt_0))]
                        return gt_avg
                    else:
                        return gt_0
                else:
                    gt_0 = self.gt_encoder(x)[-1]
                    if self.flip_avg_gt:
                        gt_flipped = self.gt_encoder(x.flip([-1]))[-1]
                        gt_avg = F.normalize(gt_flipped.flip([-1]) + gt_0, dim=1)
                        return [gt_avg]
                    else:
                        return [gt_0]
        else:
            if self.encoder_frozen:
                with torch.no_grad():
                    patch_features = self.encoder(x)
            else:
                patch_features = self.encoder(x)
            return self.decoder(patch_features)

    def downsample(self, x, mode="patch"):
        if self.downsampler is None:
            return None
        else:
            return self.downsampler(x, mode)

    def expand_dim(self, features):
        return self.dim_reduction.transform_expand(features)

    def fit_visualization(self, features, refit=True):
        return self.visualization.fit_pca(features, refit)

    def transform_visualization(self, features, norm=False, from_dim=0):
        return self.visualization.transform_pca(features, norm, from_dim)

    def fit_transform_kmeans_visualization(self, features):
        return self.visualization.fit_transform_kmeans_batch(features)

    @classmethod
    def from_conf(cls, conf):
        return cls(
            mode=conf.mode,
            decoder_arch=conf.decoder_arch,
            upsampler_arch=conf.get("upsampler_arch", None),
            downsampler_arch=conf.get("downsampler_arch", None),
            encoder_arch=conf.encoder_arch,
            encoder_freeze=conf.encoder_freeze,
            flip_avg_gt=conf.get("flip_avg_gt", False),
            dim_reduction_arch=conf.dim_reduction_arch,
            num_ch_enc=conf.get("num_ch_enc", None),
            intermediate_features=conf.get("intermediate_features", []),
            decoder_out_dim=conf.decoder_out_dim,
            dino_pca_dim=conf.dino_pca_dim,
            image_size=conf.image_size,
            key_features=conf.key_features,
            dino_version=conf.get("version", "reg"),
            separate_gt_version=conf.get("separate_gt_version", None)
        )


def _normalize_input(x):
    norm_tf = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return norm_tf(x / 2 + 0.5)


class DINOv2Encoder(nn.Module):
    def __init__(self, backbone, image_size, intermediate_features, key_features, version):
        super().__init__()

        self.image_size = image_size
        if version in ["fit3d", "v2", "reg"]:
            # "Internal" patch size 14 is resized to "External" patch size 16 for decoder!
            self.patch_size = 16
            adjusted_image_size = (image_size[0] * 14 // self.patch_size, image_size[1] * 14 // self.patch_size)
            self.resize_tf = torchvision.transforms.Resize(size=adjusted_image_size,
                                                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        elif version == "v1":
            self.patch_size = 8
            adjusted_image_size = self.image_size
            self.resize_tf = None
        elif version == "v1_16":
            self.patch_size = 16
            adjusted_image_size = self.image_size
            self.resize_tf = None
        else:
            raise NotImplementedError()
        
        self.key_features = key_features
        self.backbone = backbone
        self.version = version

        self.model, self.latent_size = self.load_model(backbone, version, adjusted_image_size, intermediate_features)
        
    def forward(self, x):
        x = _normalize_input(x)
        if self.resize_tf:
            x = self.resize_tf(x)

        output_dict = self.model(x)
        if self.version == "fit3d":
            output_dict = self.model.output_dict

        inter_keys = [output_key for output_key in output_dict if output_key.startswith("intermediate_features.")]
        result = []

        for inter_key in sorted(inter_keys):
            output = output_dict[inter_key].transpose(-1, -2)  # (L, B, C_dino, H*W)
            output_grid = output.view(*output.size()[:-1],
                                      x.size(-2) // self.model.patch_size,
                                      x.size(-1) // self.model.patch_size)
            result.append(output_grid)

        if self.key_features:
            output = output_dict['key_features'].transpose(-1, -2).flatten(1, 2)
            output = F.normalize(output, dim=1)
        else:
            output = output_dict['features_normalized'].transpose(-1, -2)
            output = F.normalize(output, dim=1)

        output_grid = output.view(*output.size()[:-1],
                                  x.size(-2) // self.model.patch_size,
                                  x.size(-1) // self.model.patch_size)
        result.append(output_grid)
        return result
    
    def load_model(self, backbone, version, image_size, intermediate_features):
        if version == "fit3d":
            if backbone == "vit-s":
                model_name = "dinov2_reg_small_fine"
            elif backbone == "vit-b":
                model_name = "dinov2_reg_base_fine"
            else:
                raise NotImplementedError()
            
            def get_features(model, key):
                def hook(blk, input, output):
                    model.output_dict[key] = output[:, 5:]
                return hook

            model = torch.hub.load("ywyue/FiT3D", model_name).to("cuda")
            model.norm.register_forward_hook(get_features(model, f"features_normalized"))
            for i, _blk in enumerate(model.blocks):
                if i in intermediate_features:
                    _blk.register_forward_hook(get_features(model, f"intermediate_features.{i}"))
            model.output_dict = {}
            model.patch_size = 14

        elif version == "v1" and backbone == "vit-s":
            model = dino_small8(image_size=image_size, intermediate_features=intermediate_features)
        elif version == "v1" and backbone == "vit-b":
            model = dino_base8(image_size=image_size, intermediate_features=intermediate_features)

        elif version == "v1_16" and backbone == "vit-s":
            model = dino_small(image_size=image_size, intermediate_features=intermediate_features)
        elif version == "v1_16" and backbone == "vit-b":
            model = dino_base(image_size=image_size, intermediate_features=intermediate_features)
            
        elif version == "v2" and backbone == "vit-s":
            model = dinov2_small(image_size=image_size, intermediate_features=intermediate_features)
        elif version == "v2" and backbone == "vit-b":
            model = dinov2_base(image_size=image_size, intermediate_features=intermediate_features)
            
        elif version == "reg" and backbone == "vit-s":
            model = dino_reg_small(image_size=image_size, intermediate_features=intermediate_features)
        elif version == "reg" and backbone == "vit-b":
            model = dino_reg_base(image_size=image_size, intermediate_features=intermediate_features)
        else:
            raise NotImplementedError()

        if backbone == "vit-s":
            latent_size = 384
        elif backbone == "vit-b":
            latent_size = 768

        return model, latent_size
