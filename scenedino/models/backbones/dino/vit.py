import os
import pathlib
from typing import Dict, Tuple, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torch.fx import GraphModule
from torchvision.models.feature_extraction import create_feature_extractor

# from ups.utils import normalize

__all__: Tuple[str, ...] = (
    "dino_small",
    "dino_base",
    "dinov2_small",
    "dinov2_base",
    "dino_reg_small",
    "dino_reg_base",
    "i_jepa_huge",
    "mae_base",
    "self_patch_small",
    "synclr_base",
    "mocov3_base",
    "msn_base",
    "vmae_large",
)


def _disable_fused_attention(model: nn.Module) -> None:
    """Function disables the use of fused attention in Timm's ViT models. (Don't use anywhere else!)

    Args:
        model (nn.Module): Timm ViT model.
    """
    # Get ViT depth
    depth: int = len(model.blocks)  # type: ignore
    # Disable fused attention for last block
    for name, module in model.named_modules():
        if "Attention" in str(type(module)):
            if str(depth - 1) in name:
                module.fused_attn = False  # type: ignore


def _load_vit(name: str, image_size: Tuple[int, int] = (224, 224), depth: int = 12) -> nn.Module:
    """Function to load ViT models from Timm.

    Args:
        name (str): Timm name of the model.
        depth (int): Depth of the model.

    Returns:
        model (nn.Module): ViT model as a nn.Module.
    """
    # Load model
    model: nn.Module = timm.create_model(name, pretrained=True, img_size=image_size, num_classes=0)
    # Force not to use fused attention to access attention maps
    # _disable_fused_attention(model)
    return model


def _interpolate_positional_embeddings(
    positional_embeddings: Tensor,
    original_image_size: Tuple[int, int],
    target_image_size: Tuple[int, int],
    patch_size: int,
    num_additional_tokens: int = 1,
) -> Tensor:
    """Function interpolates positional embeddings to a different image size.

    Args:
        positional_embeddings (Tensor): Positional embeddings of the sape [1, N, C].
        original_image_size (Tuple[int, int]): Original image size as a tuple.
        target_image_size (Tuple[int, int]): Target image size as a tuple.
        patch_size (int): Utilize patch size.
        num_additional_tokens (int): Number of additional tokens used. Default 1 (class token).

    Returns:
        positional_embeddings_interpolated (Tensor): Interpolated positional embeddings [1, N_new, C].
    """
    # Get positional embeddings for image
    if num_additional_tokens > 0:
        positional_embeddings_add_tokens: Tensor = positional_embeddings[:, :num_additional_tokens]
        positional_embeddings_image: Tensor = positional_embeddings[:, num_additional_tokens:]
    else:
        positional_embeddings_image = positional_embeddings
    # Reshape embeddings to 2D
    positional_embeddings_image = positional_embeddings_image.view(
        1, original_image_size[0] // patch_size, original_image_size[1] // patch_size, -1
    )
    # Interpolate positional embeddings
    positional_embeddings_image = F.interpolate(
        positional_embeddings_image.permute(0, 3, 1, 2),
        size=(target_image_size[0] // patch_size, target_image_size[1] // patch_size),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    ).permute(0, 2, 3, 1)
    # Stack positional embeddings again
    if num_additional_tokens > 0:
        positional_embeddings_interpolated: Tensor = torch.cat(
            (positional_embeddings_add_tokens, positional_embeddings_image.flatten(1, 2)), dim=1
        )
    else:
        positional_embeddings_interpolated = positional_embeddings_image.flatten(1, 2)
    return positional_embeddings_interpolated


class _ViT(nn.Module):
    """This class wraps Timm's ViT's and always ensures eval mode."""

    def __init__(
        self,
        vit: nn.Module,
        patch_size: int,
        registers: bool = False,
        class_token: bool = False,
        intermediate_features: List[int] = None,
    ) -> None:
        """Constructor method.

        Args:
            vit (nn.Module): Timm ViT model.
            patch_size (int): Patch size utilized.
            registers (bool): Set to true if registers are use. Default False.
            class_token (bool): Set true if class token is use. Default False.
        """
        # Call super constructor
        super(_ViT, self).__init__()
        # Save parameter
        self.patch_size: int = patch_size
        self.registers: bool = registers
        self.class_token: bool = class_token
        # Get ViT depth
        depth: int = len(vit.blocks)  # type: ignore

        return_nodes = {
            #f"blocks.{depth - 1}.attn.softmax": "attention_maps",
            "norm": "features_normalized",
            f"blocks.{depth - 1}.attn.getitem_4": "key_features",
        }
        if intermediate_features is not None:
            for idx, feat in enumerate(intermediate_features):
                return_nodes[f"blocks.{feat}"] = f"intermediate_features.{idx}"

        # Make FX graph module for feature extraction
        self.vit: GraphModule = create_feature_extractor(vit, return_nodes)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """Forward pass.

        Notes:
            attention_maps have the shape of [B, num heads, N, N].
            features have the shape of [B, N, C]
            Class token and register tokens omitted!

        Args:
            images (Tensor): Images (w/ pix. range of [0, 1]) of the shape [B, 3, H, W].

        Returns:
            output_dict (Dict[str, Tensor]): Dict of features ("attention_maps" and "features").
        """
        # Ensure model is in eval mode
        self.vit.eval()
        # Normalize images
        images_normalized: Tensor = images  # normalize(images)
        # Perform forward pass
        output_dict: Dict[str, Tensor] = self.vit(images_normalized)
        # Omit class token (and registers) from attention maps and features
        if self.registers:
            output_dict["features_normalized"] = output_dict["features_normalized"][:, 5:]
            #output_dict["attention_maps"] = output_dict["attention_maps"][..., 5:, 5:]
            output_dict["key_features"] = output_dict["key_features"][:, :, 5:]
            for output_key in output_dict:
                if output_key.startswith('intermediate_features'):
                    output_dict[output_key] = output_dict[output_key][:, 5:]
        elif self.class_token:
            output_dict["features_normalized"] = output_dict["features_normalized"][:, 1:]
            #output_dict["attention_maps"] = output_dict["attention_maps"][..., 1:, 1:]
            output_dict["key_features"] = output_dict["key_features"][:, :, 1:]
            for output_key in output_dict:
                if output_key.startswith('intermediate_features'):
                    output_dict[output_key] = output_dict[output_key][:, 1:]
        # Normalize features
        output_dict["features_normalized"] = F.normalize(output_dict["features_normalized"], p=2, dim=2)  # [B, N, C]
        return output_dict


def mae_base(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    """Builds the pre-trained MAE base model (patch size is 16 x 16).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): ViT MAE model as a nn.Module.
    """
    return _ViT(
        _load_vit(name="vit_base_patch16_224.mae", image_size=image_size),
        patch_size=16,
        class_token=True,
    )


def vmae_large(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    """Builds the pre-trained video MAE large model (patch size is 16 x 16).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): ViT MAE model as a nn.Module.
    """
    # Init model
    model = VisionTransformer(
        img_size=image_size, patch_size=16, num_classes=0, qkv_bias=True, embed_dim=1024, depth=24, num_heads=16
    )
    # Force not to use fused attention to access attention maps
    _disable_fused_attention(model)
    # Load and adapt checkpoint
    checkpoint: Dict[str, Tensor] = torch.load(
        os.path.join(pathlib.Path(__file__).parent.resolve(), "checkpoints/mae_pretrain_vit_large_k700.pth"),
        map_location="cpu",
    )["model_state"]
    checkpoint["pos_embed"] = checkpoint["pos_embed_spatial"] + checkpoint["pos_embed_temporal"].mean(
        dim=1, keepdim=True
    )
    checkpoint["pos_embed"] = torch.cat((checkpoint["pos_embed_class"], checkpoint["pos_embed"]), dim=1)
    checkpoint["patch_embed.proj.weight"] = checkpoint["patch_embed.proj.weight"][:, :, 0]
    for layer in range(24):
        checkpoint[f"blocks.{layer}.attn.qkv.weight"] = torch.cat(
            (
                checkpoint[f"blocks.{layer}.attn.q.weight"],
                checkpoint[f"blocks.{layer}.attn.k.weight"],
                checkpoint[f"blocks.{layer}.attn.v.weight"],
            ),
            dim=0,
        )
        checkpoint[f"blocks.{layer}.attn.qkv.bias"] = torch.cat(
            (
                checkpoint[f"blocks.{layer}.attn.q.bias"],
                checkpoint[f"blocks.{layer}.attn.k.bias"],
                checkpoint[f"blocks.{layer}.attn.v.bias"],
            ),
            dim=0,
        )
    checkpoint = {key: value for key, value in checkpoint.items() if key in model.state_dict().keys()}
    # Interpolated positional embeddings
    checkpoint["pos_embed"] = _interpolate_positional_embeddings(
        checkpoint["pos_embed"],
        original_image_size=(224, 224),
        target_image_size=image_size,
        patch_size=16,
        num_additional_tokens=1,
    )
    # Load checkpoint
    model.load_state_dict(checkpoint)
    return _ViT(model, patch_size=16, class_token=True)


def dino_small(image_size: Tuple[int, int] = (224, 224), intermediate_features: List[int] = None) -> nn.Module:
    """Builds the pre-trained ViT DINO small model (patch size is 16 x 16).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): ViT Dino model as a nn.Module.
    """
    return _ViT(
        _load_vit(name="vit_small_patch16_224.dino", image_size=image_size),
        patch_size=16,
        class_token=True,
        intermediate_features=intermediate_features,
    )


def dino_small8(image_size: Tuple[int, int] = (224, 224), intermediate_features: List[int] = None) -> nn.Module:
    """Builds the pre-trained ViT DINO small model (patch size is 8 x 8).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): ViT Dino model as a nn.Module.
    """
    return _ViT(
        _load_vit(name="vit_small_patch8_224.dino", image_size=image_size),
        patch_size=8,
        class_token=True,
        intermediate_features=intermediate_features,
    )


def dino_base(image_size: Tuple[int, int] = (224, 224), intermediate_features: List[int] = None) -> nn.Module:
    """Builds the pre-trained ViT DINO base model (patch size is 16 x 16).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): ViT Dino model as a nn.Module.
    """
    return _ViT(
        _load_vit(name="vit_base_patch16_224.dino", image_size=image_size),
        patch_size=16,
        class_token=True,
        intermediate_features=intermediate_features,
    )


def dino_base8(image_size: Tuple[int, int] = (224, 224), intermediate_features: List[int] = None) -> nn.Module:
    """Builds the pre-trained ViT DINO base model (patch size is 16 x 16).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): ViT Dino model as a nn.Module.
    """
    return _ViT(
        _load_vit(name="vit_base_patch8_224.dino", image_size=image_size),
        patch_size=8,
        class_token=True,
        intermediate_features=intermediate_features,
    )


def dinov2_small(image_size: Tuple[int, int] = (224, 224), intermediate_features: List[int] = None) -> nn.Module:
    """Builds the pre-trained ViT DINO V2 small model (patch size is 14 x 14).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).
        intermediate_features (List[int]): Index of intermediate layer features to return.

    Returns:
        model (nn.Module): ViT Dino model as a nn.Module.
    """
    return _ViT(
        _load_vit(name="vit_small_patch14_dinov2.lvd142m", image_size=image_size),
        patch_size=14,
        class_token=True,
        intermediate_features=intermediate_features,
    )


def dinov2_base(image_size: Tuple[int, int] = (224, 224), intermediate_features: List[int] = None) -> nn.Module:
    """Builds the pre-trained ViT DINO V2 base model (patch size is 14 x 14).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).
        intermediate_features (List[int]): Index of intermediate layer features to return.

    Returns:
        model (nn.Module): ViT Dino model as a nn.Module.
    """
    return _ViT(
        _load_vit(name="vit_base_patch14_dinov2.lvd142m", image_size=image_size),
        patch_size=14,
        class_token=True,
        intermediate_features=intermediate_features,
    )


def dino_reg_small(image_size: Tuple[int, int] = (224, 224), intermediate_features: List[int] = None) -> nn.Module:
    """Builds the pre-trained ViT DINO (w/ registers) small model (patch size is 14 x 14).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).
        intermediate_features (List[int]): Index of intermediate layer features to return.

    Returns:
        model (nn.Module): ViT Dino model as a nn.Module.
    """
    return _ViT(
        _load_vit(name="vit_small_patch14_reg4_dinov2.lvd142m", image_size=image_size),
        patch_size=14,
        registers=True,
        intermediate_features=intermediate_features,
    )


def dino_reg_base(image_size: Tuple[int, int] = (224, 224), intermediate_features: List[int] = None) -> nn.Module:
    """Builds the pre-trained ViT DINO (w/ registers) base model (patch size is 14 x 14).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).
        intermediate_features (List[int]): Index of intermediate layer features to return.

    Returns:
        model (nn.Module): ViT Dino model as a nn.Module.
    """
    return _ViT(
        _load_vit(name="vit_base_patch14_reg4_dinov2.lvd142m", image_size=image_size),
        patch_size=14,
        registers=True,
        intermediate_features=intermediate_features,

    )


def synclr_base(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    """Builds the pre-trained SynCLR ViT base model (patch size is 16 x 16).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): SynCLR ViT model as a nn.Module
    """
    # Init model
    model = _load_vit(name="vit_base_patch16_224", image_size=image_size)
    # Load and adapt checkpoint
    checkpoint: Dict[str, Tensor] = torch.load(
        os.path.join(pathlib.Path(__file__).parent.resolve(), "checkpoints/synclr_vit_b_16.pth")
    )["model"]
    checkpoint = {key.replace("module.visual.", ""): value for key, value in checkpoint.items()}
    # Interpolated positional embeddings
    if image_size != (224, 224):
        checkpoint["pos_embed"] = _interpolate_positional_embeddings(
            checkpoint["pos_embed"],
            original_image_size=(224, 224),
            target_image_size=image_size,
            patch_size=16,
            num_additional_tokens=1,
        )
    # Load checkpoint
    model.load_state_dict(checkpoint)
    return _ViT(model, patch_size=16, class_token=True)


def mocov3_base(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    """Builds the pre-trained MoCo-V3 ViT base model (patch size is 16 x 16).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): MoCo-V3 ViT model as a nn.Module
    """
    # Init model
    model = _load_vit(name="vit_base_patch16_224", image_size=image_size)
    # Load and adapt checkpoint
    checkpoint: Dict[str, Tensor] = torch.load(
        os.path.join(pathlib.Path(__file__).parent.resolve(), "checkpoints/vit-b-300ep.pth.tar")
    )["state_dict"]
    checkpoint = {
        key.replace("module.momentum_encoder.", ""): value
        for key, value in checkpoint.items()
        if ("module.momentum_encoder." in key) and ("head." not in key)
    }
    # Interpolated positional embeddings
    if image_size != (224, 224):
        checkpoint["pos_embed"] = _interpolate_positional_embeddings(
            checkpoint["pos_embed"],
            original_image_size=(224, 224),
            target_image_size=image_size,
            patch_size=16,
            num_additional_tokens=1,
        )
    # Load checkpoint
    model.load_state_dict(checkpoint)
    return _ViT(model, patch_size=16, class_token=True)


def msn_base(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    """Builds the pre-trained MSN ViT base model (patch size is 16 x 16).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): MSN ViT model as a nn.Module
    """
    # Init model
    model = _load_vit(name="vit_base_patch16_224", image_size=image_size)
    # Load and adapt checkpoint
    checkpoint: Dict[str, Tensor] = torch.load(
        os.path.join(pathlib.Path(__file__).parent.resolve(), "checkpoints/vitb16_600ep.pth.tar")
    )["target_encoder"]
    checkpoint = {
        key.replace("module.", ""): value
        for key, value in checkpoint.items()
        if key.replace("module.", "") in model.state_dict().keys()
    }
    # Interpolated positional embeddings
    if image_size != (224, 224):
        checkpoint["pos_embed"] = _interpolate_positional_embeddings(
            checkpoint["pos_embed"],
            original_image_size=(224, 224),
            target_image_size=image_size,
            patch_size=16,
            num_additional_tokens=1,
        )
    # Load checkpoint
    model.load_state_dict(checkpoint)
    return _ViT(model, patch_size=16, class_token=True)


def self_patch_small(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    """Builds the pre-trained Self-Patch ViT small model (patch size is 16 x 16).

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): Self-Patch ViT model as a nn.Module
    """
    # Init model
    model = VisionTransformer(
        img_size=image_size,
        patch_size=16,
        num_classes=0,
        class_token=False,
        qkv_bias=True,
        global_pool="avg",
        embed_dim=384,
        depth=12,
        num_heads=6,
    )
    # Force not to use fused attention to access attention maps
    _disable_fused_attention(model)
    # Load and adapt checkpoint
    checkpoint: Dict[str, Tensor] = torch.load(
        os.path.join(pathlib.Path(__file__).parent.resolve(), "checkpoints/dino_selfpatch.pth")
    )
    checkpoint = {
        key.replace("module.", ""): value for key, value in checkpoint.items() if key in model.state_dict().keys()
    }
    # Interpolated positional embeddings
    if image_size != (224, 224):
        checkpoint["pos_embed"] = _interpolate_positional_embeddings(
            checkpoint["pos_embed"],
            original_image_size=(224, 224),
            target_image_size=image_size,
            patch_size=16,
            num_additional_tokens=0,
        )
    # Load checkpoint
    model.load_state_dict(checkpoint, strict=False)
    return _ViT(model, patch_size=16, class_token=False, registers=False)


def i_jepa_huge(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    """Builds the pre-trained I-JEPA ViT huge model (patch size is 14 x 14).

    Notes:
        ViT huge is very large...

    Args:
        image_size (Tuple[int, int]): Image size to be used. Default is (224, 224).

    Returns:
        model (nn.Module): SynCLR ViT model as a nn.Module
    """
    # Init model
    model = VisionTransformer(
        img_size=image_size,
        patch_size=14,
        num_classes=0,
        class_token=False,
        global_pool="avg",
        qkv_bias=True,
        embed_dim=1280,
        depth=32,
        num_heads=16,
    )
    # Force not to use fused attention to access attention maps
    _disable_fused_attention(model)
    # Load and adapt checkpoint
    checkpoint: Dict[str, Tensor] = torch.load(
        os.path.join(pathlib.Path(__file__).parent.resolve(), "checkpoints/IN22k-vit.h.14-900e.pth.tar"),
        map_location="cpu",
    )["encoder"]
    checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    # Interpolated positional embeddings
    if image_size != (224, 224):
        checkpoint["pos_embed"] = _interpolate_positional_embeddings(
            checkpoint["pos_embed"],
            original_image_size=(224, 224),
            target_image_size=image_size,
            patch_size=14,
            num_additional_tokens=0,
        )
    # Load checkpoint
    model.load_state_dict(checkpoint, strict=False)
    return _ViT(model, patch_size=14, class_token=False, registers=False)
