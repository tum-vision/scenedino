import torch
import torchvision
from torch import nn

from bts.models.backbones.monodepth2 import Decoder


class NoDecoder(nn.Module):
    def __init__(self, image_size, interpolation, normalize_features):
        super().__init__()

        match interpolation:
            case 'nearest':
                inter_mode = torchvision.transforms.InterpolationMode.NEAREST
            case 'bilinear':
                inter_mode = torchvision.transforms.InterpolationMode.BILINEAR
            case 'bicubic':
                inter_mode = torchvision.transforms.InterpolationMode.BICUBIC
            case _:
                raise NotImplementedError(f"Interpolation mode \"{interpolation}\" not implemented!")

        self.image_size = image_size
        self.resize_tf = torchvision.transforms.Resize(size=image_size, interpolation=inter_mode)
        self.normalize_features = normalize_features

    def forward(self, x):
        features = x[-1]
        resized_features = self.resize_tf(features)

        if self.normalize_features:
            resized_features = resized_features / torch.linalg.norm(resized_features, dim=1, keepdim=True)

        return [resized_features]


class SimpleFeaturePyramidDecoder(nn.Module):
    def __init__(self,
                 latent_size,
                 num_ch_enc,
                 num_ch_dec,
                 d_out,
                 scales,
                 use_skips,
                 device):
        super().__init__()

        self.scales = scales
        self.resize_layers = [
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=num_ch_enc[0], kernel_size=8, stride=8, padding=0, device=device),
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=num_ch_enc[1], kernel_size=4, stride=4, padding=0, device=device),
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=num_ch_enc[2], kernel_size=2, stride=2, padding=0, device=device),
            nn.Conv2d(in_channels=latent_size, out_channels=num_ch_enc[3], kernel_size=3, stride=1, padding=1, device=device),
            nn.Conv2d(in_channels=latent_size, out_channels=num_ch_enc[4], kernel_size=3, stride=2, padding=1, device=device),
        ]

        num_ch_dec = [max(d_out, chns) for chns in num_ch_dec]
        self.decoder = Decoder(
            num_ch_enc=num_ch_enc,
            num_ch_dec=num_ch_dec,
            d_out=d_out,
            scales=scales,
            use_skips=use_skips,
            extra_outs=0,
        )

    def forward(self, x):
        dino_features = x[-1]
        features = []
        for resize_layer in self.resize_layers:
            features.append(resize_layer(dino_features))

        outputs = self.decoder(features)
        return [outputs[("disp", i)] for i in self.scales]