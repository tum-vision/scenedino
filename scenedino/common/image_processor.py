from collections import namedtuple

import torch
from torch import nn
import lpips
import torch.nn.functional as F

from torchvision import transforms as tfs

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from scenedino.common.geometry import compute_occlusions


def make_image_processor(config):
    type = config.get("type", "RGB").lower()
    if type == "rgb":
        ip = RGBProcessor()
    elif type == "perceptual":
        ip = PerceptualProcessor(config.get("layers", 1))
    elif type == "patch":
        ip = PatchProcessor(config.get("patch_size", 3))
    elif type == "raft":
        ip = RaftExtractor()
    elif type == "flow":
        ip = FlowProcessor()
    elif type == "flow_occlusion":
        ip = FlowOcclusionProcessor()
    else:
        raise NotImplementedError(f"Unsupported image processor type: {type}")
    return ip


class RGBProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = 3

    def forward(self, images):
        images = images * .5 + .5
        return images


class PerceptualProcessor(nn.Module):
    def __init__(self, layers=1) -> None:
        super().__init__()
        self.lpips_module = lpips.LPIPS(net="vgg")
        self._layers = layers
        self.channels = sum(self.lpips_module.chns[:self._layers])

    def forward(self, images):
        n, v, c, h, w = images.shape
        images = images.view(n*v, c, h, w)

        in_input = self.lpips_module.scaling_layer(images)

        x = self.lpips_module.net.slice1(in_input)
        h_relu1_2 = x
        x = self.lpips_module.net.slice2(x)
        h_relu2_2 = x
        x = self.lpips_module.net.slice3(x)
        h_relu3_3 = x

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3'])
        outs = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3)

        feats = []

        for kk in range(self._layers):
            f = lpips.normalize_tensor(outs[kk])
            f = F.interpolate(f, (h, w))
            feats.append(f)

        feats = torch.cat(feats, dim=1)

        feats = feats.view(n, v, self.channels, h, w)

        return feats


class PatchProcessor(nn.Module):
    def __init__(self, patch_size) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.channels = 3 * (patch_size ** 2)

        self._hps = self.patch_size // 2

    def forward(self, images):
        n, v, c, h, w = images.shape
        images = images.view(n*v, c, h, w) * .5 + .5

        images = F.pad(images, pad=(self.patch_size // 2,)*4, mode="replicate")
        h_, w_ = images.shape[-2:]

        parts = []

        for y in range(0, self.patch_size):
            for x in range(0, self.patch_size):
                parts.append(images[:, :, y:h_-(self.patch_size - y - 1), x:w_-(self.patch_size - x - 1)])

        patch_images = torch.cat(parts, dim=1)
        patch_images = patch_images.view(n, v, self.channels, h, w)

        return patch_images
    

class DinoExtractor(nn.Module):
    def __init__(self, variant):
        super().__init__()
        
        self.model = torch.hub.load('facebookresearch/dino:main', variant)
        self.model.eval()

    def load_checkpoint(self, ckpt_file, checkpoint_key="model"):
        state_dict = torch.load(ckpt_file, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded with msg: {}".format(msg))

    def forward(
        self, img: torch.Tensor, transform=True, upsample=True
    ):
        n, c, h_in, w_in = img.shape
        if transform:
            img = self.transform(img, 256)  # Nx3xHxW
        with torch.no_grad():
            out = self.model.get_intermediate_layers(img.to(self.device), n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token
            h, w = int(img.shape[2] / self.model.patch_embed.patch_size), int(
                img.shape[3] / self.model.patch_embed.patch_size
            )
            dim = out.shape[-1]
            out = out.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
            if upsample:
                out = torch.nn.functional.interpolate(out, (h_in, w_in), mode="bilinear")
        return out

    @staticmethod
    def transform(img, image_size):

        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        transforms = tfs.Compose(
            [
                tfs.Resize(image_size), 
                tfs.Normalize(MEAN, STD)]
        )
        img = transforms(img)
        return img
    
    @property
    def device(self):
        return next(self.parameters()).device
    

class RaftExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        raft_weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = raft_weights.transforms()
        self.raft = raft_large(raft_weights)
        self.raft.eval()
        for param in self.raft.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(
        self, img: torch.Tensor, upsample=True
    ):
        n, v, c, h_in, w_in = img.shape
        img = img.reshape(n * v, c, h_in, w_in)
        img, _ = self.raft_transforms(img * .5 + .5, img * .5 + .5)
        feats = self.raft.feature_encoder(img)
        if upsample:
            feats = F.interpolate(feats, (h_in, w_in), mode="bilinear")
            feats = feats.view(n, v, -1, h_in, w_in)
        else:
            feats = feats.view(n, v, -1, feats.shape[-2], feats.shape[-1])
        return feats

    @property
    def device(self):
        return next(self.parameters()).device


class FlowProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        
        raft_weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = raft_weights.transforms()
        self.raft = raft_large(raft_weights)
        self.raft.eval()
        for param in self.raft.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(
        self, img: torch.Tensor, upsample=True
    ):
        n, v, c, h, w = img.shape
        img = img.reshape(n * v // 2, 2, c, h, w)
        img0 = img[:, 0]
        img1 = img[:, 1]
        img0, img1 = self.raft_transforms(img0 * .5 + .5, img1 * .5 + .5)
        flow_fwd = self.raft(img0, img1)[-1]
        flow_bwd = self.raft(img1, img0)[-1]
        flow0_r = torch.cat((flow_fwd[:, 0:1, :, :] * 2 / w , flow_fwd[:, 1:2, :, :] * 2 / h), dim=1)
        flow1_r = torch.cat((flow_bwd[:, 0:1, :, :] * 2 / w , flow_bwd[:, 1:2, :, :] * 2 / h), dim=1)
        flow = torch.stack((flow0_r, flow1_r), dim=1)

        img = torch.cat((img, flow), dim=2)

        img = img.reshape(n, v, -1, h, w)

        return img

    @property
    def device(self):
        return next(self.parameters()).device
    

class FlowOcclusionProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        
        raft_weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = raft_weights.transforms()
        self.raft = raft_large(raft_weights)
        self.raft.eval()
        for param in self.raft.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(
        self, img: torch.Tensor, upsample=True
    ):
        n, v, c, h, w = img.shape
        img = img.reshape(n * v // 2, 2, c, h, w)
        img0 = img[:, 0]
        img1 = img[:, 1]
        img0, img1 = self.raft_transforms(img0 * .5 + .5, img1 * .5 + .5)
        flow_fwd = self.raft(img0, img1)[-1]
        flow_bwd = self.raft(img1, img0)[-1]
        occ0, occ1 = compute_occlusions(flow_fwd, flow_bwd)
        flow0_r = torch.cat((flow_fwd[:, 0:1, :, :] * 2 / w , flow_fwd[:, 1:2, :, :] * 2 / h), dim=1)
        flow1_r = torch.cat((flow_bwd[:, 0:1, :, :] * 2 / w , flow_bwd[:, 1:2, :, :] * 2 / h), dim=1)
        flow = torch.stack((flow0_r, flow1_r), dim=1)
        occ = torch.stack((occ0, occ1), dim=1)

        img = torch.cat((img, flow, occ), dim=2)

        img = img.reshape(n, v, -1, h, w)

        return img

    @property
    def device(self):
        return next(self.parameters()).device


class AutoMaskingWrapper(nn.Module):

    # Adds the corresponding color from the input frame for reference
    def __init__(self, image_processor):
        super().__init__()
        self.image_processor = image_processor

        self.channels = self.image_processor.channels + 1

    def forward(self, images, threshold):
        n, v, c, h, w = images.shape
        processed_images = self.image_processor(images)
        thresholds = threshold.view(n, 1, 1, h, w).expand(n, v, 1, h, w)
        processed_images = torch.stack((processed_images, thresholds), dim=2)
        return processed_images
