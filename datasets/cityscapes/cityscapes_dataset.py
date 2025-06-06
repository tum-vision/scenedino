
import numpy as np
import time
import torch

from torchvision import transforms
from torchvision.datasets.cityscapes import Cityscapes
from torch.utils.data import Dataset


def resize_with_padding(img, target_size, padding_value, interpolation):
    target_h, target_w = target_size
    width, height = img.size
    aspect = width / height

    if aspect > (target_w / target_h):  
        new_w = target_w
        new_h = int(target_w / aspect)
    else:
        new_h = target_h
        new_w = int(target_h * aspect)

    img = transforms.functional.resize(img, (new_h, new_w), interpolation)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

    return transforms.functional.pad(img, padding, fill=padding_value)

class CityscapesSeg(Dataset):
    def __init__(self, root, image_set, image_size=(192, 640)):
        super(CityscapesSeg, self).__init__()
        self.split = image_set
        self.root = root

        transform = transforms.Compose([
            #transforms.Lambda(lambda img: resize_with_padding(img, image_size, padding_value=0, interpolation=transforms.InterpolationMode.BILINEAR)), 

            transforms.Resize((320, 640), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
            #transforms.Lambda(lambda img: resize_with_padding(img, image_size, padding_value=-1, interpolation=transforms.InterpolationMode.NEAREST)), 

            transforms.Resize((320, 640), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.long()),
        ])

        self.inner_loader = Cityscapes(self.root, image_set,
                                       mode="fine",
                                       target_type="semantic",
                                       transform=transform,
                                       target_transform=target_transform)

    def __getitem__(self, index):
        _start_time = time.time()
        image, target = self.inner_loader[index]  # (3, h, w) / (1, h, w)
      
        image = 2.0 * image - 1.0
        poses = torch.eye(4)        # (4, 4) 
        projs = torch.eye(3)        # (3, 3) 
        target = target.squeeze(0)  # (h, w)
        
        _proc_time = time.time() - _start_time

        data = {
            "imgs": [image.numpy()],
            "poses": [poses.numpy()],
            "projs": [projs.numpy()],
            "segs": [target.numpy()],
            "t__get_item__": np.array([_proc_time]),
            "index": [np.array([index])],
        }
        return data

    def __len__(self):
        return len(self.inner_loader)
