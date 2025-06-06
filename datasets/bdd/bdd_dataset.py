
import numpy as np
import time
import torch
import os

from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset

from collections import namedtuple
from datasets.kitti_360.labels import trainId2label


Label = namedtuple(
    "Label",
    [
        "name",
        "id",
        "trainId",
        "category",
        "categoryId",
        "hasInstances",
        "ignoreInEval",
        "color",
        "to_cs27",
    ],
)

BDD_LABEL = [
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0), 255),
    Label("dynamic", 1, 255, "void", 0, False, True, (111, 74, 0), 255),
    Label("ego vehicle", 2, 255, "void", 0, False, True, (0, 0, 0), 255),
    Label("ground", 3, 255, "void", 0, False, True, (81, 0, 81), 255),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0), 255),
    Label("parking", 5, 255, "flat", 1, False, True, (250, 170, 160), 2),
    Label("rail track", 6, 255, "flat", 1, False, True, (230, 150, 140), 3),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128), 0),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232), 1),
    Label("bridge", 9, 255, "construction", 2, False, True, (150, 100, 100), 8),
    Label("building", 10, 2, "construction", 2, False, False, (70, 70, 70), 4),
    Label("fence", 11, 4, "construction", 2, False, False, (190, 153, 153), 6),
    Label("garage", 12, 255, "construction", 2, False, True, (180, 100, 180), 255),
    Label("guard rail", 13, 255, "construction", 2, False, True, (180, 165, 180), 7),
    Label("tunnel", 14, 255, "construction", 2, False, True, (150, 120, 90), 9),
    Label("wall", 15, 3, "construction", 2, False, False, (102, 102, 156), 5),
    Label("banner", 16, 255, "object", 3, False, True, (250, 170, 100), 255),
    Label("billboard", 17, 255, "object", 3, False, True, (220, 220, 250), 255),
    Label("lane divider", 18, 255, "object", 3, False, True, (255, 165, 0), 255),
    Label("parking sign", 19, 255, "object", 3, False, False, (220, 20, 60), 255),
    Label("pole", 20, 5, "object", 3, False, False, (153, 153, 153), 10),
    Label("polegroup", 21, 255, "object", 3, False, True, (153, 153, 153), 11),
    Label("street light", 22, 255, "object", 3, False, True, (220, 220, 100), 255),
    Label("traffic cone", 23, 255, "object", 3, False, True, (255, 70, 0), 255),
    Label("traffic device", 24, 255, "object", 3, False, True, (220, 220, 220), 255),
    Label("traffic light", 25, 6, "object", 3, False, False, (250, 170, 30), 12),
    Label("traffic sign", 26, 7, "object", 3, False, False, (220, 220, 0), 13),
    Label("traffic sign frame", 27, 255, "object", 3, False, True, (250, 170, 250), 255),
    Label("terrain", 28, 9, "nature", 4, False, False, (152, 251, 152), 15),
    Label("vegetation", 29, 8, "nature", 4, False, False, (107, 142, 35), 14),
    Label("sky", 30, 10, "sky", 5, False, False, (70, 130, 180), 16),
    Label("person", 31, 11, "human", 6, True, False, (220, 20, 60), 17),
    Label("rider", 32, 12, "human", 6, True, False, (255, 0, 0), 18),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32), 26),
    Label("bus", 34, 15, "vehicle", 7, True, False, (0, 60, 100), 21),
    Label("car", 35, 13, "vehicle", 7, True, False, (0, 0, 142), 19),
    Label("caravan", 36, 255, "vehicle", 7, True, True, (0, 0, 90), 22),
    Label("motorcycle", 37, 17, "vehicle", 7, True, False, (0, 0, 230), 25),
    Label("trailer", 38, 255, "vehicle", 7, True, True, (0, 0, 110), 23),
    Label("train", 39, 16, "vehicle", 7, True, False, (0, 80, 100), 24),
    Label("truck", 40, 14, "vehicle", 7, True, False, (0, 0, 70), 20),
]


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


class BDDSeg(Dataset):
    def __init__(self, root, image_set, image_size=(192, 640)):
        super(BDDSeg, self).__init__()
        self.split = image_set
        self.root = root

        self.image_transform = transforms.Compose([
            #transforms.Lambda(lambda img: resize_with_padding(img, image_size, padding_value=0, interpolation=transforms.InterpolationMode.BILINEAR)), 

            transforms.Resize((320, 640), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

        self.target_transform = transforms.Compose([
            #transforms.Lambda(lambda img: resize_with_padding(img, image_size, padding_value=-1, interpolation=transforms.InterpolationMode.NEAREST)), 

            transforms.Resize((320, 640), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.long()),
        ])

        self.images, self.targets = [], []

        image_dir = os.path.join(self.root, "images/10k", self.split)
        target_dir = os.path.join(self.root, "labels/pan_seg/bitmasks", self.split)
        for file_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, file_name)

            target_filename = os.path.splitext(file_name)[0] + ".png"
            target_path = os.path.join(target_dir, target_filename)
            assert os.path.isfile(target_path)

            self.images.append(image_path)
            self.targets.append(target_path)

        self.class_mapping = torch.Tensor([trainId2label[c.trainId].id for c in BDD_LABEL]).int()

    def __getitem__(self, index):
        _start_time = time.time()

        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index])

        image = self.image_transform(image)
        target = self.target_transform(target)
      
        image = 2.0 * image - 1.0
        poses = torch.eye(4)        # (4, 4) 
        projs = torch.eye(3)        # (3, 3) 
        target = target[0]  # ("instance", "semantic", "polygon", "color")
        target = self.class_mapping[target]
        
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
        return len(self.images)
