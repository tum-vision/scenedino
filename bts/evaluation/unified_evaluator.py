import logging
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from bts.common.io.configs import load_model_config
from bts.models import make_model


from bts.datasets import make_test_dataset
from bts.common.geometry import distance_to_z
from bts.renderer import NeRFRenderer
from bts.common.ray_sampler import ImageRaySampler, get_ray_sampler
from bts.evaluation.base_evaluator import base_evaluation

from bts.training.trainer_downstream import BTSDownstreamWrapper

IDX = 0

logger = logging.getLogger("evaluation")


class BTSWrapper(nn.Module):
    def __init__(
        self,
        renderer,
        config,
        # evaluation_fns
    ) -> None:
        super().__init__()

        self.renderer = renderer

        # TODO: have a consitent sampling range
        self.z_near = config.get("z_near", 3.0)
        self.z_far = config.get("z_far", 80.0)
        self.sampler = ImageRaySampler(self.z_near, self.z_far)

        # self.evaluation_fns = evaluation_fns

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]

    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)  # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)  # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)  # n, v, 4, 4 (-1, 1)

        B, n_frames, c, h, w = images.shape
        device = images.device

        # Use first frame as keyframe
        to_base_pose = torch.inverse(poses[:, :1, :, :])
        poses = to_base_pose.expand(-1, n_frames, -1, -1) @ poses

        # TODO: make configurable
        ids_encoder = [0]

        self.renderer.net.compute_grid_transforms(
            projs[:, ids_encoder], poses[:, ids_encoder]
        )
        self.renderer.net.encode(
            images,
            projs,
            poses,
            ids_encoder=ids_encoder,
            ids_render=ids_encoder,
            images_alt=images * 0.5 + 0.5,
        )

        all_rays, all_rgb_gt = self.sampler.sample(images * 0.5 + 0.5, poses, projs)

        data["fine"] = []
        data["coarse"] = []

        self.renderer.net.set_scale(0)
        render_dict = self.renderer(all_rays, want_weights=True, want_alphas=True)

        if "fine" not in render_dict:
            render_dict["fine"] = dict(render_dict["coarse"])

        render_dict["rgb_gt"] = all_rgb_gt
        render_dict["rays"] = all_rays

        # TODO: check if distance to z is needed
        render_dict = self.sampler.reconstruct(render_dict)
        render_dict["coarse"]["depth"] = distance_to_z(
            render_dict["coarse"]["depth"], projs
        )
        render_dict["fine"]["depth"] = distance_to_z(
            render_dict["fine"]["depth"], projs
        )

        data["fine"].append(render_dict["fine"])
        data["coarse"].append(render_dict["coarse"])
        data["rgb_gt"] = render_dict["rgb_gt"]
        data["rays"] = render_dict["rays"]

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)

        # for eval_fn in self.evaluation_fns:
        #     data["metrics"].update(eval_fn(data, model=self.renderer.net))

        return data


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize)


def get_dataflow(config):
    test_dataset = make_test_dataset(config["dataset"])
    test_loader = DataLoader(
        test_dataset,  # Subset(test_dataset, torch.randperm(test_dataset.length)[:1000]),
        batch_size=config.get("batch_size", 1),
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    return test_loader


def initialize(config: dict):
    checkpoint = Path(config["checkpoint"])
    logger.info(f"Loading model config from {checkpoint.parent}")
    load_model_config(checkpoint.parent, config)

    net = make_model(config["model"], config["downstream"])
    # net = make_model(config["model"])

    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    # TODO: attach evaluation functions rather that add them to the wrapper
    # eval_fns = []
    # for eval_conf in config["evaluations"]:
    #     eval_fn = make_eval_fn(eval_conf)
    #     if eval_fn is not None:
    #         eval_fns.append(eval_fn)

    ray_sampler = get_ray_sampler(config["training"]["ray_sampler"])
    model = BTSDownstreamWrapper(renderer, ray_sampler, config["model"])
    # model = BTSWrapper(renderer, config["model"])
    # model = BTSWrapper(renderer, config["model"], eval_fns)

    return model
