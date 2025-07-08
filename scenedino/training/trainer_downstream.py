import logging
from pathlib import Path

import ignite.distributed as idist
import torch
from torch import optim

from scenedino.losses import make_loss
from scenedino.common.ray_sampler import (
    RaySampler,
    get_ray_sampler,
)
from scenedino.common.io.configs import load_model_config

from scenedino.models import make_model
from scenedino.training.trainer import BTSWrapper, get_dataflow

from scenedino.training.base_trainer import base_training
from scenedino.common.scheduler import make_scheduler
from scenedino.renderer import NeRFRenderer
from scenedino.common import util

from torch.cuda.amp import autocast

logger = logging.getLogger("training")


class BTSDownstreamWrapper(BTSWrapper):
    def __init__(
        self, renderer: NeRFRenderer, ray_sampler: RaySampler, config, eval_nvs=False, dino_channels=None
    ) -> None:
        super().__init__(renderer, ray_sampler, config, eval_nvs, dino_channels)
        for param in super().parameters(True):
            param.requires_grad_(False)
        for param in renderer.net.downstream_head.parameters(True):
            param.requires_grad_(True)

        self.sample_radius_3d = config.get("sample_radius_3d", 0.5)

    def forward(self, data):
        with torch.no_grad():
            # TODO: CLEAN THIS UP
            if self.renderer.net.downstream_head.training and len(data["imgs"]) > 1 and torch.rand(1).item() < 0.5:
                # side view
                encode_id = torch.randint(low=4, high=8, size=(1,)).item()
                # Segmentation only present in front view
                data.pop("segs")
            else:
                encode_id = 0

            data["imgs"] = [data["imgs"][encode_id]]
            data["projs"] = [data["projs"][encode_id]]
            data["poses"] = [data["poses"][encode_id]]

            data = self.forward_downstream(data, id_encoder=0)
            if not self.renderer.net.downstream_head.training and hasattr(self, "validation_tag") and self.validation_tag == "visualization_seg":
                dino_module = self.renderer.net.encoder
                dino_module.visualization.n_kmeans_clusters = 19
                for _data_coarse in data["coarse"]:
                    with torch.amp.autocast(_data_coarse["dino_features"].device.type, enabled=False):
                        dino_module.fit_visualization(_data_coarse["dino_features"].float().flatten(0, -2))
                    _data_coarse["vis_batch_dino_features"] = [
                        dino_module.transform_visualization(_data_coarse["dino_features"], norm=True, from_dim=0),
                        dino_module.transform_visualization(_data_coarse["dino_features"], norm=True, from_dim=3),
                        dino_module.transform_visualization(_data_coarse["dino_features"], norm=True, from_dim=6),
                    ]
                    #_data_coarse["vis_batch_dino_features_kmeans"] = dino_module.fit_transform_kmeans_visualization(_data_coarse["dino_features"])

        data = self.renderer.net.downstream_head.forward_training(data, visualize=not self.training and hasattr(self, "validation_tag") and self.validation_tag == "visualization_seg")
        return data

    def forward_downstream(self, data, id_encoder):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)  # B, n_framnes, c, h, w
        poses = torch.stack(data["poses"], dim=1)  # B, n_framnes, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)  # B, n_frames, 4, 4 (-1, 1)

        n, n_frames, c, h, w = images.shape

        with autocast(enabled=False):
            to_base_pose = torch.inverse(poses[:, :1, :, :])
            poses = to_base_pose.expand(-1, n_frames, -1, -1) @ poses

        ids_encoder = [id_encoder]
        ids_loss = ids_encoder
        ids_renderer = ids_encoder

        ip = self.train_image_processor if self.training else self.val_image_processor
        images_ip = ip(images)

        self.renderer.net.compute_grid_transforms(
            projs[:, ids_encoder], poses[:, ids_encoder]
        )
        self.renderer.net.encode(
            images,
            projs,
            poses,
            ids_encoder=ids_encoder,
            ids_render=ids_renderer,
            ids_loss=ids_loss,
            images_alt=images_ip,
            combine_ids=None,
            color_frame_filter=None,
        )

        sampler = self.ray_sampler if self.training else self.val_sampler

        renderer_scale = self.renderer.net._scale
        dino_features = self.renderer.net.grid_l_loss_features[renderer_scale]

        if self.artifact_field is not None:
            dino_features = torch.cat(torch.broadcast_tensors(dino_features, self.artifact_field), dim=2)
            
        all_rays, all_rgb_gt, all_dino_gt = sampler.sample(
            images_ip[:, ids_loss], poses[:, ids_loss], projs[:, ids_loss], image_ids=ids_loss,
            dino_features=dino_features
        )

        if self.artifact_field is not None:
            all_dino_artifacts = all_dino_gt[:, :, self.artifact_field.shape[0]:]
            all_dino_gt = all_dino_gt[:, :, :self.artifact_field.shape[0]]
        else:
            all_dino_artifacts = None

        data["fine"], data["coarse"] = [], []

        scales = list(
            self.renderer.net.encoder.scales
            if self.prediction_mode == "multiscale"
            else [self.renderer.net.get_scale()]
        )

        for scale in scales:
            self.renderer.net.set_scale(scale)

            using_fine = self.renderer.renderer.using_fine

            if scale == 0:
                render_dict = self.renderer(
                    all_rays,
                    want_weights=True,
                    want_alphas=True,
                    want_rgb_samps=True,
                )
            else:
                using_fine = self.renderer.renderer.using_fine
                self.renderer.renderer.using_fine = False
                render_dict = self.renderer(
                    all_rays,
                    want_weights=True,
                    want_alphas=True,
                    want_rgb_samps=False,
                )
                self.renderer.renderer.using_fine = using_fine

            render_dict["rgb_gt"] = all_rgb_gt
            render_dict["rays"] = all_rays
            render_dict["dino_gt"] = all_dino_gt.float()

            if all_dino_artifacts is not None:
                render_dict["dino_artifacts"] = all_dino_artifacts.float()

            render_dict = sampler.reconstruct(render_dict,
                                              channels=images_ip.shape[2],
                                              dino_channels=self.renderer.net.encoder.dino_pca_dim)

            if "fine" in render_dict:
                data["fine"].append(render_dict["fine"])
            data["coarse"].append(render_dict["coarse"])
            data["rgb_gt"] = render_dict["rgb_gt"]
            data["dino_gt"] = render_dict["dino_gt"]
            if "dino_artifacts" in render_dict:
                data["dino_artifacts"] = render_dict["dino_artifacts"]
            data["rays"] = render_dict["rays"]

            dino_module = self.renderer.net.encoder
            downsampling_mode = "patch" if self.training else "image"
            for _data_coarse in data["coarse"]:
                _data_coarse["dino_features"] = dino_module.expand_dim(_data_coarse["dino_features"])
                downsampling_result = dino_module.downsample(_data_coarse["dino_features"], downsampling_mode)
                if isinstance(downsampling_result, tuple):
                    (_data_coarse["dino_features_downsampled"],
                     _data_coarse["dino_features_salience_map"],
                     _data_coarse["dino_features_weight_map"],
                     _data_coarse["dino_features_per_patch_weight"]) = downsampling_result
                elif downsampling_result is not None:
                    _data_coarse["dino_features_downsampled"] = downsampling_result

            if not self.training and hasattr(self, "validation_tag") and self.validation_tag == "visualization":
                for _data_coarse in data["coarse"]:
                    with torch.amp.autocast(_data_coarse["dino_features"].device.type, enabled=False):
                        dino_module.fit_visualization(_data_coarse["dino_features"].flatten(0, -2))
                    _data_coarse["vis_batch_dino_features"] = [
                        dino_module.transform_visualization(_data_coarse["dino_features"], norm=True, from_dim=0),
                        dino_module.transform_visualization(_data_coarse["dino_features"], norm=True, from_dim=3),
                        dino_module.transform_visualization(_data_coarse["dino_features"], norm=True, from_dim=6),
                    ]
                    #_data_coarse["vis_batch_dino_features_kmeans"] = dino_module.fit_transform_kmeans_visualization(_data_coarse["dino_features"])


        if self.training:
            data["feature_volume"] = self.renderer.net.grid_f_features[0]

        data["z_near"] = torch.tensor(self.ray_sampler.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.ray_sampler.z_far, device=images.device)

        surface_sample = self.sample_3d_crop(poses, projs, data["coarse"][0]["depth"], sample_radius=self.sample_radius_3d)
        if surface_sample is not None:
            data["sample_surface_dino_features"], data["sample_surface_sigma"] = surface_sample

        if self.training:
            self._counter += 1

        return data

    def sample_3d_crop(self, poses, projs, depth, z_far=100, n_crops=5, n_samples=576, sample_radius=0.5, sigma_threshold=0.5):
        positions_samples = []
        n = projs.size(0)

        oversampling = 4
        for n_ in range(n):
            focals = projs[n_, :1, [0, 1], [0, 1]]
            centers = projs[n_, :1, [0, 1], [2, 2]]

            _, _, height, width = depth.shape
            rays, _ = util.gen_rays(
                poses[n_, :1].view(-1, 4, 4),
                width,
                height,
                focal=focals,
                c=centers,
                z_near=0,
                z_far=0,
                norm_dir=True,
            )
            current_depth = depth[n_, 0]  # [h, w]
            limits = torch.quantile(current_depth[current_depth < z_far], torch.range(0, 1, 1/n_crops).cuda())
            
            sampled_positions = []
            for i in range(n_crops):
                valid_positions = torch.nonzero((current_depth > limits[i]) & (current_depth < limits[i+1]), as_tuple=False)
                if valid_positions.size(0) > 0:  # Not enough samples in depth range
                    sampled_positions.append(valid_positions[torch.randint(valid_positions.size(0), (1,)).item()])

            n_crops = len(sampled_positions)
            if n_crops > 0:
                sampled_positions = torch.stack(sampled_positions, dim=0)

                cam_centers = rays[0, :, :, :3]  # [h, w, 3]
                cam_raydir = rays[0, :, :, 3:6]  # [h, w, 3]

                depth_crop = current_depth[sampled_positions[:, 0], sampled_positions[:, 1]]      # [n_crops]
                cam_centers_crop = cam_centers[sampled_positions[:, 0], sampled_positions[:, 1]]  # [n_crops, 3]
                cam_raydir_crop = cam_raydir[sampled_positions[:, 0], sampled_positions[:, 1]]    # [n_crops, 3]

                positions_crop = cam_centers_crop + cam_raydir_crop * depth_crop.unsqueeze(-1)  # [n_crops, 3]
                                
                # Sample in unit sphere
                unit_vecs = torch.randn(n_crops, oversampling*n_samples, 3, device=positions_crop.device)   # [n_crops, n_samples, 3]
                unit_vecs /= torch.norm(unit_vecs, dim=2, keepdim=True)
                radii = sample_radius * torch.rand(n_crops, oversampling*n_samples, 1).cuda() ** (1/3)

                # Scale radius in view space
                # radii = radii * depth_crop[:, None, None] / 20.0

                random_shifts = unit_vecs * radii
                positions_samples.append(positions_crop.unsqueeze(1) + random_shifts)           # [n_crops, n_samples, 3]

        if not positions_samples:
            return None, None

        positions_samples = torch.stack(positions_samples, dim=0)  # [n, n_crops, n_samples, 3]

        _, _, sigma, _, state_dict = self.renderer.net(positions_samples.flatten(1, -2))  # [n, n_crops*n_samples, ...]
        sigma = sigma.view(n * n_crops, oversampling*n_samples)
        dino = state_dict["dino_features"].view(n * n_crops, oversampling * n_samples, -1)

        valid_samples = sigma > sigma_threshold
        valid_crop = valid_samples.sum(-1) > n_samples

        if valid_crop.sum() == 0:
            return None, None

        # Keep only crops with enough valid samples
        sigma = sigma[valid_crop]  
        dino = dino[valid_crop]

        # For each crop, take the first n_samples valid samples
        sigma = torch.stack([s[mask][:n_samples] for s, mask in zip(sigma, valid_samples[valid_crop])]).unsqueeze(0).unsqueeze(-1)
        dino = torch.stack([d[mask][:n_samples] for d, mask in zip(dino, valid_samples[valid_crop])]).unsqueeze(0)

        return self.renderer.net.encoder.expand_dim(dino), 1 - torch.exp(-sigma)

    def train(self, mode=True):
        super().train(False)
        self.renderer.net.downstream_head.train(mode)

    def parameters(self, recurse=True):
        return self.renderer.net.downstream_head.parameters(recurse)
    
    def parameters_lr(self):
        return self.renderer.net.downstream_head.parameters_lr()

    def update_model_eval(self, metrics):
        self.renderer.net.downstream_head.update_model_eval(metrics)


def training(local_rank, config, sweep_trial=None):
    return base_training(
        local_rank,
        config,
        get_dataflow,
        initialize,
        sweep_trial,
    )


def initialize(config: dict):
    # Continue if checkpoint already exists
    if config["training"].get("continue", False):
        prefix = "training_checkpoint_"
        ckpts = Path(config["output"]["path"]).glob(f"{prefix}*.pt")
        # TODO: probably correct logic but please check
        training_steps = [int(ckpt.stem.split(prefix)[1]) for ckpt in ckpts]
        if training_steps:
            config["training"]["resume_from"] = (
                Path(config["output"]["path"]) / f"{prefix}{max(training_steps)}.pt"
            )

    if config["training"].get("continue", False) and config["training"].get(
        "resume_from", None
    ):
        config_path = Path(config["output"]["path"])
        logger.info(f"Loading model config from {config_path}")
        load_model_config(config_path, config)

    net = make_model(config["model"], config["downstream"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    mode = config.get("mode", "depth")

    ray_sampler = get_ray_sampler(config["training"]["ray_sampler"])

    model = BTSDownstreamWrapper(renderer, ray_sampler, config["model"], mode == "nvs")
    model = idist.auto_model(model)

    # TODO: make optimizer itself configurable configurable
    if config["training"].get("optimizer", None):
        optim_args = config["training"]["optimizer"]["args"].copy()
        optim_lr = optim_args.pop("lr")
        optimizer = optim.Adam(
            [
                {"params": params, "lr": lr_factor * optim_lr}
                for lr_factor, params in model.parameters_lr()
            ],
            **optim_args
        )
        optimizer = idist.auto_optim(optimizer)
    else:
        optimizer = None

    if config["training"].get("scheduler", None):
        lr_scheduler = make_scheduler(config["training"].get("scheduler", {}), optimizer)
    else:
        lr_scheduler = None

    criterion = [
        make_loss(config_loss)
        for config_loss in config["training"].get("loss", [])
    ]

    return model, optimizer, criterion, lr_scheduler
