from copy import copy
import logging
from pathlib import Path

import ignite.distributed as idist
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Subset
from torch import profiler

import lpips

from scenedino.datasets import make_datasets
from scenedino.losses import make_loss
from scenedino.common.image_processor import make_image_processor, RGBProcessor
from scenedino.common.ray_sampler import (
    ImageRaySampler,
    PointBasedRaySampler,
    RandomRaySampler,
    RaySampler,
    get_ray_sampler,
)
from scenedino.common.io.configs import load_model_config
from scenedino.common.sampling_strategies import (
    get_encoder_sampling,
    get_loss_renderer_sampling,
)
from scenedino.models import make_model
from scenedino.models.backbones.dino.dinov2_module import OrthogonalLinearDimReduction

# TODO: change
from scenedino.training.base_trainer import base_training
from scenedino.common.scheduler import make_scheduler
from scenedino.renderer import NeRFRenderer

from torch.cuda.amp import autocast
from scenedino.common import util


logger = logging.getLogger("training")


class BTSWrapper(nn.Module):
    def __init__(
        self, renderer: NeRFRenderer, ray_sampler: RaySampler, config, eval_nvs=False, dino_channels=None
    ) -> None:
        super().__init__()
        self.renderer = renderer

        self.loss_from_single_img = config.get("loss_from_single_img", False)

        self.use_automasking = config.get("use_automasking", False)

        self.prediction_mode = config.get("prediction_mode", "multiscale")

        self.alternating_ratio = config.get("alternating_ratio", None)

        self.encoder_sampling = get_encoder_sampling(config["encoding_strategy"])
        self.eval_encoder_sampling = get_encoder_sampling(
            config["eval_encoding_strategy"]
        )
        self.loss_renderer_sampling = get_loss_renderer_sampling(
            config["loss_renderer_strategy"]
        )
        self.eval_loss_renderer_sampling = get_loss_renderer_sampling(
            config["eval_loss_renderer_strategy"]
        )

        cfg_ip = config.get("image_processor", {})
        self.train_image_processor = make_image_processor(cfg_ip)
        self.val_image_processor = RGBProcessor() if not self.renderer.renderer.render_flow else make_image_processor({"type": "flow_occlusion"})

        self.ray_sampler = ray_sampler

        if self.use_automasking:
            self.train_sampler.channels += 1

        self.val_sampler = ImageRaySampler(
            self.ray_sampler.z_near, self.ray_sampler.z_far, dino_upscaled=self.ray_sampler.dino_upscaled
        )

        self.predict_uncertainty = config.get("predict_uncertainty", False)
        self.uncertainty_predictor_res = config.get("uncertainty_predictor_res", 0)

        self.predict_consistency = config.get("predict_consistency", False)


        if self.predict_consistency:
            z_near = self.ray_sampler.z_near
            z_far = self.ray_sampler.z_far
            consistency_rays = config.get("consistency_rays", 512)

            self.random_ray_sampler = RandomRaySampler(z_near, z_far, consistency_rays)
            self.point_ray_sampler = PointBasedRaySampler(z_near, z_far, consistency_rays)

        if self.predict_uncertainty:
            assert self.renderer.net.uncertainty_predictor is not None

        self.eval_nvs = eval_nvs
        if self.eval_nvs:
            self.lpips = lpips.LPIPS(net="alex")

        self._counter = 0

        self.compensate_artifacts = config.get("compensate_artifacts", True)
        if self.compensate_artifacts:
            patch_size = renderer.net.encoder.gt_encoder.patch_size
            image_size = renderer.net.encoder.gt_encoder.image_size
            latent_size = renderer.net.encoder.gt_encoder.latent_size

            self.artifact_field = nn.Parameter(torch.zeros(latent_size, image_size[0]//patch_size, image_size[1]//patch_size))
            nn.init.normal_(self.artifact_field, mean=0.0, std=0.001)
        else:
            self.artifact_field = None

    @staticmethod
    def get_loss_metric_names():
        return [
            "loss",
            "loss_l2",
            "loss_mask",
            "loss_temporal",
            "loss_pgt",
        ]

    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)  # B, n_framnes, c, h, w
        poses = torch.stack(data["poses"], dim=1)  # B, n_framnes, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)  # B, n_frames, 4, 4 (-1, 1)
        data_index = data["index"]

        n, n_frames, c, h, w = images.shape
        device = images.device

        with autocast(enabled=False):
            to_base_pose = torch.inverse(poses[:, :1, :, :])
            poses = to_base_pose.expand(-1, n_frames, -1, -1) @ poses

        if self.training and self.alternating_ratio is not None:
            step = self._counter % (self.alternating_ratio + 1)
            if step < self.alternating_ratio:
                for params in self.renderer.net.encoder.parameters(True):
                    params.requires_grad_(True)
                for params in self.renderer.net.mlp_coarse.parameters(True):
                    params.requires_grad_(False)
            else:
                for params in self.renderer.net.encoder.parameters(True):
                    params.requires_grad_(False)
                for params in self.renderer.net.mlp_coarse.parameters(True):
                    params.requires_grad_(True)

        if self.training:
            ids_encoder = self.encoder_sampling(n_frames)
            ids_loss, ids_renderer, color_frame_filter = self.loss_renderer_sampling(n_frames)
        else:
            ids_encoder = self.eval_encoder_sampling(n_frames)
            ids_loss, ids_renderer, color_frame_filter = self.eval_loss_renderer_sampling(n_frames)

        combine_ids = None

        if self.loss_from_single_img:
            ids_loss = ids_loss[:1]

        if color_frame_filter is not None:
            color_frame_filter = torch.tensor(color_frame_filter, device=images.device)

        ip = self.train_image_processor if self.training else self.val_image_processor
        images_ip = ip(images)

        if self.predict_uncertainty:
            images_uncert = images.reshape(-1, c, h, w)
            uncertainties = self.renderer.net.uncertainty_predictor(images_uncert)
            uncertainties = F.interpolate(uncertainties[self.uncertainty_predictor_res], (h, w), mode="bilinear", align_corners=False)
            uncertainties = F.softplus(uncertainties).reshape(n, -1, 1, h, w)
            images_ip = torch.cat((images_ip, uncertainties), dim=2)

        with profiler.record_function(
            "trainer_encode-grid"
        ):
            self.renderer.net.compute_grid_transforms(
                projs[:, ids_encoder], poses[:, ids_encoder]
            )
            shift = self.renderer.net.encoder.encoder.patch_size // 2
            loss_feature_grid_shift = torch.randint(-shift, shift, (2,)) if self.training else None
            self.renderer.net.encode(
                images,
                projs,
                poses,
                ids_encoder=ids_encoder,
                ids_render=ids_renderer,
                ids_loss=ids_loss,
                images_alt=images_ip,
                combine_ids=combine_ids,
                color_frame_filter=color_frame_filter,
                loss_feature_grid_shift=loss_feature_grid_shift,
            )

        sampler = self.ray_sampler if self.training else self.val_sampler

        with autocast(enabled=False), profiler.record_function("trainer_sample-rays"):
            renderer_scale = self.renderer.net._scale
            dino_features = self.renderer.net.grid_l_loss_features[renderer_scale]

            if self.artifact_field is not None:
                dino_features = torch.cat(torch.broadcast_tensors(dino_features, self.artifact_field), dim=2)

            if loss_feature_grid_shift is not None:
                all_rays, all_rgb_gt, all_dino_gt = sampler.sample(
                    images_ip[:, ids_loss], poses[:, ids_loss], projs[:, ids_loss], image_ids=ids_loss,
                    dino_features=dino_features, loss_feature_grid_shift=loss_feature_grid_shift
                )
            else:
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
                with profiler.record_function("trainer_render"):
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

            # if "fine" not in render_dict:
            #     render_dict["fine"] = dict(render_dict["coarse"])

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
            if isinstance(dino_module.dim_reduction, OrthogonalLinearDimReduction):
                data["reduction_matrix"] = dino_module.dim_reduction.weights

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

            if not self.training and self.validation_tag == "visualization":
                logger.info("Visualizing a batch...")
                with torch.amp.autocast(render_dict["dino_gt"].device.type, enabled=False):
                    dino_module.fit_visualization(render_dict["dino_gt"].flatten(0, -2))
                data["vis_batch_dino_gt"] = [
                    dino_module.transform_visualization(data["dino_gt"], norm=True, from_dim=0),
                    dino_module.transform_visualization(data["dino_gt"], norm=True, from_dim=3),
                    dino_module.transform_visualization(data["dino_gt"], norm=True, from_dim=6),
                ]
                #data["vis_batch_dino_gt_kmeans"] = dino_module.fit_transform_kmeans_visualization(data["dino_gt"])
                for _data_coarse in data["coarse"]:
                    with torch.amp.autocast(_data_coarse["dino_features"].device.type, enabled=False):
                        dino_module.fit_visualization(_data_coarse["dino_features"].flatten(0, -2))
                    _data_coarse["vis_batch_dino_features"] = [
                        dino_module.transform_visualization(_data_coarse["dino_features"], norm=True, from_dim=0),
                        dino_module.transform_visualization(_data_coarse["dino_features"], norm=True, from_dim=3),
                        dino_module.transform_visualization(_data_coarse["dino_features"], norm=True, from_dim=6),
                    ]
                    #_data_coarse["vis_batch_dino_features_kmeans"] = dino_module.fit_transform_kmeans_visualization(_data_coarse["dino_features"])
                    if "dino_features_downsampled" in _data_coarse:
                        _data_coarse["vis_batch_dino_features_downsampled"] = [
                            dino_module.transform_visualization(_data_coarse["dino_features_downsampled"], norm=True, from_dim=0),
                            dino_module.transform_visualization(_data_coarse["dino_features_downsampled"], norm=True, from_dim=3),
                            dino_module.transform_visualization(_data_coarse["dino_features_downsampled"], norm=True, from_dim=6),
                        ]

                if "dino_artifacts" in data:
                    with torch.amp.autocast(render_dict["dino_gt"].device.type, enabled=False):
                        dino_module.fit_visualization(render_dict["dino_artifacts"].flatten(0, -2))
                    data["vis_batch_dino_artifacts"] = [
                        dino_module.transform_visualization(data["dino_artifacts"], norm=True, from_dim=0),
                        dino_module.transform_visualization(data["dino_artifacts"], norm=True, from_dim=3),
                        dino_module.transform_visualization(data["dino_artifacts"], norm=True, from_dim=6),
                    ]


        if self.training:
            data["feature_volume"] = self.renderer.net.grid_f_features[0]

        if self.predict_consistency and self.training:
            cf = 1

            data["consistency"] = []

            rays_0, rgb_gt_0 = self.random_ray_sampler.sample(
                images_ip[:, :1], poses[:, :1], projs[:, :1]
            )

            render_dict_0 = self.renderer(
                    rays_0,
                    want_weights=False,
                    want_alphas=False,
                    want_rgb_samps=False,
            )

            render_dict_0["rgb_gt"] = rgb_gt_0
            render_dict_0["rays"] = rays_0

            render_dict_0 = self.random_ray_sampler.reconstruct(render_dict_0, channels=images_ip.shape[2])

            xyz = rays_0[..., :3] + rays_0[..., 3:6] / torch.norm(rays_0[..., 3:6], keepdim=True, dim=-1) * render_dict_0["coarse"]["depth"][..., None]

            rays_1, rgb_gt_1 = self.point_ray_sampler.sample(
                images_ip[:, cf:cf+1], poses[:, cf:cf+1], projs[:, cf:cf+1], xyz
            )

            self.renderer.net.encode(
                images[:, cf:cf+1],
                projs[:, cf:cf+1],
                poses[:, cf:cf+1],
                images_alt=images_ip[:, cf:cf+1],
            )

            render_dict_1 = self.renderer(
                    rays_1,
                    want_weights=True,
                    want_alphas=False,
                    want_rgb_samps=False,       
            )

            render_dict_1["rgb_gt"] = rgb_gt_1
            render_dict_1["rays"] = rays_1

            render_dict_1 = self.point_ray_sampler.reconstruct(render_dict_1, channels=images_ip.shape[2])

            data["consistency"] = {
                "render_dict_0": render_dict_0,
                "render_dict_1": render_dict_1,
            }

        data["z_near"] = torch.tensor(self.ray_sampler.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.ray_sampler.z_far, device=images.device)

        surface_sample = self.sample_from_3d(poses, projs, data["coarse"][0]["depth"])
        if surface_sample is not None:
            data["sample_surface_dino_features"], data["sample_surface_sigma"] = surface_sample

        if self.training:
            self._counter += 1

        return data


    def sample_from_3d(self, poses, projs, depth, z_near=2, z_far=50, n_crops=5, n_samples=576, sample_radius=0.1):
        positions_samples = []
        n = projs.size(0)
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
            
            valid_positions = torch.nonzero((current_depth > z_near) & (current_depth < z_far), as_tuple=False)
            if valid_positions.size(0) < n_crops:  # Not enough samples in depth range (z_near, z_far)
                return None
            sampled_positions = valid_positions[torch.randperm(valid_positions.size(0))[:n_crops]]

            cam_centers = rays[0, :, :, :3]  # [h, w, 3]
            cam_raydir = rays[0, :, :, 3:6]  # [h, w, 3]

            depth_crop = current_depth[sampled_positions[:, 0], sampled_positions[:, 1]]      # [n_crops]
            cam_centers_crop = cam_centers[sampled_positions[:, 0], sampled_positions[:, 1]]  # [n_crops, 3]
            cam_raydir_crop = cam_raydir[sampled_positions[:, 0], sampled_positions[:, 1]]    # [n_crops, 3]

            positions_crop = cam_centers_crop + cam_raydir_crop * depth_crop.unsqueeze(-1)  # [n_crops, 3]
            random_shifts = sample_radius * torch.randn(n_crops, n_samples, 3, device=positions_crop.device)   # [n_crops, n_samples, 3]
            # random_shifts = random_shifts * depth_crop[:, None, None] / 5.0

            positions_samples.append(positions_crop.unsqueeze(1) + random_shifts)           # [n_crops, n_samples, 3]

        positions_samples = torch.stack(positions_samples, dim=0)  # [n, n_crops, n_samples, 3]

        _, _, sigma, _, state_dict = self.renderer.net(positions_samples.flatten(1, -2))  # [n, n_crops*n_samples, ...]
        sigma = sigma.view(n, n_crops, n_samples, -1)
        dino = state_dict["dino_features"].view(n, n_crops, n_samples, -1)

        return self.renderer.net.encoder.expand_dim(dino), 1 - torch.exp(-sigma)


def training(local_rank, config):
    return base_training(
        local_rank,
        config,
        get_dataflow,
        initialize,
    )


def get_subset(config, len_dataset: int):
    subset_type = config.get("type", None)
    match subset_type:
        case "random":
            return torch.sort(
                torch.randperm(len_dataset)[: config["args"]["size"]]
            )[0].tolist()
        case "range":
            return list(
                range(
                    config["args"].get("start", 0),
                    config["args"].get("end", len_dataset),
                )
            )
        case _:
            return list(range(len_dataset))


# NOTE: type hints are difficult but should be tuple[DataLoader, dict[str, DataLoader]]
def get_dataflow(config):
    # TODO: change to reflect evaluation
    # - Get train/test datasets
    if idist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the dataset
        idist.barrier()

    # REMOVE: ?
    mode = config.get("mode", "depth")

    train_dataset, test_dataset = make_datasets(config["dataset"])
    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
    )

    validation_loaders = {}
    for name, validation_config in config["validation"].items():
        dataset = copy(test_dataset)
        # TODO: check the following configs
        # dataset.frame_count = (
        #     1
        #     if isinstance(train_dataset, KittiRawDataset)
        #     or isinstance(train_dataset, KittiOdometryDataset)
        #     else 2
        # )
        dataset._left_offset = 0
        dataset.return_stereo = True
        dataset.return_depth = True

        subset = Subset(dataset, get_subset(validation_config["subset"], len(dataset)))
        validation_loaders[name] = idist.auto_dataloader(
            subset,
            batch_size=validation_config.get("batch_size", 1),
            num_workers=0,  # Find issue here
            shuffle=False,
        )

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()

    return train_loader, validation_loaders


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

    # TODO: think about this again
    if config["training"].get("continue", False) and config["training"].get(
        "resume_from", None
    ):
        config_path = Path(config["output"]["path"])
        logger.info(f"Loading model config from {config_path}")
        load_model_config(config_path, config)

    net = make_model(config["model"])

    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    mode = config.get("mode", "depth")

    ray_sampler = get_ray_sampler(config["training"]["ray_sampler"])

    model = BTSWrapper(renderer, ray_sampler, config["model"], mode == "nvs")

    model = idist.auto_model(model)

    dino_decoder_optim_args = config["training"]["optimizer"]["args"].copy()
    dino_decoder_optim_args["lr"] = dino_decoder_optim_args["lr"]

    dino_encoder_optim_args = config["training"]["optimizer"]["args"].copy()
    dino_encoder_optim_args["lr"] = dino_encoder_optim_args["lr"] / 10  # for fine-tuning

    # TODO: make optimizer itself configurable configurable
    optimizer = optim.Adam(
        [
            {"params": (p for n, p in model.named_parameters() if not (n.startswith('renderer.net.encoder.encoder.') or n.startswith('renderer.net.encoder.decoder.'))), 
             **config["training"]["optimizer"]["args"]},
            {"params": model.renderer.net.encoder.decoder.parameters(), 
             **dino_decoder_optim_args},
            {"params": model.renderer.net.encoder.encoder.parameters(), 
             **dino_encoder_optim_args},
        ]
    )
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = make_scheduler(config["training"].get("scheduler", {}), optimizer)

    # TODO: change to reflect all the losses together with the config
    # TODO: integrate lambda for all losses
    criterion = [
        make_loss(config_loss)
        for config_loss in config["training"]["loss"]
        # ReconstructionLoss(
        #     config["training"]["loss"], config["model"].get("use_automasking", False)
        # )
    ]

    return model, optimizer, criterion, lr_scheduler
