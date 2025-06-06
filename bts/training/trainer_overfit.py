from copy import copy

import ignite.distributed as idist
from torch import optim
from torch.utils.data import DataLoader, Subset

from bts.training.base_trainer import base_training

# TODO: change dataset
from bts.datasets import make_datasets
from bts.common.scheduler import make_scheduler
from bts.renderer import NeRFRenderer
from bts.models.backbones.dino.dinov2_module import *
from bts.training.trainer import BTSWrapper
from bts.models import make_model
from bts.common.ray_sampler import get_ray_sampler
from bts.losses import make_loss



class EncoderDummy(nn.Module):
    def __init__(self, size, feat_dim, num_views=1) -> None:
        super().__init__()  ## initializes this feature map as a random tensor of a specified size
        self.feats = nn.Parameter(torch.randn(num_views, feat_dim, *size))
        self.latent_size = feat_dim

    def forward(self, x):
        n = x.shape[0]
        return [self.feats.expand(n, -1, -1, -1)]


class EncoderDinoDummy(nn.Module):
    def __init__(self,
                 mode: str,                                 # downsample-prediction, upsample-gt
                 decoder_arch: str,                         # nearest, bilinear, sfp, dpt
                 upsampler_arch: Optional[str],             # nearest, bilinear, multiscale-crop
                 downsampler_arch: Optional[str],           # sample-center, featup
                 encoder_arch: str,                         # vit-s, vit-b, fit3d-s
                 separate_gt_encoder_arch: Optional[str],   # vit-s, vit-b, fit3d-s, None (reuses encoder)
                 encoder_freeze: bool,
                 dim_reduction_arch: str,                   # orthogonal-linear, mlp
                 num_ch_enc: np.array,
                 intermediate_features: List[int],
                 decoder_out_dim: int,
                 dino_pca_dim: int,
                 image_size: Tuple[int, int],
                 key_features: bool,
                 ):

        super().__init__()

        self.feats = nn.Parameter(torch.randn(1, decoder_out_dim, *image_size))
        self.latent_size = decoder_out_dim

        if separate_gt_encoder_arch is None:
            self.gt_encoder = build_encoder(encoder_arch, image_size, [], key_features)  # ONLY IN OVERFIT DUMMY!
        else:
            self.gt_encoder = build_encoder(separate_gt_encoder_arch, image_size, [], key_features)

        for p in self.gt_encoder.parameters(True):
            p.requires_grad = False

        # General way of creating loss
        if mode == "downsample-prediction":
            assert upsampler_arch is None
            self.downsampler = build_downsampler(downsampler_arch, self.gt_encoder.latent_size)
            self.gt_wrapper = self.gt_encoder

        elif mode == "upsample-gt":
            assert downsampler_arch is None
            self.downsampler = None
            self.gt_wrapper = build_gt_upsampling_wrapper(upsampler_arch, self.gt_encoder, image_size)

        else:
            raise NotImplementedError

        self.extra_outs = 0
        self.latent_size = decoder_out_dim

        self.dino_pca_dim = dino_pca_dim
        self.dim_reduction = build_dim_reduction(dim_reduction_arch, self.gt_encoder.latent_size, dino_pca_dim)
        self.visualization = VisualizationModule(self.gt_encoder.latent_size)

    def forward(self, x, ground_truth=False):
        if ground_truth:
            return self.gt_wrapper(x)

        return [self.feats.expand(x.shape[0], -1, -1, -1)]

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
            separate_gt_encoder_arch=conf.get("separate_gt_encoder_arch", None),
            encoder_freeze=conf.encoder_freeze,
            dim_reduction_arch=conf.dim_reduction_arch,
            num_ch_enc=conf.get("num_ch_enc", None),
            intermediate_features=conf.get("intermediate_features", []),
            decoder_out_dim=conf.decoder_out_dim,
            dino_pca_dim=conf.dino_pca_dim,
            image_size=conf.image_size,
            key_features=conf.key_features,
        )


class BTSWrapperOverfit(BTSWrapper):
    def __init__(self, renderer, ray_sampler, config, eval_nvs=False, size=None) -> None:
        super().__init__(renderer, ray_sampler, config, eval_nvs)

        if config["predict_dino"]:
            encoder_dummy = EncoderDinoDummy.from_conf(config["encoder"])
        else:
            encoder_dummy = EncoderDummy(
                size,
                config["encoder"]["d_out"],
            )

        self.renderer.net.encoder = encoder_dummy


def training(local_rank, config):
    return base_training(
        local_rank,
        config,
        get_dataflow,
        initialize,
    )


def get_dataflow(config):
    # - Get train/test datasets
    if idist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the datasetMVBTSNet
        idist.barrier()

    train_dataset_full = make_datasets(config["dataset"])[0]
    train_dataset = Subset(
        train_dataset_full,
        [config.get("example", config["dataset"].get("skip", 0))],
    )

    train_dataset.dataset._skip = config["dataset"].get("skip", 0)

    validation_datasets = {}
    for name, validation_config in config["validation"].items():
        dataset = copy(train_dataset)
        dataset.dataset.return_depth = True
        validation_datasets[name] = dataset

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()  ## Once the dataset has been downloaded, the barrier is invoked, and only then are the other processes allowed to proceed.
        ## By using this method, you can control the order of execution in a distributed setting and ensure that certain
        ## steps are not performed multiple times by different processes. This can be very useful when working with shared
        ## resources or when coordination is required between different processes.

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader_full = DataLoader(train_dataset_full)
    train_loader = DataLoader(train_dataset)

    validation_loaders = {}
    for name, dataset in validation_datasets.items():
        validation_loaders[name] = DataLoader(dataset)

    return (train_loader, train_loader_full), validation_loaders


def initialize(config: dict):
    net = make_model(config["model"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    mode = config.get("mode", "depth")
    ray_sampler = get_ray_sampler(config["training"]["ray_sampler"])

    model = BTSWrapperOverfit(
        renderer,
        ray_sampler,
        config["model"],
        mode == "nvs",
        size=config["dataset"].get("image_size", (192, 640)),
    )

    model = idist.auto_model(model)
    optimizer = optim.Adam(model.parameters(), **config["training"]["optimizer"]["args"])
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = make_scheduler(config["training"].get("scheduler", {}), optimizer)

    criterion = [
        make_loss(config_loss)
        for config_loss in config["training"]["loss"]
    ]

    return model, optimizer, criterion, lr_scheduler
