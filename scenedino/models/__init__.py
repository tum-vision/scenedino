from scenedino.common.positional_encoding import PositionalEncoding
from .backbones import make_backbone
from .prediction_heads import make_head
from .bts import BTSNet

from scenedino.downstream_head import make_downstream_head


def make_model(config, downstream_config=None):
    arch = config.get("arch", "BTSNet")

    sample_color = config.get("sample_color", True)
    predict_dino = config.get("predict_dino", False)
    dino_dims = config.get("dino_dims", 16)
    if sample_color and predict_dino:
        d_out = 1 + dino_dims
    elif sample_color:
        d_out = 1
    else:
        d_out = 4

    uncertainty_predictor_conf = config.get("uncertainty_predictor", None)
    if uncertainty_predictor_conf is not None:
        uncertainty_predictor = make_backbone(uncertainty_predictor_conf)
    else:
        uncertainty_predictor = None

    match arch:
        case "BTSNet":
            code_xyz = PositionalEncoding.from_conf(config["code"], d_in=3)
            encoder = make_backbone(config["encoder"])
            d_in = encoder.latent_size + code_xyz.d_out

            split_dino_heads = config.get("split_dino_heads", False)
            if split_dino_heads:
                heads = {
                    head_conf["name"]: make_head(head_conf, d_in, 1 if head_conf["name"] == "normal_head" else dino_dims)
                    for head_conf in config["decoder_heads"]
                }
            else:
                heads = {
                    head_conf["name"]: make_head(head_conf, d_in, d_out)
                    for head_conf in config["decoder_heads"]
                }

            if downstream_config is not None:
                downstream_head = make_downstream_head(downstream_config)
            else:
                downstream_head = None

            # TODO: check ren_nc
            return BTSNet(
                config,
                encoder,
                code_xyz,
                heads,
                config.get("final_pred_head", None),
                uncertainty_predictor=uncertainty_predictor,
                ren_nc=None,
                downstream_head=downstream_head
            )
        case _:
            raise NotImplementedError("Model architecture was not implemented yet")
