from pathlib import Path

import hydra
from omegaconf import OmegaConf


def load_model_config(output_path: Path, config):
    # @hydra.main(
    #     version_base=None, config_path=str(output_path), config_name="training_config"
    # )
    # def load_model_conf(cfg):
    #     return cfg

    # cfg = load_model_conf()
    cfg = OmegaConf.load(output_path / "training_config.yaml")
    if cfg:
        config["model"] = cfg["model"]


def save_hydra_config(path: Path, config, force: bool = False):
    # TODO: Save config as yaml. Check
    if not path.is_file() and not force:
        OmegaConf.save(config, path)
