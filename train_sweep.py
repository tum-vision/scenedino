import ignite.distributed as idist
import hydra
from omegaconf import DictConfig, OmegaConf
import os

import torch
import optuna
from optuna.samplers import TPESampler

from bts.training.trainer import training as full_training
from bts.training.trainer_overfit import training as overfit_training
from bts.training.trainer_downstream import training as downstream_training

import numpy as np


def custom_gamma(x: int) -> int:
    return min(int(np.ceil(np.sqrt(x) / 4)), 25)


@hydra.main(version_base=None, config_path="configs", config_name="exp_kitti_360_DFT")
def main(config: DictConfig):
    study_name = config["sweep"]["study_name"]
    storage_url = config["sweep"]["storage_url"]
    direction = config["sweep"]["direction"]
    n_trials = config["sweep"]["n_trials"]

    OmegaConf.set_struct(config, False)
    config["output"]["original_path"] = config["output"]["path"]
    config["output"]["original_unique_id"] = config["output"]["unique_id"]

    study = optuna.create_study(
        study_name=study_name,
        sampler=TPESampler(n_startup_trials=10, gamma=custom_gamma),
        storage=storage_url,
        direction=direction,
        load_if_exists=True,
        pruner=optuna.pruners.PercentilePruner(75.0, interval_steps=300, n_min_trials=20)  # NopPruner(),
    )

    if config["sweep"]["start_original_param"] and len(study.get_trials()) == 0:
        current_hparams = {}
        for sweep_hparam in config["sweep"]["hparams"]:
            config_value = OmegaConf.select(config, sweep_hparam["key"])
            config_name = sweep_hparam["kwargs"]["name"]
            current_hparams[config_name] = config_value
        study.enqueue_trial(current_hparams)

    def _objective(trial: optuna.Trial) -> float:
        config["eval_seed"] = config["seed"]
        config["seed"] = trial.number
        config["output"]["path"] = config["output"]["original_path"]
        original_unique_id = config["output"]["original_unique_id"]
        config["output"]["unique_id"] = f"{original_unique_id}_{trial.number}"

        for sweep_hparam in config["sweep"]["hparams"]:
            trial_method = getattr(trial, sweep_hparam["method"])
            config_key = sweep_hparam["key"]
            config_value = trial_method(**sweep_hparam["kwargs"])

            OmegaConf.update(config, config_key, config_value, merge=True)
            print(f"{config_key}: {config_value}")

        os.environ["NCCL_DEBUG"] = "INFO"
        torch.autograd.set_detect_anomaly(False)

        training = globals()[config["training_type"]]
        best_metric = training(0, config, trial)
        return best_metric

    study.optimize(_objective, n_trials=n_trials)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
