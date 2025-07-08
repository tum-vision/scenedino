import ignite.distributed as idist
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from scenedino.evaluation.unified_evaluator import evaluation


@hydra.main(version_base=None, config_path="configs", config_name="evaluation")
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)

    os.environ["NCCL_DEBUG"] = "INFO"
    # torch.autograd.set_detect_anomaly(True)

    backend = config.get("backend", None)
    nproc_per_node = config.get("nproc_per_node", None)
    with_amp = config.get("with_amp", False)
    spawn_kwargs = {}

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(evaluation, config)


if __name__ == "__main__":
    main()
