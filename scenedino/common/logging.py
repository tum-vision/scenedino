from typing import Any

import ignite
import ignite.distributed as idist
from ignite.engine import Engine, Events, EventsList
import torch
from omegaconf import OmegaConf


# TODO: move to utils or similar
def event_list_from_config(config) -> EventsList:
    events = EventsList()
    if isinstance(config, int):
        events = events | Events.EPOCH_COMPLETED(every=config) | Events.COMPLETED
    else:
        for event in config:
            if event["args"]:
                events = events | Events[event["type"]](**event["args"])
            else:
                events = events | Events[event["type"]]

    return events


def global_step_fn(trainer: Engine, config: dict[str, Any]):
    match config.get("type", None):
        case "trainer epoch":
            return lambda engine, event_name: trainer.state.epoch
        case "trainer iteration":
            return lambda engine, event_name: trainer.state.iteration
        case _:
            raise ValueError(f"Unknown global step type: {config['type']}")

    # trainer iteration
    gst = lambda engine, event_name: trainer.state.iteration

    # # iteration per epoch
    # gst_it_epoch = (
    #     lambda engine, event_name: (trainer.state.epoch - 1)
    #     * engine.state.epoch_length
    #     + engine.state.iteration
    #     - 1
    # )
    # gst_it_iters = (
    #     lambda engine, event_name: (
    #         (
    #             (trainer.state.epoch - 1) * trainer.state.epoch_length
    #             + trainer.state.iteration
    #         )
    #         // every
    #     )
    #     * engine.state.epoch_length
    #     + engine.state.iteration
    #     - 1
    # )
    # gst_ep_iters = lambda engine, event_name: (
    #     (
    #         (trainer.state.epoch - 1) * trainer.state.epoch_length
    #         + trainer.state.iteration
    #     )
    #     // every
    # )


def log_basic_info(logger, config):
    logger.info(f"Run {config['name']}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDNN version: {cudnn.version()}")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")

    logger.info("\n")
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(config)}")
