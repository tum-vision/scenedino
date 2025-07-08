from datetime import datetime
from pathlib import Path

import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.utils import manual_seed, setup_logger
from torch.cuda.amp import autocast
from scenedino.common.logging import log_basic_info

from scenedino.common.array_operations import to

# from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.handlers import TensorboardLogger

from scenedino.common.metrics import DictMeanMetric, SegmentationMetric, ConcatenateMetric
from scenedino.training.handlers import VisualizationHandler
from scenedino.visualization.vis_2d import tb_visualize

from .wrapper import make_eval_fn


def base_evaluation(
    local_rank,
    config,
    get_dataflow,
    initialize,
):
    rank = idist.get_rank()
    if "eval_seed" in config:
        manual_seed(config["eval_seed"] + rank)
    else:
        manual_seed(config["seed"] + rank)
    device = idist.device()

    model_name = config["name"]
    logger = setup_logger(
        name=model_name, format="%(levelname)s: %(message)s"
    )  ## default

    output_path = config["output"]["path"]
    if rank == 0:
        unique_id = config["output"].get(
            "unique_id", datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        folder_name = unique_id

        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)

        config["output"]["path"] = output_path.as_posix()
        logger.info(f"Output path: {config['output']['path']}")

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)
    tb_logger = TensorboardLogger(log_dir=output_path)

    log_basic_info(logger, config)

    # Setup dataflow, model, optimizer, criterion
    test_loader = get_dataflow(config)  ## default

    if hasattr(test_loader, "dataset"):
        logger.info(f"Dataset length: Test: {len(test_loader.dataset)}")

    config["dataset"]["steps_per_epoch"] = len(test_loader)

    # ===================================================== MODEL =====================================================
    model = initialize(config)

    cp_path = config.get("checkpoint", None)

    if cp_path is not None:
        if not cp_path.endswith(".pt"):
            cp_path = Path(cp_path)
            cp_path = next(cp_path.glob("training*.pt"))
        checkpoint = torch.load(cp_path, map_location=device)
        logger.info(f"Loading checkpoint from path: {cp_path}")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        logger.warning("Careful, no model is loaded")
    model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    evaluator = create_evaluator(model, config=config, logger=logger, vis_logger=tb_logger)

    # evaluator.add_event_handler(
    #     Events.ITERATION_COMPLETED(every=config["log_every"]),
    #     log_metrics_current(logger, metrics),
    # )

    try:
        state = evaluator.run(test_loader, max_epochs=1)
        log_metrics(logger, state.times["COMPLETED"], "Test", state.metrics)
        logger.info(f"Checkpoint: {str(cp_path)}")
    except Exception as e:
        logger.exception("")
        raise e


# def log_metrics_current(logger, metrics):
#     def f(engine):
#         out_str = "\n" + "\t".join(
#             [
#                 f"{v.compute():.3f}".ljust(8)
#                 for v in metrics.values()
#                 if v._num_examples != 0
#             ]
#         )
#         out_str += "\n" + "\t".join([f"{k}".ljust(8) for k in metrics.keys()])
#         logger.info(out_str)

#     return f


def log_metrics(logger, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEvaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


# def create_evaluator(model, metrics, config, tag="val"):
def create_evaluator(model, config, logger=None, vis_logger=None, tag="val"):
    with_amp = config["with_amp"]
    device = idist.device()

    metrics = {}
    for eval_config in config["evaluations"]:
        agg_type = eval_config.get("agg_type", None)
        if agg_type == "unsup_seg":
            metrics[eval_config["type"]] = SegmentationMetric(
                eval_config["type"], make_eval_fn(model, eval_config), assign_pseudo=True
            )
        elif agg_type == "sup_seg":
            metrics[eval_config["type"]] = SegmentationMetric(
                eval_config["type"], make_eval_fn(model, eval_config), assign_pseudo=False
            )
        elif agg_type == "concat":
            metrics[eval_config["type"]] = ConcatenateMetric(
                eval_config["type"], make_eval_fn(model, eval_config)
            )
        else:
            metrics[eval_config["type"]] = DictMeanMetric(
                eval_config["type"], make_eval_fn(model, eval_config)
            )

    @torch.no_grad()
    def evaluate_step(engine: Engine, data):
        # if not engine.state_dict["iteration"] % 10 == 0:      ## to prevent iterating whole testset for viz purpose
        model.eval()
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        data = to(data, device)

        with autocast(enabled=with_amp):
            data = model(data)  ## ! This is where the occupancy prediction is made.

        loss_metrics = {}

        return {
            "output": data,
            "loss_dict": loss_metrics,
            "timings_dict": timing,
            "metrics_dict": {},
        }

    evaluator = Engine(evaluate_step)
    evaluator.logger = logger  ##

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    eval_visualize = config.get("eval_visualize", [])
    if eval_visualize and vis_logger is not None:
        for name, vis_config in config["validation"].items():
            if "visualize" in vis_config:
                visualize = tb_visualize(
                    (model.renderer.net if hasattr(model, "renderer") else model.module.renderer.net),
                    None,
                    vis_config["visualize"],
                )
                def vis_wrapper(*args, **kwargs):
                    with autocast(enabled=with_amp):
                        return visualize(*args, **kwargs)

                def custom_vis_filter(engine, event):
                    return engine.state.iteration-1 in eval_visualize

                vis_logger.attach(
                    evaluator,
                    VisualizationHandler(
                        tag=tag,
                        visualizer=vis_wrapper,
                    ),
                    Events.ITERATION_COMPLETED(event_filter=custom_vis_filter),
                )

    if idist.get_rank() == 0 and (not config.get("with_clearml", False)):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator
