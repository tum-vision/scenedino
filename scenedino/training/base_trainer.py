import time
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

import ignite.distributed as idist
import numpy as np
import torch
from torch.utils.data import DataLoader
from ignite.contrib.engines import common
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine, Events, EventsList
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.utils import manual_seed, setup_logger
from torch.cuda.amp import autocast, GradScaler

from scenedino.common.logging import event_list_from_config, global_step_fn, log_basic_info
from scenedino.common.io.configs import save_hydra_config
from scenedino.common.io.model import load_checkpoint
from scenedino.evaluation.wrapper import make_eval_fn
from scenedino.losses.base_loss import BaseLoss
from scenedino.training.handlers import (
    MetricLoggingHandler,
    VisualizationHandler,
    add_time_handlers,
)

from scenedino.common.array_operations import to
from scenedino.common.metrics import DictMeanMetric, MeanMetric, SegmentationMetric, ConcatenateMetric
from scenedino.visualization.vis_2d import tb_visualize

import optuna


def base_training(local_rank, config, get_dataflow, initialize, sweep_trial=None):
    # ============================================ LOGGING AND OUTPUT SETUP ============================================
    # TODO: figure out rank
    rank = (
        idist.get_rank()
    )  ## rank of the current process within a group of processes: each process could handle a unique subset of the data, based on its rank
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
        # folder_name = f"{model_name}_backend-{idist.backend()}-{idist.get_world_size()}_{unique_id}"

        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)

        config["output"]["path"] = output_path.as_posix()
        logger.info(f"Output path: {config['output']['path']}")

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)
    log_basic_info(logger, config)
    tb_logger = TensorboardLogger(log_dir=output_path)

    # ================================================== DATASET SETUP =================================================
    # TODO: think about moving the dataset setup to the create validators and create trainer functions
    train_loader, val_loaders = get_dataflow(config)

    if isinstance(train_loader, tuple):
        train_loader = train_loader[0]

    if hasattr(train_loader, "dataset"):
        val_loader_lengths = "\n".join(
            [
                f"{name}: {len(val_loader.dataset)}"
                for name, val_loader in val_loaders.items()
                if hasattr(val_loader, "dataset")
            ]
        )
        logger.info(
            f"Dataset lengths:\nTrain: {len(train_loader.dataset)}\n{val_loader_lengths}"
        )
    config["dataset"]["steps_per_epoch"] = len(train_loader)

    # ============================================= MODEL AND OPTIMIZATION =============================================
    model, optimizer, criterion, lr_scheduler = initialize(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Create trainer for current task
    trainer = create_trainer(
        model,
        optimizer,
        criterion,
        lr_scheduler,
        train_loader.sampler if hasattr(train_loader, "sampler") else None,
        config,
        logger,
        metrics={},
    )
    if rank == 0:
        tb_logger.attach(
            trainer,
            MetricLoggingHandler("train", optimizer),
            Events.ITERATION_COMPLETED(every=config.get("log_every_iters", 1)),
        )

    # ========================================= EVALUTATION, AND VISUALIZATION =========================================
    validators: dict[str, tuple[Engine, EventsList]] = create_validators(
        config,
        model,
        val_loaders,
        criterion,
        tb_logger,
        trainer,
    )

    # NOTE: not super elegant as val_loaders has to have the same name but should work
    def run_validation(name: str, validator: Engine):
        def _run(engine: Engine):
            epoch = trainer.state.epoch
            state = validator.run(val_loaders[name])
            log_metrics(logger, epoch, state.times["COMPLETED"], name, state.metrics)

            if sweep_trial is not None and name == "validation":
                sweep_trial.report(trainer.state.best_metric, trainer.state.iteration)
                if sweep_trial.should_prune():
                    raise optuna.TrialPruned()

        return _run

    for name, validator in validators.items():
        trainer.add_event_handler(validator[1], run_validation(name, validator[0]))

    # ================================================ SAVE FINAL CONFIG ===============================================
    if rank == 0:
        # Plot config to tensorboard
        config_yaml = OmegaConf.to_yaml(config)
        config_yaml = "".join("\t" + line for line in config_yaml.splitlines(True))
        tb_logger.writer.add_text("config", text_string=config_yaml, global_step=0)
    save_hydra_config(output_path / "training_config.yaml", config, force=False)

    # ================================================= TRAINING LOOP ==================================================
    # In order to check training resuming we can stop training on a given iteration
    if config.get("stop_iteration", None):

        @trainer.on(Events.ITERATION_STARTED(once=config["stop_iteration"]))
        def _():
            logger.info(f"Stop training on {trainer.state.iteration} iteration")
            trainer.terminate()

    try:  ## train_loader == models.bts.trainer_overfit.DataloaderDummy object
        trainer.run(train_loader,
                    max_epochs=config["training"]["num_epochs"],
                    epoch_length=config["training"].get("epoch_length", None))
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()

    return trainer.state.best_metric


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


def create_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterions: list[Any],
    lr_scheduler,
    train_sampler,
    config,
    logger,
    metrics={},
):
    device = idist.device()
    model = model.to(device)

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    with_amp = config["with_amp"]
    gradient_accum_factor = config.get("gradient_accum_factor", 1)

    scaler = GradScaler(enabled=with_amp)

    def train_step(engine, data: dict):
        if "t__get_item__" in data:
            timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
        else:
            timing = {}

        _start_time = time.time()

        data = to(data, device)

        timing["t_to_gpu"] = time.time() - _start_time

        model.train()
        model.validation_tag = None

        _start_time = time.time()

        with autocast(enabled=with_amp):
            data = model(data)

        timing["t_forward"] = time.time() - _start_time

        loss_metrics = {}
        if optimizer is not None:
            _start_time = time.time()
            overall_loss = torch.tensor(0.0, device=device)
            for criterion in criterions:
                losses = criterion(data)
                names = criterion.get_loss_metric_names()

                overall_loss += losses[names[0]]
                loss_metrics.update({name: loss for name, loss in losses.items()})

            timing["t_loss"] = time.time() - _start_time

            ## make same scale for gradients. Note: it's not ignite built-in func. (c.f. https://wandb.ai/wandb_fc/tips/reports/How-To-Use-GradScaler-in-PyTorch--VmlldzoyMTY5MDA5)
            _start_time = time.time()

            # optimizer.zero_grad()
            # scaler.scale(overall_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # Gradient accumulation
            overall_loss = overall_loss / gradient_accum_factor
            scaler.scale(overall_loss).backward()
            if engine.state.iteration % gradient_accum_factor == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            timing["t_backward"] = time.time() - _start_time

        return {
            "output": data,
            "loss_dict": loss_metrics,
            "timings_dict": timing,
            "metrics_dict": {},
        }

    trainer = Engine(train_step)
    trainer.logger = logger

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    # TODO: maybe save only the network not the whole wrapper
    # TODO: Make adaptable
    to_save = {
        "trainer": trainer,
        "model": model,
        # "optimizer": optimizer,
        # "lr_scheduler": lr_scheduler,
    }

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["training"]["checkpoint_every"],
        save_handler=DiskSaver(config["output"]["path"], require_empty=False),
        lr_scheduler=lr_scheduler,
        output_names=None,
        with_pbars=False,
        clear_cuda_cache=False,
        log_every_iters=config.get("log_every_iters", 100),
        n_saved=1,
    )

    # NOTE: don't move to initialization, as to save is also needed here
    if config["training"].get("resume_from", None):
        ckpt_path = Path(config["training"]["resume_from"])
        logger.info(f"Resuming from checkpoint: {str(ckpt_path)}")

        load_checkpoint(ckpt_path, to_save, strict=False)

    if config["training"].get("from_pretrained", None):
        ckpt_path = Path(config["training"]["from_pretrained"])
        logger.info(f"Pretrained from checkpoint: {str(ckpt_path)}")

        to_save = {"model": to_save["model"]}

        load_checkpoint(ckpt_path, to_save, strict=False)

    if idist.get_rank() == 0:
        common.ProgressBar(desc=f"Training", persist=False).attach(trainer)

    return trainer


def create_validators(
    config,
    model: torch.nn.Module,
    dataloaders: dict[str, DataLoader],
    criterions: list[BaseLoss],
    tb_logger: TensorboardLogger,
    trainer: Engine,
) -> dict[str, tuple[Engine, EventsList]]:
    # TODO: change model object to evaluator object that has a different ray sampler
    with_amp = config["with_amp"]
    device = idist.device()

    def _create_validator(
        tag: str,
        validation_config,
    ) -> tuple[Engine, EventsList]:
        # TODO: make eval functions configurable from config
        metrics = {}
        for metric_config in validation_config["metrics"]:
            agg_type = metric_config.get("agg_type", None)
            if agg_type == "unsup_seg":
                metrics[metric_config["type"]] = SegmentationMetric(
                    metric_config["type"], make_eval_fn(model, metric_config), assign_pseudo=True
                )
            elif agg_type == "sup_seg":
                metrics[metric_config["type"]] = SegmentationMetric(
                    metric_config["type"], make_eval_fn(model, metric_config), assign_pseudo=False
                )
            elif agg_type == "concat":
                metrics[metric_config["type"]] = ConcatenateMetric(
                    metric_config["type"], make_eval_fn(model, metric_config)
                )
            else:
                metrics[metric_config["type"]] = DictMeanMetric(
                    metric_config["type"], make_eval_fn(model, metric_config)
                )

        loss_during_validation = validation_config.get("log_loss", True)
        if loss_during_validation:
            metrics_loss = {}
            for criterion in criterions:
                metrics_loss.update(
                    {
                        k: MeanMetric((lambda y: lambda x: x["loss_dict"][y])(k))
                        for k in criterion.get_loss_metric_names()
                    }
                )
            eval_metrics = {**metrics, **metrics_loss}
        else:
            eval_metrics = metrics

        @torch.no_grad()
        def validation_step(engine: Engine, data):
            model.eval()
            model.validation_tag = tag
            if "t__get_item__" in data:
                timing = {"t__get_item__": torch.mean(data["t__get_item__"]).item()}
            else:
                timing = {}

            data = to(data, device)

            with autocast(enabled=with_amp):
                data = model(data)

            overall_loss = torch.tensor(0.0, device=device)
            loss_metrics = {}
            if loss_during_validation:
                for criterion in criterions:
                    losses = criterion(data)
                    names = criterion.get_loss_metric_names()

                    overall_loss += losses[names[0]]
                    loss_metrics.update({name: loss for name, loss in losses.items()})
            else:
                loss_metrics = {}

            return {
                "output": data,
                "loss_dict": loss_metrics,
                "timings_dict": timing,
                "metrics_dict": {},
            }

        validator = Engine(validation_step)

        add_time_handlers(validator)

        # ADD METRICS
        for name, metric in eval_metrics.items():
            metric.attach(validator, name)

        # ADD LOGGING HANDLER
        # TODO: split up handlers
        tb_logger.attach(
            validator,
            MetricLoggingHandler(
                tag,
                log_loss=False,
                global_step_transform=global_step_fn(
                    trainer, validation_config["global_step"]
                ),
            ),
            Events.EPOCH_COMPLETED,
        )

        # ADD VISUALIZATION HANDLER
        if validation_config.get("visualize", None):
            visualize = tb_visualize(
                (model.renderer.net if hasattr(model, "renderer") else model.module.renderer.net),
                dataloaders[tag].dataset.dataset,
                validation_config["visualize"],
            )

            def vis_wrapper(*args, **kwargs):
                with autocast(enabled=with_amp):
                    return visualize(*args, **kwargs)

            tb_logger.attach(
                validator,
                VisualizationHandler(
                    tag=tag,
                    visualizer=vis_wrapper,
                    global_step_transform=global_step_fn(
                        trainer, validation_config["global_step"]
                    ),
                ),
                Events.ITERATION_COMPLETED(every=1),
            )

        if "save_best" in validation_config:
            save_best_config = validation_config["save_best"]
            metric_name = save_best_config["metric"]
            sign = save_best_config.get("sign", 1.0)
            update_model = save_best_config.get("update_model", False)
            dry_run = save_best_config.get("dry_run", False)

            best_model_handler = Checkpoint(
                {"model": model},
                # NOTE: fixes a problem with log_dir or logdir
                DiskSaver(Path(config["output"]["path"]), require_empty=False),
                # DiskSaver(tb_logger.writer.log_dir, require_empty=False),
                filename_prefix=f"{metric_name}_best",
                n_saved=1,
                global_step_transform=global_step_from_engine(trainer),
                score_name=metric_name,
                score_function=Checkpoint.get_default_score_fn(
                    metric_name, score_sign=sign
                ),
            )

            def event_handler(engine):
                if update_model:
                    model.update_model_eval(engine.state.metrics)
                if not dry_run:
                    best_model_handler(engine)
                    trainer.state.best_metric = best_model_handler._saved[0].priority
                
            validator.add_event_handler(Events.COMPLETED, event_handler)

        if idist.get_rank() == 0 and (not validation_config.get("with_clearml", False)):
            common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(
                validator
            )

        return validator, event_list_from_config(validation_config["events"])

    return {
        name: _create_validator(name, config)
        for name, config in config["validation"].items()
    }
