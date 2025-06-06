import time
from typing import Mapping, Union

from ignite.contrib.handlers import TensorboardLogger
from ignite.handlers import global_step_from_engine
from ignite.contrib.handlers.base_logger import BaseHandler
from ignite.engine import Engine, EventEnum, Events

import torch


def add_time_handlers(engine: Engine):
    iteration_time_handler = TimeHandler("iter", freq=True, period=True)
    batch_time_handler = TimeHandler("get_batch", freq=False, period=True)
    engine.add_event_handler(
        Events.ITERATION_STARTED, iteration_time_handler.start_timing
    )
    engine.add_event_handler(
        Events.ITERATION_COMPLETED, iteration_time_handler.end_timing
    )
    engine.add_event_handler(Events.GET_BATCH_STARTED, batch_time_handler.start_timing)
    engine.add_event_handler(Events.GET_BATCH_COMPLETED, batch_time_handler.end_timing)


class MetricLoggingHandler(BaseHandler):
    def __init__(
        self,
        tag,
        optimizer=None,
        log_loss=True,
        log_metrics=True,
        log_timings=True,
        global_step_transform=None,
    ):
        self.tag = tag
        self.optimizer = optimizer
        self.log_loss = log_loss
        self.log_metrics = log_metrics
        self.log_timings = log_timings
        self.gst = global_step_transform
        super(MetricLoggingHandler, self).__init__()

    def __call__(
        self,
        engine: Engine,
        logger: TensorboardLogger,
        event_name: Union[str, EventEnum],
    ):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError(
                "Handler 'MetricLoggingHandler' works only with TensorboardLogger"
            )

        if self.gst is None:
            gst = global_step_from_engine(engine)
        else:
            gst = self.gst
        global_step = gst(engine, event_name)  # type: ignore[misc]

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        writer = logger.writer

        # Optimizer parameters
        if self.optimizer is not None:
            params = {
                k: float(param_group["lr"])
                for k, param_group in enumerate(self.optimizer.param_groups)
            }

            for k, param in params.items():
                writer.add_scalar(f"lr-{self.tag}/{k}", param, global_step)

        if self.log_loss:
            # Plot losses
            loss_dict = engine.state.output["loss_dict"]
            for k, v in loss_dict.items():
                # TODO: is this needed?
                # if not isinstance(v, (float, int)):
                #     print(f"{k}: {type(v)}")
                writer.add_scalar(f"loss-{self.tag}/{k}", v, global_step)

        if self.log_metrics:
            # Plot metrics
            metrics_dict = engine.state.metrics
            metrics_dict_custom = engine.state.output["metrics_dict"]

            for k, v in metrics_dict.items():
                # Avoid dictionaries because of weird ignite handling of Mapping metrics
                if isinstance(v, Mapping) or k.endswith("assignment"):  # TODO: Remove hard-coded assignment
                    continue
                if isinstance(v, torch.Tensor) and v.ndim > 0:
                    writer.add_histogram(f"metrics-{self.tag}/{k}", v, global_step)
                else:
                    writer.add_scalar(f"metrics-{self.tag}/{k}", v, global_step)

            for k, v in metrics_dict_custom.items():
                if isinstance(v, Mapping):
                    continue
                if isinstance(v, torch.Tensor) and v.ndim > 0:
                    writer.add_histogram(f"metrics-{self.tag}/{k}", v, global_step)
                else:
                    writer.add_scalar(f"metrics-{self.tag}/{k}", v, global_step)

        if self.log_timings:
            # Plot timings
            timings_dict = engine.state.times
            timings_dict_custom = engine.state.output["timings_dict"]
            for k, v in timings_dict.items():
                if k == "COMPLETED":
                    continue
                writer.add_scalar(f"timing-{self.tag}/{k}", v, global_step)
            for k, v in timings_dict_custom.items():
                writer.add_scalar(f"timing-{self.tag}/{k}", v, global_step)

        engine.state.output = None  # For memory efficiency, val results do not need to stay in the state


class TimeHandler:
    def __init__(self, name: str, freq: bool = False, period: bool = False) -> None:
        self.name = name
        self.freq = freq
        self.period = period
        if not self.period and not self.freq:
            print(f"Warning: No timings logged for {name}")
        self._start_time = None

    def start_timing(self, engine):
        self._start_time = time.time()

    def end_timing(self, engine):
        if self._start_time is None:
            period = 0
            freq = 0
        else:
            period = max(time.time() - self._start_time, 1e-6)
            freq = 1 / period
        if not hasattr(engine.state, "times"):
            engine.state.times = {}
        else:
            if self.period:
                engine.state.times[f"secs_per_{self.name}"] = period
            if self.freq:
                engine.state.times[f"num_{self.name}_per_sec"] = freq


class VisualizationHandler(BaseHandler):
    def __init__(self, tag, visualizer, global_step_transform=None):
        self.tag = tag
        self.visualizer = visualizer
        self.gst = global_step_transform
        super(VisualizationHandler, self).__init__()

    def __call__(
        self,
        engine: Engine,
        logger: TensorboardLogger,
        event_name: Union[str, EventEnum],
    ) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError(
                "Handler 'VisualizationHandler' works only with TensorboardLogger"
            )

        if self.gst is None:
            gst = global_step_from_engine(engine)
        else:
            gst = self.gst
        global_step = gst(engine, event_name)  # type: ignore[misc]

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        self.visualizer(engine, logger, global_step, self.tag)
