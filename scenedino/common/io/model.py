from pathlib import Path
from typing import Any

import torch
from ignite.handlers import Checkpoint


def load_checkpoint(ckpt_path: Path, to_save: dict[str, Any], strict: bool = False):
    assert ckpt_path.exists(), f"__Checkpoint '{str(ckpt_path)}' is not found"
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    to_save = {"model": to_save["model"]}
    checkpoint = {"model": checkpoint["model"]}

    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint, strict=strict)
