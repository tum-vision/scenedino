from enum import Enum
from .semantic_kitti import save_semantic_kitti


# TODO: add more formats
class VoxelFormats(Enum):
    """Enum for voxel grid formats."""

    semantic_kitti = "semantic_kitti"


def save_voxel_grid(voxel_grid, path, format: VoxelFormats | str):
    """Save a voxel grid to a bin file."""
    if isinstance(format, str):
        format = VoxelFormats(format)

    match format:
        case VoxelFormats.semantic_kitti:
            save_semantic_kitti(voxel_grid, path, format)
        case _:
            raise NotImplementedError(f"Format {format} not implemented.")
