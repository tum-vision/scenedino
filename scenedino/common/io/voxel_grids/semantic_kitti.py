# code for saving voxel grid
import numpy as np

# TODO: adapt to semantic voxel grid


def unpack(compressed):
    """given a bit encoded voxel grid, make a normal voxel grid out of it."""
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed


def pack(uncompressed: np.ndarray) -> np.ndarray:
    """convert a boolean array into a bitwise array."""
    uncompressed_r = uncompressed.reshape(-1, 8)
    compressed = uncompressed_r.dot(
        1 << np.arange(uncompressed_r.shape[-1] - 1, -1, -1)
    )
    return compressed


def save_semantic_kitti(voxel_grid, path, format):
    """Save a voxel grid to a bin file."""
    pack(np.flip(voxel_grid, (0, 1, 2)).reshape(-1)).astype(np.uint8).tofile(path)
