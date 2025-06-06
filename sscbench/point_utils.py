
import numpy as np
from fusion import TSDFVolume, rigid_transform


def get_fov_mask():
    calib = read_calib()
    T_velo_2_cam = calib["Tr"]
    _, fov_mask = generate_point_grid(vox_origin=np.array([0, -25.6, -2]),
                              scene_size=(51.2, 51.2, 6.4),
                              voxel_size=0.2,
                              cam_E=T_velo_2_cam,
                              cam_k=get_cam_k())

    return fov_mask.reshape(256, 256, 32)

def generate_point_grid(cam_E, vox_origin, voxel_size, scene_size, cam_k, img_W=1408, img_H=376):
    """
    compute the 2D projection of voxels centroids

    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2

    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = vox_origin
    vol_bnds[:, 1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
        range(vol_dim[0]),
        range(vol_dim[1]),
        range(vol_dim[2]),
        indexing='ij'
    )
    print(xv.shape, yv.shape, zv.shape)
    vox_coords = np.concatenate([
        xv.reshape(1, -1),
        yv.reshape(1, -1),
        zv.reshape(1, -1)
    ], axis=0).astype(int).T

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0,
                              np.logical_and(pix_x < img_W,
                                             np.logical_and(pix_y >= 0,
                                                            np.logical_and(pix_y < img_H,
                                                                           pix_z > 0))))

    return cam_pts, fov_mask

def read_calib():
    """
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    P = np.array(
        [
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]
    ).reshape(3, 4)

    cam2velo = np.array(
        [
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
        ]
    ).reshape(3, 4)
    C2V = np.concatenate(
        [cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0
    )
    # print("C2V: ", C2V)
    V2C = np.linalg.inv(C2V)
    # print("V2C: ", V2C)
    V2C = V2C[:3, :]
    # print("V2C: ", V2C)

    # reshape matrices
    calib_out = {}
    # 3x4 projection matrix for left camera
    calib_out["P2"] = P
    calib_out["Tr"] = np.identity(4)  # 4x4 matrix
    calib_out["Tr"][:3, :4] = V2C
    return calib_out


def get_cam_k():
    cam_k = np.array(
        [
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]
    ).reshape(3, 4)
    return cam_k[:3, :3]