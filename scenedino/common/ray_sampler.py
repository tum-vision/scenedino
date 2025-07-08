from math import isqrt
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from omegaconf import ListConfig

from scenedino.common import util
from scenedino.common.cameras.pinhole import outside_frustum, project_to_image, pts_into_camera


class RaySampler:
    def __init__(self, z_near: float, z_far: float) -> None:
        self.z_near = z_near
        self.z_far = z_far

    def sample(self, images, poses, projs):
        raise NotImplementedError

    def reconstruct(self, render_dict):
        raise NotImplementedError


class RandomRaySampler(RaySampler):
    def __init__(
        self, z_near: float, z_far: float, ray_batch_size: int, channels: int = 3
    ) -> None:
        super().__init__(z_near, z_far)
        self.ray_batch_size = ray_batch_size
        self.channels = channels

    def sample(self, images, poses, projs, image_ids=None):
        n, v, c, h, w = images.shape

        all_rgb_gt = []
        all_rays = []

        for n_ in range(n):
            focals = projs[n_, :, [0, 1], [0, 1]]
            centers = projs[n_, :, [0, 1], [2, 2]]

            rays, xy = util.gen_rays(
                poses[n_].view(-1, 4, 4),
                w,
                h,
                focal=focals,
                c=centers,
                z_near=self.z_near,
                z_far=self.z_far,
            )

            # Append frame id to the ray
            if image_ids is None:
                ids = torch.arange(v, device=images.device, dtype=images.dtype)
            else:
                ids = torch.tensor(image_ids, device=images.device, dtype=images.dtype)
            ids = ids.view(v, 1, 1, 1).expand(v, h, w, 1)
            rays = torch.cat((rays, ids), dim=-1)
            rays = torch.cat((rays, xy), dim=-1)
            r_dim = rays.shape[-1]
            rays = rays.view(-1, r_dim)

            rgb_gt = images[n_].view(-1, c, h, w)
            rgb_gt = rgb_gt.permute(0, 2, 3, 1).contiguous().reshape(-1, c)

            pix_inds = torch.randint(0, v * h * w, (self.ray_batch_size,))

            rgb_gt = rgb_gt[pix_inds]
            rays = rays[pix_inds]

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)
        all_rays = torch.stack(all_rays)

        return all_rays, all_rgb_gt

    def reconstruct(self, render_dict, channels=None):
        for name, render_dict_part in render_dict.items():
            if not type(render_dict_part) == dict or not "rgb" in render_dict_part:
                continue

            if channels is None:
                channels = self.channels

            rgb = render_dict_part["rgb"]  # n, n_pts, v * 3
            depth = render_dict_part["depth"]
            invalid = render_dict_part["invalid"]

            rgb_gt = render_dict["rgb_gt"]

            n, n_pts, v_c = rgb.shape
            v = v_c // channels
            n_smps = invalid.shape[-2]

            render_dict_part["rgb"] = rgb.view(n, n_pts, v, channels)
            render_dict_part["depth"] = depth.view(n, n_pts)
            render_dict_part["invalid"] = invalid.view(n, n_pts, n_smps, v)

            if "invalid_features" in render_dict_part:
                invalid_features = render_dict_part["invalid_features"]
                render_dict_part["invalid_features"] = invalid_features.view(n, n_pts, n_smps, v)

            if "weights" in render_dict_part:
                weights = render_dict_part["weights"]
                render_dict_part["weights"] = weights.view(n, n_pts, n_smps)

            if "alphas" in render_dict_part:
                alphas = render_dict_part["alphas"]
                render_dict_part["alphas"] = alphas.view(n, n_pts, n_smps)

            if "z_samps" in render_dict_part:
                z_samps = render_dict_part["z_samps"]
                render_dict_part["z_samps"] = z_samps.view(n, n_pts, n_smps)

            if "rgb_samps" in render_dict_part:
                rgb_samps = render_dict_part["rgb_samps"]
                render_dict_part["rgb_samps"] = rgb_samps.view(n, n_pts, n_smps, v, channels)
            
            if "ray_info" in render_dict_part:
                ri_shape = render_dict_part["ray_info"].shape[-1]
                ray_info = render_dict_part["ray_info"]
                render_dict_part["ray_info"] = ray_info.view(n, n_pts, ri_shape)

            if "extras" in render_dict_part:
                extras_shape = render_dict_part["extras"].shape[-1]
                extras = render_dict_part["extras"]
                render_dict_part["extras"] = extras.view(n, n_pts, extras_shape)

            render_dict[name] = render_dict_part
        render_dict["rgb_gt"] = rgb_gt.view(n, n_pts, channels)

        return render_dict


class PatchRaySampler(RaySampler):
    def __init__(
        self,
        z_near: float,
        z_far: float,
        ray_batch_size: int,
        patch_size: int,
        channels: int = 3,
        snap_to_grid: bool = False,
        dino_upscaled: bool = False,
    ) -> None:
        super().__init__(z_near, z_far)
        self.ray_batch_size = ray_batch_size
        self.channels = channels
        self.snap_to_grid = snap_to_grid
        self.dino_upscaled = dino_upscaled

        if isinstance(patch_size, int):
            self.patch_size_x, self.patch_size_y = patch_size, patch_size
        elif (
            isinstance(patch_size, tuple)
            or isinstance(patch_size, list)
            or isinstance(patch_size, ListConfig)
        ):
            self.patch_size_y = patch_size[0]
            self.patch_size_x = patch_size[1]
        else:
            raise ValueError(f"Invalid format for patch size")
        assert (ray_batch_size % (self.patch_size_x * self.patch_size_y)) == 0
        self._patch_count = self.ray_batch_size // (
            self.patch_size_x * self.patch_size_y
        )

    def sample(
        self, images, poses, projs, image_ids=None, dino_features=None, loss_feature_grid_shift=None,
    ):  ### dim(images) == nv (ids_loss nv randomly sampled)
        n, v, c, h, w = images.shape

        self.channels = c

        if dino_features is not None:
            _, _, dino_channels, _, _ = dino_features.shape
            dino_features = dino_features.permute(0, 1, 3, 4, 2)

        device = images.device

        images = images.permute(0, 1, 3, 4, 2)

        all_rgb_gt, all_rays, all_dino_gt = [], [], []

        for n_ in range(n):
            focals = projs[n_, :, [0, 1], [0, 1]]
            centers = projs[n_, :, [0, 1], [2, 2]]

            rays, xy = util.gen_rays(
                poses[n_].view(-1, 4, 4),
                w,
                h,
                focal=focals,
                c=centers,
                z_near=self.z_near,
                z_far=self.z_far,
            )

            # Append frame id to the ray
            if image_ids is None:
                ids = torch.arange(v, device=images.device, dtype=images.dtype)
            else:
                ids = torch.tensor(image_ids, device=images.device, dtype=images.dtype)
            ids = ids.view(v, 1, 1, 1).expand(v, h, w, 1)
            rays = torch.cat((rays, ids), dim=-1)
            rays = torch.cat((rays, xy), dim=-1)
            r_dim = rays.shape[-1]

            patch_coords_v = torch.randint(0, v, (self._patch_count,))

            if self.snap_to_grid:
                if loss_feature_grid_shift is not None:
                    patch_coords_y = torch.randint(0, h // self.patch_size_y - 1, (self._patch_count,))
                    patch_coords_x = torch.randint(0, w // self.patch_size_x - 1, (self._patch_count,))
                else:
                    patch_coords_y = torch.randint(0, h // self.patch_size_y, (self._patch_count,))
                    patch_coords_x = torch.randint(0, w // self.patch_size_x, (self._patch_count,))
            else:
                patch_coords_y = torch.randint(0, h - self.patch_size_y, (self._patch_count,))
                patch_coords_x = torch.randint(0, w - self.patch_size_x, (self._patch_count,))

            sample_rgb_gt = []
            sample_rays = []
            sample_dino_gt = []

            for v_, coord_y, coord_x in zip(patch_coords_v, patch_coords_y, patch_coords_x):
                if self.snap_to_grid:
                    patch_y, patch_x = coord_y, coord_x
                    if loss_feature_grid_shift is not None:
                        y = (loss_feature_grid_shift[0] % self.patch_size_y) + self.patch_size_y * coord_y
                        x = (loss_feature_grid_shift[1] % self.patch_size_x) + self.patch_size_x * coord_x
                        if loss_feature_grid_shift[0] < 0:
                            patch_y += 1
                        if loss_feature_grid_shift[1] < 0:
                            patch_x += 1
                    else:
                        y = self.patch_size_y * coord_y
                        x = self.patch_size_x * coord_x
                else:
                    raise NotImplementedError

                rgb_gt_patch = images[n_][
                    v_, y : y + self.patch_size_y, x : x + self.patch_size_x, :
                ].reshape(-1, self.channels)
                rays_patch = rays[
                    v_, y : y + self.patch_size_y, x : x + self.patch_size_x, :
                ].reshape(-1, r_dim)

                sample_rgb_gt.append(rgb_gt_patch)
                sample_rays.append(rays_patch)

                if dino_features is not None:
                    if self.dino_upscaled:
                        dino_gt_patch = dino_features[n_][
                            v_, y: y + self.patch_size_y, x: x + self.patch_size_x, :
                        ].reshape(-1, dino_channels)
                    else:
                        dino_gt_patch = dino_features[n_][
                            v_, patch_y, patch_x, :
                        ].reshape(-1, dino_channels)
                    sample_dino_gt.append(dino_gt_patch)

            sample_rgb_gt = torch.cat(sample_rgb_gt, dim=0)
            sample_rays = torch.cat(sample_rays, dim=0)
            all_rgb_gt.append(sample_rgb_gt)
            all_rays.append(sample_rays)

            if dino_features is not None:
                sample_dino_gt = torch.cat(sample_dino_gt, dim=0)
                all_dino_gt.append(sample_dino_gt)

        all_rgb_gt = torch.stack(all_rgb_gt)
        all_rays = torch.stack(all_rays)

        if dino_features is not None:
            all_dino_gt = torch.stack(all_dino_gt)
            return all_rays, all_rgb_gt, all_dino_gt
        else:
            return all_rays, all_rgb_gt

    def reconstruct(self, render_dict, channels=None, dino_channels=None):

        for name, render_dict_part in render_dict.items():
            if not type(render_dict_part) == dict or not "rgb" in render_dict_part:
                continue

            if channels is None:
                channels = self.channels

            rgb_gt = render_dict["rgb_gt"]
            dino_gt = render_dict["dino_gt"]

            n, n_pts, v_c = render_dict_part["rgb"].shape
            v = v_c // channels
            n_smps = render_dict_part["weights"].shape[-1]
            # (This can be a different v from the sample method)

            render_dict_part["rgb"] = render_dict_part["rgb"].view(
                n, self._patch_count, self.patch_size_y, self.patch_size_x, v, channels
            )
            render_dict_part["weights"] = render_dict_part["weights"].view(
                n, self._patch_count, self.patch_size_y, self.patch_size_x, n_smps
            )
            render_dict_part["depth"] = render_dict_part["depth"].view(
                n, self._patch_count, self.patch_size_y, self.patch_size_x
            )
            render_dict_part["invalid"] = render_dict_part["invalid"].view(
                n, self._patch_count, self.patch_size_y, self.patch_size_x, n_smps, v
            )

            # TODO: Figure out DINO invalid policy
            # if "invalid_features" in render_dict_part:
            #     render_dict_part["invalid_features"] = render_dict_part["invalid_features"].view(
            #         n, self._patch_count, self.patch_size_y, self.patch_size_x, n_smps, v
            #     )

            if "alphas" in render_dict_part:
                render_dict_part["alphas"] = render_dict_part["alphas"].view(
                    n, self._patch_count, self.patch_size_y, self.patch_size_x, n_smps
                )

            if "z_samps" in render_dict_part:
                render_dict_part["z_samps"] = render_dict_part["z_samps"].view(
                    n, self._patch_count, self.patch_size_y, self.patch_size_x, n_smps
                )

            if "rgb_samps" in render_dict_part:
                render_dict_part["rgb_samps"] = render_dict_part["rgb_samps"].view(
                    n,
                    self._patch_count,
                    self.patch_size_y,
                    self.patch_size_x,
                    n_smps,
                    v,
                    channels,
                )

            if "ray_info" in render_dict_part:
                ri_shape = render_dict_part["ray_info"].shape[-1]
                render_dict_part["ray_info"] = render_dict_part["ray_info"].view(
                    n, self._patch_count, self.patch_size_y, self.patch_size_x, ri_shape
                )
                
            if "extras" in render_dict_part:
                extras_shape = render_dict_part["extras"].shape[-1]
                render_dict_part["extras"] = render_dict_part["extras"].view(
                    n, self._patch_count, self.patch_size_y, self.patch_size_x, extras_shape
                )

            if "dino_features" in render_dict_part:
                dino_shape = render_dict_part["dino_features"].shape[-1]
                render_dict_part["dino_features"] = render_dict_part["dino_features"].view(
                    n, self._patch_count, self.patch_size_y, self.patch_size_x, 1, dino_shape
                )

            render_dict[name] = render_dict_part

        render_dict["rgb_gt"] = rgb_gt.view(
            n, self._patch_count, self.patch_size_y, self.patch_size_x, channels
        )
        dino_gt_shape = dino_gt.shape[-1]
        if self.dino_upscaled:
            render_dict["dino_gt"] = dino_gt.view(
                n, self._patch_count, self.patch_size_y, self.patch_size_x, dino_gt_shape
            )
        else:
            render_dict["dino_gt"] = dino_gt.view(
                n, self._patch_count, dino_gt_shape
            )

        if "dino_artifacts" in render_dict:
            render_dict["dino_artifacts"] = render_dict["dino_artifacts"].view(
                n, self._patch_count, dino_gt_shape
            )

        return render_dict


class PointBasedRaySampler(RandomRaySampler):
    def sample(
        self, images, poses, projs, xyz, image_ids=None
    ):  ### dim(images) == nv (ids_loss nv randomly sampled)
        n, v, c, h, w = images.shape

        assert v == 1

        with autocast(enabled=False):
            poses_w2c = torch.inverse(poses)
            inv_K = torch.inverse(projs[:, :, :3, :3])

        B, n_pts, _ = xyz.shape

        xyz_projected = pts_into_camera(xyz, poses_w2c)
        distance = torch.norm(xyz_projected, dim=-2, keepdim=True)
        xy, z = project_to_image(xyz_projected, projs)

        xy = xy[:, 0]
        z = z[:, 0]
        distance = distance[:, 0, 0, :, None]

        # For numerical stability with AMP. Should not affect training outcome
        xy = xy.clamp(-2, 2)

        # Build rays
        cam_centers = poses[:, 0, None, :3, 3].expand(-1, n_pts, -1)
        ray_dir = ((poses[:, 0, :3, :3] @ inv_K[:, 0]) @ torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

        cam_nears = torch.ones_like(cam_centers[..., :1]) * self.z_near
        cam_fars = torch.ones_like(cam_centers[..., :1]) * self.z_far

        ids = torch.zeros_like(cam_nears)

        rays = torch.cat((cam_centers, ray_dir, cam_nears, cam_fars, ids, xy, distance), dim=-1)

        rgb_gt = F.grid_sample(images[:, 0], xy.reshape(n, -1, 1, 2).to(images.dtype), padding_mode="border", align_corners=False)[..., 0].permute(0, 2, 1)

        return rays, rgb_gt


class ImageRaySampler(RaySampler):
    def __init__(
        self,
        z_near: float,
        z_far: float,
        height: int | None = None,
        width: int | None = None,
        channels: int = 3,
        norm_dir: bool = True,
        dino_upscaled: bool = False,
    ) -> None:
        super().__init__(z_near, z_far)
        self.height = height
        self.width = width
        self.channels = channels
        self.norm_dir = norm_dir
        self.dino_upscaled = dino_upscaled

    def sample(self, images, poses, projs, image_ids=None, dino_features=None, dino_artifacts=None):
        n, v, _, _ = poses.shape
        device = poses.device
        dtype = poses.dtype

        if images is not None:
            self.channels = images.shape[2]

        if self.height is None:
            self.height, self.width = images.shape[-2:]

        if dino_features is not None:
            _, _, dino_channels, _, _ = dino_features.shape

        h = self.height
        w = self.width

        all_rgb_gt = []
        all_dino_gt = []
        all_rays = []

        for n_ in range(n):
            focals = projs[n_, :, [0, 1], [0, 1]]
            centers = projs[n_, :, [0, 1], [2, 2]]

            rays, xy = util.gen_rays(
                poses[n_].view(-1, 4, 4),
                self.width,
                self.height,
                focal=focals,
                c=centers,
                z_near=self.z_near,
                z_far=self.z_far,
                norm_dir=self.norm_dir,
            )

            # Append frame id to the ray
            if image_ids is None:
                ids = torch.arange(v, device=device, dtype=dtype)
            else:
                ids = torch.tensor(image_ids, device=device, dtype=dtype)
            ids = ids.view(v, 1, 1, 1).expand(v, h, w, 1)
            rays = torch.cat((rays, ids), dim=-1)
            rays = torch.cat((rays, xy), dim=-1)
            r_dim = rays.shape[-1]
            rays = rays.view(-1, r_dim)
            
            all_rays.append(rays)

            if images is not None:
                rgb_gt = images[n_].view(-1, self.channels, self.height, self.width)
                rgb_gt = (
                    rgb_gt.permute(0, 2, 3, 1).contiguous().reshape(-1, self.channels)
                )
                all_rgb_gt.append(rgb_gt)

            if dino_features is not None:
                patch_h, patch_w = dino_features[n_].shape[-2], dino_features[n_].shape[-1]
                dino_gt = dino_features[n_].view(-1, dino_channels, patch_h, patch_w)
                dino_gt = (
                    dino_gt.permute(0, 2, 3, 1).contiguous().reshape(-1, dino_channels)
                )
                all_dino_gt.append(dino_gt)

        all_rays = torch.stack(all_rays)
        if images is not None:
            all_rgb_gt = torch.stack(all_rgb_gt)
        else:
            all_rgb_gt = None

        if dino_features is not None:
            all_dino_gt = torch.stack(all_dino_gt)
            return all_rays, all_rgb_gt, all_dino_gt
        else:
            return all_rays, all_rgb_gt

    def reconstruct(self, render_dict, channels=None, dino_channels=None):
        for name, render_dict_part in render_dict.items():
            if not type(render_dict_part) == dict or not "rgb" in render_dict_part:
                continue

            if channels is None:
                channels = self.channels

            rgb = render_dict_part["rgb"]  # n, n_pts, v * 3
            weights = render_dict_part["weights"]
            depth = render_dict_part["depth"]
            invalid = render_dict_part["invalid"]

            n, n_pts, v_c = rgb.shape
            v_in = n_pts // (self.height * self.width)
            v_render = v_c // channels
            n_smps = weights.shape[-1]
            # (This can be a different v from the sample method)

            render_dict_part["rgb"] = rgb.view(n, v_in, self.height, self.width, v_render, channels)
            render_dict_part["weights"] = weights.view(n, v_in, self.height, self.width, n_smps)
            render_dict_part["depth"] = depth.view(n, v_in, self.height, self.width)
            render_dict_part["invalid"] = invalid.view(
                n, v_in, self.height, self.width, n_smps, v_render
            )

            if "invalid_features" in render_dict_part:
                invalid_features = render_dict_part["invalid_features"]
                render_dict_part["invalid_features"] = invalid_features.view(
                    n, v_in, self.height, self.width, n_smps, v_render
                )

            if "alphas" in render_dict_part:
                alphas = render_dict_part["alphas"]
                render_dict_part["alphas"] = alphas.view(n, v_in, self.height, self.width, n_smps)

            if "z_samps" in render_dict_part:
                z_samps = render_dict_part["z_samps"]
                render_dict_part["z_samps"] = z_samps.view(
                    n, v_in, self.height, self.width, n_smps
                )

            if "rgb_samps" in render_dict_part:
                rgb_samps = render_dict_part["rgb_samps"]
                render_dict_part["rgb_samps"] = rgb_samps.view(
                    n, v_in, self.height, self.width, n_smps, v_render, channels
                )
            
            if "ray_info" in render_dict_part:
                ri_shape = render_dict_part["ray_info"].shape[-1]
                ray_info = render_dict_part["ray_info"]
                render_dict_part["ray_info"] = ray_info.view(n, v_in, self.height, self.width, ri_shape)
                
            if "extras" in render_dict_part:
                ex_shape = render_dict_part["extras"].shape[-1]
                extras = render_dict_part["extras"]
                render_dict_part["extras"] = extras.view(n, v_in, self.height, self.width, ex_shape)

            if "dino_features" in render_dict_part:
                dino_shape = render_dict_part["dino_features"].shape[-1]
                dino = render_dict_part["dino_features"]
                render_dict_part["dino_features"] = dino.view(n, v_in, self.height, self.width, 1, dino_shape)

            render_dict[name] = render_dict_part

        if "rgb_gt" in render_dict:
            rgb_gt = render_dict["rgb_gt"]
            render_dict["rgb_gt"] = rgb_gt.view(
                n, v_in, self.height, self.width, channels
            )

        if "dino_gt" in render_dict:
            dino_gt = render_dict["dino_gt"]
            dino_gt_shape = dino_gt.shape[-1]
            if self.dino_upscaled:
                render_dict["dino_gt"] = dino_gt.view(
                    n, v_in, self.height, self.width, dino_gt_shape
                )
            else:
                # TODO: patch size should not be inferred like this, but parameter
                patch_size = isqrt((n * v_in * self.height * self.width * dino_gt_shape) // dino_gt.numel())
                render_dict["dino_gt"] = dino_gt.view(
                    n, v_in, self.height // patch_size, self.width // patch_size, dino_gt_shape
                )
            if "dino_artifacts" in render_dict:
                dino_artifacts = render_dict["dino_artifacts"]
                # TODO: patch size should not be inferred like this, but parameter
                patch_size = isqrt((n * v_in * self.height * self.width * dino_gt_shape) // dino_artifacts.numel())
                render_dict["dino_artifacts"] = dino_artifacts.view(
                    n, v_in, self.height // patch_size, self.width // patch_size, dino_gt_shape
                )

        return render_dict
    

class JitteredPatchRaySampler(PatchRaySampler):
    def __init__(
        self,
        z_near: float,
        z_far: float,
        ray_batch_size: int,
        patch_size: int,
        jitter_strength: float, # In pixels, max [0, 1)
        channels: int = 3,
    ) -> None:
        super().__init__(z_near, z_far, ray_batch_size, patch_size, channels)

        assert 0 <= jitter_strength < 1, "Jitter strength is invalid."

        self.jitter_strength = jitter_strength
        
        x = torch.arange(0, self.patch_size_x).view(1, 1, -1, 1).expand(-1, self.patch_size_y, -1, -1)
        y = torch.arange(0, self.patch_size_y).view(1, -1, 1, 1).expand(-1, -1, self.patch_size_x, -1)
        self._grid = torch.cat((x, y), dim=-1)

    def sample(
        self, images, poses, projs, image_ids=None
    ):  ### dim(images) == nv (ids_loss nv randomly sampled)
        n, v, c, h, w = images.shape
        device = images.device

        all_rgb_gt, all_rays = [], []

        xy_offset = ((torch.rand(2) - .5) * self.jitter_strength)

        for n_ in range(n):
            focals = projs[n_, :, [0, 1], [0, 1]]
            centers = projs[n_, :, [0, 1], [2, 2]]

            rays, xy = util.gen_rays(
                poses[n_].view(-1, 4, 4),
                w,
                h,
                focal=focals,
                c=centers,
                z_near=self.z_near,
                z_far=self.z_far,
                xy_offset=xy_offset,
            )

            # Append frame id to the ray
            if image_ids is None:
                ids = torch.arange(v, device=images.device, dtype=images.dtype)
            else:
                ids = torch.tensor(image_ids, device=images.device, dtype=images.dtype)
            ids = ids.view(v, 1, 1, 1).expand(v, h, w, 1)
            rays = torch.cat((rays, ids), dim=-1)
            r_dim = rays.shape[-1]

            patch_coords_v = torch.randint(0, v, (self._patch_count,))
            patch_coords_y = torch.randint(
                0, h - self.patch_size_y, (self._patch_count,)
            )
            patch_coords_x = torch.randint(
                0, w - self.patch_size_x, (self._patch_count,)
            )

            sample_rgb_gt = []
            sample_rays = []

            for v_, y, x in zip(patch_coords_v, patch_coords_y, patch_coords_x):
                xy_patch = torch.tensor((x, y)).view(1, 1, 1, 2)
                patch_grid = self._grid + xy_patch + xy_offset.view(1, 1, 1, 2) + .5
                patch_grid = patch_grid.to(images.device)
                patch_grid[..., 0] = (patch_grid[..., 0] / w) * 2 - 1
                patch_grid[..., 1] = (patch_grid[..., 1] / h) * 2 - 1

                rgb_gt_patch = F.grid_sample(images[n_:n_+1, v_], patch_grid, padding_mode="border", align_corners=False)
                rgb_gt_patch = rgb_gt_patch.permute(0, 2, 3, 1).reshape(-1, self.channels)

                rays_patch = rays[
                    v_, y : y + self.patch_size_y, x : x + self.patch_size_x, :
            ].reshape(-1, r_dim)

                sample_rgb_gt.append(rgb_gt_patch)
                sample_rays.append(rays_patch)

            sample_rgb_gt = torch.cat(sample_rgb_gt, dim=0)
            sample_rays = torch.cat(sample_rays, dim=0)
            all_rgb_gt.append(sample_rgb_gt)
            all_rays.append(sample_rays)

        all_rgb_gt = torch.stack(all_rgb_gt)
        all_rays = torch.stack(all_rays)

        return all_rays, all_rgb_gt


def get_ray_sampler(config) -> RaySampler:
    z_near = config["z_near"]
    z_far = config["z_far"]
    sample_mode = config.get("sample_mode", "random")

    # TODO: check channel size
    match sample_mode:
        case "random":
            return RandomRaySampler(z_near, z_far, **config["args"])
        case "patch":
            return PatchRaySampler(z_near, z_far, **config["args"])
        case "jitteredpatch":
            return JitteredPatchRaySampler(z_near, z_far, **config["args"])
        case "image":
            return ImageRaySampler(z_near, z_far)
        case _:
            raise NotImplementedError
