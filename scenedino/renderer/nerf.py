"""
NeRF differentiable renderer.
References:
https://github.com/bmild/nerf
https://github.com/kwea123/nerf_pl
"""
import torch
import torch.autograd.profiler as profiler
from dotmap import DotMap


class _RenderWrapper(torch.nn.Module):
    def __init__(self, net, renderer, simple_output):
        super().__init__()
        self.net = net
        self.renderer = renderer
        self.simple_output = simple_output

    def forward(
        self,
        rays,
        want_weights=False,
        want_alphas=False,
        want_z_samps=False,
        want_rgb_samps=False,
        sample_from_dist=None,
    ):
        if rays.shape[0] == 0:
            return (
                torch.zeros(0, 3, device=rays.device),
                torch.zeros(0, device=rays.device),
            )

        outputs = self.renderer(
            self.net,
            rays,
            want_weights=want_weights and not self.simple_output,
            want_alphas=want_alphas and not self.simple_output,
            want_z_samps=want_z_samps and not self.simple_output,
            want_rgb_samps=want_rgb_samps and not self.simple_output,
            sample_from_dist=sample_from_dist,
        )
        if self.simple_output:
            if self.renderer.using_fine:
                rgb = outputs.fine.rgb
                depth = outputs.fine.depth
            else:
                rgb = outputs.coarse.rgb
                depth = outputs.coarse.depth
            return rgb, depth
        else:
            # Make DotMap to dict to support DataParallel
            return outputs.toDict()


class NeRFRenderer(torch.nn.Module):
    """
    NeRF differentiable renderer
    :param n_coarse number of coarse (binned uniform) samples
    :param n_fine number of fine (importance) samples
    :param n_fine_depth number of expected depth samples
    :param noise_std noise to add to sigma. We do not use it
    :param depth_std noise for depth samples
    :param eval_batch_size ray batch size for evaluation
    :param white_bkgd if true, background color is white; else black
    :param lindisp if to use samples linear in disparity instead of distance
    :param sched ray sampling schedule. list containing 3 lists of equal length.
    sched[0] is list of iteration numbers,
    sched[1] is list of coarse sample numbers,
    sched[2] is list of fine sample numbers
    """

    def __init__(
        self,
        n_coarse=128,
        n_fine=0,
        n_fine_depth=0,
        noise_std=0.0,
        depth_std=0.01,
        eval_batch_size=100000,
        white_bkgd=False,
        lindisp=False,
        sched=None,  # ray sampling schedule for coarse and fine rays
        hard_alpha_cap=False,
        render_mode="volumetric",
        surface_sigmoid_scale=.1,
        render_flow=False,
        normalize_dino=False,
    ):
        super().__init__()
        self.n_coarse, self.n_fine = n_coarse, n_fine
        self.n_fine_depth = n_fine_depth

        self.noise_std = noise_std
        self.depth_std = depth_std

        self.eval_batch_size = eval_batch_size
        self.white_bkgd = white_bkgd
        self.lindisp = lindisp
        if lindisp:
            print("Using linear displacement rays")
        self.using_fine = n_fine > 0
        self.sched = sched
        if sched is not None and len(sched) == 0:
            self.sched = None
        self.register_buffer(
            "iter_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "last_sched", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.hard_alpha_cap = hard_alpha_cap

        assert render_mode in ("volumetric", "surface", "neus")
        self.render_mode = render_mode
        self.only_surface_color = (self.render_mode == "surface")
        self.surface_sigmoid_scale = surface_sigmoid_scale
        self.render_flow = render_flow
        self.normalize_dino = normalize_dino

    def sample_coarse(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, 6:7], rays[:, 7:8]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        if not self.lindisp:  # Use linear sampling in depth space
            return near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)

        # Use linear sampling in depth space
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)

    def sample_coarse_from_dist(self, rays, weights, z_samp):
        device = rays.device
        B = rays.shape[0]

        num_bins = weights.shape[-1]
        num_samples = self.n_coarse

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)

        u = torch.rand(B, num_samples, dtype=torch.float32, device=device)  # (B, Kf)
        interval_ids = torch.searchsorted(cdf, u, right=True) - 1  # (B, Kf)
        interval_ids = torch.clamp(interval_ids, 0, num_samples - 1)
        interval_interp = torch.rand_like(interval_ids, dtype=torch.float32)

        # z_samps describe the centers of the respective histogram bins. Therefore, we have to extend them to the left and right
        if self.lindisp:
            z_samp = 1 / z_samp

        centers = 0.5 * (z_samp[:, 1:] + z_samp[:, :-1])
        interval_borders = torch.cat((z_samp[:, :1], centers, z_samp[:, -1:]), dim=-1)

        left_border = torch.gather(interval_borders, dim=-1, index=interval_ids)
        right_border = torch.gather(interval_borders, dim=-1, index=interval_ids + 1)

        z_samp_new = (
            left_border * (1 - interval_interp) + right_border * interval_interp
        )

        if self.lindisp:
            z_samp_new = 1 / z_samp_new

        assert not torch.any(torch.isnan(z_samp_new))

        return z_samp_new

    def sample_fine(self, rays, weights):
        """min
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        device = rays.device
        B = rays.shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)

        u = torch.rand(
            B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device
        )  # (B, Kf)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
        inds = torch.clamp_min(inds, 0.0)

        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse  # (B, Kf)

        near, far = rays[:, 6:7], rays[:, 7:8]  # (B, 1)
        if not self.lindisp:  # Use linear sampling in depth space
            z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)

        assert not torch.any(torch.isnan(z_samp))

        return z_samp

    def sample_fine_depth(self, rays, depth):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param depth (B)
        :return (B, Kfd)
        """
        z_samp = depth.unsqueeze(1).repeat((1, self.n_fine_depth))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays[:, 7:8]), rays[:, 6:7])

        assert not torch.any(torch.isnan(z_samp))

        return z_samp

    def composite(self, model, rays, z_samp, coarse=True, sb=0):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp z positions sampled for each ray (B, K)
        :param coarse whether to evaluate using coarse NeRF
        :param sb super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        with profiler.record_function("renderer_composite"):
            B, K = z_samp.shape
            r_dim = rays.shape[-1]

            deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
            delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
            # delta_inf = rays[:, -1:] - z_samp[:, -1:]
            deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

            # (B, K, 3)
            points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
            points = points.reshape(-1, 3)  # (B*K, 3)

            if r_dim > 8:
                ray_info = rays[:, None, 8:].expand(-1, K, -1)
            else:
                ray_info = None

            if hasattr(model, "use_viewdirs"):
                use_viewdirs = model.use_viewdirs
            else:
                use_viewdirs = None

            viewdirs_all = []
            rgbs_all, invalid_all, sigmas_all, extras_all, state_dicts_all = [], [], [], [], []

            if sb > 0:
                points = points.reshape(
                    sb, -1, 3
                )  # (SB, B'*K, 3) B' is real ray batch size
                
                if ray_info is not None:
                    ray_info = ray_info.reshape(sb, -1, ray_info.shape[-1])

                eval_batch_dim = 1
                eval_batch_size = (self.eval_batch_size - 1) // sb + 1
            else:
                eval_batch_size = self.eval_batch_size
                eval_batch_dim = 0

            split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
            
            if ray_info is not None:
                split_ray_infos = torch.split(ray_info, eval_batch_size, dim=eval_batch_dim)
            else:
                split_ray_infos = [None for _ in split_points]

            if use_viewdirs:
                dim1 = K
                viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)
                if sb > 0:
                    viewdirs = viewdirs.reshape(sb, -1, 3)  # (SB, B'*K, 3)
                else:
                    viewdirs = viewdirs.reshape(-1, 3)  # (B*K, 3)
                split_viewdirs = torch.split(
                    viewdirs, eval_batch_size, dim=eval_batch_dim
                )

                for i, pnts in enumerate(split_points):
                    dirs = split_viewdirs[i]
                    infos = split_ray_infos[i]

                    rgbs, invalid, sigmas, extras, state_dict = model(
                        pnts, coarse=coarse, viewdirs=dirs, only_density=self.only_surface_color, ray_info=ray_info, render_flow=self.render_flow
                    )
                    rgbs_all.append(rgbs)
                    invalid_all.append(invalid)
                    sigmas_all.append(sigmas)
                    extras_all.append(extras)

                    viewdirs_all.append(dirs)
                    if state_dict is not None:
                        state_dicts_all.append(state_dict)
            else:
                for i, pnts in enumerate(split_points):
                    infos = split_ray_infos[i]

                    rgbs, invalid, sigmas, extras, state_dict = model(pnts, coarse=coarse, only_density=self.only_surface_color, ray_info=infos, render_flow=self.render_flow)
                    rgbs_all.append(rgbs)
                    invalid_all.append(invalid)
                    sigmas_all.append(sigmas)
                    extras_all.append(extras)

                    if state_dict is not None:
                        state_dicts_all.append(state_dict)

            points, viewdirs = None, None

            # (B*K, 4) OR (SB, B'*K, 4)
            if not self.only_surface_color:
                rgbs = torch.cat(rgbs_all, dim=eval_batch_dim)
            else:
                rgbs = None
            invalid = torch.cat(invalid_all, dim=eval_batch_dim)
            sigmas = torch.cat(sigmas_all, dim=eval_batch_dim)

            if not extras_all[0] is None:
                extras = torch.cat(extras_all, dim=eval_batch_dim)
            else:
                extras = None

            deltas = deltas.float()
            sigmas = sigmas.float()

            if (
                state_dicts_all is not None and len(state_dicts_all) != 0
            ):  ## not empty in a list
                state_dicts = {
                    key: torch.cat(
                        [state_dicts[key] for state_dicts in state_dicts_all],
                        dim=eval_batch_dim,
                    )
                    for key in state_dicts_all[0].keys()
                }
            else:
                state_dicts = None

            if rgbs is not None:
                rgbs = rgbs.reshape(B, K, -1)  # (B, K, 4 or 5)
            invalid = invalid.reshape(B, K, -1)
            sigmas = sigmas.reshape(B, K)

            if extras is not None:
                extras = extras.reshape(B, K, -1)

            if state_dicts is not None:
                state_dicts = {
                    key: value.reshape(B, K, *value.shape[2:])
                    for key, value in state_dicts.items()
                }  # BxKx... (BxKxn_viewsx...)

            if self.training and self.noise_std > 0.0:
                sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

            alphas = 1 - torch.exp(
                -deltas.abs() * torch.relu(sigmas)
            )  # (B, K) (delta should be positive anyways)

            if self.hard_alpha_cap:
                alphas[:, -1] = 1

            deltas, sigmas = None, None

            alphas_shifted = torch.cat(
                [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
            )  # (B, K+1) = [1, a1, a2, ...]
            T = torch.cumprod(alphas_shifted, -1)  # (B)
            weights = alphas * T[:, :-1]  # (B, K)
            # alphas = None
            alphas_shifted = None

            depth_final = torch.sum(weights * z_samp, -1)  # (B)
            state_dicts["dino_features"] = torch.sum(state_dicts["dino_features"].mul_(weights.unsqueeze(-1)), -2)

            if self.render_mode == "neus":
                # dist_from_surf = z_samp - depth_final[..., None]
                indices = torch.arange(0, weights.shape[-1], device=weights.device, dtype=weights.dtype).unsqueeze(0)
                surface_index = torch.sum(weights * indices, dim=-1, keepdim=True)
                dist_from_surf = surface_index - indices
                weights = torch.exp(-.5 * (dist_from_surf * self.surface_sigmoid_scale) ** 2)
                weights = weights / torch.sum(weights, dim=-1, keepdim=True) 

            if not self.only_surface_color:
                rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
            else:
                surface_points = rays[:, None, :3] + depth_final[:, None, None] * rays[:, None, 3:6]
                surface_points = surface_points.reshape(sb, -1, 3)

                if ray_info is not None:
                    ray_info = ray_info.reshape(sb, -1, K, ray_info.shape[-1])[:, :, 0, :]

                rgb_final, invalid_colors = model.sample_colors(surface_points, ray_info=ray_info, render_flow=self.render_flow)
                rgb_final = rgb_final.permute(0, 2, 1, 3).reshape(B, -1)
                invalid_colors = invalid_colors.permute(0, 2, 1, 3).reshape(B, 1, -1)
                invalid = ((invalid > .5) | invalid_colors).float()

            if self.white_bkgd:
                # White background
                pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
                rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)

            if extras is not None:
                extras_final = torch.sum(weights.unsqueeze(-1) * extras, -2)  # (B, extras)
            else:
                extras_final = None

            for name, x in [("weights", weights), ("rgb_final", rgb_final), ("depth_final", depth_final), ("alphas", alphas), ("invalid", invalid), ("z_samp", z_samp)]:
                if torch.any(torch.isnan(x)):
                    print(f"Detected NaN in {name} ({x.dtype}):")
                    print(x)
                    exit()

            if ray_info is not None:
                ray_info = rays[:, None, 8:]

            # return (weights, rgb_final, depth_final, alphas, invalid, z_samp, rgbs, viewdirs)
            return (
                weights,
                rgb_final,
                depth_final,
                alphas,
                invalid,
                z_samp,
                rgbs,
                ray_info,
                extras_final,
                state_dicts,
            )

    def forward(
        self,
        model,
        rays,
        want_weights=False,
        want_alphas=False,
        want_z_samps=False,
        want_rgb_samps=False,
        sample_from_dist=None,
    ):
        """
        :model nerf model, should return (SB, B, (r, g, b, sigma))
        when called with (SB, B, (x, y, z)), for multi-object:
        SB = 'super-batch' = size of object batch,
        B  = size of per-object ray batch.
        Should also support 'coarse' boolean argument for coarse NeRF.
        :param rays ray spec [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
        :param want_weights if true, returns compositing weights (SB, B, K)
        :return render dict
        """
        with profiler.record_function("renderer_forward"):
            if self.sched is not None and self.last_sched.item() > 0:
                self.n_coarse = self.sched[1][self.last_sched.item() - 1]
                self.n_fine = self.sched[2][self.last_sched.item() - 1]

            assert len(rays.shape) == 3
            superbatch_size = rays.shape[0]
            r_dim = rays.shape[-1]

            rays = rays.reshape(-1, r_dim)  # (SB * B, 8)

            if sample_from_dist is None:
                z_coarse = self.sample_coarse(rays)  # (B, Kc)
            else:
                prop_weights, prop_z_samp = sample_from_dist
                n_samples = prop_weights.shape[-1]
                prop_weights = prop_weights.reshape(-1, n_samples)
                prop_z_samp = prop_z_samp.reshape(-1, n_samples)
                z_coarse = self.sample_coarse_from_dist(rays, prop_weights, prop_z_samp)
                z_coarse, _ = torch.sort(z_coarse, dim=-1)
            coarse_composite = self.composite(
                model,
                rays,
                z_coarse,
                coarse=True,
                sb=superbatch_size,
            )

            outputs = DotMap(
                coarse=self._format_outputs(
                    coarse_composite,
                    superbatch_size,
                    want_weights=want_weights,
                    want_alphas=want_alphas,
                    want_z_samps=want_z_samps,
                    want_rgb_samps=want_rgb_samps,
                ),
            )
            outputs.state_dict = coarse_composite[-1]

            if self.using_fine:
                all_samps = [z_coarse]
                if self.n_fine - self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine(rays, coarse_composite[0].detach())
                    )  # (B, Kf - Kfd)
                if self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine_depth(rays, coarse_composite[2])
                    )  # (B, Kfd)
                z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)
                fine_composite = self.composite(
                    model,
                    rays,
                    z_combine_sorted,
                    coarse=False,
                    sb=superbatch_size,
                )
                outputs.fine = self._format_outputs(
                    fine_composite,
                    superbatch_size,
                    want_weights=want_weights,
                    want_alphas=want_alphas,
                    want_z_samps=want_z_samps,
                    want_rgb_samps=want_rgb_samps,
                )

            return outputs

    def _format_outputs(
        self,
        rendered_outputs,
        superbatch_size,
        want_weights=False,
        want_alphas=False,
        want_z_samps=False,
        want_rgb_samps=False,
    ):
        (
            weights,
            rgb_final,
            depth,
            alphas,
            invalid,
            z_samps,
            rgb_samps,
            ray_info,
            extras,
            state_dict,
        ) = rendered_outputs
        n_smps = weights.shape[-1]
        out_d_rgb = rgb_final.shape[-1]
        out_d_i = invalid.shape[-1]
        out_d_dino = state_dict["dino_features"].shape[-1]

        if superbatch_size > 0:
            rgb_final = rgb_final.reshape(superbatch_size, -1, out_d_rgb)
            depth = depth.reshape(superbatch_size, -1)
            invalid = invalid.reshape(superbatch_size, -1, n_smps, out_d_i)
        ret_dict = DotMap(rgb=rgb_final, depth=depth, invalid=invalid)
        if ray_info is not None:
            ri_shape = ray_info.shape[-1]
            ray_info = ray_info.reshape(superbatch_size, -1, ri_shape)
            ret_dict.ray_info = ray_info
        if extras is not None:
            extras_shape = extras.shape[-1]
            extras = extras.reshape(superbatch_size, -1, extras_shape)
            ret_dict.extras = extras
        if want_weights:
            weights = weights.reshape(superbatch_size, -1, n_smps)
            ret_dict.weights = weights
        if want_alphas:
            alphas = alphas.reshape(superbatch_size, -1, n_smps)
            ret_dict.alphas = alphas
        if want_z_samps:
            z_samps = z_samps.reshape(superbatch_size, -1, n_smps)
            ret_dict.z_samps = z_samps
        if want_rgb_samps:
            rgb_samps = rgb_samps.reshape(superbatch_size, -1, n_smps, out_d_rgb)
            ret_dict.rgb_samps = rgb_samps
        if "dino_features" in state_dict:
            dino_features = state_dict["dino_features"].reshape(superbatch_size, -1, out_d_dino)
            ret_dict.dino_features = dino_features
        if "invalid_features" in state_dict:
            invalid_features = state_dict["invalid_features"].reshape(superbatch_size, -1, n_smps, out_d_i)
            ret_dict.invalid_features = invalid_features
        return ret_dict

    def sched_step(self, steps=1):
        """
        Called each training iteration to update sample numbers
        according to schedule
        """
        if self.sched is None:
            return
        self.iter_idx += steps
        while (
            self.last_sched.item() < len(self.sched[0])
            and self.iter_idx.item() >= self.sched[0][self.last_sched.item()]
        ):
            self.n_coarse = self.sched[1][self.last_sched.item()]
            self.n_fine = self.sched[2][self.last_sched.item()]
            print(
                "INFO: NeRF sampling resolution changed on schedule ==> c",
                self.n_coarse,
                "f",
                self.n_fine,
            )
            self.last_sched += 1

    @classmethod
    def from_conf(cls, conf, white_bkgd=False, eval_batch_size=100000):
        return cls(
            conf.get("n_coarse", 128),
            conf.get("n_fine", 0),
            n_fine_depth=conf.get("n_fine_depth", 0),
            noise_std=conf.get("noise_std", 0.0),
            depth_std=conf.get("depth_std", 0.01),
            white_bkgd=conf.get("white_bkgd", white_bkgd),
            lindisp=conf.get("lindisp", True),
            eval_batch_size=conf.get("eval_batch_size", eval_batch_size),
            sched=conf.get("sched", None),
            hard_alpha_cap=conf.get("hard_alpha_cap", False),
            render_mode=conf.get("render_mode", "volumetric"),
            surface_sigmoid_scale=conf.get("surface_sigmoid_scale", 1),
            render_flow=conf.get("render_flow", False),
            normalize_dino=conf.get("normalize_dino", False),
        )

    def bind_parallel(self, net, gpus=None, simple_output=False):
        """
        Returns a wrapper module compatible with DataParallel.
        Specifically, it renders rays with this renderer
        but always using the given network instance.
        Specify a list of GPU ids in 'gpus' to apply DataParallel automatically.
        :param net A PixelNeRF network
        :param gpus list of GPU ids to parallize to. If length is 1,
        does not parallelize
        :param simple_output only returns rendered (rgb, depth) instead of the
        full render output map. Saves data tranfer cost.
        :return torch module
        """
        wrapped = _RenderWrapper(net, self, simple_output=simple_output)
        if gpus is not None and len(gpus) > 1:
            print("Using multi-GPU", gpus)
            wrapped = torch.nn.DataParallel(wrapped, gpus, dim=1)
        return wrapped
