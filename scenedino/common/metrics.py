import sys
import math
from typing import Callable, Mapping

import skimage.metrics as sk_metrics
import torch
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

import pulp


def median_scaling(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
):
    # TODO: ensure this works for any batch size
    mask = depth_gt > 0

    depth_gt[mask] = torch.nan
    depth_pred[mask] = torch.nan
    scaling = torch.nanmedian(depth_gt.flatten(-2, -1), dim=-1) / torch.nanmedian(
        depth_pred.flatten(-2, -1), dim=-1
    )
    depth_pred = scaling[..., None, None] * depth_pred
    return depth_pred


def l2_scaling(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
):
    # TODO: ensure this works for any batch size
    mask = depth_gt > 0
    depth_pred = depth_pred
    depth_gt_ = depth_gt[mask]
    depth_pred_ = depth_pred[mask]
    depth_pred_ = torch.stack((depth_pred_, torch.ones_like(depth_pred_)), dim=-1)
    x = torch.linalg.lstsq(
        depth_pred_.to(torch.float32), depth_gt_.unsqueeze(-1).to(torch.float32)
    ).solution.squeeze()
    depth_pred = depth_pred * x[0] + x[1]
    return depth_pred


def compute_depth_metrics(
    depth_gt: torch.Tensor,
    depth_pred: torch.Tensor,
    scaling_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
):
    # TODO: find out if dim -3 is dummy dimension or part of the batch
    # TODO: Test if works for batches of images
    if scaling_fn:
        depth_pred = scaling_fn(depth_gt, depth_pred)

    depth_pred = torch.clamp(depth_pred, 1e-3, 80)
    mask = depth_gt != 0

    max_ratio = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
    a_scores = {}
    for name, thresh in {"a1": 1.25, "a2": 1.25**2, "a3": 1.25**3}.items():
        within_thresh = (max_ratio < thresh).to(torch.float)
        within_thresh[~mask] = 0.0
        a_scores[name] = within_thresh.flatten(-2, -1).sum(dim=-1) / mask.to(
            torch.float
        ).flatten(-2, -1).sum(dim=-1)

    square_error = (depth_gt - depth_pred) ** 2
    square_error[~mask] = 0.0

    log_square_error = (torch.log(depth_gt) - torch.log(depth_pred)) ** 2
    log_square_error[~mask] = 0.0

    abs_error = torch.abs(depth_gt - depth_pred)
    abs_error[~mask] = 0.0

    rmse = (
        square_error.flatten(-2, -1).sum(dim=-1)
        / mask.to(torch.float).flatten(-2, -1).sum(dim=-1)
    ) ** 0.5

    rmse_log = (
        log_square_error.flatten(-2, -1).sum(dim=-1)
        / mask.to(torch.float).flatten(-2, -1).sum(dim=-1)
    ) ** 0.5

    abs_rel = abs_error / depth_gt
    abs_rel[~mask] = 0.0
    abs_rel = (
        abs_rel.flatten(-2, -1).sum(dim=-1)
        / mask.to(torch.float).flatten(-2, -1).sum(dim=-1)
    ) ** 0.5

    sq_rel = square_error / depth_gt
    sq_rel[~mask] = 0.0
    sq_rel = (
        sq_rel.flatten(-2, -1).sum(dim=-1)
        / mask.to(torch.float).flatten(-2, -1).sum(dim=-1)
    ) ** 0.5

    metrics_dict = {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a_scores["a1"],
        "a2": a_scores["a2"],
        "a3": a_scores["a3"],
    }
    return metrics_dict


def compute_occ_metrics(
    occupancy_pred: torch.Tensor, occupancy_gt: torch.Tensor, is_visible: torch.Tensor
):
    # Only not visible points can be occupied
    occupancy_gt &= ~is_visible

    is_occupied_acc = (occupancy_pred == occupancy_gt).float().mean().item()
    is_occupied_prec = occupancy_gt[occupancy_pred].float().mean().item()
    is_occupied_rec = occupancy_pred[occupancy_gt].float().mean().item()

    not_occupied_not_visible_ratio = (
        ((~occupancy_gt) & (~is_visible)).float().mean().item()
    )

    total_ie = ((~occupancy_gt) & (~is_visible)).float().sum().item()

    ie_acc = (occupancy_pred == occupancy_gt)[(~is_visible)].float().mean().item()
    ie_prec = (~occupancy_gt)[(~occupancy_pred) & (~is_visible)].float().mean()
    ie_rec = (~occupancy_pred)[(~occupancy_gt) & (~is_visible)].float().mean()
    total_no_nop_nv = (
        ((~occupancy_gt) & (~occupancy_pred))[(~is_visible) & (~occupancy_gt)]
        .float()
        .sum()
    )

    return {
        "o_acc": is_occupied_acc,
        "o_rec": is_occupied_rec,
        "o_prec": is_occupied_prec,
        "ie_acc": ie_acc,
        "ie_rec": ie_rec,
        "ie_prec": ie_prec,
        "ie_r": not_occupied_not_visible_ratio,
        "t_ie": total_ie,
        "t_no_nop_nv": total_no_nop_nv,
    }


def compute_nvs_metrics(data, lpips):
    # TODO: This is only correct for batchsize 1!
    # Following tucker et al. and others, we crop 5% on all sides

    # idx of stereo frame (the target frame is always the "stereo" frame).
    sf_id = data["rgb_gt"].shape[1] // 2

    imgs_gt = data["rgb_gt"][:1, sf_id : sf_id + 1]
    imgs_pred = data["fine"][0]["rgb"][:1, sf_id : sf_id + 1]

    imgs_gt = imgs_gt.squeeze(0).permute(0, 3, 1, 2)
    imgs_pred = imgs_pred.squeeze(0).squeeze(-2).permute(0, 3, 1, 2)

    n, c, h, w = imgs_gt.shape
    y0 = int(math.ceil(0.05 * h))
    y1 = int(math.floor(0.95 * h))
    x0 = int(math.ceil(0.05 * w))
    x1 = int(math.floor(0.95 * w))

    imgs_gt = imgs_gt[:, :, y0:y1, x0:x1]
    imgs_pred = imgs_pred[:, :, y0:y1, x0:x1]

    imgs_gt_np = imgs_gt.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    imgs_pred_np = imgs_pred.detach().squeeze().permute(1, 2, 0).cpu().numpy()

    ssim_score = sk_metrics.structural_similarity(
        imgs_pred_np, imgs_gt_np, multichannel=True, data_range=1, channel_axis=-1
    )
    psnr_score = sk_metrics.peak_signal_noise_ratio(
        imgs_pred_np, imgs_gt_np, data_range=1
    )
    lpips_score = lpips(imgs_pred, imgs_gt, normalize=False).mean()

    metrics_dict = {
        "ssim": torch.tensor([ssim_score], device=imgs_gt.device),
        "psnr": torch.tensor([psnr_score], device=imgs_gt.device),
        "lpips": torch.tensor([lpips_score], device=imgs_gt.device),
    }
    return metrics_dict


def compute_dino_metrics(data):
    dino_gt = data["dino_gt"]
    if "dino_features_downsampled" in data["coarse"][0]:
        dino_pred = data["coarse"][0]["dino_features_downsampled"].squeeze(-2)
    else:
        dino_pred = data["coarse"][0]["dino_features"].squeeze(-2)

    l1_loss = F.l1_loss(dino_pred, dino_gt, reduction="none").mean(dim=(0, 2, 3, 4))
    l2_loss = F.mse_loss(dino_pred, dino_gt, reduction="none").mean(dim=(0, 2, 3, 4))
    cos_sim = F.cosine_similarity(dino_pred, dino_gt, dim=-1).mean(dim=(0, 2, 3))

    metrics_dict = {
        "l1": torch.tensor([l1_loss.mean()], device=dino_gt.device),
        "l2": torch.tensor([l2_loss.mean()], device=dino_gt.device),
        "cos_sim": torch.tensor([cos_sim.mean()], device=dino_gt.device)
    }
    for i in range(len(l1_loss)):
        metrics_dict[f"l1_{i}"] = torch.tensor([l1_loss[i]], device=dino_gt.device)
        metrics_dict[f"l2_{i}"] = torch.tensor([l2_loss[i]], device=dino_gt.device)
        metrics_dict[f"cos_sim_{i}"] = torch.tensor([cos_sim[i]], device=dino_gt.device)
    return metrics_dict


def compute_stego_metrics(data):
    if "stego_corr" not in data["segmentation"]:
        return {}

    metrics_dict = {
        "stego_self_corr": data["segmentation"]["stego_corr"]["stego_self_corr"],
        "stego_nn_corr": data["segmentation"]["stego_corr"]["stego_nn_corr"],
        "stego_random_corr": data["segmentation"]["stego_corr"]["stego_random_corr"],
    }
    return metrics_dict


def compute_seg_metrics(data, n_classes, gt_classes):
    segs_gt = data["segmentation"]["target"].flatten()
    valid_mask = segs_gt >= 0
    segs_gt = segs_gt[valid_mask]

    metrics_dict = {}
    for result_key, result in data["segmentation"]["results"].items():
        if "pseudo_segs_pred" in result:
            segs_pred = result["pseudo_segs_pred"][:, 0].flatten()
        else:
            segs_pred = result["segs_pred"][:, 0].flatten()

        segs_pred = segs_pred[valid_mask]
        confusion_matrix = torch.bincount(n_classes * segs_gt + segs_pred,
                                          minlength=n_classes * gt_classes).reshape(gt_classes, n_classes)
        metrics_dict[result_key] = confusion_matrix
    
    return metrics_dict


class MeanMetric(Metric):
    def __init__(self, output_transform=lambda x: x["output"], device="cpu"):
        super(MeanMetric, self).__init__(
            output_transform=output_transform, device=device
        )
        self._sum = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._num_examples = 0
        self.required_output_keys = ()

    @reinit__is_reduced
    def reset(self):
        self._sum = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._num_examples = 0
        super(MeanMetric, self).reset()

    @reinit__is_reduced
    def update(self, value):
        if torch.any(torch.isnan(torch.tensor(value))):
            raise ValueError("NaN values present in metric!")
        self._sum += value
        self._num_examples += 1

    @sync_all_reduce("_num_examples:SUM", "_sum:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )
        return self._sum.item() / self._num_examples

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(
            engine.state.output
        )  ## engine.state.output.keys() == dict_keys(['output', 'loss_dict', 'timings_dict', 'metrics_dict'])
        self.update(output)


class DictMeanMetric(Metric):
    def __init__(self, name: str, output_transform=lambda x: x["output"], device="cpu"):
        self._name = name
        self._sums: dict[str, torch.Tensor] = {}
        self._num_examples = 0
        self.required_output_keys = ()
        super(DictMeanMetric, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self):
        self._sums = {}
        self._num_examples = 0
        super(DictMeanMetric, self).reset()

    @reinit__is_reduced
    def update(self, value):
        num_examples = None
        for key, metric in value.items():
            if not key in self._sums:
                self._sums[key] = torch.tensor(
                    0, device=self._device, dtype=torch.float32
                )
            if torch.any(torch.isnan(metric)):
                # TODO: integrate into logging
                print(f"Warining: Metric {self._name}/{key} has a nan value")
                continue
            self._sums[key] += metric.sum().to(self._device)
            # TODO: check if this works with batches
            if num_examples is None:
                num_examples = metric.shape[0]
        self._num_examples += 1

    @sync_all_reduce("_num_examples:SUM", "_sum:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )
        return {
            f"{self._name}_{key}": metric.item() / self._num_examples
            for key, metric in self._sums.items()
        }

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output["output"])
        self.update(output)

    def completed(self, engine: Engine, name: str) -> None:
        """Helper method to compute metric's value and put into the engine. It is automatically attached to the
        `engine` with :meth:`~ignite.metrics.metric.Metric.attach`. If metrics' value is torch tensor, it is
        explicitly sent to CPU device.

        Args:
            engine: the engine to which the metric must be attached
            name: the name of the metric used as key in dict `engine.state.metrics`

        .. changes from default implementation:
            don't add whole result dict to engine state, but only the values

        """
        result = self.compute()
        if isinstance(result, Mapping):
            if name in result.keys():
                raise ValueError(
                    f"Argument name '{name}' is conflicting with mapping keys: {list(result.keys())}"
                )

            for key, value in result.items():
                engine.state.metrics[key] = value
        else:
            if isinstance(result, torch.Tensor):
                if len(result.size()) == 0:
                    result = result.item()
                elif "cpu" not in result.device.type:
                    result = result.cpu()

            engine.state.metrics[name] = result


class SegmentationMetric(DictMeanMetric):
    def __init__(self, name: str, output_transform=lambda x: x["output"], device="cpu", assign_pseudo=True):
        super(SegmentationMetric, self).__init__(
            name, output_transform, device
        )
        self.assign_pseudo = assign_pseudo

        # [road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle]
        self.weights = torch.Tensor([4, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1])
        self.weights = self.weights / self.weights.mean()
    
    @reinit__is_reduced
    def update(self, value):
        for key, metric in value.items():
            if not key in self._sums:
                self._sums[key] = torch.zeros(metric.shape, device=self._device, dtype=torch.int32)
            if torch.any(torch.isnan(metric)):
                print(f"Warining: Metric {self._name}/{key} has a nan value")
                continue
            self._sums[key] += metric.to(self._device)
        self._num_examples += 1

    @sync_all_reduce("_num_examples:SUM", "_sum:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )
        result = {}
        for key, _sum in self._sums.items():
            if self.assign_pseudo:
                assignment = self._calculate_pseudo_label_assignment(_sum)
                gt_classes = _sum.size(0)
                confusion_matrix = torch.zeros((gt_classes, gt_classes), dtype=_sum.dtype)
                confusion_matrix.scatter_add_(
                    1,
                    assignment.unsqueeze(0).expand(gt_classes, -1),
                    _sum
                )
                result[key + "_assignment"] = assignment
            else:
                confusion_matrix = _sum

            # confusion_matrix axes: (actual, prediction)

            true_positives = confusion_matrix.diag()
            false_negatives = torch.sum(confusion_matrix, dim=1) - true_positives
            false_positives = torch.sum(confusion_matrix, dim=0) - true_positives
            denominator = true_positives + false_positives + false_negatives
            per_class_iou = torch.where(denominator > 0, true_positives / denominator,
                                        torch.zeros_like(denominator))

            result[key + "_per_class_iou"] = per_class_iou
            result[key + "_miou"] = per_class_iou.mean().item()
            result[key + "_weighted_miou"] = (per_class_iou * self.weights).mean().item()
            result[key + "_acc"] = confusion_matrix.diag().sum().item() / confusion_matrix.sum().item()

            result[key + "_confusion_matrix"] = confusion_matrix

        return result

    def _calculate_pseudo_label_assignment(self, metric_matrix):
        """Implemented this way to generalize to over-segmentation"""
        gt_classes, n_classes = metric_matrix.size()
        costs = metric_matrix.cpu().numpy()

        problem = pulp.LpProblem("CapacitatedAssignment", pulp.LpMaximize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(n_classes)] for i in
             range(gt_classes)]

        problem += pulp.lpSum(costs[i][j] * x[i][j] for i in range(gt_classes) for j in range(n_classes))
        for j in range(n_classes):
            problem += pulp.lpSum(x[i][j] for i in range(gt_classes)) == 1, f"AssignPseudoLabel_{j}"

        for i in range(gt_classes):
            problem += pulp.lpSum(x[i][j] for j in range(n_classes)) >= 1, f"MinAssignActualLabel_{i}"

        problem.solve()

        print("Status:", pulp.LpStatus[problem.status])
        print("Objective:", pulp.value(problem.objective))

        assignment = torch.zeros(n_classes, dtype=torch.int64)
        for j in range(n_classes):
            assignment[j] = next(i for i in range(gt_classes) if pulp.value(x[i][j]) == 1)

        return assignment


class ConcatenateMetric(DictMeanMetric):
    @reinit__is_reduced
    def update(self, value, every_nth=100):
        n_bins = 50
        for key, metric in value.items():
            if not key in self._sums:
                self._sums[key] = torch.zeros((n_bins,), device=self._device, dtype=torch.int32)
            if torch.any(torch.isnan(metric)):
                print(f"Warning: Metric {self._name}/{key} has a nan value")
                continue

            metric_flat = metric.flatten().to(self._device)[::every_nth]
            if key in self._sums:
                self._sums[key] = torch.cat([self._sums[key], metric_flat])
            else:
                self._sums[key] = metric_flat

        self._num_examples += 1

    @sync_all_reduce("_num_examples:SUM", "_sum:SUM")
    def compute(self):
        return self._sums


class FG_ARI(Metric):
    def __init__(self, output_transform=lambda x: x["output"], device="cpu"):
        self._sum_fg_aris = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._num_examples = 0
        self.required_output_keys = ()
        super(FG_ARI, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_fg_aris = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._num_examples = 0
        super(FG_ARI, self).reset()

    @reinit__is_reduced
    def update(self, data):
        true_masks = data["segs"]  # fc [n, h, w]
        pred_masks = data["slot_masks"]  # n, fc, sc, h, w

        n, fc, sc, h, w = pred_masks.shape

        true_masks = [
            F.interpolate(tm.to(float).unsqueeze(1), (h, w), mode="nearest")
            .squeeze(1)
            .to(int)
            for tm in true_masks
        ]

        for i in range(n):
            for f in range(fc):
                true_mask = true_masks[f][i]
                pred_mask = pred_masks[i, f]

                true_mask = true_mask.view(-1)
                pred_mask = pred_mask.view(sc, -1)

                if torch.max(true_mask) == 0:
                    continue

                foreground = true_mask > 0
                true_mask = true_mask[foreground]
                pred_mask = pred_mask[:, foreground].permute(1, 0)

                true_mask = F.one_hot(true_mask)

                # Filter out empty true groups
                not_empty = torch.any(true_mask, dim=0)
                true_mask = true_mask[:, not_empty]

                # Filter out empty predicted groups
                not_empty = torch.any(pred_mask, dim=0)
                pred_mask = pred_mask[:, not_empty]

                true_mask.unsqueeze_(0)
                pred_mask.unsqueeze_(0)

                _, n_points, n_true_groups = true_mask.shape
                n_pred_groups = pred_mask.shape[-1]
                if n_points <= n_true_groups and n_points <= n_pred_groups:
                    print(
                        "adjusted_rand_index requires n_groups < n_points.",
                        file=sys.stderr,
                    )
                    continue

                true_group_ids = torch.argmax(true_mask, -1)
                pred_group_ids = torch.argmax(pred_mask, -1)
                true_mask_oh = true_mask.to(torch.float32)
                pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).to(
                    torch.float32
                )

                n_points = torch.sum(true_mask_oh, dim=[1, 2]).to(torch.float32)

                nij = torch.einsum("bji,bjk->bki", pred_mask_oh, true_mask_oh)
                a = torch.sum(nij, dim=1)
                b = torch.sum(nij, dim=2)

                rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
                aindex = torch.sum(a * (a - 1), dim=1)
                bindex = torch.sum(b * (b - 1), dim=1)
                expected_rindex = aindex * bindex / (n_points * (n_points - 1))
                max_rindex = (aindex + bindex) / 2
                ari = (rindex - expected_rindex) / (
                    max_rindex - expected_rindex + 0.000000000001
                )

                _all_equal = lambda values: torch.all(
                    torch.eq(values, values[..., :1]), dim=-1
                )
                both_single_cluster = torch.logical_and(
                    _all_equal(true_group_ids), _all_equal(pred_group_ids)
                )

                self._sum_fg_aris += torch.where(
                    both_single_cluster, torch.ones_like(ari), ari
                ).squeeze()
                self._num_examples += 1

    @sync_all_reduce("_num_examples:SUM", "_sum_fg_aris:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "CustomAccuracy must have at least one example before it can be computed."
            )
        return self._sum_fg_aris.item() / self._num_examples

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)
