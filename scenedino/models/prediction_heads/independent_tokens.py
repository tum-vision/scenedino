from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_independent_token(conf, attn_feat):
    token_type = conf.get("type", "FixedViewIndependentToken")
    if token_type == "FixedViewIndependentToken":
        return FixedViewIndependentToken(attn_feat)
    elif token_type == "DataViewIndependentToken":
        return DataViewIndependentToken(attn_feat)
    elif token_type == "NeuRayIndependentToken":
        return NeuRayIndependentToken(att_feat=attn_feat, **conf["args"])
    else:
        raise NotImplementedError("Unsupported Token type")


class BaseIndependentToken(nn.Module):
    def __init__(self, attn_feat: int) -> None:
        super().__init__()

        self.attn_feat = attn_feat
        self.require_bottleneck_feats = False

    @abstractmethod
    def forward(self, view_dependent_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class FixedViewIndependentToken(BaseIndependentToken):
    def __init__(self, attn_feat: int) -> None:
        super().__init__(attn_feat)
        self.require_bottleneck_feats = False

        self.readout_token = nn.Parameter(torch.rand(1, 1, attn_feat), requires_grad=True)

    def forward(self, view_dependent_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.readout_token.expand(view_dependent_tokens.shape[0], -1, -1)  ### (n_pts, 1, 16)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=-2, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=-2, keepdim=True)
    return mean, var


class DataViewIndependentToken(BaseIndependentToken):
    def __init__(self, attn_feat: int) -> None:
        super().__init__(attn_feat)
        self.require_bottleneck_feats = False

        self.eps = 1.0e-9
        self.layer = nn.Linear(2 * attn_feat, attn_feat, bias=True)

    # def forward(self, view_dependent_tokens: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
    def forward(self, view_dependent_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        mask = 1 - kwargs["invalid_features"].float()
        # mask = 1 - invalid_mask
        weights = mask / (torch.sum(mask, dim=-1, keepdim=True) + 1e-8)
        mean, var = fused_mean_variance(view_dependent_tokens, weights.unsqueeze(-1))
        # num_valid_tokens = torch.sum((1 - invalid_mask), dim=-1, keepdim=True) + self.eps
        # mean = torch.sum(view_dependent_tokens * (1 - invalid_mask).unsqueeze(-1), dim=-2) / num_valid_tokens
        # var = torch.sum((view_dependent_tokens - mean)**2 * (1 - invalid_mask).unsqueeze(-1), dim=-2) / num_valid_tokens
        return nn.ELU()(self.layer(torch.cat([mean, var], dim=-1)))


class NeuRayIndependentToken(BaseIndependentToken):
    def __init__(
        self,
        n_points_per_ray: int,
        # neuray_in_dim: int = 32,
        in_feat_ch: int = 32,
        n_samples: int = 64,
        att_feat: int = 16,
        d_model: int = 103,
        rbs: int = 2048,
        **kwargs
    ):
        super().__init__(att_feat)

        self.n_points_per_ray = n_points_per_ray
        self.require_bottleneck_feats = True
        # self.args = args
        self.anti_alias_pooling = False
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(
            inplace=True
        )  ## (+): Mean Outputs Closer to Zero: want activations with mean outputs closer to zero.        ## nn.LeakyReLU: (+): faster convergence, When the distribution of the negative values in your dataset is meaningful and shouldn't be discarded.
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(
            nn.Linear(4, 16),  ## defualt: 4
            activation_func,
            nn.Linear(16, in_feat_ch),  ## default: in_feat_ch + 3
            activation_func,
        )

        self.base_fc = nn.Sequential(
            nn.Linear((in_feat_ch) * 5 + att_feat, 64),  ## default: ((in_feat_ch+3)*5+neuray_in_dim, 64)
            activation_func,
            nn.Linear(64, 32),
            activation_func,
        )

        self.vis_fc = nn.Sequential(
            nn.Linear(32, 32),
            activation_func,
            nn.Linear(32, 33),
            activation_func,
        )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32), activation_func, nn.Linear(32, 1), nn.Sigmoid())

        self.geometry_fc = nn.Sequential(
            nn.Linear(32 * 2 + 1, att_feat * 2),  ## default: (32*2+1, 64)
            activation_func,
            nn.Linear(att_feat * 2, att_feat),
            activation_func,
        )

        # self.ray_attention = MultiHeadAttention(nhead, att_feat, 4, 4)              ## default: (4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16), activation_func, nn.Linear(16, 1), nn.ReLU())

        self.rgb_fc = nn.Sequential(
            nn.Linear(32 + 1 + 4, 16), activation_func, nn.Linear(16, 8), activation_func, nn.Linear(8, 1)
        )

        self.neuray_fc = nn.Sequential(
            nn.Linear(
                att_feat,
                8,
            ),
            activation_func,
            nn.Linear(8, 1),
        )

        self.img_feat2low = nn.Sequential(
            nn.Linear(rbs, rbs // 4),  ## TODO: replace this hard coded with the flexible
            activation_func,
            # nn.Linear(rbs // 4, d_model),
            nn.Linear(rbs // 4, in_feat_ch),
        )

        # self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        # self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.neuray_fc.apply(weights_init)

    def forward(self, view_dependent_tokens, bottleneck_feats, ray_diff, invalid_features, **kwargs):
        """ibrnet dim e.g. [6, 64, 8, 35]
        :param rgb_feat:    rgbs and image features [n_rays, n_samples, n_views, n_feat] == img_feat
        :param neuray_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat] == viz_feat
        :param ray_diff: ray direction difference   [n_rays, n_samples, n_views, 4], first 3 channels are directions, ## tensor encodes information about how rays in the novel view differ from rays in the source views
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        """
        """ibrnet dim e.g. [6, 64, 8, 35]
        :param view_dependent_tokens: (B*n_pts, n_views, C) = (B*num_rays*point_per_ray, n_views, C)
        :param bottleneck_features: (B*n_pts, n_views, C_bottleneck) = (B*num_rays*point_per_ray, n_views, C)
        :param ray_diff: (B*n_pts, n_views, 4) = (B*num_rays*point_per_ray, n_views, 4)
        :param invalid_features: (B*n_pts, n_views) = (B*num_rays*point_per_ray, n_views)
        :return: rgb and density output, [n_rays, n_samples, 4]
        """

        view_dependent_tokens = view_dependent_tokens.reshape(
            (-1, self.n_points_per_ray) + view_dependent_tokens.shape[-2:]
        )  # (B*num_rays, point_per_ray, n_views, C)
        bottleneck_feats = bottleneck_feats.reshape(
            (-1, self.n_points_per_ray) + bottleneck_feats.shape[-2:]
        )  # (B*num_rays, point_per_ray, n_views, C_bottleneck)
        ray_diff = ray_diff.reshape(
            (-1, self.n_points_per_ray) + ray_diff.shape[-2:]
        )  # (B*num_rays, point_per_ray, n_views, 4)
        invalid_features = invalid_features.reshape(
            (-1, self.n_points_per_ray) + invalid_features.shape[-1:]
        )  # (B*num_rays, point_per_ray, n_views)

        ## Assumption: rgb_feat already contains image feature + dir_feat / this can be implemented further
        mask = ~invalid_features.unsqueeze(-1)
        num_views = bottleneck_feats.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        # rgb_in = rgb_feat[..., :3]            ## no used in both original code and necessary code here
        bottleneck_feats = self.img_feat2low(bottleneck_feats)
        bottleneck_feats = bottleneck_feats + direction_feat

        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (
                torch.sum(weight, dim=2, keepdim=True) + 1e-8
            )  # means it will trust the one more with more consistent view point
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # neuray layer 0 ## == feature aggregation networks (M) above pipeline from fig. 19
        weight0 = torch.sigmoid(self.neuray_fc(view_dependent_tokens)) * weight  # [rn,dn,rfn,f]
        mean0, var0 = fused_mean_variance(bottleneck_feats, weight0)  # [n_rays, n_samples, 1, n_feat]    ## 2nd one
        mean1, var1 = fused_mean_variance(bottleneck_feats, weight)  # [n_rays, n_samples, 1, n_feat]    ## 1st one
        globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat(
            [globalfeat.expand(-1, -1, num_views, -1), bottleneck_feats, view_dependent_tokens], dim=-1
        )  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)  ## after concat it gives input for net A

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask  ## above one from Network A from Fig. 19
        weight = vis / (
            torch.sum(vis, dim=2, keepdim=True) + 1e-8
        )  ## normalized: weighed mean and var ## weight == buttom from net A [N, K, 32]

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat(
            [mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1
        )  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, att_feat] ## MLP for input transformer

        # num_valid_obs = torch.sum(mask, dim=2)
        # num_valid_obs = num_valid_obs > torch.mean(num_valid_obs, dtype=float)  ## making boolean

        return globalfeat.flatten(0, 1).unsqueeze(-2)  # (B*num_rays*point_per_ray, 1, C)
        # return globalfeat, num_valid_obs


# class IBRNetWithNeuRay(nn.Module):
#     def __init__(
#         self, neuray_in_dim=32, in_feat_ch=32, n_samples=64, att_feat=16, d_model=103, rbs=2048, nhead=4, **kwargs
#     ):
#         super().__init__()
#         # self.args = args
#         self.anti_alias_pooling = False
#         if self.anti_alias_pooling:
#             self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
#         activation_func = nn.ELU(
#             inplace=True
#         )  ## (+): Mean Outputs Closer to Zero: want activations with mean outputs closer to zero.        ## nn.LeakyReLU: (+): faster convergence, When the distribution of the negative values in your dataset is meaningful and shouldn't be discarded.
#         self.n_samples = n_samples
#         self.ray_dir_fc = nn.Sequential(
#             nn.Linear(4, 16),  ## defualt: 4
#             activation_func,
#             nn.Linear(16, in_feat_ch),  ## default: in_feat_ch + 3
#             activation_func,
#         )

#         self.base_fc = nn.Sequential(
#             nn.Linear((in_feat_ch) * 5 + neuray_in_dim, 64),  ## default: ((in_feat_ch+3)*5+neuray_in_dim, 64)
#             activation_func,
#             nn.Linear(64, 32),
#             activation_func,
#         )

#         self.vis_fc = nn.Sequential(
#             nn.Linear(32, 32),
#             activation_func,
#             nn.Linear(32, 33),
#             activation_func,
#         )

#         self.vis_fc2 = nn.Sequential(nn.Linear(32, 32), activation_func, nn.Linear(32, 1), nn.Sigmoid())

#         self.geometry_fc = nn.Sequential(
#             nn.Linear(32 * 2 + 1, att_feat * 2),  ## default: (32*2+1, 64)
#             activation_func,
#             nn.Linear(att_feat * 2, att_feat),
#             activation_func,
#         )

#         # self.ray_attention = MultiHeadAttention(nhead, att_feat, 4, 4)              ## default: (4, 16, 4, 4)
#         self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16), activation_func, nn.Linear(16, 1), nn.ReLU())

#         self.rgb_fc = nn.Sequential(
#             nn.Linear(32 + 1 + 4, 16), activation_func, nn.Linear(16, 8), activation_func, nn.Linear(8, 1)
#         )

#         self.neuray_fc = nn.Sequential(
#             nn.Linear(
#                 neuray_in_dim,
#                 8,
#             ),
#             activation_func,
#             nn.Linear(8, 1),
#         )

#         self.img_feat2low = nn.Sequential(
#             nn.Linear(rbs, rbs // 4),  ## TODO: replace this hard coded with the flexible
#             activation_func,
#             nn.Linear(rbs // 4, d_model),
#         )

#         # self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

#         self.base_fc.apply(weights_init)
#         self.vis_fc2.apply(weights_init)
#         self.vis_fc.apply(weights_init)
#         # self.geometry_fc.apply(weights_init)
#         self.rgb_fc.apply(weights_init)
#         self.neuray_fc.apply(weights_init)

#     def forward(self, rgb_feat, neuray_feat, ray_diff, mask):
#         """ibrnet dim e.g. [6, 64, 8, 35]
#         :param rgb_feat:    rgbs and image features [n_rays, n_samples, n_views, n_feat] == img_feat
#         :param neuray_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat] == viz_feat
#         :param ray_diff: ray direction difference   [n_rays, n_samples, n_views, 4], first 3 channels are directions, ## tensor encodes information about how rays in the novel view differ from rays in the source views
#         last channel is inner product
#         :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
#         :return: rgb and density output, [n_rays, n_samples, 4]
#         """

#         ## Assumption: rgb_feat already contains image feature + dir_feat / this can be implemented further
#         num_views = rgb_feat.shape[2]
#         direction_feat = self.ray_dir_fc(ray_diff)
#         # rgb_in = rgb_feat[..., :3]            ## no used in both original code and necessary code here
#         rgb_feat = self.img_feat2low(rgb_feat)
#         rgb_feat = rgb_feat + direction_feat

#         if self.anti_alias_pooling:
#             _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
#             exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
#             weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
#             weight = weight / (
#                 torch.sum(weight, dim=2, keepdim=True) + 1e-8
#             )  # means it will trust the one more with more consistent view point
#         else:
#             weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

#         # neuray layer 0 ## == feature aggregation networks (M) above pipeline from fig. 19
#         weight0 = torch.sigmoid(self.neuray_fc(neuray_feat)) * weight  # [rn,dn,rfn,f]
#         mean0, var0 = fused_mean_variance(rgb_feat, weight0)  # [n_rays, n_samples, 1, n_feat]    ## 2nd one
#         mean1, var1 = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]    ## 1st one
#         globalfeat = torch.cat([mean0, var0, mean1, var1], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

#         x = torch.cat(
#             [globalfeat.expand(-1, -1, num_views, -1), rgb_feat, neuray_feat], dim=-1
#         )  # [n_rays, n_samples, n_views, 3*n_feat]
#         x = self.base_fc(x)  ## after concat it gives input for net A

#         x_vis = self.vis_fc(x * weight)
#         x_res, vis = torch.split(x_vis, [x_vis.shape[-1] - 1, 1], dim=-1)
#         vis = F.sigmoid(vis) * mask
#         x = x + x_res
#         vis = self.vis_fc2(x * vis) * mask  ## above one from Network A from Fig. 19
#         weight = vis / (
#             torch.sum(vis, dim=2, keepdim=True) + 1e-8
#         )  ## normalized: weighed mean and var ## weight == buttom from net A [N, K, 32]

#         mean, var = fused_mean_variance(x, weight)
#         globalfeat = torch.cat(
#             [mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1
#         )  # [n_rays, n_samples, 32*2+1]
#         globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, att_feat] ## MLP for input transformer

#         # num_valid_obs = torch.sum(mask, dim=2)
#         # num_valid_obs = num_valid_obs > torch.mean(num_valid_obs, dtype=float)  ## making boolean

#         return globalfeat
#         # return globalfeat, num_valid_obs
