from itertools import chain
import torch
from torch import nn
import torch.nn.functional as F

from datasets.kitti_360.labels import labels as kitti_labels
from datasets.kitti_360.labels import trainId2label

# from pykeops.torch import LazyTensor

from multiprocessing import Pool
from .crf import dense_crf


def _five_crop(features, sample_factor=1):
    _, _, h, w, _, _ = features.shape
    assert h % (4*sample_factor) == 0 and w % (4*sample_factor) == 0

    center_shift = sample_factor // 2
    crop_length = min(h, w) // 4
    crop_centers = [
        (h//2,   w//2),
        (3*h//4, w//4),
        (3*h//4, 3*w//4),
        (h//4,   w//4),
        (h//4,   3*w//4),
    ]
    result = torch.cat([
        features[:, :, 
                 crop_center[0]-crop_length+center_shift : crop_center[0]+crop_length+center_shift : sample_factor, 
                 crop_center[1]-crop_length+center_shift : crop_center[1]+crop_length+center_shift : sample_factor]
        for crop_center in crop_centers
    ])
    return result


def _norm(x):
    return F.normalize(x, dim=-1, eps=1e-10)


class SemanticHead(nn.Module):
    def __init__(self,
                 n_classes,
                 gt_classes,
                 input_dim,
                 code_dim,
                 buffer_size,
                 patch_sample_size,
                 knn_neighbors,
                 mode,
                 mlp_head,
                 apply_crf,
                 ):
        
        super().__init__()
        self.n_classes = n_classes
        self.gt_classes = gt_classes

        self.input_dim = input_dim
        self.code_dim = code_dim
        self.knn_neighbors = knn_neighbors
        self.mode = mode

        self.apply_crf = apply_crf

        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.buffer_filled = 1
        
        self.dino_patch_buffer = torch.zeros((buffer_size, patch_sample_size, input_dim), device="cuda")
        self.dino_gap_buffer = torch.zeros((buffer_size, input_dim), device="cuda")
            
        self.direct_cluster_head = KMeansParamHead(n_classes, gt_classes, input_dim)
        self.stego_head = StegoClusterHead(input_dim, code_dim)
        self.stego_cluster_head = KMeansParamHead(n_classes, gt_classes, code_dim)

        if mlp_head:
            self.direct_linear_head = MLPHead(input_dim, gt_classes)
            self.stego_linear_head = MLPHead(code_dim, gt_classes)
        else:
            self.direct_linear_head = LinearHead(input_dim, gt_classes)
            self.stego_linear_head = LinearHead(code_dim, gt_classes)

        self.label_colors = [torch.Tensor(trainId2label[train_id].color) for train_id in range(gt_classes)]
        self.label_colors.append(torch.Tensor([0, 0, 0]))
        self.label_colors = torch.stack(self.label_colors, dim=0).to("cuda") / 255.0

        self.dropout = nn.Dropout2d(p=.1)
        self.dropout1d = nn.Dropout1d(p=.1)
        

    @classmethod
    def from_conf(cls, config):
        return SemanticHead(
            n_classes=config.n_classes,
            gt_classes=config.gt_classes,
            input_dim=config.input_dim,
            code_dim=config.code_dim,
            buffer_size=config.buffer_size,
            patch_sample_size=config.patch_sample_size,
            knn_neighbors=config.knn_neighbors,
            mode=config.get("mode", "2d"),
            mlp_head=config.get("mlp_head", False),
            apply_crf=config.get("apply_crf", False),
        )

    def forward(self, features, mode="stego_kmeans"):
        features = _norm(features)
        if mode == "stego_kmeans":
            stego_features = self.stego_head(features)
            return self.stego_cluster_head(stego_features)["segs_pred"]
        elif mode == "stego_linear":
            stego_features = self.stego_head(features)
            return self.stego_linear_head(stego_features)["segs_pred"]
        elif mode == "direct_kmeans":
            return self.direct_cluster_head(features)["segs_pred"]
        elif mode == "direct_linear":
            return self.direct_linear_head(features)["segs_pred"]
        else:
            raise NotImplementedError(f"Mode '{mode}' is not known!")

    def forward_training(self, data, visualize=False, sample_factor=4):  # TODO: visualization
        rgb_image = data["coarse"][0]["rgb"].detach()
        dino_features = data["coarse"][0]["dino_features"].detach()  # [n, v, h, w, 1, c]
        dino_features = _norm(dino_features)

        n, v, h, w, _, c = dino_features.shape
        reshaped_dino_features = dino_features.squeeze(-2).flatten(0, 1)  # [n*v, h, w, c]
        stego_features = self.stego_head(reshaped_dino_features).reshape(n, v, h, w, 1, -1)

        dino_features = self.dropout(dino_features.reshape(n*v, h, w, c).permute(0, 3, 1, 2))
        dino_features = dino_features.permute(0, 2, 3, 1).reshape(n, v, h, w, 1, c)
        dino_features = dino_features.reshape(n, v, h, w, 1, c)

        data["segmentation"] = {}
        if data["sample_surface_sigma"] is not None:
            if self.mode == "3d":
                cropped_dino_features = data["sample_surface_dino_features"].detach().squeeze(0)
                cropped_dino_features = _norm(cropped_dino_features)

                cropped_dino_features = self.dropout1d(cropped_dino_features.swapaxes(-2, -1)).swapaxes(-2, -1)
                stego_self_features = self.stego_head(cropped_dino_features.unsqueeze(1)).squeeze(1)

            elif self.mode == "2d":
                dino_features = dino_features[:, :1]
                stego_features = stego_features[:, :1]

                # Single view
                cropped_dino_features = _five_crop(dino_features, sample_factor).flatten(0, 1).flatten(1, 2).squeeze(-2)
                stego_self_features = _five_crop(stego_features, sample_factor).flatten(0, 1).flatten(1, 2).squeeze(-2)

            dino_feature_gap = cropped_dino_features.mean(dim=-2)
            dino_feature_gap = _norm(dino_feature_gap)


            # Just in training: update knn buffer
            if self.training:
                new_idx = self._update_buffer(self.dino_patch_buffer, cropped_dino_features)
                assert new_idx == self._update_buffer(self.dino_gap_buffer, dino_feature_gap)

                if new_idx < self.buffer_idx:
                    self.buffer_filled = self.buffer_size
                else:
                    self.buffer_filled = max(new_idx, self.buffer_filled)
                self.buffer_idx = new_idx


            # Calculate from buffer - "kNN", "random"
            pairwise_cos_sims = torch.einsum("nf,mf->nm", dino_feature_gap, self.dino_gap_buffer)
            topk_indices = torch.topk(pairwise_cos_sims, self.knn_neighbors+1, dim=1)[1][:, 1:]  # (n, k_nn)

            n = cropped_dino_features.size(0)
            random_nn_indices = topk_indices[torch.arange(n), torch.randint(self.knn_neighbors, (n,))]
            dino_nn_features = self.dino_patch_buffer[random_nn_indices].detach()
            stego_nn_features = self.stego_head(dino_nn_features)

            random_indices = torch.randint(self.buffer_filled, (n,))
            dino_random_features = self.dino_patch_buffer[random_indices].detach()
            stego_random_features = self.stego_head(dino_random_features)

            stego_corr = {
                "dino_self_corr":    self._compute_stego_correlation(cropped_dino_features, cropped_dino_features),
                "stego_self_corr":   self._compute_stego_correlation(stego_self_features,   stego_self_features),
                "dino_nn_corr":      self._compute_stego_correlation(cropped_dino_features, dino_nn_features),
                "stego_nn_corr":     self._compute_stego_correlation(stego_self_features,   stego_nn_features),
                "dino_random_corr":  self._compute_stego_correlation(cropped_dino_features, dino_random_features),
                "stego_random_corr": self._compute_stego_correlation(stego_self_features,   stego_random_features),
            }

            data["segmentation"]["stego_corr"] = stego_corr

        else:
            data["sample_surface_sigma"] = torch.Tensor([0.0])
            data["sample_surface_dino_features"] = torch.Tensor([0.0])


        # IMPORTANT, train heads after detaching features!
        dino_features = dino_features.detach()
        stego_features = stego_features.detach()

        direct_cluster_result = self.direct_cluster_head(dino_features)
        stego_cluster_result = self.stego_cluster_head(stego_features)

        data["segmentation"]["results"] = {
            "direct_cluster": direct_cluster_result,
            "stego_cluster": stego_cluster_result,
        }

        data["segmentation"]["visualization"] = {
            "direct_cluster": self.visualize(direct_cluster_result["segs_pred"]),
            "stego_cluster": self.visualize(stego_cluster_result["segs_pred"])
        }

        if "segs" in data:
            seg_target = self.map_kitti_id_to_train_id(data["segs"][0]).to(stego_features.device)
            direct_linear_result = self.direct_linear_head(dino_features, seg_target)
            stego_linear_result = self.stego_linear_head(stego_features, seg_target)

            data["segmentation"]["target"] = seg_target
            data["segmentation"]["results"]["direct_linear"] = direct_linear_result
            data["segmentation"]["results"]["stego_linear"] = stego_linear_result
            data["segmentation"]["visualization"]["target"] = self.visualize(seg_target)

        if self.apply_crf:
            result_names = list(data["segmentation"]["results"].keys())
            for result_name in result_names:
                pred_no_crf = data["segmentation"]["results"][result_name]["segs_pred"]
                pred_crf = self.forward_crf(pred_no_crf, rgb_image)
                
                data["segmentation"]["results"][result_name + "_crf"] = {"segs_pred": pred_crf}

        for result_name, result in data["segmentation"]["results"].items():
            data["segmentation"]["visualization"][result_name] = self.visualize(result["segs_pred"])

        return data
    
    def forward_crf(self, pred_no_crf, rgb_image):
        pred_no_crf_logits = F.one_hot(pred_no_crf.squeeze()).permute(2,0,1).float()
        pred_crf = torch.Tensor(dense_crf(rgb_image.squeeze().permute(2,0,1), pred_no_crf_logits)).to(pred_no_crf.device)
        pred_crf = pred_crf.argmax(dim=0).reshape(pred_no_crf.shape)
        return pred_crf
    
    def update_model_eval(self, metrics):
        self.direct_cluster_head.pseudo_assignment[:] = metrics["direct_cluster_assignment"]
        self.stego_cluster_head.pseudo_assignment[:] = metrics["stego_cluster_assignment"]
    
    def map_kitti_id_to_train_id(self, labels):
        result = torch.zeros(labels.shape).long()
        for kitti_label in kitti_labels:
            result[labels == kitti_label.id] = kitti_label.trainId

        result[result == 255] = -1
        return result
    
    def visualize(self, labels):
        label_map = self.label_colors[labels.long()]
        return label_map

    def parameters_lr(self):
        return [
            (1.0, self.stego_head.parameters()),
            (10.0, self.direct_cluster_head.parameters()),
            (10.0, self.stego_cluster_head.parameters()),
            (10.0, self.direct_linear_head.parameters()),
            (10.0, self.stego_linear_head.parameters()),
        ]

    def _compute_stego_correlation(self, tensor1, tensor2):
        corr = torch.einsum("npf,nqf->npq", _norm(tensor1), _norm(tensor2))
        return corr

    def _update_buffer(self, buffer, x):
        n = x.size(0)
        if n >= self.buffer_size:
            buffer[:] = x[-self.buffer_size:]
            new_buffer_idx = 0  # Reset write index
        else:
            indices = (torch.arange(n) + self.buffer_idx) % self.buffer_size
            buffer[indices] = x
            new_buffer_idx = (self.buffer_idx + n) % self.buffer_size

        return new_buffer_idx


class StegoClusterHead(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        
        self.linear_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.Dropout2d(p=.1),
        )
        self.nonlinear_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, (1, 1)),
            nn.Dropout2d(p=.1),
        )

    def forward(self, x):
        x = x.swapaxes(-1, -3)
        result = self.linear_path(x) + self.nonlinear_path(x)
        return _norm(result.swapaxes(-1, -3)).to(x.dtype)


class KMeansParamHead(nn.Module):
    def __init__(self,
                 n_classes: int,
                 gt_classes: int,
                 dim: int,
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.init_type = "random"
        self.cluster_centers = torch.nn.Parameter(torch.randn(self.n_classes, self.dim))
        self.centroids_initialized = False
        self.register_buffer("pseudo_assignment", torch.arange(0, n_classes).remainder(gt_classes))

    def forward(self, features, weight=None):
        features_flat = features.flatten(0, -2)  # (n, d)
        if weight is not None:
            weight_flat = weight.flatten()
        else:
            weight_flat = torch.ones(features_flat.size(0), device=features.device)

        # K-means++ init
        if not self.centroids_initialized and self.training:
            if self.init_type == "kmeans++":
                cluster_centers = torch.empty(self.n_classes, self.dim, device=features.device)
                first_idx = torch.randint(0, features_flat.size(0), (1,))
                cluster_centers[0] = features_flat[first_idx]
                for k in range(1, self.n_classes):
                    current_centroids = cluster_centers[:k]  # (k, d)
                    similarity = (current_centroids @ features_flat.transpose(1, 0))  # (k, n)
                    max_similarity = similarity.max(dim=0).values  # Closest centroid per point
                    distances = 1 - max_similarity
                    probabilities = distances ** 2
                    probabilities /= probabilities.sum()
                    next_idx = torch.multinomial(probabilities, 1)
                    cluster_centers[k] = features_flat[next_idx]
                self.cluster_centers.data = cluster_centers
            else:
                self.cluster_centers.data = torch.randn(self.n_classes, self.dim, device=self.cluster_centers.device)
            self.centroids_initialized = True
        
        class_labels, cluster_loss, _ = self._kmeans_cosine(features_flat)
        pseudo_segs_pred = class_labels.view(*features.shape[:-1])
        result = {
            "pseudo_segs_pred": pseudo_segs_pred,
            "segs_pred": self._assign_pseudo_labels(pseudo_segs_pred),
            "loss": torch.mean(cluster_loss * weight_flat)
        }
        return result

    def _assign_pseudo_labels(self, pseudo_labels):
        return self.pseudo_assignment[pseudo_labels.cpu()].long()

    def _kmeans_cosine(self, features):
        normed_clusters = F.normalize(self.cluster_centers, dim=1)
        normed_features = F.normalize(features, dim=1)
        inner_products = normed_features.matmul(normed_clusters.t())

        class_labels = torch.argmax(inner_products, dim=1)

        # cluster_probs = F.softmax(inner_products, dim=1)
        cluster_probs = F.one_hot(class_labels, normed_clusters.shape[0]).to(torch.float32)
        cluster_loss = -(cluster_probs * inner_products).sum(1)

        # return nn.functional.log_softmax(inner_products * alpha, dim=1)
        return class_labels, cluster_loss, cluster_probs


class KMeansIterHead(nn.Module):
    def __init__(self,
                 n_classes: int,
                 gt_classes: int,
                 dim: int,
                 reassignment_threshold: int = 5000,
                 kmeans_update_factor: float = 1.0,
                 training_chunk: int = 100000,
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim

        self.reassignment_threshold = reassignment_threshold
        self.kmeans_update_factor = kmeans_update_factor
        self.training_chunk = training_chunk

        self.centroids_initialized = False
        self.register_buffer("cluster_centers", torch.empty(self.n_classes, self.dim, device="cuda"))
        self.register_buffer("pseudo_assignment", torch.arange(0, n_classes).remainder(gt_classes))

    def forward(self, features):
        features_flat = features.flatten(0, -2)  # (n, d)
        
        # K-means++ init
        if not self.centroids_initialized and self.training:
            first_idx = torch.randint(0, features_flat.size(0), (1,))
            self.cluster_centers[0] = features_flat[first_idx]
            for k in range(1, self.n_classes):
                current_centroids = self.cluster_centers[:k]  # (k, d)
                similarity = (current_centroids @ features_flat.transpose(1, 0))  # (k, n)
                max_similarity = similarity.max(dim=0).values  # Closest centroid per point
                distances = 1 - max_similarity
                probabilities = distances ** 2
                probabilities /= probabilities.sum()
                next_idx = torch.multinomial(probabilities, 1)
                self.cluster_centers[k] = features_flat[next_idx]
            self.centroids_initialized = True
        
        class_labels = self._kmeans_cosine(features_flat)
        pseudo_segs_pred = class_labels.view(*features.shape[:-1])
        result = {
            "pseudo_segs_pred": pseudo_segs_pred,
            "segs_pred": self._assign_pseudo_labels(pseudo_segs_pred),
        }
        return result

    def _assign_pseudo_labels(self, pseudo_labels):
        return self.pseudo_assignment[pseudo_labels.cpu()].long()

    def _kmeans_cosine(self, features):
        """Implements Lloyd's algorithm for the Cosine similarity metric."""
        features = F.normalize(features, dim=1, p=2)
        n, d = features.shape

        x_i = LazyTensor(features.view(n, 1, d).contiguous())  # (n, 1, d) samples
        c_j = LazyTensor(self.cluster_centers.view(1, self.n_classes, d).contiguous())  # (1, n_classes, d) centroids

        s_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
        class_labels = s_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

        if self.training:
            class_labels_count = class_labels.bincount(minlength=self.n_classes)
            cluster_center_update = torch.zeros_like(self.cluster_centers)

            if self.training_chunk:
                for i in range(0, n, self.training_chunk):
                    if i + self.training_chunk < n:
                        cluster_center_update.scatter_add_(0, class_labels[i:i+self.training_chunk, None].repeat(1, d), features)
                    else:
                        cluster_center_update.scatter_add_(0, class_labels[i:, None].repeat(1, d), features)
            else:
                cluster_center_update.scatter_add_(0, class_labels[:, None].repeat(1, d), features)

            cluster_center_update = F.normalize(cluster_center_update)

            update_factor = self.kmeans_update_factor * (class_labels_count > self.reassignment_threshold)
            update_factor = update_factor.unsqueeze(-1)

            self.cluster_centers[:] = F.normalize(cluster_center_update * update_factor + self.cluster_centers * (1-update_factor))

        return class_labels


class LinearHead(nn.Module):
    def __init__(self,
                 dim: int,
                 gt_classes: int
                 ):
        super().__init__()
        self.linear = torch.nn.Linear(dim, gt_classes)

    def forward(self, features, target=None):
        logit = self.linear(features).float()
        result = {
            "segs_pred": logit.argmax(-1),
        }
        if target is not None:
            target = target.long().to(logit.device)
            result["loss"] = F.cross_entropy(logit[:, 0].movedim(-1, 1).squeeze(-1), target, ignore_index=-1)

        return result


class MLPHead(nn.Module):
    def __init__(self,
                 dim: int,
                 gt_classes: int
                 ):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, 2*dim)
        self.linear2 = torch.nn.Linear(2*dim, gt_classes)
        self.activation = torch.nn.ReLU()

    def forward(self, features, target=None):
        features = self.linear1(features)
        features = self.activation(features)
        logit = self.linear2(features).float()
        result = {
            "segs_pred": logit.argmax(-1),
        }
        if target is not None:
            target = target.long().to(logit.device)
            result["loss"] = F.cross_entropy(logit[:, 0].movedim(-1, 1).squeeze(-1), target, ignore_index=-1)

        return result
