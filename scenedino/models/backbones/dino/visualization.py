from pykeops.torch import LazyTensor
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor


class VisualizationModule(nn.Module):
    def __init__(self, in_channels, reduce_images=3):
        super().__init__()
        self.batch_rgb_mean = torch.zeros(in_channels)
        self.batch_rgb_comp = torch.eye(in_channels, 3)
        self.reduce_images = reduce_images
        self.fitted_pca = False

        self.n_kmeans_clusters = 8
        self.kmeans_cluster_centers = torch.zeros(self.n_kmeans_clusters, in_channels)
        self.cmap_kmeans = plt.get_cmap("tab10")

    def fit_pca(self, batch_features, refit):
        if batch_features.dim() > 2:
            raise ValueError(f"Wrong dims for PCA: {batch_features.shape}")
        if not self.fitted_pca or refit:
            # filter nan values
            batch_features = batch_features[~torch.isnan(batch_features).any(dim=1)]
            self._pca_fast(batch_features, num_components=3*self.reduce_images)
            self.fitted_pca = True

    def transform_pca(self, features, norm, from_dim):
        features = features - self.batch_rgb_mean
        if norm:
            features = features / torch.linalg.norm(features, dim=-1, keepdim=True)
        return features @ self.batch_rgb_comp[..., from_dim:from_dim+3]

    def _pca_fast(self, data: Tensor, num_components: int = 3) -> Tuple[Tensor, Tensor]:
        """Function implements PCA using PyTorch fast low-rank approximation.

        Args:
            data (Tensor): Data matrix of the shape [N, C] or [B, N, C].
            num_components (int): Number of principal components to be used.

        Returns:
            data_pca (Tensor): Transformed low-dimensional data of the shape [N, num_components] or [B, N, num_components].
            pca_components (Tensor): Principal components of the shape [num_components, C] or [B, num_components, C].
        """
        # Normalize data
        data_mean = data.mean(dim=-2, keepdim=True)
        data_normalize = (data - data_mean) / (data.std(dim=-2, keepdim=True) + 1e-08)
        # Perform fast low-rank PCA
        u, _, v = torch.pca_lowrank(data_normalize, q=max(num_components, 6), niter=2, center=True)
        v = v.transpose(-1, -2)
        # Perform SVD flip
        u, v = self._svd_flip(u, v)  # type: Tensor, Tensor
        # Transpose PCA components to match scikit-learn
        if data_normalize.ndim == 2:
            pca_components = v[:num_components]
        else:
            pca_components = v[:, :num_components]

        self.batch_rgb_mean = data_mean
        self.batch_rgb_comp = pca_components.transpose(-1, -2)

    def _svd_flip(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform SVD flip to solve sign issue of SVD.

        Args:
            u (Tensor): u matrix of the shape [N, C] or [B, N, C].
            v (Tensor): v matrix of the shape [C, C] or [B, C, C].

        Returns:
            u (Tensor): Fixed u matrix of the shape [N, C] or [B, N, C].
            v (Tensor): Fixed v matrix of the shape [C, C] or [B, C, C].
        """
        max_abs: Tensor = torch.abs(u).argmax(dim=-2)
        indexes: Tensor = torch.arange(u.shape[-1], device=u.device)
        if u.ndim == 2:
            signs: Tensor = torch.sign(u[max_abs, indexes])
            u = u * signs
            v = v * signs.unsqueeze(dim=-1)
        else:
            # Maybe fix looping the future...
            signs = torch.stack(
                [torch.sign(u[batch_index, max_abs[batch_index], indexes]) for batch_index in range(u.shape[0])], dim=0
            )
            u = u * signs.unsqueeze(dim=1)
            v = v * signs.unsqueeze(dim=-1)
        return u, v

    def old_fit_transform_kmeans_batch(self, batch_features, subsample_size=20000):
        feats_map_flattened = batch_features.flatten(1, -2)
        from torch_kmeans import KMeans, CosineSimilarity
        kmeans_engine = KMeans(n_clusters=self.n_kmeans_clusters, distance=CosineSimilarity)

        n = feats_map_flattened.size(1)
        if subsample_size is not None and subsample_size < n:
            indices = torch.randperm(n)[:subsample_size]
            feats_map_subsampled = feats_map_flattened[:, indices]
            kmeans_engine.fit(feats_map_subsampled)
        else:
            kmeans_engine.fit(feats_map_flattened)

        labels = kmeans_engine.predict(feats_map_flattened)
        labels = labels.reshape(batch_features.shape[:-1]).float().cpu().numpy()

        label_map = self.cmap_kmeans(labels / (self.n_kmeans_clusters - 1))[..., :3]
        label_map = torch.Tensor(label_map).squeeze(-2)

        return label_map

    def fit_transform_kmeans_batch(self, batch_features):
        feats_map_flattened = batch_features.flatten(0, -2)

        with torch.no_grad():
            cl, c = self._KMeans_cosine(feats_map_flattened.float(), K=self.n_kmeans_clusters)
        self.kmeans_cluster_centers = c

        labels = cl.reshape(batch_features.shape[:-1]).float().cpu().numpy()

        label_map = self.cmap_kmeans(labels / (self.n_kmeans_clusters - 1))[..., :3]
        label_map = torch.Tensor(label_map).squeeze(-2)

        return label_map

    def _KMeans_cosine(self, x, K=19, Niter=100):
        """Implements Lloyd's algorithm for the Cosine similarity metric."""
        N, D = x.shape  # Number of samples, dimension of the ambient space

        c = x[:K, :].clone()  # Simplistic initialization for the centroids
        # Normalize the centroids for the cosine similarity:
        c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

        x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
        c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

        # K-means loop:
        # - x  is the (N, D) point cloud,
        # - cl is the (N,) vector of class labels
        # - c  is the (K, D) cloud of cluster centroids
        for i in range(Niter):
            # E step: assign points to the closest cluster -------------------------
            S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
            cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

            # M step: update the centroids to the normalized cluster average: ------
            # Compute the sum of points per cluster:
            c.zero_()
            c.scatter_add_(0, cl[:, None].repeat(1, D), x)

            # Normalize the centroids, in place:
            c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

        return cl, c