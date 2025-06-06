from bts.losses.base_loss import BaseLoss
import torch
import torch.nn.functional as F


class StegoLoss(BaseLoss):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.random_weight = config.get("random_weight", 1.0)
        self.knn_weight = config.get("knn_weight", 1.0)
        self.self_weight = config.get("self_weight", 1.0)

        self.random_shift = config.get("random_shift", 0.0)
        self.knn_shift = config.get("knn_shift", 0.0)
        self.self_shift = config.get("self_shift", 0.0)

        self.pointwise = config.get("pointwise", True)

    def get_loss_metric_names(self) -> list[str]:
        return [
            "total_loss", 
            "self_loss", "knn_loss", "random_loss", 
            "direct_cluster_loss", "direct_linear_loss", "stego_cluster_loss", "stego_linear_loss"
        ]

    def __call__(self, data) -> dict[str, torch.Tensor]:

        if "stego_corr" not in data["segmentation"]:
            self_loss, knn_loss, random_loss, total_loss = 0, 0, 0, 0
        else:
            dino_self_corr = data["segmentation"]["stego_corr"]["dino_self_corr"]
            stego_self_corr = data["segmentation"]["stego_corr"]["stego_self_corr"]

            dino_nn_corr = data["segmentation"]["stego_corr"]["dino_nn_corr"]
            stego_nn_corr = data["segmentation"]["stego_corr"]["stego_nn_corr"]

            dino_random_corr = data["segmentation"]["stego_corr"]["dino_random_corr"]
            stego_random_corr = data["segmentation"]["stego_corr"]["stego_random_corr"]

            self_loss = self._compute_stego_loss(dino_self_corr, stego_self_corr,
                                                self.self_weight, self.self_shift)
            knn_loss = self._compute_stego_loss(dino_nn_corr, stego_nn_corr,
                                                self.knn_weight, self.knn_shift)
            random_loss = self._compute_stego_loss(dino_random_corr, stego_random_corr,
                                                self.random_weight, self.random_shift)
            total_loss = self_loss + knn_loss + random_loss
        
        direct_cluster_loss = data["segmentation"]["results"]["direct_cluster"].get("loss", 0.0)
        stego_cluster_loss = data["segmentation"]["results"]["stego_cluster"].get("loss", 0.0)

        # If linear heads present
        direct_linear_loss = data["segmentation"]["results"].get("direct_linear", {}).get("loss", 0.0)
        stego_linear_loss = data["segmentation"]["results"].get("stego_linear", {}).get("loss", 0.0)

        total_loss += direct_cluster_loss + direct_linear_loss + stego_cluster_loss + stego_linear_loss

        losses = {
            "total_loss": total_loss,

            "self_loss": self_loss,
            "knn_loss": knn_loss,
            "random_loss": random_loss,

            "direct_cluster_loss": direct_cluster_loss,
            "direct_linear_loss": direct_linear_loss,
            "stego_cluster_loss": stego_cluster_loss,
            "stego_linear_loss": stego_linear_loss,
        }
        return losses

    def _compute_stego_loss(self, dino_corr, stego_corr, weight, shift):
        if self.pointwise:
            old_mean = dino_corr.mean()
            dino_corr -= dino_corr.mean(dim=-1, keepdim=True)
            dino_corr = dino_corr - dino_corr.mean() + old_mean

        loss = -weight * stego_corr.clamp(0) * (dino_corr - shift)

        return loss.mean()
