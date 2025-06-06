import torch
import torch.nn.functional as F
import torchvision


class BilinearDownsampler(torch.nn.Module):
    def __init__(
        self,
        patch_size,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, tuple):
            self.patch_size = patch_size

    def forward(self, x, mode):
        n, v, h, w, _, c = x.shape

        assert h % self.patch_size[0] == 0
        target_h = h // self.patch_size[0]
        assert w % self.patch_size[1] == 0
        target_w = w // self.patch_size[1]

        x = x.permute(0, 1, 4, 5, 2, 3).flatten(0, 2)
        x = F.interpolate(x, size=(target_h, target_w), mode="bilinear")
        x = x.reshape(n, v, -1, c, target_h, target_w).permute(0, 1, 4, 5, 2, 3)
        return x.squeeze(2, 3)


class PatchSalienceDownsampler(torch.nn.Module):
    def __init__(
        self,
        channels,
        patch_size,
        normalize_features,
    ):
        super().__init__()

        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, tuple):
            self.patch_size = patch_size

        self.conv = torch.nn.Conv2d(channels, 1, kernel_size=1)
        self.patch_weight = torch.nn.Parameter(torch.ones(self.patch_size))
        self.patch_bias = torch.nn.Parameter(torch.zeros(self.patch_size))

        self.normalize_features = normalize_features

        torch.nn.init.kaiming_normal_(self.conv.weight, a=0, mode="fan_in")
        torch.nn.init.zeros_(self.conv.bias)
        torch.nn.init.normal_(self.patch_weight, mean=1.0, std=0.01)
        torch.nn.init.normal_(self.patch_bias, mean=0.0, std=0.01)

    def forward(self, x, mode):
        if mode == "patch":
            return self.forward_patches(x)

        elif mode == "image":
            n, v, h, w, _, c = x.shape
            patch_h, patch_w = self.patch_size[0], self.patch_size[1]
            no_patches_h, no_patches_w = h // patch_h, w // patch_w

            patches = x.reshape(n, v, no_patches_h, patch_h, no_patches_w, patch_w, 1, c)
            patches = patches.swapaxes(3, 4).flatten(1, 3)

            patched_result, salience_map, weight_map, patch_weight_bias = self.forward_patches(patches)
            patched_result = patched_result.reshape(n, v, no_patches_h, no_patches_w, 1, c)

            salience_map = salience_map.reshape(n, v, no_patches_h, no_patches_w, patch_h, patch_w, 1, 1)
            salience_map = salience_map.swapaxes(3, 4).reshape(n, v, h, w, 1, 1)

            weight_map = weight_map.reshape(n, v, no_patches_h, no_patches_w, patch_h, patch_w, 1, 1)
            weight_map = weight_map.swapaxes(3, 4).reshape(n, v, h, w, 1, 1)

            return patched_result, salience_map, weight_map, patch_weight_bias
        else:
            return None

    def forward_patches(self, x):
        n, p, patch_h, patch_w, _, c = x.shape
        x_flat = x.reshape(-1, patch_h, patch_w, c).permute(0, 3, 1, 2)

        salience_map = self.conv(x_flat).squeeze(1)

        weight_map = salience_map * self.patch_weight + self.patch_bias
        weight_map = torch.nn.functional.softmax(weight_map.reshape(-1, patch_h * patch_w), dim=1)
        weight_map = weight_map.reshape(n, p, patch_h, patch_w, 1, 1)

        patched_features = torch.sum(weight_map * x, dim=(2, 3))
        if self.normalize_features:
            patched_features = patched_features / torch.linalg.norm(patched_features, dim=-1, keepdim=True)

        return (patched_features,
                salience_map.reshape(n, p, patch_h, patch_w, 1, 1),
                weight_map,
                torch.cat([self.patch_weight, self.patch_bias], dim=1))
