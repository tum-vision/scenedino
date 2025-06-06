import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from . import mlp
from .resnetfc import ResnetFC
from .independent_tokens import BaseIndependentToken, make_independent_token

"""
# BTS model: The BTS model is used to predict the density field for each view of the input images. 
The predicted density fields are stored in a list.

# Stacking density fields: The predicted density fields from the BTS model are stacked along a new dimension, 
creating a tensor of shape (batch_size, num_views, height, width).

# Flattening and embedding: The density fields tensor is reshaped to (batch_size, num_views, height * width), e.g. features are stacked along the row, 
and then passed through an embedding layer that converts the density field values into a suitable format for the Transformer. 
The embedding layer is a linear layer that maps the input features to the desired dimension d_model.

# Transformer encoder: The embedded features are processed by a Transformer encoder, which consists of multiple layers
 of multi-head self-attention and feedforward sublayers. The Transformer encoder is designed to learn and capture
  relationships between the multiple views by attending to the most relevant parts of the input features. (density field as geometric consistency)
  The output of the Transformer encoder has the same shape as the input, (batch_size, num_views, d_model).

# Density field prediction: The transformed features are passed through a density field prediction layer, 
which is a sequential model containing a linear layer followed by a ReLU activation function. This layer predicts 
the accumulated density field for each pixel. The output shape is (batch_size, num_views, 1).

# Reshaping: The accumulated density field tensor is reshaped back to its original spatial dimensions (batch_size, height, width).
"""


def make_attn_layers(config, ndim: int) -> nn.Module:
    num_layers = config.get("n_layers", 3)
    n_heads = config.get("n_heads", 4)
    use_built_in = config.get("IBRAttn", False)
    if use_built_in:
        transformer_enlayer = mlp.EncoderLayer(ndim, ndim, n_heads, ndim, ndim)
        return mlp.TrEnLayer(
            transformer_enlayer, num_layers
        )  ## TODO: replace MHA module with IBRNet network and complete integretable encoder part of transformer
    elif not use_built_in:
        transformer_enlayer = TransformerEncoderLayer(
            ndim, n_heads, dim_feedforward=ndim, batch_first=True
        )
        return TransformerEncoder(transformer_enlayer, num_layers)
    else:
        raise NotImplementedError(f"__unrecognized use_built_in: {use_built_in}")


class MultiViewHead(nn.Module):
    def __init__(
        self,
        emb_encoder: nn.Module | None,
        independent_token_net: BaseIndependentToken,
        attn_layers: nn.Module,
        density_head: nn.Module,
        do_: float = 0.0,
        do_mvh: bool = False,
    ):
        """Attention based feature aggregation module for multi-view density prediction.

        Args:
            emb_encoder (nn.Module, optional): small network to compress the per view feature vectors to a lower dimensional representation. Defaults to Optional[nn.Module].
            independent_feature_net (nn.Module, optional): module to generate the view independent token from the view dependent tokens. Defaults to nn.Module.
            attn_layers (nn.Module, optional): attention layers of the module responsible for information sharing between the views. Defaults to nn.Module.
            density_head (nn.Module, optional): final network layers to predict the density from the view independent token. Defaults to nn.Module.
            do_ (float, optional): probability of dropping out a single view for training. Defaults to 0.0.
            do_mvh (bool, optional): to decide whether the first view feature map should be droppout due to pgt_loss computation. Defaults to 0.0.
        """

        super(MultiViewHead, self).__init__()
        self.emb_encoder = emb_encoder

        self.independent_token_net = independent_token_net
        self.require_bottleneck_feats = (
            self.independent_token_net.require_bottleneck_feats
        )
        self.attn_layers = attn_layers

        self.dropout = nn.Dropout(do_)
        self.do_mvh = do_mvh

        self.density_head = density_head

    def forward(
        self, sampled_features, **kwargs
    ):  ### [n_, nv_, M, C1+C_pos_emb], [nv_==2, M==100000, C==1]
        ## invalid_features: invalid features to mask the features to let model learn without occluded points in the camera's view
        invalid_features = kwargs.get("invalid_features", None)
        assert isinstance(
            invalid_features, torch.Tensor
        ), f"__The {invalid_features} is not a torch.Tensor."
        assert (
            invalid_features.dtype == torch.bool
        ), f"The elements of the {invalid_features} are not boolean."
        # invalid_features = (invalid_features > 0.5)  ## round the each of values of 3D points simply by step function within the range of std_var [0,1]

        if (
            self.dropout.p != 0 and self.do_mvh
        ):  ## dropping out except first view feature map due to pgt_loss computation
            invalid_features = torch.concat(
                [
                    invalid_features[:, :1],
                    1 - self.dropout((1 - invalid_features[:, 1:].float())),
                ],
                dim=1,
            )
        elif self.dropout.p != 0 and not self.do_mvh:
            invalid_features = 1 - self.dropout(
                (1 - invalid_features.float())
            )  ## Note: after dropping out NeuRay, the values of elements are 2. ## randomly zero out the valid sampled_features' matrix. i.e. (1-invalid_features)
        elif self.dropout.p == 0 and not self.do_mvh:
            pass
        else:
            raise NotImplementedError(
                f"__unrecognized self.dropout: {self.dropout}, self.do_mvh: {self.do_mvh} condition"
            )

        if self.emb_encoder is not None:
            encoded_features = self.emb_encoder(
                sampled_features.flatten(0, -2)
            ).reshape(
                sampled_features.shape[:-1] + (-1,)
            )  ### [M*n==100000, nv_==6, 32]   ## Embedding to Transformer arch.
        else:
            encoded_features = sampled_features.flatten(0, -2).reshape(
                sampled_features.shape[:-1] + (-1,)
            )

        ## Process the embedded features with the Transformer
        view_independent_feature = self.independent_token_net(
            encoded_features, **kwargs
        ).to(encoded_features.device)

        # padding
        padded_features = torch.concat(
            [view_independent_feature, encoded_features], dim=1
        )  ### (B*n_pts, nv_+1, 103) == ([100000, 2+1, 103]): padding along the num_token dim. B*n_pts:=Batch size or number of data points being processed.
        padded_invalid = torch.concat(  ## Note: view_independent_feature is 1st index in Tensor (:,0,:)
            [
                torch.zeros(invalid_features.shape[0], 1, device="cuda"),
                invalid_features,
            ],
            dim=1,
        )

        transformed_features = self.attn_layers(
            src=padded_features, src_key_padding_mask=padded_invalid
        )[
            :, 0, :
        ]  # [n_pts, C] ##Note: remember the tensor shape is batch-first mode, sequence length is determined by the size of the first dimension of the input tensor
        ## ## first token refers to the readout token where it stores the feature information accumulated from the layers
        ## TODO: GeoNeRF: Identify readout token belongs to single ray: M should be divisable by nhead, so that it can feed into AE, Note: make sure sampled points are in valid in the mask. (camera frustum)
        ## !TODO: Q K^T V each element of which is a density field prediction for a corresponding 3D point.
        density_field = self.density_head(transformed_features)

        return density_field

    @classmethod
    def from_conf(cls, conf, d_in, d_out):
        d_enc = conf["embedding_encoder"].get("d_out", d_in)
        embedding_encoder = mlp.make_embedding_encoder(
            conf["embedding_encoder"], d_in, d_enc
        )
        attn_layers = make_attn_layers(conf["attn_layers"], d_enc)
        independent_token = make_independent_token(conf["independent_token"], d_enc)
        probing_layer = nn.Sequential(
            nn.Linear(d_enc, d_enc // 2), nn.ELU(), nn.Linear(d_enc // 2, d_out)
        )  ## This FFNet is how the final density field scalar element is inferred.
        return cls(
            embedding_encoder,
            independent_token,
            attn_layers,
            probing_layer,
            conf.get("dropout_views_rate", 0.0),
            conf.get("dropout_multiviewhead", False),
        )


class SimpleMultiViewHead(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        do_: float = 0.0,
        do_mvh: bool = True,
    ):
        """Attention based feature aggregation module for multi-view density prediction.

        Args:
            emb_encoder (nn.Module, optional): small network to compress the per view feature vectors to a lower dimensional representation. Defaults to Optional[nn.Module].
            independent_feature_net (nn.Module, optional): module to generate the view independent token from the view dependent tokens. Defaults to nn.Module.
            attn_layers (nn.Module, optional): attention layers of the module responsible for information sharing between the views. Defaults to nn.Module.
            density_head (nn.Module, optional): final network layers to predict the density from the view independent token. Defaults to nn.Module.
            do_ (float, optional): probability of dropping out a single view for training. Defaults to 0.0.
            do_mvh (bool, optional): to decide whether the first view feature map should be droppout due to pgt_loss computation. Defaults to 0.0.
        """

        super(SimpleMultiViewHead, self).__init__()

        self.dropout = nn.Dropout(do_)
        self.do_mvh = do_mvh

        self.mlp = mlp

    def forward(
        self, sampled_features, **kwargs
    ):  ### [n_, nv_, M, C1+C_pos_emb], [nv_==2, M==100000, C==1]
        ## invalid_features: invalid features to mask the features to let model learn without occluded points in the camera's view
        invalid_features = kwargs.get("invalid_features", None)
        assert isinstance(
            invalid_features, torch.Tensor
        ), f"__The {invalid_features} is not a torch.Tensor."
        assert (
            invalid_features.dtype == torch.bool
        ), f"The elements of the {invalid_features} are not boolean."
        # invalid_features = (invalid_features > 0.5)  ## round the each of values of 3D points simply by step function within the range of std_var [0,1]

        if (
            self.dropout.p != 0 and self.do_mvh
        ):  ## dropping out except first view feature map due to pgt_loss computation
            invalid_features = torch.concat(
                [
                    invalid_features[:, :1],
                    1 - self.dropout((1 - invalid_features[:, 1:].float())),
                ],
                dim=1,
            )
        elif self.dropout.p != 0 and not self.do_mvh:
            invalid_features = 1 - self.dropout(
                (1 - invalid_features.float())
            )  ## Note: after dropping out NeuRay, the values of elements are 2. ## randomly zero out the valid sampled_features' matrix. i.e. (1-invalid_features)
        elif self.dropout.p == 0 and not self.do_mvh:
            pass
        else:
            raise NotImplementedError(
                f"__unrecognized self.dropout: {self.dropout}, self.do_mvh: {self.do_mvh} condition"
            )

        output = self.mlp(sampled_features)

        weights = torch.nn.functional.softmax(
            output[..., 0].masked_fill(invalid_features == 1, -1e9), dim=-1
        )

        density_field = torch.sum(output[..., 1:] * weights.unsqueeze(-1), dim=-2)

        return density_field

    @classmethod
    def from_conf(cls, conf, d_in, d_out):
        mlp = ResnetFC.from_conf(conf["mlp"]["args"], d_in, d_out + 1)
        return cls(
            mlp,
            conf.get("dropout_views_rate", 0.0),
            conf.get("dropout_multiviewhead", False),
        )


class MultiViewHead2(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        do_: float = 0.0,
        do_mvh: bool = True,
        attn_layers: nn.Module | None = None,
        independent_token_net: BaseIndependentToken | None = None,
        mlp2: nn.Module | None = None,
    ):
        """Attention based feature aggregation module for multi-view density prediction.

        Args:
            emb_encoder (nn.Module, optional): small network to compress the per view feature vectors to a lower dimensional representation. Defaults to Optional[nn.Module].
            independent_feature_net (nn.Module, optional): module to generate the view independent token from the view dependent tokens. Defaults to nn.Module.
            attn_layers (nn.Module, optional): attention layers of the module responsible for information sharing between the views. Defaults to nn.Module.
            density_head (nn.Module, optional): final network layers to predict the density from the view independent token. Defaults to nn.Module.
            do_ (float, optional): probability of dropping out a single view for training. Defaults to 0.0.
            do_mvh (bool, optional): to decide whether the first view feature map should be droppout due to pgt_loss computation. Defaults to 0.0.
        """

        super(MultiViewHead2, self).__init__()

        self.dropout = nn.Dropout(do_)
        self.do_mvh = do_mvh

        self.mlp = mlp

        self.attn_layers = attn_layers
        self.independent_token = independent_token_net
        self.mlp2 = mlp2

    def forward(
        self, sampled_features, **kwargs
    ):  ### [n_, nv_, M, C1+C_pos_emb], [nv_==2, M==100000, C==1]
        ## invalid_features: invalid features to mask the features to let model learn without occluded points in the camera's view
        invalid_features = kwargs.get("invalid_features", None)
        assert isinstance(
            invalid_features, torch.Tensor
        ), f"__The {invalid_features} is not a torch.Tensor."
        assert (
            invalid_features.dtype == torch.bool
        ), f"The elements of the {invalid_features} are not boolean."
        # invalid_features = (invalid_features > 0.5)  ## round the each of values of 3D points simply by step function within the range of std_var [0,1]

        if (
            self.dropout.p != 0 and self.do_mvh
        ):  ## dropping out except first view feature map due to pgt_loss computation
            invalid_features = torch.concat(
                [
                    invalid_features[:, :1],
                    1 - self.dropout((1 - invalid_features[:, 1:].float())),
                ],
                dim=1,
            )
        elif self.dropout.p != 0 and not self.do_mvh:
            invalid_features = 1 - self.dropout(
                (1 - invalid_features.float())
            )  ## Note: after dropping out NeuRay, the values of elements are 2. ## randomly zero out the valid sampled_features' matrix. i.e. (1-invalid_features)
        elif self.dropout.p == 0 and not self.do_mvh:
            pass
        else:
            raise NotImplementedError(
                f"__unrecognized self.dropout: {self.dropout}, self.do_mvh: {self.do_mvh} condition"
            )

        encoded_features = self.mlp(sampled_features)

        if self.independent_token is not None:
            view_independent_feature = self.independent_token(
                encoded_features, **kwargs
            ).to(encoded_features.device)

            # padding
            encoded_features = torch.concat(
                [view_independent_feature, encoded_features], dim=1
            )  ### (B*n_pts, nv_+1, 103) == ([100000, 2+1, 103]): padding along the num_token dim. B*n_pts:=Batch size or number of data points being processed.
            invalid_features = torch.concat(  ## Note: view_independent_feature is 1st index in Tensor (:,0,:)
                [
                    torch.zeros(invalid_features.shape[0], 1, device="cuda"),
                    invalid_features,
                ],
                dim=1,
            )

        if self.attn_layers is not None:
            encoded_features = self.attn_layers(
                encoded_features, src_key_padding_mask=invalid_features
            )

        if self.independent_token is not None:
            if self.mlp2 is not None:
                return self.mlp2(encoded_features[..., 0, :])
            else:
                return encoded_features[..., 0, 1:]
        else:
            if self.mlp2 is not None:
                encoded_features = self.mlp2(encoded_features)

            weights = torch.nn.functional.softmax(
                encoded_features[..., 0].masked_fill(invalid_features == 1, -1e9),
                dim=-1,
            )
            return torch.sum(encoded_features[..., 1:] * weights.unsqueeze(-1), dim=-2)

        # return density_field

    @classmethod
    def from_conf(cls, conf, d_in, d_out):
        if conf["mlp2"] is not None:
            d_out_mlp = conf["mlp2"]["d_in"]
        else:
            d_out_mlp = d_out + 1
        mlp = ResnetFC.from_conf(conf["mlp"]["args"], d_in, d_out_mlp)

        if conf["attn_layers"] is not None:
            attn_layers = make_attn_layers(conf["attn_layers"], d_out_mlp)
        else:
            attn_layers = None

        if conf["independent_token"] is not None:
            independent_token = make_independent_token(
                conf["independent_token"], d_out_mlp
            )
        else:
            independent_token = None

        if conf["mlp2"] is not None:
            if conf["independent_token"] is not None:
                d_out_mlp2 = d_out
            else:
                d_out_mlp2 = d_out + 1
            mlp2 = ResnetFC.from_conf(conf["mlp2"]["args"], d_out_mlp, d_out_mlp2)
        else:
            mlp2 = None

        return cls(
            mlp,
            conf.get("dropout_views_rate", 0.0),
            conf.get("dropout_multiviewhead", False),
            attn_layers,
            independent_token,
            mlp2,
        )


class MultiViewHead3(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        mlp2: nn.Module,
        do_: float = 0.0,
        do_mvh: bool = True,
    ):
        """Attention based feature aggregation module for multi-view density prediction.

        Args:
            emb_encoder (nn.Module, optional): small network to compress the per view feature vectors to a lower dimensional representation. Defaults to Optional[nn.Module].
            independent_feature_net (nn.Module, optional): module to generate the view independent token from the view dependent tokens. Defaults to nn.Module.
            attn_layers (nn.Module, optional): attention layers of the module responsible for information sharing between the views. Defaults to nn.Module.
            density_head (nn.Module, optional): final network layers to predict the density from the view independent token. Defaults to nn.Module.
            do_ (float, optional): probability of dropping out a single view for training. Defaults to 0.0.
            do_mvh (bool, optional): to decide whether the first view feature map should be droppout due to pgt_loss computation. Defaults to 0.0.
        """

        super(MultiViewHead3, self).__init__()

        self.dropout = nn.Dropout(do_)
        self.do_mvh = do_mvh

        self.mlp = mlp

        self.mlp2 = mlp2

    def forward(
        self, sampled_features, **kwargs
    ):  ### [n_, nv_, M, C1+C_pos_emb], [nv_==2, M==100000, C==1]
        ## invalid_features: invalid features to mask the features to let model learn without occluded points in the camera's view
        invalid_features = kwargs.get("invalid_features", None)
        assert isinstance(
            invalid_features, torch.Tensor
        ), f"__The {invalid_features} is not a torch.Tensor."
        assert (
            invalid_features.dtype == torch.bool
        ), f"The elements of the {invalid_features} are not boolean."
        # invalid_features = (invalid_features > 0.5)  ## round the each of values of 3D points simply by step function within the range of std_var [0,1]

        if (
            self.dropout.p != 0 and self.do_mvh
        ):  ## dropping out except first view feature map due to pgt_loss computation
            invalid_features = torch.concat(
                [
                    invalid_features[:, :1],
                    1 - self.dropout((1 - invalid_features[:, 1:].float())),
                ],
                dim=1,
            )
        elif self.dropout.p != 0 and not self.do_mvh:
            invalid_features = 1 - self.dropout(
                (1 - invalid_features.float())
            )  ## Note: after dropping out NeuRay, the values of elements are 2. ## randomly zero out the valid sampled_features' matrix. i.e. (1-invalid_features)
        elif self.dropout.p == 0 and not self.do_mvh:
            pass
        else:
            raise NotImplementedError(
                f"__unrecognized self.dropout: {self.dropout}, self.do_mvh: {self.do_mvh} condition"
            )

        encoded_features = self.mlp(sampled_features)

        weights = torch.nn.functional.softmax(
            encoded_features[..., 0].masked_fill(invalid_features == 1, -1e9), dim=-1
        )

        density_feature = torch.sum(
            encoded_features[..., 1:] * weights.unsqueeze(-1), dim=-2
        )

        return self.mlp2(density_feature)

    @classmethod
    def from_conf(cls, conf, d_in, d_out):
        mlp = ResnetFC.from_conf(conf["mlp"]["args"], d_in, conf["mlp2"]["d_in"] + 1)

        mlp2 = ResnetFC.from_conf(conf["mlp2"]["args"], conf["mlp2"]["d_in"], d_out)

        return cls(
            mlp,
            mlp2,
            conf.get("dropout_views_rate", 0.0),
            conf.get("dropout_multiviewhead", False),
        )
