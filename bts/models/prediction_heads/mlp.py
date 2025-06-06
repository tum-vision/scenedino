from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bts.common import util


class ImplicitNet(nn.Module):
    """
    Represents a MLP;
    Original code from IGR
    """

    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        d_out=4,
        geometric_init=True,
        radius_init=0.3,
        beta=0.0,
        output_init_gain=2.0,
        num_position_inputs=3,
        sdf_scale=1.0,
        dim_excludes_skip=False,
        combine_layer=1000,
        combine_type="average",
    ):
        """
        :param d_in input size
        :param dims dimensions of hidden layers. Num hidden layers == len(dims)
        :param skip_in layers with skip connections from input (residual)
        :param d_out output size
        :param geometric_init if true, uses geometric initialization
               (to SDF of sphere)
        :param radius_init if geometric_init, then SDF sphere will have
               this radius
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        :param output_init_gain output layer normal std, only used for
                                output dimension >= 1, when d_out >= 1
        :param dim_excludes_skip if true, dimension sizes do not include skip
        connections
        """
        super().__init__()

        dims = [d_in] + dims + [d_out]
        if dim_excludes_skip:
            for i in range(1, len(dims) - 1):
                if i in skip_in:
                    dims[i] += d_in

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.dims = dims
        self.combine_layer = combine_layer
        self.combine_type = combine_type

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            # if true preform geometric initialization
            if geometric_init:
                if layer == self.num_layers - 2:
                    # Note our geometric init is negated (compared to IDR)
                    # since we are using the opposite SDF convention:
                    # inside is +
                    nn.init.normal_(
                        lin.weight[0],
                        mean=-np.sqrt(np.pi) / np.sqrt(dims[layer]) * sdf_scale,
                        std=0.00001,
                    )
                    nn.init.constant_(lin.bias[0], radius_init)
                    if d_out > 1:
                        # More than SDF output
                        nn.init.normal_(lin.weight[1:], mean=0.0, std=output_init_gain)
                        nn.init.constant_(lin.bias[1:], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                if d_in > num_position_inputs and (layer == 0 or layer in skip_in):
                    # Special handling for input to allow positional encoding
                    nn.init.constant_(lin.weight[:, -d_in + num_position_inputs :], 0.0)
            else:
                nn.init.constant_(lin.bias, 0.0)
                nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

            setattr(self, "lin" + str(layer), lin)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            # Vanilla ReLU
            self.activation = nn.ReLU()

    def forward(self, x, combine_inner_dims=(1,)):
        """
        :param x (..., d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        x_init = x
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer == self.combine_layer:
                x = util.combine_interleaved(x, combine_inner_dims, self.combine_type)
                x_init = util.combine_interleaved(
                    x_init, combine_inner_dims, self.combine_type
                )

            if layer < self.combine_layer and layer in self.skip_in:
                x = torch.cat([x, x_init], -1) / np.sqrt(2)

            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x

    @classmethod
    def from_conf(cls, conf, d_in, d_out):
        return cls(d_in=d_in, d_out=d_out, **conf)

    # @classmethod
    # def from_conf(cls, conf, d_in, **kwargs):
    #     # PyHocon construction
    #     return cls(
    #         d_in,
    #         conf.get_list("dims"),
    #         skip_in=conf.get_list("skip_in"),
    #         beta=conf.get_float("beta", 0.0),
    #         dim_excludes_skip=conf.get_bool("dim_excludes_skip", False),
    #         combine_layer=conf.get_int("combine_layer", 1000),
    #         combine_type=conf.get_string("combine_type", "average"),  # average | max
    #         **kwargs,
    #     )


"""
GeoNeRF
https://github.com/idiap/GeoNeRF/blob/e6249fdae5672853c6bbbd4ba380c4c166d02c95/model/self_attn_renderer.py#L60
"""


# Custom TransposeLayer to perform transpose operation
class TransposeLayer(nn.Module):
    def __init__(self):
        super(TransposeLayer, self).__init__()

    def forward(self, x):
        print("x_shape before transpose: ", x.shape)
        return x.transpose(1, 2)


#
# class CNN2AE(nn.Module):
#     def __init__(self, num_channels, num_features, desired_spatial_output): ## reduced mapping: num_points |-> num_features
#         super(CNN2AE, self).__init__()
#         self.conv1 = nn.Conv1d(num_channels, num_channels*2, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(num_channels*4, num_channels*8, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
#         self.desired_spatial_output = desired_spatial_output
#         # self.fc = nn.Linear(num_channels*4 * num_features, num_features)  # Fully connected layer to further reduce dimension
#         # self.fc = nn.Linear(num_channels*4 * (num_features // 4), num_channels)  # Fully connected layer to reduce dimension
#
#     def forward(self, x):   ## input_tensor's shape: (batch_size=1, C=num_channels, M=num_points)
#         _, num_channels, num_features = x.shape
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         x = self.pool(nn.functional.relu(self.conv3(x)))
#         x = x.view(x.size(0), num_channels, self.desired_spatial_output)  # Reshape to (batch_size, num_channels, reduced_features)
#         return x

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Use GPU if available, else CPU


class CNN2AE(
    nn.Module
):  ## convolute density sampled features along a ray from end of cam's frustum to the end. ( n_coarse==16 x att_feat==32 x (8x8) )
    def __init__(self, num_channels: int = 32, num_features: int = 64):
        super(CNN2AE, self).__init__()
        self.n_coarse = num_features
        self.conv1 = nn.Conv1d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1
        )
        # self.conv2 = nn.Conv1d(num_channels*2, num_channels*4, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        # self.fc = nn.Linear(num_channels * num_features, num_features)  # Fully connected layer to further reduce dimension
        # self.fc = None  # We will initialize this later

    def forward(self, x):  ## , desired_spatial_output):
        assert (
            x.size(0) % self.n_coarse
        ) == 0, f"__given points should be dividable by n_coarse: {self.n_coarse},but points given: {x.size(0)}"
        # x = x.to(device)  # Move the input data to the device
        # B_, C_, M_ = x.shape  # Get the new number of channels and points
        x = self.pool(F.relu(self.conv1(x)))  # Apply first conv layer and pool
        x = self.pool(F.relu(self.conv1(x)))  # Apply second conv layer and pool

        # if self.fc is None:
        #     # Initialize the fully connected layer now that we know the input size
        #     self.fc = nn.Linear(C_ * M_, C_ * desired_spatial_output).to(device)

        # x = x.view(B_, C_ * M_)  # Reshape to (batch_size, C * M)
        # x = self.fc(x)  # Apply fully connected layer
        # x = x.view(B_, C_, desired_spatial_output)  # Reshape to (batch_size, num_channels, desired_spatial_output)
        return x


## Auto-encoder network
class ConvAutoEncoder(nn.Module):  ## purpose: to enforce the geometric generalization
    def __init__(
        self, num_ch: int = 32, S_: int = 64
    ):  ## S:= Sequence length of the input tensor. i.e. nb_samples_per_ray
        super(ConvAutoEncoder, self).__init__()
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_ch, num_ch * 2, 3, stride=1, padding=1),
            # TransposeLayer(),  # Use the custom TransposeLayer to transpose the output
            nn.LayerNorm(
                S_, elementwise_affine=False
            ),  ## RuntimeError: Given normalized_shape=[64], expected input with shape [*, 64], but got input of size[1, 64, 100000]
            nn.ELU(alpha=1.0, inplace=True),
            # TransposeLayer(),  # Use the custom TransposeLayer to transpose the output
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch * 4, 3, stride=1, padding=1),
            # TransposeLayer(),  # Use the custom TransposeLayer to transpose the output
            nn.LayerNorm(S_ // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            # TransposeLayer(),  # Use the custom TransposeLayer to transpose the output
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_ch * 4, num_ch * 4, 3, stride=1, padding=1),
            # TransposeLayer(),  # Use the custom TransposeLayer to transpose the output
            nn.LayerNorm(S_ // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            # TransposeLayer(),  # Use the custom TransposeLayer to transpose the output
            nn.MaxPool1d(2),
        )

        # Decoder
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch * 4, 4, stride=2, padding=1),
            nn.LayerNorm(S_ // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 8, num_ch * 2, 4, stride=2, padding=1),
            nn.LayerNorm(S_ // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch, 4, stride=2, padding=1),
            nn.LayerNorm(S_, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        # Output
        self.conv_out = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch, 3, stride=1, padding=1),
            nn.LayerNorm(S_, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        conv1_out = x
        x = self.conv2(x)
        conv2_out = x
        x = self.conv3(x)

        x = self.t_conv1(x)
        x = self.t_conv2(torch.cat([x, conv2_out], dim=1))
        x = self.t_conv3(torch.cat([x, conv1_out], dim=1))

        x = self.conv_out(torch.cat([x, input], dim=1))

        return x


"""
Transformer encoder part from IBRNet network
https://github.com/googleinterns/IBRNet/blob/master/ibrnet/mlp_network.py
"""


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(
            q / self.temperature, k.transpose(2, 3)
        )  ### ?? [32768, 4, 7, 7]

        if mask is not None:  ### [32768, 1, 7]
            mask = mask.unsqueeze(-1)  ##
            mask = mask.expand(
                -1, attn.shape[1], -1, attn.shape[-1]
            )  ##  TODO: matrix should be investiated to validate the operator
            mask = 1.0 - (
                (1.0 - mask) * (1.0 - mask.transpose(-2, -1))
            )  ### As being symmetric of the mask matrix => the info of masked info won't give result: 2 problems: 1) computation bottleneck demand, eval_batch_size=25000 decreasing (setup pipeline using smaller pipeline nerf.py)
            attn = attn.masked_fill(
                mask == 1, -1e9
            )  ## masking should be done when the value of invalidity as boolean is 1 by making the value of element zero (numerical stability)
            # attn = attn * mask
            """
            def masked_fill(self, mask, value):
                result = self.clone()  # Start with a copy of the original data
                result[mask] = value   # Replace values where the mask is true
                return result
            """

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class PoswiseFF_emb4enc(nn.Module):
    """A two-feed-forward-layer module (tailored to encoder for DFT model's input) inspired code from Transformer's encoder"""

    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_out)  # position-wise
        self.w_match = nn.Linear(d_in, d_out)  # position-wise
        # self.post_layer_norm = nn.LayerNorm(d_out, eps=1e-6)
        self.pre_layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # embedding for residual input
        emb_residual = self.w_match(x)

        # Pre-layer normalization
        x = self.pre_layer_norm(x)

        # Transform the (normalized) input
        x = self.w_2(
            F.elu(self.w_1(x))
        )  ## default: ReLU | or F.leaky_relu, LeakyReLU used to handle dying gradients, espeically when dense outputs are expected, so that it wouldn't lose expressiveness for Transformer due to lack of info
        # x = self.dropout(x)

        # Post-layer normaliation
        # x = self.post_layer_norm(x)

        # Residual connection
        x += emb_residual

        return x


class PreLNPositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.layer_norm(x)

        x = self.w_2(F.leaky_relu(self.w_1(x)))  ## default: F.relu
        # x = self.dropout(x)
        x += residual

        return x


def make_embedding_encoder(
    config, input_channels: int, output_channels: int
) -> Optional[nn.Module]:
    emb_enc_type = config.get("type", "none")
    non_linearity = nn.ELU()  # make configurable
    if emb_enc_type == "none":
        return None
    elif emb_enc_type == "pwf":
        return PoswiseFF_emb4enc(input_channels, 2 * output_channels, output_channels)
    elif emb_enc_type == "ff":
        return nn.Sequential(
            nn.Linear(input_channels, 2 * output_channels, bias=True),
            non_linearity,
            nn.Linear(2 * output_channels, output_channels, bias=True),
        )  ## default: ReLU |  nn.LeakyReLU()
    elif emb_enc_type == "ffh":
        return nn.Sequential(
            nn.Linear(input_channels, output_channels, bias=True)
        )  ## default: ReLU |  nn.LeakyReLU()
    elif emb_enc_type == "hpwf":
        return nn.Sequential(  ## == mlp.PositionwiseFeedForward
            nn.Linear(input_channels, 2 * output_channels, bias=True),
            non_linearity,
            nn.LayerNorm(2 * output_channels, eps=1e-6),
            nn.Linear(2 * output_channels, output_channels, bias=True),
        )
    else:
        raise NotImplementedError(
            "__unrecognized input for emb_enc, not using an embedding encoder."
        )
        return None


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PreLNMultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.layer_norm(q)
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        return q, attn


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(
        self, d_model, d_inner, n_head, d_k, d_v, dropout=0, pre_ln: bool = False
    ):
        super(EncoderLayer, self).__init__()
        if pre_ln:
            self.slf_attn = PreLNMultiHeadAttention(
                n_head, d_model, d_k, d_v, dropout=dropout
            )
            self.pos_ffn = PreLNPositionwiseFeedForward(
                d_model, d_inner, dropout=dropout
            )
        else:
            self.slf_attn = MultiHeadAttention(
                n_head, d_model, d_k, d_v, dropout=dropout
            )
            self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


"""(modified) Transformer arch from Pytorch library
to be compatible with nn.TransformerEncoder() as input arg"""


class TrEnLayer(nn.Module):
    r"""
    Args:
    encoder_layer: an instance of the TransformerEncoderLayer() class (required).
    num_layers: the number of sub-encoder-layers in the encoder (required).
    norm: the layer normalization component (optional).
    enable_nested_tensor: if True, input will automatically convert to nested tensor
        (and convert back on output). This will improve the overall performance of
        TransformerEncoder when padding rate is high. Default: ``True`` (enabled).
    """

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,
    ):
        super(TrEnLayer, self).__init__()
        # self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(num_layers)])
        self.layers = TTF._get_clones(encoder_layer, num_layers)  ## deep copy
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"

        # if not isinstance(first_layer, EncoderLayer):
        #     why_not_sparsity_fast_path = f"{str_first_layer} was not IBR EncoderLayer"
        # elif first_layer.norm_first :
        #     why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        # elif first_layer.training:
        #     why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        # elif not first_layer.self_attn.batch_first:
        #     why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        # elif not first_layer.self_attn._qkv_same_embed_dim:
        #     why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        # elif not first_layer.activation_relu_or_gelu:
        #     why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        # elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
        #     why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        # elif not src.dim() == 3:
        #     why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        # elif not self.enable_nested_tensor:
        #     why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        # elif src_key_padding_mask is None:
        #     why_not_sparsity_fast_path = "src_key_padding_mask was None"
        # elif (((not hasattr(self, "mask_check")) or self.mask_check)
        #         and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
        #     why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        # elif output.is_nested:
        #     why_not_sparsity_fast_path = "NestedTensor input is not supported"
        # elif mask is not None:
        #     why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        # elif first_layer.self_attn.num_heads % 2 == 1:
        #     why_not_sparsity_fast_path = "num_head is odd"
        # elif torch.is_autocast_enabled():
        #     why_not_sparsity_fast_path = "autocast is enabled"
        #
        # if not why_not_sparsity_fast_path:
        #     tensor_args = (
        #         src,
        #         first_layer.self_attn.in_proj_weight,
        #         first_layer.self_attn.in_proj_bias,
        #         first_layer.self_attn.out_proj.weight,
        #         first_layer.self_attn.out_proj.bias,
        #         first_layer.norm1.weight,
        #         first_layer.norm1.bias,
        #         first_layer.norm2.weight,
        #         first_layer.norm2.bias,
        #         first_layer.linear1.weight,
        #         first_layer.linear1.bias,
        #         first_layer.linear2.weight,
        #         first_layer.linear2.bias,
        #     )
        #
        #     if torch.overrides.has_torch_function(tensor_args):
        #         why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
        #     elif not (src.is_cuda or 'cpu' in str(src.device)):
        #         why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
        #     elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
        #         why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
        #                                       "input/output projection weights or biases requires_grad")
        #
        #     if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
        #         convert_to_nested = True
        #         output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
        #         src_key_padding_mask_for_layers = None

        for mod in self.layers:
            # output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)
            output = mod(output, slf_attn_mask=src_key_padding_mask_for_layers)[0]

        if convert_to_nested:
            output = output.to_padded_tensor(0.0)

        if self.norm is not None:
            output = self.norm(output)

        return output


# class TrEnLayer(torch.nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
#                  activation="relu", batch_first=True, norm_first=False,
#                  activation_relu_or_gelu=True):
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
#
#         self.norm1 = torch.nn.LayerNorm(d_model)
#         self.norm2 = torch.nn.LayerNorm(d_model)
#         self.dropout1 = torch.nn.Dropout(dropout)
#         self.dropout2 = torch.nn.Dropout(dropout)
#
#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             self.activation = _get_activation_fn(activation)
#         else:
#             self.activation = activation
#
#         self.pos_ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
#
#         self.self_attn.batch_first = batch_first
#         self.self_attn._qkv_same_embed_dim = True  # assuming d_model is the same for query, key, value
#         self.norm_first = norm_first
#         self.activation_relu_or_gelu = activation_relu_or_gelu
#
#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         src2 = self.self_attn(src, src, src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         if self.norm_first:
#             src = src + self.dropout1(src2)
#             src = self.norm1(src)
#             src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#             src = src + self.dropout2(src2)
#             src = self.norm2(src)
#         else:
#             src = self.norm1(src)
#             src = src + self.dropout1(src2)
#             src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#             src = self.norm2(src)
#             src = src + self.dropout2(src2)
#         return src

# '''
# c.f. nn.transformer.py
# '''
# def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
#     if activation == "relu":
#         return F.relu
#     elif activation == "gelu":
#         return F.gelu
#
#     raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
#
# def _get_clones(module, N):
#     return ModuleList([copy.deepcopy(module) for i in range(N)])
