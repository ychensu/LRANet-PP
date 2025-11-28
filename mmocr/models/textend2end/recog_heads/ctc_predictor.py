import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_
from mmocr.models.builder import ENCODERS
from mmcv.cnn import  ConvModule

class GELU(nn.Module):

    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):

    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'hard_sigmoid':
            self.act = nn.Hardsigmoid(inplace)
        elif act_type == 'hard_swish':
            self.act = nn.Hardswish(inplace=inplace)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=inplace)
        elif act_type == 'gelu':
            self.act = GELU(inplace=inplace)
        elif act_type == 'swish':
            self.act = Swish(inplace=inplace)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)


def drop_path(x,
              drop_prob: float = 0.0,
              training: bool = False,
              scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = self.norm1(x + self.drop_path(self.attn(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x



class SubSample1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, [H, W]


class IdentitySize(nn.Module):

    def forward(self, x, sz):
        return x, sz


class CTCPredictor(nn.Module):

    def __init__(self,
                 in_channels=512,
                 out_channels=97,
                 **kwargs):
        super(CTCPredictor, self).__init__()
        self.char_token = nn.Parameter(
            torch.zeros([1, 1, in_channels], dtype=torch.float32),
            requires_grad=True,
        )
        trunc_normal_(self.char_token, mean=0, std=0.02)
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            bias=True,
        )
        self.fc_kv = nn.Linear(
            in_channels,
            2 * in_channels,
            bias=True,
        )
        self.w_atten_block = Block(dim=in_channels,
                                   num_heads=in_channels // 32,
                                   mlp_ratio=4.0,
                                   qkv_bias=False)
        self.out_channels = out_channels

    def forward(self, x):

        B, C, H, W = x.shape
        x = self.w_atten_block(x.permute(0, 2, 3,
                                         1).reshape(-1, W, C)).reshape(
                                             B, H, W, C).permute(0, 3, 1, 2)
        # B, D, 8, 32
        x_kv = self.fc_kv(x.flatten(2).transpose(1, 2)).reshape(
            B, H * W, 2, C).permute(2, 0, 3, 1)  # 2, b, c, hw
        x_k, x_v = x_kv.unbind(0)  # b, c, hw
        char_token = self.char_token.tile([B, 1, 1])
        attn_ctc2d = char_token @ x_k  # b, 1, hw
        attn_ctc2d = attn_ctc2d.reshape([-1, 1, H, W])
        attn_ctc2d = F.softmax(attn_ctc2d, 2)  # b, 1, h, w
        attn_ctc2d = attn_ctc2d.permute(0, 3, 1, 2)  # b, w, 1, h
        x_v = x_v.reshape(B, C, H, W)
        # B, W, H, C
        feats = attn_ctc2d @ x_v.permute(0, 3, 2, 1)  # b, w, 1, c
        feats = feats.squeeze(2)  # b, w, c
        predicts = self.fc(feats)

        return predicts
    

class CoordConvBlock(nn.Module):
    def __init__(self, conv_dim):
        super(CoordConvBlock, self).__init__()
        convs = []
        convs.append(ConvModule(258, conv_dim, 3, 1, padding=1, norm_cfg=dict(type='BN')))
        self.coord_convs = nn.Sequential(*convs)

    def forward(self, features):
        x_range = torch.linspace(-1, 1, features.shape[-1], device=features.device)
        y_range = torch.linspace(-1, 1, features.shape[-2], device=features.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([features.shape[0], 1, -1, -1])
        x = x.expand([features.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_features = torch.cat([features, coord_feat], dim=1)
        output_features = self.coord_convs(ins_features)
        return output_features


class EncoderStage(nn.Module):

    def __init__(self,
                 dim=64,
                 out_dim=256,
                 depth=3,
                 sub_k=[2, 1],
                 num_heads=2,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path=[0.1] * 3,
                 norm_layer=nn.LayerNorm,
                 act=nn.GELU,
                 eps=1e-6,
                 downsample=None,
                 **kwargs):
        super().__init__()
        self.dim = dim

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=act,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    eps=eps,
                ))

        if downsample:
            self.downsample = SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = IdentitySize()

    def forward(self, x, sz):
        for blk in self.blocks:
            x = blk(x)
        x, sz = self.downsample(x, sz)
        return x, sz



class Feat2D(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, sz):
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        return x, sz



@ENCODERS.register_module()
class TransformerCTCPredictor(nn.Module):

    def __init__(self,
                 voc_size=96,
                 depths=[3, 3, 3],
                 dims=[256, 384, 512],
                 sub_k=[[1, 1], [2, 1], [-1, -1]],
                 num_heads=[8, 12, 16],
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 act=nn.GELU,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]
        dpr = np.linspace(0, drop_path_rate,
                          sum(depths))  # stochastic depth decay rule

        self.stages = nn.ModuleList()
        for i_stage in range(num_stages):
            stage = EncoderStage(
                dim=dims[i_stage],
                out_dim=dims[i_stage + 1] if i_stage < num_stages - 1 else 0,
                depth=depths[i_stage],
                sub_k=sub_k[i_stage],
                num_heads=num_heads[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=norm_layer,
                act=act,
                downsample=False if i_stage == num_stages - 1 else True,
                eps=eps,
            )
            self.stages.append(stage)

        self.out_channels = self.num_features
        self.pos_embed = CoordConvBlock(256)
        self.stages.append(Feat2D())
        self.ctc_fc = CTCPredictor(in_channels=self.num_features, out_channels=voc_size+1)
        self.apply(self._init_weights)


    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'downsample'}

    def forward(self, x):
        x = self.pos_embed(x)
        sz = [x.shape[2:][0], x.shape[2:][1]]
        x = x.flatten(2).transpose(1, 2)
        for stage in self.stages:
            x, sz = stage(x, sz)
        
        logits = self.ctc_fc(x)
        
        return logits
