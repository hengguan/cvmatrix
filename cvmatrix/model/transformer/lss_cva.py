import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
from omegaconf.listconfig import ListConfig

from ..build import TRANSFORMER_REGISTRY


__all__ = ['CrossViewTransformer']

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w 
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


# class Normalize(nn.Module):
#     def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#         super().__init__()

#         self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
#         self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

#     def forward(self, x):
#         return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(blocks))
        w = bev_width // (2 ** len(blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d m)
        k: (b n d h w)
        v: (b n d h w)
        """
        b, n, _, m = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d m -> b (n m) d')
        k = rearrange(k, 'b n d h w -> b (n h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)
        # dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)
        z = rearrange(z, 'b (n m) d -> b n m d', n=n, m=m)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b n d m -> b n m d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b n m d -> b n d m')

        return z


class LayerConvert(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        # no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_proj = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        # self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(3, dim, 1, bias=False)
        # self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        pos_emb: torch.FloatTensor,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        # E_inv: torch.FloatTensor,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape

        # camera location embeding
        # c = E_inv[..., -1:]                                                     # b n 4 1
        # c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        # c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        # unprojected pixel coordinates embeding
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = rearrange(cam, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n) 3 h w
        cam = cam / (cam.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w
        img_embed = self.img_embed(cam)

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w
        val_flat = self.feature_proj(feature_flat)

        key_flat = img_embed + val_flat              # (b n) d h w

        # Expand + refine the BEV embedding
        query = x + pos_emb
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        return self.cross_attend(query, key, val, skip=x if self.skip else None)


@TRANSFORMER_REGISTRY.register()
class CrossViewAttention(nn.Module):
    def __init__(
            self,
            *,
            # backbone,
            dim=128,
            scale=1.0,
            middle=[],
            cross_view=None,
            depth_conf=[1., 49., 0.25],
            feats_shapes=None
    ):
        super(CrossViewAttention, self).__init__()

        # self.norm = Normalize()
        # self.backbone = backbone
        sigma = 1.

        if scale < 1.0:
            self.down = lambda x: F.interpolate(
                x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(feats_shapes) == len(middle)
        image_height = cross_view['image_height']
        image_width = cross_view['image_width']
        
        cross_views = list()
        layers = list()
        for feat_shape, num_layers in zip(feats_shapes, middle):

            if isinstance(feat_shape, ListConfig):
                feat_shape = tuple(feat_shape)
                
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = LayerConvert(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            # layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            # layers.append(layer)

        # self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.pos = torch.arange(*depth_conf)[:, None].cuda()
        self.depth_embed = nn.Linear(1, dim)
        self.depth_prior = nn.Parameter(sigma * torch.randn(dim, self.pos.shape[0]))    # d h w
        self.cross_views = nn.ModuleList(cross_views)
        # self.layers = nn.ModuleList(layers)
        
    def forward(self, feats, I):
        
        I_inv = I.inverse()           # b n 3 3
        # E_inv = camera['extrinsics'].inverse()           # b n 4 4
        b, n, _, _ = I_inv.shape

        features = [self.down(y) for y in feats]

        x = self.depth_prior              # d H W
        x = repeat(x, '... -> b n ...', b=b, n=n)              # b d H W

        depth_pos = repeat(self.pos, '... -> b n ...', b=b, n=n) # b n m d
        depth_emb = self.depth_embed(depth_pos)
        depth_emb = rearrange(depth_emb, 'b n m d -> b n d m')
        depth_emb = depth_emb / (depth_emb.norm(dim=2, keepdim=True) + 1e-7)    # (b n) d h w

        for cross_view, feature in zip(self.cross_views, features):#, layer, self.layers
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(x, depth_emb, feature, I_inv)
            # x = layer(x)

        return x
