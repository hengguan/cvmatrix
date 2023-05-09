import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
from omegaconf.listconfig import ListConfig

from ..build import TRANSFORMER_REGISTRY


__all__ = ['CVTPolar']

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


def get_polar_grid(ray_x, ray_y, rad_step=18, distance=50, distance_step=25):
    b, n, _ = ray_x.shape
    rads = torch.atan2(ray_y, ray_x)
    max_rad, _ = rads.max(axis=2)
    min_rad, _ = rads.min(axis=2)  
    step = torch.arange(rad_step, dtype=torch.float32, device=max_rad.device) / (rad_step - 1) 
    step = step.view(1, 1, rad_step)
    rads_grid = min_rad[..., None] + step * (max_rad - min_rad)[..., None]
    dist = torch.linspace(0, distance, distance_step, dtype=torch.float32, device=max_rad.device)
    dist = repeat(dist, '... -> b n ...', b=b, n=n)
    cam_x = torch.sin(rads_grid)[..., None] @ dist.unsqueeze(2)    # b n rad dist
    cam_z = torch.cos(rads_grid)[..., None] @ dist.unsqueeze(2)
    cam_coord = torch.stack([cam_x, cam_z], axis=2)       # b n 2 18 25
    return cam_coord, dist


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
        self.height = bev_height
        self.width = bev_width

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

        self.polars_feature = nn.Parameter(sigma * torch.randn(6, dim, 18, 25))    # d h w

    def get_prior(self):
        return self.learned_features


class PolarAttention(nn.Module):
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
        # self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        b, n, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b n (h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        # dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, '(b n) d h w -> b n (h w) d', b=b, n=n)

        # z = self.prenorm(z)
        # z = z + self.mlp(z)
        # z = self.postnorm(z)
        # z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return rearrange(z, 'b n (h w) d -> b n d h w', h=H, w=W)


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
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b d H W -> b (H W) d')
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

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class CrossViewAttention(nn.Module):
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
        no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(3, dim, 1, bias=False)
        # self.polar_embed = nn.Conv2d(2, dim, 1)
        self.cam_embed = nn.Conv2d(2, dim, 1)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.polar_attend = PolarAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        px: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
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
        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        # c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        # unprojected pixel coordinates embeding
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
        d = E_inv @ cam                                                         # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w

        rays = d_flat[:, :3, :, :] - c_flat[:, :3, :, :]           # (b n) 3 h w
        eff = repeat(torch.Tensor([1, 1, 1, 1, -1, 1]).to(rays.device), '... -> b ...', b=b)
        eff = rearrange(eff, 'b n ... -> (b n) ...').view(-1, 1, 1, 1)
        eff_rays = eff * rays[:, :2, ...]
        rad_step=18
        distance=50
        distance_step=25
        rads = torch.atan2(eff_rays[:, 1, :, :], eff_rays[:, 0, :, :])  # (b n) h w
        min_rad = rads[:, h//2, 0]
        max_rad = rads[:, h//2, -1]
        mid_rad = (min_rad + max_rad) * 0.5
        rads_norm = rads - mid_rad[..., None, None]
        rays_norm = torch.stack([torch.cos(rads_norm), torch.sin(rads_norm), rays[:, 2, :, :]], axis=1)  # b n 3 (h w)

        step = torch.arange(rad_step, dtype=torch.float32, device=max_rad.device) / (rad_step - 1) 
        # step = step.view(1, 1, rad_step)
        rads_grid = step[None] * (max_rad - min_rad)[..., None] + min_rad[..., None]   # (b n) r
        # rads_grid_norm = rads_grid - mid_rad[..., None]
        dist = torch.linspace(1, distance-1, distance_step, dtype=torch.float32, device=max_rad.device)
        dist = repeat(dist, '... -> bn ...', bn=b*n).unsqueeze(1)   # (b n) 1 d
        # y_norm = torch.sin(rads_grid_norm)[..., None] * dist    # (b n) r d
        # x_norm = torch.cos(rads_grid_norm)[..., None] * dist
        # xy_norm = torch.stack([x_norm, y_norm], axis=1)       # (b n) 2 r d

        yy = torch.sin(rads_grid)[..., None] * dist    # (b n) r d
        xx = torch.cos(rads_grid)[..., None] * dist
        xy = torch.stack([xx, yy], axis=1)
        xy = eff * xy + c_flat[:, :2, :, :]      # (b n) 2 r d : in bev coordinate

        rays = rays_norm / (rays_norm.norm(dim=1, keepdim=True) + 1e-7)    # (b n) 3 h w
        # rays = rearrange(rays, 'b n d (h w) -> (b n) d h w', h=h, w=w)
        # rays = rays + c_flat
        img_embed = self.img_embed(rays)

        # polars = xy_norm / (xy_norm.norm(dim=1, keepdim=True) + 1e-7)
        # cam_polars = rearrange(cam_polars, 'b n d h w -> (b n) d h w')           # (b n) 4 h w
        # polar_embed = self.polar_embed(polars)

        bev_coord_norm = xy.norm(dim=1, keepdim=True) + 1e-7
        bev_coord = xy / bev_coord_norm
        # bev_pos = rearrange(bev_pos, 'b n d h w -> (b n) d h w')
        cam_embed = self.cam_embed(bev_coord)
    
        # d_embed = self.img_embed(d_flat)                                        # (b n) d h w

        # Camera-aware positional encoding
        # img_embed = d_embed - c_embed                                           # (b n) d h w
        # img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        world = repeat(bev.grid[:2], '... -> b ...', b=b)                                                    # 2 H W
        # world = F.pad(world[None], (0, 0, 0, 0, 0, 1, 0, 0), value=1)       # 1 3 H W
        # bevs = world[None] - c_flat[:, [0, 1], :, :]                            # (b n) 3 H W
        # bevs = bevs / (bevs.norm(dim=1, keepdim=True) + 1e-7)    # (b n) 2 H W
        world_norm = world.norm(dim=1, keepdim=True) + 1e-7
        bevs = world / world_norm

        bev_embed = self.bev_embed(bevs)                                  # b d H W
        # query_pos = torch.cat([query_pos, world_], axis=1)                 # b d H W

        # w_embed = self.bev_embed(world[None])                                   # 1 d H W
        # bev_embed = w_embed - c_embed                                           # (b n) d H W
        # bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
        # query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
        else:
            key_flat = img_embed                                                # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

        # Expand + refine the BEV embedding
        # query = query_pos + x[:, None]                                          # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        polar_q = px + cam_embed
        p_query = rearrange(polar_q, '(b n) ... -> b n ...', b=b, n=n)
        px = self.polar_attend(p_query, key, val, skip=px if self.skip else None)

        cam_embed = rearrange(cam_embed, '(b n) d h w -> b n d h w', b=b, n=n)
        x = self.cross_attend(x + bev_embed, px+cam_embed, px, skip=x if self.skip else None)

        return x, rearrange(px, 'b n ... -> (b n) ...')


@TRANSFORMER_REGISTRY.register()
class CVTPolar(nn.Module):
    def __init__(
            self,
            *,
            # backbone,
            dim=128,
            scale=1.0,
            middle=[],
            cross_view=None,
            bev_embedding=None,
            feats_shapes=None
    ):
        super().__init__()

        # self.norm = Normalize()
        # self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(
                x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(feats_shapes) == len(middle)
        
        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(feats_shapes, middle):

            if isinstance(feat_shape, ListConfig):
                feat_shape = tuple(feat_shape)
                
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, feats, camera):
        
        I_inv = camera['intrinsics'].inverse()           # b n 3 3
        E_inv = camera['extrinsics'].inverse()           # b n 4 4

        features = [self.down(y) for y in feats]
        b = features[0].shape[0] // 6
        n = 6

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        px = self.bev_embedding.polars_feature
        px = repeat(px, '... -> b ...', b=b)
        px = rearrange(px, 'b n ... -> (b n) ...')

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x, px = cross_view(x, px, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)

        return x
