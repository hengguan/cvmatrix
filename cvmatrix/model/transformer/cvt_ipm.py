import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
from omegaconf.listconfig import ListConfig

from cvmatrix.model.head.cvt_head import DecoderBlock
from ..build import TRANSFORMER_REGISTRY


__all__ = ['CVTIPM']

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


class MyBottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MyBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

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


class BEVGrid(nn.Module):
    def __init__(
        self,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        blocks: list,
    ):
        super().__init__()
        # bev coordinates
        # _, bev_height, bev_width, h_meters, w_meters, offset, _ = **bev_embedding
        grid = generate_grid(bev_height, bev_height).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=bev_height, w=bev_height)                    # 3 h w

        world = grid[:2]                                                    # 2 H W
        world_ = F.pad(world, (0, 0, 0, 0, 0, 1), value=-2)       # 3 H W
        world_ = F.pad(world_, (0, 0, 0, 0, 0, 1), value=1)       # 4 H W
        world_ = rearrange(world_, 'd H W -> d (H W)')
        # egocentric frame
        self.register_buffer('world', world_, persistent=False)                    # 3 h w


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
    #     self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    # def get_prior(self):
    #     return self.learned_features


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
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
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
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
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
        bev_embedding,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        self.bev = BEVEmbedding(dim, **bev_embedding)
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
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        # self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        # bev: BEVEmbedding,
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
        rays = d_flat - c_flat           # (b n) 4 h w
        rays = rays / (rays.norm(dim=1, keepdim=True) + 1e-7)    # (b n) 4 h w
        # rays = rays + c_flat
        img_embed = self.img_embed(rays)
        # d_embed = self.img_embed(d_flat)                                        # (b n) d h w

        # Camera-aware positional encoding
        # img_embed = d_embed - c_embed                                           # (b n) d h w
        # img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        world = self.bev.grid[:2]                                                    # 2 H W
        # world = F.pad(world[None], (0, 0, 0, 0, 0, 1, 0, 0), value=1)       # 1 3 H W
        bevs = world[None] - c_flat[:, [0, 1], :, :]                            # (b n) 3 H W
        bevs = bevs / (bevs.norm(dim=1, keepdim=True) + 1e-7)    # (b n) 3 H W
        # bevs = F.pad(bevs, (0, 0, 0, 0, 0, 1, 0, 0), value=0)    # (b n) 4 H W
        # bevs = bevs + c_flat[:, :2, :, :]
        # bn = bevs.shape[0]
        # world_ = repeat(world, '... -> b ...', b=bn)                     # bn 2 H W
        # bevs = torch.cat([bevs, world_], axis=1)       # bn 4 H W
        # bevs = rearrange(bevs, '(b n) c H W -> b (n c) H W', b=b, n=n)      # b H W (6 4)
        bev_embed = self.bev_embed(bevs)
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b d H W

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
        query = query_pos + x[:, None]                                          # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        return self.cross_attend(query, key, val, skip=x if self.skip else None)


@TRANSFORMER_REGISTRY.register()
class CVTIPM(nn.Module):
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

        self.bev_height, self.bev_width = bev_embedding.bev_height, bev_embedding.bev_width
        self.bevgrid = BEVGrid(**bev_embedding)
        self.delta = torch.tensor([
            [-1., 0., 1., -1., 0., 1., -1., 0., 1.],
            [-1., -1., -1., 0., 0., 0., 1., 1., 1.] 
        ]).long().cuda()        # 2x9
        self.kernel = (torch.tensor([
            [1., 2., 1., 2., 8., 2., 1., 2., 1.]
        ]) * (1. / 20.)).cuda()  # 1x9

        if scale < 1.0:
            self.down = lambda x: F.interpolate(
                x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(feats_shapes) == len(middle)
        
        cross_views = list()
        layers = list()

        # w, h = None, None
        # fdim = None
        for feat_shape, num_layers in zip(feats_shapes, middle):

            if isinstance(feat_shape, ListConfig):
                feat_shape = tuple(feat_shape)
                
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, bev_embedding, **cross_view)
            cross_views.append(cva)
            bev_embedding.blocks = bev_embedding.blocks[1:]

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        # self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

        self.layer1 = self._make_layer(MyBottleneck, feat_dim, 128, 1, stride=1)
        self.layer2 = self._make_layer(MyBottleneck, 128, 128, 1, stride=2)
        self.layer3 = self._make_layer(MyBottleneck, 128, 128, 1, stride=2)
        self.layer4 = self._make_layer(MyBottleneck, 128, 128, 1, stride=2)

        # up
        self.up1 = DecoderBlock(128, 128, dim, True, 1)
        self.up2 = DecoderBlock(128, 128, dim, True, 1)
        # self.upsample = [up1, up2]

        self.lat1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.lat2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # self.image_height = cross_view.image_height
        # self.latlayer = [lat1, lat2]
        self.minpixel = torch.tensor([0, 0])[None, None, :, None].cuda()
        self.maxpixel = torch.tensor([feat_width-1, feat_height-1])[None, None, :, None].cuda()
        self.dim0, self.dim1 = None, None

    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, feats, camera):
        
        I_inv = camera['intrinsics'].inverse()           # b n 3 3
        E_inv = camera['extrinsics'].inverse()           # b n 4 4

        features = [self.down(y) for y in feats]
        bn, c, h, w = features[1].shape
        b, n = bn // 6, 6

        E = camera['extrinsics']
        I = camera['intrinsics']  # stride

        cams = E @ self.bevgrid.world   # b n 4 (H W)
        pixels = I @ cams[:, :, :3, :] # b n 3 (H W)
        uv = pixels[:, :, :2, :] / pixels[:, :, 2:, :] # b n 2 (H W)
        flag = pixels[:, :, 2:, :] > 0
        uv = uv.float() / 16.
        uv = uv.long() * flag.long()  # b n 2 (H W)
        inds = (uv>0) & (uv<self.maxpixel)
        inds = inds[:, :, :1, :] & inds[:, :, 1:, :] # b n 1 (H W)
        inds = rearrange(inds, 'b n c (H W) -> b n c H W', H=self.bev_height, W=self.bev_width)

        uv = uv[:, :, :, None, :] + self.delta[None, None, :, :, None] # b n 2 9 (H W)
        uv = uv.clamp(min=self.minpixel[..., None], max=self.maxpixel[..., None])

        feature = rearrange(features[1], '(b n) ... -> b n ...', b=b, n=n) # b n d h w
        if self.dim0 is None or self.dim1 is None or self.dim0.shape[0]!=b:
            bev_size = self.bev_height * self.bev_width
            dim0 = torch.arange(b)[:, None, None, None]
            self.dim0 = dim0.expand(-1, n, 9, bev_size).to(feature.device)
            dim1 = torch.arange(n)[None, :, None, None]
            self.dim1 = dim1.expand(b, -1, 9, bev_size).to(feature.device)

        feats = feature[self.dim0, self.dim1, :, uv[:, :, 1, :, :], uv[:, :, 0, :, :]]
        feats = rearrange(feats, 'b n k (H W) c -> b n c k H W', H=self.bev_height, W=self.bev_width)
        feats = feats * self.kernel[None, None, :, :, None, None]
        feats = (feats.sum(dim=3) * inds.long()).sum(dim=1) / (inds.long().sum(dim=1) + 1e-7)  # b c H W

        c0 = self.layer1(feats)
        c1 = self.layer2(c0)  # b c 100 100
        c2 = self.layer3(c1)  # b c 50 50
        c3 = self.layer4(c2)  # b c 25 25
        
        refine = [c2, c1]

        # x = self.bev_embedding.get_prior()              # d H W
        # x = repeat(x, '... -> b ...', b=b)              # b d H W
        x = c3
        upsamples = [self.up1, self.up2]
        lats = [self.lat1, self.lat2]
        for cross_view, feature, layer, up, lat, c in zip(
            self.cross_views, features, self.layers, upsamples, lats, refine):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(x, feature, I_inv, E_inv) # self.bev_embedding, 
            x = layer(x)
            x = up(x, x) + lat(c)
        
        return x
