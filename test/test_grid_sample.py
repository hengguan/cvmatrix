import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from einops import repeat, rearrange
from PIL import Image
import cv2


transform = transforms.Compose([
    transforms.ToTensor()
])
im = Image.open('test/01.jpg')
im = im.resize((960, 540))
im = im.crop((224, 14, 736, 526))
im_tensor = transform(im)
im_tensor[:, 256:, :] = 0

theta = -np.pi / 3
rot_mat = torch.Tensor([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
])


xs = torch.linspace(0, 511, 512)
ys = torch.linspace(0, 511, 512)
xy = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0) - 256
print(xy)

# data1 = repeat(torch.arange(25).view(5, 5), '... -> n ...', n=3)
# data2 = torch.zeros_like(data1)
# data = torch.cat([data1, data2], axis=1)
# print(data)

rot_xy = rot_mat[:2, :2] @ xy.view(2, -1) + rot_mat[:2, 2:3]
rot_xy = rot_xy.view(2, 512, 512) / 256.
grid = rearrange(rot_xy, 'c h w -> h w c')
print(rot_xy)


im_sample = F.grid_sample(im_tensor[None], grid[None], mode='bicubic')
img = im_sample[0].numpy().transpose((1, 2, 0)) * 255
img = img.astype(np.uint8)

cv2.imshow('viz', img)
cv2.waitKey(0)
