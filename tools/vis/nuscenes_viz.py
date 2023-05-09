import torch
import numpy as np

from .common import resize, to_image, get_colors, colorize


STATIC = ['lane', 'road_segment']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

CLASSES = STATIC + DIVIDER + DYNAMIC
NUM_CLASSES = len(CLASSES)


class NuScenesViz:
    SEMANTICS = CLASSES

    def __init__(self, label_indices=None, colormap='inferno'):
        self.label_indices = label_indices
        self.colors = get_colors(self.SEMANTICS)
        self.colormap = colormap

    def visualize_pred(self, bev, pred, threshold=None):
        """
        (c, h, w) torch float {0, 1}
        (c, h, w) torch float [0-1]
        """
        if isinstance(bev, torch.Tensor):
            bev = bev.cpu().numpy().transpose(1, 2, 0)

        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy().transpose(1, 2, 0)

        if self.label_indices is not None:
            bev = [bev[..., idx].max(-1) for idx in self.label_indices]
            bev = np.stack(bev, -1)

        if threshold is not None:
            pred = (pred > threshold).astype(np.float32)

        result = colorize((255 * pred.squeeze(2)).astype(np.uint8), self.colormap)

        return result

        h, w, c = pred.shape

        img = np.zeros((h, w, 3), dtype=np.float32)
        img[...] = 0.5
        colors = np.float32([
            [0, .6, 0],
            [1, .7, 0],
            [1,  0, 0]
        ])
        tp = (pred > threshold) & (bev > threshold)
        fp = (pred > threshold) & (bev < threshold)
        fn = (pred <= threshold) & (bev > threshold)

        for channel in range(c):
            for i, m in enumerate([tp, fp, fn]):
                img[m[..., channel]] = colors[i][None]

        return (255 * img).astype(np.uint8)

    def visualize_bev(self, bev):
        """
        (c, h, w) torch [0, 1] float

        returns (h, w, 3) np.uint8
        """
        if isinstance(bev, torch.Tensor):
            bev = bev.cpu().numpy().transpose(1, 2, 0)

        h, w, c = bev.shape

        assert c == len(self.SEMANTICS)

        # Prioritize higher class labels
        eps = (1e-5 * np.arange(c))[None, None]
        idx = (bev + eps).argmax(axis=-1)
        val = np.take_along_axis(bev, idx[..., None], -1)

        # Spots with no labels are light grey
        empty = np.uint8(COLORS['nothing'])[None, None]

        result = (val * self.colors[idx]) + ((1 - val) * empty)
        result = np.uint8(result)

        return result

    def visualize_custom(self, batch, pred, b):
        return []

    @torch.no_grad()
    def visualize(self, batch, pred=None, b_max=8, **kwargs):
        bev = batch['bev']
        batch_size = bev.shape[0]

        for b in range(min(batch_size, b_max)):
            if pred is not None:
                right = self.visualize_pred(bev[b], pred['bev'][b].sigmoid())
            else:
                right = self.visualize_bev(bev[b])

            right = [right] + self.visualize_custom(batch, pred, b)
            right = [x for x in right if x is not None]
            right = np.hstack(right)

            image = None if not hasattr(batch.get('image'), 'shape') else batch['image']

            if image is not None:
                imgs = [to_image(image[b][i]) for i in range(image.shape[1])]

                if len(imgs) == 6:
                    a = np.hstack(imgs[:3])
                    b = np.hstack(imgs[3:])
                    left = resize(np.vstack((a, b)), right)
                else:
                    left = np.hstack([resize(x, right) for x in imgs])

                yield np.hstack((left, right))
            else:
                yield right

    def __call__(self, batch=None, pred=None, **kwargs):
        return list(self.visualize(batch=batch, pred=pred, **kwargs))