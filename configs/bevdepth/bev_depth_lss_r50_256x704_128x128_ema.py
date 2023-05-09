from ..common.models.bev_depth_lss_base import model
from ..common.train import train, test
from ..common.solver import solver

H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)
x_bound_ = [-51.2, 51.2, 0.8]
y_bound_ = [-51.2, 51.2, 0.8]
z_bound_ = [-5, 3, 8]
d_bound_ = [2.0, 58.0, 0.5]

model['lss_conf'].update(dict(
    x_bound=x_bound_,
    y_bound=y_bound_,
    z_bound=z_bound_,
    d_bound=d_bound_,
    final_dim = final_dim,
))

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim': final_dim,
    'rot_lim': (-5.4, 5.4),
    'H': H,
    'W': W,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.0),
    'Ncams': 6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}

# solver config
solver.optimizer_name = "AdamW"
solver.lr_scheduler_name = "OneCycleLR"
solver.max_iter = 140650
solver.optimizer = dict(
    lr=4e-3,
    weight_decay=1e-4,
)
solver.scheduler = dict(
    div_factor=10,                                  # starts at lr / 10
    pct_start=0.3,                                  # reaches lr at 30% of total steps
    final_div_factor=4,                            # ends at lr / 10 / 10
    max_lr=solver.optimizer.lr,
    total_steps=solver.max_iter,
    cycle_momentum=False,
)
