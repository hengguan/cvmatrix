from ..common.solver import solver
from ..common.dataset.nuscenes import datasets
from ..common.dataloader import dataloader
from ..common.train import train, test

image_=dict(
    h=224,
    w=480,
    top_crop=46
)
dim_ = 128
bev_ = dict(
    bev_height=200,
    bev_width=200,
    h_meters=100.0,
    w_meters=100.0,
    offset=0.0
)
blocks_ = [128, 64]
label_indices_ = [[4, 5, 6, 7, 8, 10, 11]]


model = dict(
    type="CrossViewTransformerExp",
    model_type = 'bev',
    dim_last=128,
    outputs=dict(
        bev=[0, 1],
        center=[1, 2]
    ),
    input_size=["${image_.h}", "${image_.w}"],
    backbone=dict(
        type="EfficientNetExtractor",
        model_name="efficientnet-b4",
        output_layers=['reduction_3', 'reduction_4'],
    ),
    transformer=dict(
        type="EfficientFormer",
        dim=dim_,
        scale=1.0,
        middle=[2, 2],
        cross_view=dict(
            heads=4,
            dim_head=64,
            qkv_bias=True,
            skip=True,
            no_image_features=False,
            image_height="${image_.h}",
            image_width="${image_.w}",
        ),
        bev_embedding=dict(
            sigma=1.0,
            **bev_,
            blocks=blocks_,
        ),
        feats_shapes=None
    ),
    head=dict(
        type="CrossViewTransformerHead",
        dim=dim_,
        blocks=blocks_,
        residual=True,
        factor=2,
    ),
    losses=dict(
        bev_loss=dict(
            type="BinarySegmentationLoss",
            label_indices=label_indices_,
            gamma=2.0,
            alpha=-1.0,
            min_visibility=2,
            weight=1.0,
        ),
        center_loss=dict(
            type="CenterLoss",
            weight=0.1,
            gamma=2.0,
            min_visibility=2,
        )
    )
)

train.uuid = "20230520-133022"
train.output_dir = "./outputs"

# solver config
solver.max_iter = 70325
solver.optimizer = dict(
    type="AdamW",
    lr=4e-3,
    weight_decay=1e-7,
)
solver.lr_scheduler = dict(
    type="OneCycleLR",
    div_factor=10,                                  # starts at lr / 10
    pct_start=0.3,                                  # reaches lr at 30% of total steps
    final_div_factor=10,                            # ends at lr / 10 / 10
    max_lr=solver.optimizer.lr,
    total_steps=solver.max_iter,
    cycle_momentum=False,
)

data_type_ = "NuscenesGenerated"
data_dir_="/home/gh/workspace/data/nuscenes/trainval/"
labels_root_="/home/gh/workspace/data/nuscenes/cvt_labels"
labels_dir_name_="cvt_labels_nuscenes_v2"
normalize_=dict(
    norm_resnet=False,
    mean=[0.485, 0.456, 0.406],     
    std=[0.229, 0.224, 0.225],
    to_rgb=False
)
train_loader = dict(
    batch_size=8,
    num_workers=6,
    pin_memory=True,
    prefetch_factor=4,
    shuffle=True,
    dataset=dict(
        type=data_type_,
        version='',
        normalize=normalize_,
        split='train',
        data_dir= data_dir_,
        labels_root  = labels_root_,
        labels_dir_name  = labels_dir_name_,
        image = image_,
        cameras=None,
        augment='none',
    )
)

evaluator = dict(
    iou=dict(
        type='IoUMetric',
        label_indices=label_indices_,
        min_visibility=2
    ),
    iou_with_occlusions=dict(
        type='IoUMetric',
        label_indices=label_indices_
    )
)
