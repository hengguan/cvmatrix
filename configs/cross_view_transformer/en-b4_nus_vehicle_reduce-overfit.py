from ..common.solver import solver
from ..common.dataset.nuscenes import datasets
from ..common.dataloader import dataloader
from ..common.train import train, test

image_height_ = 224
image_width_ = 480
top_crop_ = 46
dim_ = 128
bev_ = dict(
    bev_height=200,
    bev_width=200,
    h_meters=100.0,
    w_meters=100.0,
    offset=0.0
)
blocks_ = [128]
label_indices_ = [[4, 5, 6, 7, 8, 10, 11]]


model = dict(
    type="CrossViewTransformerExp",
    model_type = 'bev',
    dim_last=128,
    outputs=dict(
        bev=[0, 1],
        center=[1, 2]
    ),
    input_size=[image_height_, image_width_],
    backbone=dict(
        type="EfficientNetExtractor",
        model_name="efficientnet-b4",
        output_layers=['reduction_3', 'reduction_4'],
    ),
    transformer=dict(
        type="CrossViewTransformer",
        dim=dim_,
        scale=1.0,
        middle=[1, 1],
        cross_view=dict(
            heads=4,
            dim_head=32,
            qkv_bias=True,
            skip=True,
            no_image_features=False,
            image_height=image_height_,
            image_width=image_width_,
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

train.uuid = ""

# solver config
solver.optimizer_name = "AdamW"
solver.lr_scheduler_name = "OneCycleLR"
solver.max_iter = 70326
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

datasets.update(dict(
    normalize=dict(
        norm_resnet=False,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        to_rgb=False
    ),
    labels_root="/tmp/dataset/dataset-693-version-1/cvt_labels",
    labels_dir_name="cvt_labels_nuscenes_v2",
    image=dict(
        h=image_height_,
        w=image_width_,
        top_crop=top_crop_
    ),
    augment='none'
))

dataloader.batch_size=4
dataloader.num_workers=6

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
