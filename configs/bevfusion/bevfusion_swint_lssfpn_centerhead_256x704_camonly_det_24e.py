from ..common.models.bevfusion_swint_lssfpn_centerhead_det import model, image_size
from ..common.solver import solver
from ..common.dataloader import dataloader
from ..common.train import train, test


image_size = (256, 704)
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]

model['encoders']['camera']['vtransform'].update(
    dict(
        type='LSSTransform',
        image_size=image_size,
    )
)

model.update(
    heads=dict(
        object=dict(
            type='CenterHead',
            in_channels=256,
            train_cfg=dict(
                point_cloud_range=point_cloud_range,
                grid_size=[1024, 1024, 1],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
            ),
            test_cfg=dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2,
                nms_type=['circle', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
                nms_scale=[[1.0], [1.0, 1.0], [1.0, 1.0], [1.0], [1.0, 1.0], [2.5, 4.0]]
            ),
            tasks=[
                ["car"],
                ["truck", "construction_vehicle"],
                ["bus", "trailer"],
                ["barrier"],
                ["motorcycle", "bicycle"],
                ["pedestrian", "traffic_cone"]
            ],
            common_heads=dict(reg=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
            share_conv_channel=64,
            bbox_coder=dict(
                pc_range=point_cloud_range,
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=9
            ),
            separate_head=dict(
                type='SeparateHead',
                init_bias=-2.19,
                final_kernel=3,
            ),
            loss_cls=dict(
                type='GaussianFocalLoss',
                reduction='mean'
            ),
            loss_bbox=dict(
                type='L1Loss',
                reduction='mean',
                loss_weight=0.25,
            ),
            norm_bbox=True
        )
    ),
    fuser=None
)

dataset_type="NuScenesDataset"
dataset_root="/home/gh/workspace/data/nuscenes/trainval/"
gt_paste_stop_epoch=-1
reduce_beams=32
load_dim=5
use_dim=5
object_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
map_classes=[
    "drivable_area",
    # "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    # "road_divider",
    # "lane_divider",
    "divider"
]
  
input_modality=dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

augment2d=dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(prob=0.0, fixed_prob=True)
)

augment3d=dict(
    scale=[0.9, 1.1],
    rotate=[-0.78539816, 0.78539816],
    translate=0.5
)
load_augmented=None
max_epochs=24

train_pipelines=[
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    # dict(type="LoadPointsFromFile",
    #     coord_type="LIDAR",
    #     load_dim=load_dim,
    #     use_dim=use_dim,
    #     reduce_beams=reduce_beams,
    #     load_augmented=load_augmented),
    # dict(type="LoadPointsFromMultiSweeps",
    #     sweeps_num=9,
    #     load_dim=load_dim,
    #     use_dim=use_dim,
    #     reduce_beams=reduce_beams,
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     load_augmented=load_augmented,),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type="ObjectPaste",
        stop_epoch=gt_paste_stop_epoch,
        db_sampler=dict(
            dataset_root=dataset_root,
            info_path=dataset_root + "nuscenes_dbinfos_train.pkl",
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5, truck=5, bus=5, trailer=5, construction_vehicle=5,
                    traffic_cone=5, barrier=5, motorcycle=5, bicycle=5, pedestrian=5
                )
            ),
            classes=object_classes,
            sample_groups=dict(
                car=2, truck=3, construction_vehicle=7, bus=4, trailer=6,
                barrier=2, motorcycle=6, bicycle=6, pedestrian=2, traffic_cone=2
            ),
            # points_loader=dict(
            #     type="LoadPointsFromFile",
            #     coord_type="LIDAR",
            #     load_dim=load_dim,
            #     use_dim=use_dim,
            #     reduce_beams=reduce_beams,
            # )
        )
    ),
    dict(
        type="ImageAug3D",
        final_dim=image_size,
        resize_lim=augment2d['resize'][0],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d["rotate"],
        rand_flip=True,
        is_train=True
    ),
    dict(
        type="GlobalRotScaleTrans",
        resize_lim=augment3d["scale"],
        rot_lim=augment3d["rotate"],
        trans_lim=augment3d["translate"],
        is_train=True
    ),
    # dict(
    #     type="LoadBEVSegmentation",
    #     dataset_root=dataset_root,
    #     xbound=[-50.0, 50.0, 0.5],
    #     ybound=[-50.0, 50.0, 0.5],
    #     classes=map_classes,
    # ),
    dict(type="RandomFlip3D"),
    # dict(type="PointsRangeFilter",  point_cloud_range=point_cloud_range,),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range,),
    dict(type="ObjectNameFilter", classes=object_classes,),
    dict(type="ImageNormalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type="GridMask", use_h=True, use_w=True,
        max_epoch=max_epochs, rotate=1, offset=False,
        ratio=0.5, mode=1,
        prob=augment2d["gridmask"]["prob"],
        fixed_prob=augment2d["gridmask"]["fixed_prob"],
    ),
    # dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", classes=object_classes,),
    dict(type="Collect3D", 
        keys=["img", "gt_bboxes_3d", "gt_labels_3d"],#, "gt_masks_bev"
        meta_keys=["camera_intrinsics", "camera2ego", "lidar2ego", "lidar2camera", 
                "camera2lidar", "lidar2image", "img_aug_matrix", "lidar_aug_matrix"]
    )
]
test_pipelines=[
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    # dict(type="LoadPointsFromFile",
    #     coord_type="LIDAR",
    #     load_dim=load_dim,
    #     use_dim=use_dim,
    #     reduce_beams=reduce_beams,
    #     load_augmented=load_augmented),
    # dict(type="LoadPointsFromMultiSweeps",
    #     sweeps_num=9,
    #     load_dim=load_dim,
    #     use_dim=use_dim,
    #     reduce_beams=reduce_beams,
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     load_augmented=load_augmented,),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type="ImageAug3D",
        final_dim=image_size,
        resize_lim=augment2d['resize'][1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0., 0.],
        rand_flip=False,
        is_train=False
    ),
    dict(
        type="GlobalRotScaleTrans",
        resize_lim=[1., 1.],
        rot_lim=[0., 0.],
        trans_lim=0.,
        is_train=False
    ),
    # dict(
    #     type="LoadBEVSegmentation",
    #     dataset_root=dataset_root,
    #     xbound=[-50.0, 50.0, 0.5],
    #     ybound=[-50.0, 50.0, 0.5],
    #     classes=map_classes,
    # ),
    # dict(type="PointsRangeFilter",  point_cloud_range=point_cloud_range,),
    dict(type="ImageNormalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type="DefaultFormatBundle3D", classes=object_classes,),
    dict(type="Collect3D", 
        keys=["img", "gt_bboxes_3d", "gt_labels_3d"],#, "gt_masks_bev"
        meta_keys=["camera_intrinsics", "camera2ego", "lidar2ego", "ego2global", "lidar2camera", 
                "camera2lidar", "lidar2image", "img_aug_matrix", "lidar_aug_matrix"]
    )
]
# datasets = dict(
#     train=dict(
        
#     )
# )

train_loader=dict(
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    prefetch_factor=4,
    shuffle=True,
    dataset=dict(
        type="CBGSDataset",
        dataset=dict(
            type=dataset_type,
            dataset_root=dataset_root,
            ann_file=dataset_root+"nuscenes_infos_train.pkl",
            pipeline=train_pipelines,
            object_classes=None,
            map_classes=map_classes,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d="LiDAR"
        )
    )
)

val_loader=dict(
    batch_size=1,
    num_workers=1,
    # pin_memory=True,
    # prefetch_factor=4,
    shuffle=False,
    dataset=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root+"nuscenes_infos_val.pkl",
        pipeline=test_pipelines,
        object_classes=None,
        load_interval=1,
        map_classes=map_classes,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR"
    )
)

train.uuid = '20230518-193552'
train.output_dir = 'outputs'
solver.optimizer=dict(
    type='AdamW',
    lr=2.e-4,
    weight_decay=0.01,
    # paramwise_cfg=dict(
    #     custom_keys=dict(
    #         absolute_pos_embed=dict(
    #             decay_mult=0
    #         ),
    #         relative_position_bias_table=dict(
    #             decay_mult=0
    #         ),
    #         # TODO=
    #         # encoders.camera.backbone=dict(
    #         #     lr_mult=0.1
    #         # )
    #     )
    # ) 
)

solver.max_iter = 140650
solver.lr_scheduler=dict(
    type='OneCycleLR',
    div_factor=10,                                  # starts at lr / 10
    pct_start=0.4,                                  # reaches lr at 30% of total steps
    final_div_factor=5,                            # ends at lr / 10 / 10
    max_lr=solver.optimizer.lr,
    total_steps=solver.max_iter,
    cycle_momentum=False,
)
solver.momentum=dict(
    policy='cyclic',
    cyclic_times=1,
    step_ratio_up=0.4
)


evaluation=dict(
    type='NuscDetEval',
    jsonfile_prefix='outputs/bevfusion_swint_lssfpn_centerhead_256x704_camonly_det_24e/20230518-193552/test',
    eval_modes=['boxes_3d'],  # boxes_3d/masks_bev
    eval_version="detection_cvpr_2019",
    modality=input_modality,
    nusc_config=dict(
        version='v1.0-trainval',
        dataroot=dataset_root,
        verbose=False
    )
)
