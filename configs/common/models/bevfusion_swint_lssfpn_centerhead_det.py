image_size = [256, 704]

model = dict(
    type="BEVFusion",
    model_type='bev',
    encoders=dict(
        camera=dict(
            backbone=dict(
                type="SwinTransformer",
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=[1, 2, 3],
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(
                    type="Pretrained",
                    checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
                )
            ),
            neck=dict(
                type="GeneralizedLSSFPN",
                in_channels=[192, 384, 768],
                out_channels=256,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(
                    type="BN2d",
                    requires_grad=True 
                ),
                act_cfg=dict(
                    type="ReLU",
                    inplace=True
                ),
                upsample_cfg=dict(
                    mode='bilinear',
                    align_corners=False
                )  
            ),
            vtransform=dict(
                in_channels=256,
                out_channels=80,
                feature_size=[image_size[0] // 8, image_size[1] // 8],
                xbound=[-51.2, 51.2, 0.4],
                ybound=[-51.2, 51.2, 0.4],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 0.5],
                downsample=2
            )
        )
    ),
    decoder=dict(
        backbone=dict(
            type="GeneralizedResNet",
            in_channels=80,
            blocks=[
               [2, 256, 2],
            #    [2, 256, 2],
               [2, 512, 2],
            ]
        ),
        neck=dict(
            type="LSSFPN",
            in_indices=[-1, 0],
            in_channels=[512, 256],
            out_channels=256,
            scale_factor=2
        )
    )
)