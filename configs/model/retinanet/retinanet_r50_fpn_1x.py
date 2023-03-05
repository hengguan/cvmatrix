
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='FPN',
        stem=dict(
            type='BasicStem',
            in_channels=3,
            out_channels=64,
            norm="FrozenBN"
        ),
        stages=dict(
            type=ResNet.make_default_stages,
            depth=50,
            stride_in_1x1=True,
            norm="FrozenBN",
        ),
        out_features=["res3", "res4", "res5"],
    ),
    in_features=["res3", "res4", "res5"],
    out_channels=256,
    top_block=dict(
        type='LastLevelP6P7',
        in_channels=2048, 
        out_channels="${..out_channels}"
    ),
)