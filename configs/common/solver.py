solver = dict(
    type="WarmupMultiStepLR",
    max_iter=4000,
    # base_lr=0.001,
    # The end lr, only used by WarmupCosineLR
    # base_lr_end=0.0,  
    # momentum=0.9,
    # nesterov=False,
    # weight_decay=0.0001,
    # The weight decay that's applied to parameters of normalization layers
    # (typically the affine transformation)
    # weight_decay_norm=0.0,
    # gamma=0.1,
    # steps=[30000, 0],
    # warmup_factor=1.0 / 1000,
    # warmup_iters=1000,
    # warmup_method="linear",
    # Save a checkpoint after every this number of iterations
    checkpoint_period=4000,
    # Number of images per batch across all machines. This is also the number
    # of training images per step (i.e. per iteration). If we use 16 GPUs
    # and IMS_PER_BATCH: 32, each GPU will see 2 images per batch.
    # May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
    # ims_per_batch=16,
    # The reference number of workers (GPUs) this config is meant to train with.
    # It takes no effect when set to 0.
    # With a non-zero value, it will be used by DefaultTrainer to compute a desired
    # per-worker batch size, and then scale the other related configs (total batch size,
    # learning rate, etc) to match the per-worker batch size.
    # See documentation of `DefaultTrainer.auto_scale_workers` for details:
    reference_world_size=0,
    # Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
    # biases. This is not useful (at least for recent models). You should avoid
    # changing these and they exist only to reproduce Detectron v1 training if
    # desired.
    bias_lr_factor=1.0,
    weight_decay_bias=None,# None means following WEIGHT_DECAY
    # Gradient clipping
    clip_gradients=dict(
        # Type of gradient clipping, currently 2 values are supported:
        # - "value": the absolute values of elements of each gradients are clipped
        # - "norm": the norm of the gradient for each parameter is clipped thus
        #   affecting all elements in the parameter
        clip_type="value",
        # Maximum absolute value used for clipping gradients
        clip_value=1.0,
        # Floating point number p for L-p norm to be used with the "norm"
        # gradient clipping type; for L-inf, please specify .inf
        norm_type=2.0,
        enabled=False,
    ),
    # Enable automatic mixed precision for training
    # Note that this does not change model's inference behavior.
    # To use AMP in inference, run inference under autocast()
    amp=dict(
        enabled=False
    )
)
  