# Common training-related configs that are designed for "tools/lazyconfig_train_net.py"
# You can use your own instead, together with your own train_net.py
train = dict(
    output_dir="/workspace",
    init_checkpoint="",
    uuid="",
    max_iter=90000,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=2000, max_to_keep=10),  # options for PeriodicCheckpointer
    eval_period=5000,
    log_period=20,
    device="cuda"
    # ...
)

# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
test=dict(
    expected_results=[]
)