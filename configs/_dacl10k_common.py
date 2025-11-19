# data_root = "/Users/diegovelasco/Desktop/Diego/FING/DLFCV-FinalProject/data_clean" LOCAL
data_root = "/content/data_clean" # COLAB

CLASSES = [
    'Crack',
    'ExposedRebars',
    'Spalling',
    'Rust',
    'ACrack',
    'Rockpocket',
    'Hollowareas',
    'Efflorescence',
    'Cavity',
    'Wetspot',
    'Weathering',
    'Restformwork',
    'Graffiti',
]

PALETTE = [
    [220, 20, 60],
    [119, 11, 32],
    [0, 0, 142],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
]


metainfo = dict(classes=CLASSES, palette=PALETTE)

backend_args = None

img_scale = (1024, 1024)  # drop to (512,512) if VRAM is tight

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2,              # start safe for your PC
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(
            img_path="images/train",
            seg_map_path="masks/train",
        ),
        pipeline=train_pipeline,
        seg_map_suffix=".png",
        reduce_zero_label=False,
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(
            img_path="images/validation",
            seg_map_path="masks/validation",
        ),
        pipeline=test_pipeline,
        seg_map_suffix=".png",
        reduce_zero_label=False,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(
            img_path="images/test",
            seg_map_path="masks/test",
        ),
        pipeline=test_pipeline,
        seg_map_suffix=".png",
        reduce_zero_label=False,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    ignore_index=255,
)

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    ignore_index=255,
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-4, weight_decay=0.01),
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False,
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,
    val_interval=2000,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2000, max_keep_ckpts=3, save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=2000),
)

default_scope = "mmseg"
