# configs/dacl10k_unet.py
_base_ = ["_dacl10k_common.py"]

num_classes = 14  # background + 13 defects

model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr2rgb=True,
        pad_val=0,
        seg_pad_val=255,
    ),
    backbone=dict(
        type="UNet",
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 2, 2, 2, 2),
        enc_kernel_sizes=(3, 3, 3, 3, 3),
        dec_kernel_sizes=(3, 3, 3, 3),
        downsamples=(True, True, True, True),
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type="ReLU"),
    ),
    decode_head=dict(
        type="UNetHead",
        in_channels=64,
        channels=64,
        num_classes=num_classes,
        loss_decode=[
            dict(type="CrossEntropyLoss", loss_weight=1.0),
            dict(type="DiceLoss", naive_dice=True, eps=1e-6, loss_weight=0.5),
        ],
        ignore_index=255,
        align_corners=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)