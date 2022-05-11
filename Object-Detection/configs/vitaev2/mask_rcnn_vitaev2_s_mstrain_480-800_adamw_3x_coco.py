_base_ = [
    '../_base_/models/mask_rcnn_vitaev2_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        in_chans=3,
        RC_tokens_type=['window', 'window', 'transformer', 'transformer'], 
        NC_tokens_type=['window', 'window', 'transformer', 'transformer'], 
        embed_dims=[64, 64, 128, 256], 
        token_dims=[64, 128, 256, 512], 
        downsample_ratios=[4, 2, 2, 2],
        NC_depth=[2, 2, 8, 2], 
        NC_heads=[1, 2, 4, 8], 
        RC_heads=[1, 1, 2, 4], 
        mlp_ratio=4., 
        NC_group=[1, 32, 64, 128], 
        RC_group=[1, 16, 32, 64],
        use_checkpoint=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        window_size=7,
    ),
    neck=dict(in_channels=[64, 128, 256, 512]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      # [722, 754, 786, 818, 850, 882, 914, 946, 978, 1010, 1042]
                      img_scale=[(690, 1920), (722, 1920), (768, 1920), (818, 1920),
                                 (850, 1920), (882, 1920), (914, 1920), (946, 1920),
                                 (978, 1920), (1010, 1920), (1042, 1920)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(658, 1920), (758, 1920), (858, 1920)], # similar
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(768, 1200), # double
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(690, 1920), (722, 1920), (768, 1920), (818, 1920),
                                 (850, 1920), (882, 1920), (914, 1920), (946, 1920),
                                 (978, 1920), (1010, 1920), (1042, 1920)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
    type='MultiScaleFlipAug',
    img_scale=(1042, 1920),
    flip=False,
    transforms=[
    dict(type='Resize', keep_ratio=True),
    dict(type='RandomFlip'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
    ])
]

data = dict(samples_per_gpu=1,
            train=dict(pipeline=train_pipeline, ann_file="data/coco/annotations/instances_train2017.json", img_prefix="data/coco/train2017"),
            val=dict(ann_file="data/coco/annotations/instances_val2017.json", img_prefix="data/coco/val2017"),
            test=dict(ann_file="data/coco/annotations/instances_val2017.json", img_prefix="data/coco/val2017"))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
work_dir = "/valohai/outputs/"
data_root = "data/coco/"
workflow = [('train', 1), ('val', 1)]
find_unused_parameters=True
# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
