# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/cascade_mask_rcnn_convnext_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        in_chans=3,
        depths=[3, 3, 27, 3], 
        dims=[256, 512, 1024, 2048], 
        drop_path_rate=0.8,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
    ),
    neck=dict(in_channels=[256, 512, 1024, 2048]),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=498,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=498,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=498,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='AutoAugment',
#          policies=[
#              [
#                  dict(type='Resize',
#                       img_scale=[(480, 864), (512, 864), (544, 864), (576, 864),
#                                  (608, 864), (640, 864), (672, 864), (704, 864),
#                                  (736, 864), (768, 864), (800, 864)],
#                       multiscale_mode='value',
#                       keep_ratio=True)
#              ],
#              [
#                  dict(type='Resize',
#                       img_scale=[(400, 1333), (500, 1333), (600, 1333)],
#                       multiscale_mode='value',
#                       keep_ratio=True),
#                  dict(type='RandomCrop',
#                       crop_type='absolute_range',
#                       crop_size=(384, 600),
#                       allow_negative_crop=True),
#                  dict(type='Resize',
#                       img_scale=[(480, 864), (512, 864), (544, 864),
#                                  (576, 864), (608, 864), (640, 864),
#                                  (672, 864), (704, 864), (736, 864),
#                                  (768, 864), (800, 864)],
#                       multiscale_mode='value',
#                       override=True,
#                       keep_ratio=True)
#              ]
#          ]),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 640), (512, 640), (544, 640), (576, 640),
                                 (608, 640), (640, 640), (672, 640), (704, 640),
                                 (736, 640), (768, 640), (800, 640)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 640), (512, 640), (544, 640),
                                 (576, 640), (608, 640), (640, 640),
                                 (672, 640), (704, 640), (736, 640),
                                 (768, 640), (800, 640)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', _delete_=True, type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.7,
                                'decay_type': 'layer_wise',
                                'num_layers': 12})
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)
