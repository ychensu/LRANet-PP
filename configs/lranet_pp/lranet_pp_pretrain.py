find_unused_parameters=True
num_coefficients = 14
path_lra = './orthanchors/syntext_14.npz'

model = dict(
    type='LRANet_PP',
    from_p2=True,
    backbone=dict(
        type='ResNetV1d',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=2, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/ychensu/LRANet-PP/releases/download/model_zoo/resnet50vd_pretrained.pth'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
        act_cfg=None),
    bbox_head=dict(
        type='LRANet_PP_DetHead',
        in_channels=256,
        scales=(8, 16, 32),
        sample_size=(8,32),
        num_coefficients=num_coefficients,
        loss=dict(type='LRALoss', num_coefficients=num_coefficients,
                  path_lra = path_lra,),
        path_lra=path_lra,
        num_convs_stu=1,
        num_convs_tea=4),
    recog_head=dict(
        type='LRANet_PP_RecogHead',
        recognizer=dict(
            type='TransformerCTCPredictor',
            voc_size=96,
            depths=[3, 3, 3],
            dims=[256, 384, 512],
        ),
        num_sample_per_ins=1,
        image_size=(960,960),
        sample_size=(8,32),
        max_num=120,
        path_lra=path_lra,
        convertor=dict(type='CTCConvertor', lower=True, max_seq_len=25,
                       dict_list=[' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']),
    )
)

train_cfg = None
test_cfg = None

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile',
         # to_float32=True
         ),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomScaling', size=960, scale=(1. / 10, 6. / 2)),
    dict(
        type='RandomCropPolyInstancesWithText',
        instance_key='gt_masks',
        crop_ratio=0.8,
        min_side_ratio=0.1),
    dict(
        type='RandomRotatePolyInstances',
        rotate_ratio=0.5,
        max_angle=60,
        pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=960, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.0, direction='horizontal'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='LRATargets',
        level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0)),
        tps_size=(0.25,1),
        with_area=True,
        gauss_center=False,
        num_coefficients=num_coefficients,
        num_samples=3,
    ),
    dict(
        type='CustomFormatBundle',
        keys=['polygons_area','gt_texts','lv_tps_coeffs', 'lra_polys'],
        visualize=dict(flag=False, boundary_key=None)),
    dict(type='Collect', keys=['img', 'p3_maps', 'p4_maps', 'p5_maps', 'polygons_area', 'gt_texts','lv_tps_coeffs', 'lra_polys'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1800, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1280, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'IcdarE2EDataset'
data_root = 'data/'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=[
            data_root + 'ic13/ic13_train.json',
            data_root + 'ic15/ic15_train.json',
            data_root + 'textocr/textocr1_train.json',
            data_root + 'textocr/textocr2_train.json',
            data_root + 'syntext1/syntext1_train.json',
            data_root + 'mlt/mlt_train.json',
            data_root + 'syntext2/syntext2_train.json',
            data_root + 'totaltext/totaltext_train.json',
                  ],
        img_prefix=[
            data_root + 'ic13/train_images',
            data_root + 'ic15/train_images',
            data_root + 'textocr/train_images',
            data_root + 'textocr/train_images',
            data_root + 'syntext1/train_images',
            data_root + 'mlt/train_images',
            data_root + 'syntext2/train_images',
            data_root + 'totaltext/training',
            ],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'totaltext/totaltext_test.json',
        img_prefix=data_root + 'totaltext',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'totaltext/totaltext_test.json',
        img_prefix=data_root + 'totaltext',
        pipeline=test_pipeline,))
evaluation = dict(interval=10000, metric='hmean-e2e',by_epoch=False)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
        )


optimizer_config = dict(grad_clip=None)


# lr_config = dict(
#     policy='step', step=[160000, 220000],by_epoch=False,
#                  warmup='linear',warmup_iters=500,warmup_ratio=0.001,
#                  )
# runner = {
#             'type': 'IterBasedRunner',
#             'max_iters': 250000
#         }


lr_config = dict(
    policy='step', step=[360000, 430000],by_epoch=False,
                 warmup='linear',warmup_iters=500,warmup_ratio=0.001,
                 )
runner = {
            'type': 'IterBasedRunner',
            'max_iters': 450000
        }


checkpoint_config = dict(interval=10000, by_epoch=False)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook')

    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
