_base_ = ['configs/base_modified.py']

# Adjusting point cloud range and voxel size
pc_range = [-52.2, -52.2, -5.5, 52.2, 52.2, 3.5]
voxel_dimensions = [0.25, 0.25, 8]

# Image normalization configuration
image_normalization_config = dict(
    avg=[124.675, 117.28, 104.53], variance=[59.395, 58.12, 58.375], to_rgb_format=True)

# Object classes for detection
DETECTION_CLASSES = [
    'automobile', 'heavy_truck', 'utility_vehicle', 'coach', 'semi_trailer', 'road_barrier',
    'motorbike', 'cycle', 'walker', 'traffic_marker'
]

# Input modality changes
input_type = dict(
    lidar_sensor=False,
    camera_sensor=True,
    radar_sensor=False,
    map_data=False,
    external_data=True)

embedding_dimension = 256
position_dimension = embedding_dimension // 2
feed_forward_network_dim = embedding_dimension * 2
hierarchy_levels = 1
bev_height = 60
bev_width = 60
sequence_length = 3

# Model configuration adjustments
model_config = dict(
    model_type='BEVFormerV2',
    grid_mask_enabled=True,
    video_mode=True,
    serial_feature_extraction=True,
    base_model=dict(img='torchvision://resnet50_v2'),
    image_main_backbone=dict(
        backbone_type='ResNetV2',
        layer_depth=50,
        stage_count=4,
        output_indices=(4, ),
        freeze_stages=1,
        normalization_config=dict(type='BatchNorm', trainable=False),
        evaluation_norm=True,
        pytorch_style=True,
        residual_zero_init=True),
    image_intermediate_neck=dict(
        neck_type='FPN',
        input_channels=[2048],
        output_channels=embedding_dimension,
        starting_level=0,
        add_convs='output_based',
        num_output_levels=hierarchy_levels,
        use_relu_convs=True),
    pts_bbox_head_config=dict(
        head_type='BEVFormerHeadV2',
        bev_map_height=bev_height,
        bev_map_width=bev_width,
        query_count=900,
        class_count=10,
        input_channel_dimension=embedding_dimension,
        synchronize_cls_avg_factor=True,
        box_refinement=True,
        two_stage_process=False,
        perception_transformer=dict(
            transformer_type='PerceptionTransformerV2',
            rotate_bev=True,
            shift_operation=True,
            can_bus_enabled=True,
            embedding_dims=embedding_dimension,
            encoder_config=dict(
                encoder_type='BEVFormerEncoderV2',
                layer_count=3,
                point_cloud_range=pc_range,
                pillar_point_count=4,
                intermediate_return=False,
                transformer_layers_config=dict(
                    layer_type='BEVFormerLayerV2',
                    attention_configs=[
                        dict(
                            attention_type='TemporalSelfAttentionV2',
                            embedding_dimensions=embedding_dimension,
                            level_count=1),
                        dict(
                            attention_type='SpatialCrossAttentionV2',
                            point_cloud_range=pc_range,
                            deformable_attention=dict(
                                attention_variant='MSDeformableAttention3DV2',
                                embedding_dimensions=embedding_dimension,
                                point_count=8,
                                level_count=hierarchy_levels),
                            embedding_dimensions=embedding_dimension,
                        )
                    ],
                    ffn_configs=dict(
                        ffn_type='FFNV2',
                        embedding_dimensions=256,
                        feedforward_channel_dimension=feed_forward_network_dim,
                        fc_count=2,
                        ffn_dropout=0.1,
                        activation_config=dict(type='ReLU', in_place=True),
                    ),
                    operation_sequence=('self_attention', 'normalization', 'cross_attention', 'normalization',
                                        'feed_forward_network', 'normalization'))),
            decoder_config=dict(
                decoder_type='Detr3DTransformerDecoderV2',
                layer_count=6,
                intermediate_layers=True,
                transformer_layers=dict(
                    decoder_layer_type='DetrTransformerDecoderLayerV2',
                    attention_configs=[
                        dict(
                            attention_type='MultiheadAttentionV2',
                            embedding_dimensions=embedding_dimension,
                            head_count=8,
                            dropout_rate=0.1),
                        dict(
                            attention_type='CustomMSDeformableAttentionV2',
                            embedding_dimensions=embedding_dimension,
                            level_count=1),
                    ],
                    ffn_configs=dict(
                        ffn_type='FFNV2',
                        embedding_dimensions=256,
                        feedforward_channel_dimension=feed_forward_network_dim,
                        fc_count=2,
                        ffn_dropout=0.1,
                        activation_config=dict(type='ReLU', in_place=True),
                    ),
                    operation_sequence=('self_attention', 'normalization', 'cross_attention', 'normalization',
                                        'feed_forward_network', 'normalization')))),
        bbox_coder_config=dict(
            coder_type='NMSFreeBBoxCoderV2',
            post_center_range=[-62.2, -62.2, -11.0, 62.2, 62.2, 11.0],
            point_cloud_range=pc_range,
            max_object_count=300,
            voxel_dimensions=voxel_dimensions,
            class_count=10),
        position_encoding_config=dict(
            encoding_type='LearnedPositionalEncodingV2',
            feature_count=pos_dimension,
            row_embedding_count=bev_height,
            column_embedding_count=bev_width,
        ),
        class_loss_config=dict(
            loss_type='FocalLossV2',
            sigmoid_activation=True,
            gamma_factor=2.0,
            alpha_factor=0.25,
            loss_weight_factor=2.0),
        bbox_loss_config=dict(loss_type='L1LossV2', loss_weight_factor=0.25),
        iou_loss_config=dict(loss_type='GIoULossV2', loss_weight_factor=0.0)),
    # Training and testing settings
    train_configuration=dict(
        points=dict(
            grid_dimension=[512, 512, 1],
            voxel_dimension=voxel_dimensions,
            point_cloud_range=pc_range,
            output_factor=4,
            assigner_config=dict(
                assigner_type='HungarianBBoxAssigner3DV2',
                class_cost=dict(cost_type='FocalLossCostV2', weight_factor=2.0),
                reg_cost=dict(cost_type='BBox3DL1CostV2', weight_factor=0.25),
                iou_cost=dict(
                    cost_type='IoUCostV2', weight_factor=0.0
                )))))

# Dataset configuration changes
dataset_config = 'NuScenesDatasetV2'
data_directory = 'data/nuScenes_v2/nuscenes-v2.0/'

# Adjusting training pipeline
training_pipeline = [
    dict(op_type='ScaleAdjustmentMultiView', scale_factors=[0.55]),
    dict(op_type='PhotoMetricDistortionMultiView'),
    dict(op_type='PointCloudRangeFilter', pc_range=pc_range),
    dict(op_type='ObjectNameFilterV2', object_classes=DETECTION_CLASSES),
    dict(op_type='ImageNormalization', **image_normalization_config),
    dict(op_type='ImagePadding', size_divisor=34),
    dict(op_type='DataFormatBundle3DV2', class_names=DETECTION_CLASSES),
    dict(
        op_type='DataCollector3D',
        data_keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
        meta_keys=('file_name', 'original_shape', 'image_shape', 'lidar_to_img',
                   'depth_to_img', 'camera_to_img', 'padded_shape', 'scaling_factor', 'flipping',
                   'horizontal_flip_pcd', 'vertical_flip_pcd', '3d_box_mode',
                   'box_3d_type', 'image_norm_config', 'point_cloud_transform', 'sample_index',
                   'previous_index', 'next_index', 'pcd_scaling_factor', 'pcd_rotation_angle',
                   'point_cloud_file', 'transformation_3d_flow', 'scene_identifier',
                   'can_bus_data'))
]

# Adjusting testing pipeline
testing_pipeline = [
    dict(op_type='ImageNormalization', **image_normalization_config),
    dict(
        op_type='MultiScaleFlipAugmentation3D',
        image_resolution=(1600, 920),
        point_scale_ratio=1.05,
        flipping=False,
        transformation_steps=[
            dict(op_type='ScaleAdjustmentMultiView', scale_factors=[0.55]),
            dict(op_type='ImagePadding', size_divisor=34),
            dict(
                op_type='DataFormatBundle3DV2',
                class_names=DETECTION_CLASSES,
                include_label=False),
            dict(
                op_type='DataCollector3D',
                data_keys=['img'],
                meta_keys=('file_name', 'original_shape', 'image_shape', 'lidar_to_img',
                           'depth_to_img', 'camera_to_img', 'padded_shape', 'scaling_factor',
                           'flipping', 'horizontal_flip_pcd', 'vertical_flip_pcd',
                           '3d_box_mode', 'box_3d_type', 'image_norm_config',
                           'point_cloud_transform', 'sample_index', 'previous_index', 'next_index',
                           'pcd_scaling_factor', 'pcd_rotation_angle', 'point_cloud_file',
                           'transformation_3d_flow', 'scene_identifier', 'can_bus_data'))
        ])
]

# Data configuration adjustments
data_config = dict(
    images_per_gpu=1,
    workers_per_gpu=4,
    enable_pin_memory=True,
    train=dict(
        dataset_type=dataset_config,
        data_source=dict(
            source_type='Det3dSourceNuScenesV2',
            data_root=data_directory,
            annotation_file=data_directory + 'nuscenes_infos_temporal_train_v2.pkl',
            data_pipeline=[
                dict(
                    op_type='LoadMultiViewImages',
                    convert_to_float32=True,
                    image_backend='turbojpeg_v2'),
                dict(
                    op_type='Load3DAnnotations',
                    include_bbox_3d=True,
                    include_label_3d=True,
                    attribute_label_inclusion=False)
            ],
            object_classes=DETECTION_CLASSES,
            modality=input_type,
            testing_mode=False,
            use_valid_flag=True,
            box_3d_type='LiDAR'),
        pipeline=training_pipeline,
        sequence_length=sequence_length,
    ),
    val=dict(
        images_per_gpu=1,
        dataset_type=dataset_config,
        data_source=dict(
            source_type='Det3dSourceNuScenesV2',
            data_root=data_directory,
            annotation_file=data_directory + 'nuscenes_infos_temporal_val_v2.pkl',
            data_pipeline=[
                dict(
                    op_type='LoadMultiViewImages',
                    convert_to_float32=True,
                    image_backend='turbojpeg_v2')
            ],
            object_classes=DETECTION_CLASSES,
            modality=input_type,
            testing_mode=True),
        pipeline=testing_pipeline))

# Optimizer configuration adjustments
optimizer_config = dict(
    opt_type='AdamWOptimizer',
    learning_rate=2e-4,
    paramwise_options={'image_main_backbone': {'learning_rate_multiplier': 0.1}},
    weight_decay_rate=0.01)

# Gradient clipping configuration
grad_clip_config = dict(max_norm_value=35, norm_type_value=2)

# Learning rate schedule configuration
lr_schedule_config = dict(
    schedule_policy='CosineAnnealingSchedule',
    warmup_strategy='linear',
    warmup_iterations=500,
    warmup_ratio=1.0 / 3,
    minimum_lr_ratio=1e-3)
total_training_epochs = 24

# Evaluation and logging configurations
evaluation_config = dict(start_evaluation=False, evaluation_interval=1, gpu_based_collection=False)
evaluation_pipeline = [
    dict(
        evaluation_mode='test',
        data_configuration=data_config['val'],
        distributed_evaluation=True,
        evaluation_metrics=[
            dict(
                evaluator_type='NuScenesMetricEvaluator',
                object_classes=DETECTION_CLASSES,
                result_tags=['pts_bbox'])
        ],
    )
]

logging_config = dict(
    log_interval=50,
    logging_hooks=[dict(hook_type='TextLoggingHook'),
                   dict(hook_type='TensorboardLoggingHook')])

checkpointing_config = dict(save_interval=1)
cudnn_optimization = True
export_config = dict(
    export_variant='blade_v2',
    blade_configuration=dict(
        enable_fp16_computation=True,
        fp16_fallback_ratio=0.0,
        custom_operation_blacklist=[
            'aten::select_v2', 'aten::index_v2', 'aten::slice_v2', 'aten::view_v2',
            'aten::upsample_v2', 'aten::clamp_v2'
        ]))
# Continuing with additional configurations for data handling

# Additional training data augmentation
additional_training_augmentation = [
    dict(op_type='RandomFlip3D', flip_ratio=0.5),
    dict(op_type='GlobalRotScaleTrans', rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
    dict(op_type='PointsRangeFilter', point_range=pc_range),
    dict(op_type='ObjectSample', sample_range=pc_range),
    dict(op_type='DefaultFormatBundle3D'),
    dict(op_type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# Additional validation data augmentation
additional_validation_augmentation = [
    dict(op_type='DefaultFormatBundle3D'),
    dict(op_type='Collect3D', keys=['points'])
]

# Incorporating additional augmentations into the data configuration
data_config['train']['additional_pipeline'] = additional_training_augmentation
data_config['val']['additional_pipeline'] = additional_validation_augmentation

# Adjusting checkpoint configuration
checkpoint_configuration = dict(
    save_strategy='epoch_based',
    save_interval=1,
    save_best_only=False
)

# Adjusting runtime settings
runtime_config = dict(
    deterministic_mode=False,
    cudnn_benchmark_enabled=cudnn_optimization,
    work_directory='work_dirs/bevformer_experiment',
    gpu_ids=range(1)
)

# Model training and validation settings
train_val_settings = dict(
    model=model_config,
    train_data=data_config['train'],
    val_data=data_config['val'],
    optimizer=optimizer_config,
    optimizer_config=grad_clip_config,
    lr_config=lr_schedule_config,
    total_epochs=total_training_epochs,
    log_config=logging_config,
    checkpoint_config=checkpoint_configuration,
    runtime_config=runtime_config,
    eval_config=evaluation_config,
    eval_pipelines=evaluation_pipeline
)

# Export settings for model deployment
model_export_settings = dict(
    export_mode='inference',
    export_directory='exported_models',
    export_name='bevformer_v2_model'
)

# Finalizing the configuration
final_configuration = {
    'base': _base_,
    'model': train_val_settings,
    'data': data_config,
    'optimizer': optimizer_config,
    'lr_schedule': lr_schedule_config,
    'log': logging_config,
    'checkpoint': checkpoint_configuration,
    'runtime': runtime_config,
    'export': model_export_settings
}

# Continuing with additional configurations for data handling
# Additional training data augmentation
additional_training_augmentation = [
    dict(op_type='RandomFlip3D', flip_ratio=0.5),
    dict(op_type='GlobalRotScaleTrans', rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0], translation_std=[0, 0, 0]),
    dict(op_type='PointsRangeFilter', point_range=pc_range),
    dict(op_type='ObjectSample', sample_range=pc_range),
    dict(op_type='DefaultFormatBundle3D'),
    dict(op_type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# Additional validation data augmentation
additional_validation_augmentation = [
    dict(op_type='DefaultFormatBundle3D'),
    dict(op_type='Collect3D', keys=['points'])
]

# Incorporating additional augmentations into the data configuration
data_config['train']['additional_pipeline'] = additional_training_augmentation
data_config['val']['additional_pipeline'] = additional_validation_augmentation

# Adjusting checkpoint configuration
checkpoint_configuration = dict(
    save_strategy='epoch_based',
    save_interval=1,
    save_best_only=False
)

# Adjusting runtime settings
runtime_config = dict(
    deterministic_mode=False,
    cudnn_benchmark_enabled=cudnn_optimization,
    work_directory='work_dirs/bevformer_experiment',
    gpu_ids=range(1)
)

# Model training and validation settings
train_val_settings = dict(
    model=model_config,
    train_data=data_config['train'],
    val_data=data_config['val'],
    optimizer=optimizer_config,
    optimizer_config=grad_clip_config,
    lr_config=lr_schedule_config,
    total_epochs=total_training_epochs,
    log_config=logging_config,
    checkpoint_config=checkpoint_configuration,
    runtime_config=runtime_config,
    eval_config=evaluation_config,
    eval_pipelines=evaluation_pipeline
)

# Export settings for model deployment
model_export_settings = dict(
    export_mode='inference',
    export_directory='exported_models',
    export_name='bevformer_v2_model'
)

# Finalizing the configuration
final_configuration = {
    'base': _base_,
    'model': train_val_settings,
    'data': data_config,
    'optimizer': optimizer_config,
    'lr_schedule': lr_schedule_config,
    'log': logging_config,
    'checkpoint': checkpoint_configuration,
    'runtime': runtime_config,
    'export': model_export_settings
}