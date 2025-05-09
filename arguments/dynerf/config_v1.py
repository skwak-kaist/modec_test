ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2],
    defor_depth = 0,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False,

    anchor_deform = True,
    local_context_feature_deform = True,
    grid_offsets_deform = False,
    grid_scale_deform = False,
    
    deform_feat_dim = 32, # ModelParams의 feat_dim과 일치시켜야 함
    deform_n_offsets = 20, # ModelParams의 n_offsets과 일치시켜야 함

)

ModelParams = dict(
    appearance_dim = 16,
    n_offsets = 20, 
    use_feat_bank = True, 
    voxel_size = 0.001,
    feat_dim = 32, # ModelHiddenParams의 feat_dim과 일치시켜야 함
    
)

OptimizationParams = dict(
    dataloader=True,
    iterations = 14000,
    batch_size=4,
    coarse_iterations = 3000,
    densify_until_iter = 10_000,
    opacity_reset_interval = 60000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    
    min_opacity = 0.005,
    success_threshold = 0.8,
    densify_grad_threshold = 0.0002,
    
    # pruning_interval = 2000
)
