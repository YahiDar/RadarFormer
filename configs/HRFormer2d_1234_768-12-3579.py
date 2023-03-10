dataset_cfg = dict(
    dataset_name='ROD2021',
    base_root="./Pickletest",
    data_root="./Pickletest",
    anno_root="./Pickletest",
    anno_ext='.txt',
    train=dict(
        subdir='train',
        # seqs=[],  # can choose from the subdir folder
    ),
    valid=dict(
        subdir='valid',
        # seqs=[],
    ),
    test=dict(
        subdir='test',
        # seqs=[],  # can choose from the subdir folder
    ),
    demo=dict(
        subdir='demo',
        seqs=[],
    ),
)

model_cfg = dict(
    type='hrformer2d',
    name='hrformer2d-1234-768-12-3579',
    max_dets=20,
    peak_thres=0.3,
    ols_thres=0.3,
    stacked_num=1,
    mnet_cfg=(4, 32),                        
    patch_size = 8, 
    hidden_size = 760, 
    channels_features = (1,2,3,4),
    receptive_field = [3,5,7,9],
    mlp_dim = 4800,
    num_layers = 12, 
    norm_layer = 'batch',
    num_heads = 20,
    dcn=False,
)

confmap_cfg = dict(
    confmap_sigmas={
        'pedestrian': 15,
        'cyclist': 20,
        'car': 30,
        # 'van': 40,
        # 'truck': 50,
    },
    confmap_sigmas_interval={
        'pedestrian': [5, 15],
        'cyclist': [8, 20],
        'car': [10, 30],
        # 'van': [15, 40],
        # 'truck': [20, 50],
    },
    confmap_length={
        'pedestrian': 1,
        'cyclist': 2,
        'car': 3,
        # 'van': 4,
        # 'truck': 5,
    }
)

train_cfg = dict(
    n_epoch=100,
    batch_size=4,
    lr=0.0001,
    lr_step=10,
    lr_factor=0.1,  # lr will decrease 10 times after lr_step epoches
    win_size=32,
    train_step=1,
    train_stride=4,
    log_step=10,
    save_step=10000,
)
test_cfg = dict(
    test_step=1,
    test_stride=8,
    rr_min=1.0,  # min radar range
    rr_max=20.0,  # max radar range
    ra_min=-60.0,  # min radar angle
    ra_max=60.0,  # max radar angle
)
