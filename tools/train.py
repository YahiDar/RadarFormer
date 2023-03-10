import os
import time
import json
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cruw import CRUW

from rodnet.datasets.CRDataset import CRDataset
from rodnet.datasets.CRDatasetSM import CRDatasetSM
from rodnet.datasets.CRDataLoader import CRDataLoader
from rodnet.datasets.loaders import list_pkl_filenames, list_pkl_filenames_from_prepared
from rodnet.datasets.collate_functions import cr_collate
from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.solve_dir import create_dir_for_new_model
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict
from rodnet.utils.visualization import visualize_train_img
from rodnet.models.backbones.T_RODNet import T_RODNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train RODNet.')

    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--sensor_config', type=str, default='sensor_config_rod2021')
    parser.add_argument('--data_dir', type=str, default='./data/', help='directory to the prepared data')
    parser.add_argument('--log_dir', type=str, default='./checkpoints/', help='directory to save trained model')
    parser.add_argument('--resume_from', type=str, default=None, help='path to the trained model')
    parser.add_argument('--save_memory', action="store_true", help="use customized dataloader to save memory")
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")
    parser.add_argument('--vis_train', type=int, default = 0, help="Choose whether to visualize the training images") 
    parser.add_argument('--validate', type=int, default  = None,help="Choose validation split, default is zero (all train). Split is only 6, 10, 20, and 30 sequences. ") 

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args

def count_params(model):
    """Count trainable parameters of a PyTorch Model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    return nb_params


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)  # update configs by args
    vis_train = args.vis_train
    if vis_train == 1:
        vis_train = True
    else:
        vis_train = False
    print(vis_train)
    validate = args.validate
    ### PRE-SHUFFLED PREVIOUSLY
    if validate is not None:
        list_train = list_pkl_filenames_from_prepared(args.data_dir, 'train')
        if validate == 6:
            valid_subset = ['2019_05_29_PM2S015.pkl', '2019_04_09_CMS1002.pkl', '2019_04_30_MLMS000.pkl', '2019_05_23_PM1S015.pkl', '2019_09_29_ONRD006.pkl', '2019_04_09_PMS1001.pkl']        
        elif validate == 10:
            valid_subset = ['2019_05_29_PM2S015.pkl', '2019_04_09_CMS1002.pkl', '2019_04_30_MLMS000.pkl', '2019_05_23_PM1S015.pkl', '2019_09_29_ONRD006.pkl', '2019_04_09_PMS1001.pkl',
                            '2019_04_30_PM2S004.pkl', '2019_09_29_ONRD001.pkl', '2019_05_29_PCMS005.pkl', '2019_05_29_BM1S016.pkl']
        elif validate == 20:
            valid_subset = ['2019_05_29_PM2S015.pkl', '2019_04_09_CMS1002.pkl', '2019_04_30_MLMS000.pkl', '2019_05_23_PM1S015.pkl', '2019_09_29_ONRD006.pkl', '2019_04_09_PMS1001.pkl',
                            '2019_04_30_PM2S004.pkl', '2019_09_29_ONRD001.pkl', '2019_05_29_PCMS005.pkl', '2019_05_29_BM1S016.pkl', '2019_05_09_MLMS003.pkl', '2019_04_30_PBMS002.pkl', 
                            '2019_04_30_MLMS001.pkl', '2019_04_30_PM2S003.pkl', '2019_09_29_ONRD002.pkl', '2019_05_09_PBMS004.pkl', '2019_04_09_PMS2000.pkl', '2019_05_23_PM1S012.pkl', 
                            '2019_04_30_PCMS001.pkl', '2019_05_29_BCMS000.pkl',]
        elif validate == 39:
            valid_subset = ['2019_04_09_BMS1000.pkl', '2019_04_09_BMS1001.pkl', '2019_04_09_BMS1002.pkl', '2019_04_09_CMS1002.pkl', '2019_04_09_PMS1000.pkl', '2019_04_09_PMS1001.pkl', 
                            '2019_04_09_PMS2000.pkl', '2019_04_09_PMS3000.pkl', '2019_04_30_MLMS000.pkl', '2019_04_30_MLMS001.pkl', '2019_04_30_MLMS002.pkl', '2019_04_30_PBMS002.pkl', 
                            '2019_04_30_PBMS003.pkl', '2019_04_30_PCMS001.pkl', '2019_04_30_PM2S003.pkl', '2019_04_30_PM2S004.pkl', '2019_05_09_BM1S008.pkl', '2019_05_09_CM1S004.pkl', 
                            '2019_05_09_MLMS003.pkl', '2019_05_09_PBMS004.pkl', '2019_05_09_PCMS002.pkl', '2019_05_23_PM1S012.pkl', '2019_05_23_PM1S013.pkl', '2019_05_23_PM1S014.pkl', 
                            '2019_05_23_PM1S015.pkl', '2019_05_23_PM2S011.pkl', '2019_05_29_BCMS000.pkl', '2019_05_29_BM1S016.pkl', '2019_05_29_BM1S017.pkl', '2019_05_29_MLMS006.pkl', 
                            '2019_05_29_PBMS007.pkl', '2019_05_29_PCMS005.pkl', '2019_05_29_PM2S015.pkl', '2019_05_29_PM3S000.pkl', '2019_09_29_ONRD001.pkl', '2019_09_29_ONRD002.pkl', 
                            '2019_09_29_ONRD005.pkl', '2019_09_29_ONRD006.pkl', '2019_09_29_ONRD011.pkl']

        else:
            valid_subset = ['2019_05_29_PM2S015.pkl', '2019_04_09_CMS1002.pkl', '2019_04_30_MLMS000.pkl', '2019_05_23_PM1S015.pkl', '2019_09_29_ONRD006.pkl', '2019_04_09_PMS1001.pkl', 
                            '2019_04_30_PM2S004.pkl', '2019_09_29_ONRD001.pkl', '2019_05_29_PCMS005.pkl', '2019_05_29_BM1S016.pkl', '2019_05_09_MLMS003.pkl', '2019_04_30_PBMS002.pkl', 
                            '2019_04_30_MLMS001.pkl', '2019_04_30_PM2S003.pkl', '2019_09_29_ONRD002.pkl', '2019_05_09_PBMS004.pkl', '2019_04_09_PMS2000.pkl', '2019_05_23_PM1S012.pkl', 
                            '2019_04_30_PCMS001.pkl', '2019_05_29_BCMS000.pkl', '2019_05_29_MLMS006.pkl', '2019_05_09_PCMS002.pkl', '2019_04_09_PMS3000.pkl', '2019_09_29_ONRD005.pkl', 
                            '2019_05_09_BM1S008.pkl', '2019_05_29_PBMS007.pkl', '2019_04_09_BMS1002.pkl', '2019_05_09_CM1S004.pkl', '2019_09_29_ONRD011.pkl', '2019_04_09_BMS1001.pkl']
    
        train_subset = [x for x in list_train if x not in valid_subset]

        print('Validation split is: %2d' %(len(valid_subset)))
        print('Train split is: %2d' %(len(train_subset)))
    else:
        print('all data will be used for training')
    torch.autograd.set_detect_anomaly(True)
    # dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'])
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    model_cfg = config_dict['model_cfg']
    if model_cfg['type'] == 'CDC':
        from rodnet.models import RODNetCDC as RODNet
    elif model_cfg['type'] == 'HG':
        from rodnet.models import RODNetHG as RODNet
    elif model_cfg['type'] == 'HGwI':
        from rodnet.models import RODNetHGwI as RODNet
    elif model_cfg['type'] == 'CDCv2':
        from rodnet.models import RODNetCDCDCN as RODNet
    elif model_cfg['type'] == 'HGv2':
        from rodnet.models import RODNetHGDCN as RODNet
    elif model_cfg['type'] == 'HGwIv2':
        from rodnet.models import RODNetHGwIDCN as RODNet
    elif model_cfg['type'] == 'HGwIv2_2d':
        from rodnet.models import RODNetHGwIDCN_2d as RODNet
    elif model_cfg['type'] == 'unetr_2d':
        from rodnet.models import UNETR_2d as RODNet
    elif model_cfg['type'] == 'unetr_2d_res_final':
        from rodnet.models import UNETR_2d_res_final as RODNet
    elif model_cfg['type'] == 'hrformer2d':
        from rodnet.models import HRFormer2d as RODNet
    elif model_cfg['type'] == 'maxvit2':
        from rodnet.models import MaxVIT2 as RODNet
    else:
        raise NotImplementedError

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    train_model_path = args.log_dir

    # create / load models
    cp_path = None
    epoch_start = 0
    iter_start = 0
    if args.resume_from is not None and os.path.exists(args.resume_from):
        cp_path = args.resume_from
        model_dir, model_name = create_dir_for_new_model(model_cfg['name'], train_model_path)
    else:
        model_dir, model_name = create_dir_for_new_model(model_cfg['name'], train_model_path)

    train_viz_path = os.path.join(model_dir, 'train_viz')
    if not os.path.exists(train_viz_path):
        os.makedirs(train_viz_path)

    writer = SummaryWriter(model_dir)
    save_config_dict = {
        'args': vars(args),
        'config_dict': config_dict,
    }
    config_json_name = os.path.join(model_dir, 'config-' + time.strftime("%Y%m%d-%H%M%S") + '.json')
    with open(config_json_name, 'w') as fp:
        json.dump(save_config_dict, fp)
    train_log_name = os.path.join(model_dir, "train.log")
    with open(train_log_name, 'w'):
        pass

    n_class = dataset.object_cfg.n_class
    if 'seed' in config_dict['train_cfg']:
        
        from numpy.random import Generator
        seed_hold = config_dict['train_cfg']['seed']

        torch.manual_seed(seed_hold)    
        random.seed(seed_hold)
        np.random.seed(seed_hold)
        rng = np.random.default_rng(seed_hold)
        print('Seed number is %i' %seed_hold)
    else:
        seed_flag = False
        print('No seed specified.')
    n_epoch = config_dict['train_cfg']['n_epoch']
    batch_size = config_dict['train_cfg']['batch_size']
    if validate == True:
        batch_size += 1
    lr = config_dict['train_cfg']['lr']
    if 'stacked_num' in model_cfg:
        stacked_num = model_cfg['stacked_num']
    else:
        stacked_num = None

    print("Building dataloader ... (Mode: %s)" % ("save_memory" if args.save_memory else "normal"))
    
    if not args.save_memory:
        if validate is not None:

            crdata_train = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train',
                                    noise_channel=args.use_noise_channel, subset = train_subset, testing_state=0)
            seq_names = crdata_train.seq_names
            index_mapping = crdata_train.index_mapping
            dataloader = DataLoader(crdata_train, batch_size, shuffle=True, num_workers=0, collate_fn=cr_collate)

            crdata_valid = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train',
                                    noise_channel=args.use_noise_channel, subset = valid_subset, testing_state=0)
            dataloader_valid = DataLoader(crdata_valid, batch_size, shuffle=False, num_workers=0, collate_fn=cr_collate)
        else:
            crdata_train = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train',
                                    noise_channel=args.use_noise_channel, testing_state=0)
            seq_names = crdata_train.seq_names
            index_mapping = crdata_train.index_mapping
            dataloader = DataLoader(crdata_train, batch_size, shuffle=True, num_workers=0, collate_fn=cr_collate)            
        # crdata_valid = CRDataset(os.path.join(args.data_dir, 'data_details'),
        #                          os.path.join(args.data_dir, 'confmaps_gt'),
        #                          win_size=win_size, set_type='valid', stride=8)
        # seq_names_valid = crdata_valid.seq_names
        # index_mapping_valid = crdata_valid.index_mapping
        # dataloader_valid = DataLoader(crdata_valid, batch_size=batch_size, shuffle=True, num_workers=0)

    else:
        crdata_train = CRDatasetSM(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='train',
                                   noise_channel=args.use_noise_channel)
        seq_names = crdata_train.seq_names
        index_mapping = crdata_train.index_mapping
        dataloader = CRDataLoader(crdata_train, shuffle=True, noise_channel=args.use_noise_channel)

        # crdata_valid = CRDatasetSM(os.path.join(args.data_dir, 'data_details'),
        #                          os.path.join(args.data_dir, 'confmaps_gt'),
        #                          win_size=win_size, set_type='train', stride=8, is_Memory_Limit=True)
        # seq_names_valid = crdata_valid.seq_names
        # index_mapping_valid = crdata_valid.index_mapping
        # dataloader_valid = CRDataLoader(crdata_valid, batch_size=batch_size, shuffle=True)

    if args.use_noise_channel:
        n_class_train = n_class + 1
    else:
        n_class_train = n_class

    print("Building model ... (%s)" % model_cfg)
    if model_cfg['type'] == 'CDC':
        rodnet = RODNet(in_channels=2, n_class=n_class_train).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HG':
        rodnet = RODNet(in_channels=2, n_class=n_class_train, stacked_num=stacked_num).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HGwI':
        rodnet = RODNet(in_channels=2, n_class=n_class_train, stacked_num=stacked_num).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'CDCv2':
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn']).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HGv2':
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn']).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HGwIv2':
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn']).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'HGwIv2_2d':
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        win_size=config_dict['train_cfg']['win_size'],
                        dcn=config_dict['model_cfg']['dcn']).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'unetr_v0' or model_cfg['type'] == 'unetr_v1':
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn'],
                        win_size = config_dict['train_cfg']['win_size'],
                        norm_layer = config_dict['model_cfg']['norm_layer'],
                        patch_size = config_dict['model_cfg']['patch_size'], 
                        hidden_size = config_dict['model_cfg']['hidden_size'], 
                        mlp_dim = config_dict['model_cfg']['mlp_dim'],
                        num_layers = config_dict['model_cfg']['num_layers'], 
                        num_heads = config_dict['model_cfg']['num_heads']).cuda()
        criterion = nn.BCELoss()
    elif (model_cfg['type'] == 'unetr_2d'):        
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn'],
                        win_size = config_dict['train_cfg']['win_size'],
                        norm_layer = config_dict['model_cfg']['norm_layer'],
                        patch_size = config_dict['model_cfg']['patch_size'], 
                        hidden_size = config_dict['model_cfg']['hidden_size'], 
                        receptive_field = config_dict['model_cfg']['receptive_field'],
                        mlp_dim = config_dict['model_cfg']['mlp_dim'],
                        num_layers = config_dict['model_cfg']['num_layers'], 
                        num_heads = config_dict['model_cfg']['num_heads']).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'hrformer2d':        
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn'],
                        win_size = config_dict['train_cfg']['win_size'],
                        norm_layer = config_dict['model_cfg']['norm_layer'],
                        patch_size = config_dict['model_cfg']['patch_size'], 
                        hidden_size = config_dict['model_cfg']['hidden_size'], 
                        receptive_field = config_dict['model_cfg']['receptive_field'],
                        channels_features = config_dict['model_cfg']['channels_features'],
                        mlp_dim = config_dict['model_cfg']['mlp_dim'],
                        num_layers = config_dict['model_cfg']['num_layers'], 
                        num_heads = config_dict['model_cfg']['num_heads']).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'unetr_2d_res_final':
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn'],
                        win_size = config_dict['train_cfg']['win_size'],
                        norm_layer = config_dict['model_cfg']['norm_layer'],
                        patch_size = config_dict['model_cfg']['patch_size'], 
                        hidden_size = config_dict['model_cfg']['hidden_size'], 
                        receptive_field = config_dict['model_cfg']['receptive_field'],
                        mlp_dim = config_dict['model_cfg']['mlp_dim'],
                        out_head = config_dict['model_cfg']['out_head'],
                        num_layers = config_dict['model_cfg']['num_layers'], 
                        num_heads = config_dict['model_cfg']['num_heads']).cuda()
        criterion = nn.BCELoss()
    elif model_cfg['type'] == 'maxvit2':
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn'],
                        win_size = config_dict['train_cfg']['win_size'],
                        patch_size = config_dict['model_cfg']['patch_size'], 
                        hidden_size = config_dict['model_cfg']['hidden_size'], 
                        receptive_field = config_dict['model_cfg']['receptive_field'],
                        out_head = config_dict['model_cfg']['out_head'],
                        num_layers = config_dict['model_cfg']['num_layers']).cuda()
        criterion = nn.BCELoss()
    else:
        raise TypeError
    optimizer = optim.Adam(rodnet.parameters(), lr=lr)
    if 'lr_type' in config_dict['train_cfg']:
        if config_dict['train_cfg']['lr_type'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, 
                                         T_max=config_dict['train_cfg']['t_max'],
                                         eta_min = config_dict['train_cfg']['lr_min'], 
                                         last_epoch = -1)
            print('Scheduler: Cosine Annealing')
        else:
            print('Scheduler: Step')
            scheduler = StepLR(optimizer, step_size=config_dict['train_cfg']['lr_step'], gamma=config_dict['train_cfg']['lr_factor'])
    else:
        print('No scheduler specified, setting to: Step')
        scheduler = StepLR(optimizer, step_size=config_dict['train_cfg']['lr_step'], gamma=config_dict['train_cfg']['lr_factor'])
    
    #scheduler = ReduceLROnPlateau(optimizer, mode = 'min', 
    #                             factor = 0.3, patience = 2, 
    #                             threshold = 1e-3, verbose = True)
    print(rodnet)
    iter_count = 0
    loss_ave = 0
    
    printing_flag = 0
    if cp_path is not None:
        checkpoint = torch.load(cp_path)
        print('loading checkpoint')
        if 'optimizer_state_dict' in checkpoint:
            rodnet.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch'] + 1
            iter_start = checkpoint['iter'] + 1
            loss_cp = checkpoint['loss']
            if 'iter_count' in checkpoint:
                iter_count = checkpoint['iter_count']
            if 'loss_ave' in checkpoint:
                loss_ave = checkpoint['loss_ave']
        else:
            rodnet.load_state_dict(checkpoint)
        del checkpoint
        for step in range((epoch_start-1)%config_dict['train_cfg']['lr_step']):
            scheduler.step()
            print(step)

    # print training configurations
    print("Model name: %s" % model_name)
    print("Number of sequences to train: %d" % crdata_train.n_seq)
    print("Training dataset length: %d" % len(crdata_train))
    print("Batch size: %d" % batch_size)
    print("Number of iterations in each epoch: %d" % int(len(crdata_train) / batch_size))
    print('optimizer currently is: ',optimizer)
    print('Number of trainable parameters: %s' % str(count_params(rodnet)))

    start_time = time.time()
    
    for epoch in range(epoch_start, n_epoch):
        print('Current optimizer is:',optimizer)
        tic_load = time.time()
        # if epoch == epoch_start:
        #     dataloader_start = iter_start
        # else:
        #     dataloader_start = 0
        
        for iter, data_dict in enumerate(dataloader):

            data = data_dict['radar_data']
            confmap_gt = data_dict['anno']['confmaps']
            image_paths = data_dict['image_paths']

            if not data_dict['status']:
                # in case load npy fail
                print("Warning: Loading NPY data failed! Skip this iteration")
                tic_load = time.time()
                continue

            tic = time.time()
            optimizer.zero_grad()  # zero the parameter gradients
            confmap_preds = rodnet(data.float().cuda())

            loss_confmap = 0
            if stacked_num is not None:
                if stacked_num != 1:
                    for i in range(stacked_num):
                        loss_cur = criterion(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap += loss_cur
                    loss_confmap.backward()
                    optimizer.step()
                else:
                    loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda())
                    loss_confmap.backward()
                    optimizer.step()
            else:
                loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda())
                loss_confmap.backward()
                optimizer.step()
            tic_back = time.time()
            if vis_train == False:
                del confmap_preds
            loss_ave = np.average([loss_ave, loss_confmap.item()], weights=[iter_count, 1])

            if iter % config_dict['train_cfg']['log_step'] == 0:
                # print statistics
                load_time = tic - tic_load
                back_time = tic_back - tic
                running_time = (tic_back - start_time)/3600
                print('epoch %2d, iter %4d: loss: %.4f (%.4f) | load time: %.2f | back time: %.2f | total time: %.2f | LR: %.8f' %
                     (epoch + 1, iter + 1, loss_confmap.item(), loss_ave, load_time, back_time, running_time, optimizer.param_groups[0]['lr']))

                with open(train_log_name, 'a+') as f_log:
                    if iter == 1 & printing_flag == 0:
                        printing_flag = 1
                        f_log.write("\nModel name: %s" % model_name)
                        f_log.write('\nNumber of trainable parameters: %s' % str(count_params(rodnet)))
                        f_log.write("\nNumber of sequences to train: %d" % crdata_train.n_seq)
                        f_log.write("\nTraining dataset length: %d" % len(crdata_train))
                        f_log.write("\nBatch size: %d" % batch_size)
                        f_log.write("\nNumber of iterations in each epoch: %d\n" % int(len(crdata_train) / batch_size))
                    f_log.write('epoch %2d, iter %4d: loss: %.4f (%.4f) | load time: %.2f | back time: %.2f | total time: %.2f | LR: %.8f\n' %
                                    (epoch + 1, iter + 1, loss_confmap.item(), loss_ave, load_time, back_time, running_time, optimizer.param_groups[0]['lr']))
                writer.add_scalar('loss/loss_all', loss_confmap.item(), iter_count)
                writer.add_scalar('loss/loss_ave', loss_ave, iter_count)
                writer.add_scalar('time/time_load', load_time, iter_count)
                writer.add_scalar('time/time_back', back_time, iter_count)
                #writer.add_scalar('param/param_lr', scheduler.get_last_lr()[0], iter_count)




                # draw train images
                
            if (iter + 1) % config_dict['train_cfg']['save_step'] == 0:
                # validate current model
                # print("validing current model ...")
                # validate()

                # save current model
                print("saving current model ...")
                status_dict = {
                    'model_name': model_name,
                    'epoch': epoch + 1,
                    'iter': iter + 1,
                    'model_state_dict': rodnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_confmap.item(),
                    'loss_ave': loss_ave,
                    'iter_count': iter_count,
                }
                save_model_path = '%s/epoch_%02d_iter_%010d.pkl' % (model_dir, epoch + 1, iter_count + 1)
                torch.save(status_dict, save_model_path)

            iter_count += 1
            
            #if iter_count % int(int(len(crdata_train) / batch_size)/6) == 0:
                #scheduler.step(loss_ave)
                #print(int(int(len(crdata_train) / batch_size))/6)
            tic_load = time.time()


        # save current model
        print("saving current epoch model ...")
        status_dict = {
            'model_name': model_name,
            'epoch': epoch,
            'iter': iter,
            'model_state_dict': rodnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_confmap.item(),
            'loss_ave': loss_ave,
            'iter_count': iter_count,
        }
        save_model_path = '%s/epoch_%02d_final.pkl' % (model_dir, epoch + 1)
        torch.save(status_dict, save_model_path)
        
        
        scheduler.step()
        print('Current optimizer is: ', optimizer)

        ############## VALIDATION STEP
        if validate is not None:
            with open(train_log_name, 'a+') as f_log:
                f_log.write("\n-----Validating-----")
                f_log.write("\nNumber of sequences to validate: %d" % crdata_valid.n_seq)
                f_log.write("\nValidating dataset length: %d" % len(crdata_valid))
                f_log.write("\nBatch size: %d" % batch_size)
                f_log.write("\nNumber of iterations in each epoch: %d" % int(len(crdata_valid) / batch_size))
            valid_loss_ave =0
            for iter, data_dict in enumerate(dataloader_valid):
                    
                data = data_dict['radar_data']
                confmap_gt = data_dict['anno']['confmaps']  
                image_paths = data_dict['image_paths']
                valid_confmap_preds = rodnet(data.float().cuda())
                valid_loss_confmap = criterion(valid_confmap_preds, confmap_gt.float().cuda())
                valid_loss_ave = np.average([valid_loss_ave, valid_loss_confmap.item()], weights=[iter, 1])
                
                if iter % config_dict['train_cfg']['log_step'] == 0:
                    # print statistics
                    print('\nepoch %2d, iter %4d: | valid_loss: %.4f (%.4f) | total time %.4f'
                        %(epoch +1, iter+1, valid_loss_confmap.item(),valid_loss_ave,running_time))
                
                if vis_train == True:
                    if stacked_num != 1:
                        confmap_pred = valid_confmap_preds[stacked_num - 1].cpu().detach().numpy()
                    else:
                        confmap_pred = valid_confmap_preds.cpu().detach().numpy()
                    if 'mnet_cfg' in model_cfg:
                        chirp_amp_curr = chirp_amp(data.numpy()[0, :, 0, 0, :, :], radar_configs['data_type'])
                    else:
                        chirp_amp_curr = chirp_amp(data.numpy()[0, :, 0, :, :], radar_configs['data_type'])
                    fig_name = os.path.join(train_viz_path,
                                        '%03d_%010d_%06d.png' % (epoch + 1, iter_count, iter + 1))
                    img_path = image_paths[0][0]
                    
                    visualize_train_img(fig_name, img_path, chirp_amp_curr,
                                    confmap_pred[0, :n_class, 0, :, :],
                                    confmap_gt[0, :n_class, 0, :, :])


                with open(train_log_name, 'a+') as f_log:
                    f_log.write('\nepoch %2d, iter %4d: | valid_loss: %.4f (%.4f) | total time %.4f'
                        %(epoch +1, iter+1, valid_loss_confmap.item(),valid_loss_ave,running_time))
                    if iter == int(len(crdata_valid) / batch_size):
                        
                        f_log.write("\n-----End of Validation-----\n")

    print('Training Finished.')
