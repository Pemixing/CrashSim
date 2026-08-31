# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import argparse
import configargparse

dmode_list = ['x2x-cond-z', 
              'x2x2d-cond-z',
              'z2z-cond-none-2L', 
              'z2z-cond-z',
              'z2z-cond-z-2L',
              'x2x-cond-z-2L',
              'x2x2d-cond-z-2L',
              'z2z-cond-zh',]

def parse_diffusion_mode(config_dict: dict):
    
    dim_values = [(4, 0, config_dict['latent_size'], 'l2'),
                  (2, 0, config_dict['latent_size'], 'l2'),
                  (1, 0, 0, 'l2'),
                  (1, 0, config_dict['latent_size'], 'l2'),
                  (1, 0, config_dict['latent_size'], 'l2'),
                  (4, 0, config_dict['latent_size'], 'l2'),
                  (2, 0, config_dict['latent_size'], 'l2'),
                  (1, 0, config_dict['latent_size'] + config_dict['map_feat_size'] + config_dict['past_feat_size'], 'l2'),
                  ]
    dim_dict = {key: value for key, value in zip(dmode_list, dim_values)} 

    action_dim, observation_dim, cond_dim, loss_type = dim_dict[config_dict['dmode']]
    horizon = config_dict['latent_size'] if config_dict['dmode'][:3] == 'z2z' else config_dict['future_len']
    transition_dim = action_dim + observation_dim

    # train_both = ('z2z' in config_dict['dmode']) and ('2L' in config_dict['dmode'])
    train_both = False

    config_dict.update({'action_dim': action_dim, 
                        'observation_dim': observation_dim, 
                        'cond_dim': cond_dim, 
                        'loss_type': loss_type, 
                        'horizon': horizon, 
                        'transition_dim': transition_dim,
                        'train_both': train_both,})
    return 

def get_parser(description):
    parser = configargparse.ArgParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                                      config_file_parser_class=configargparse.YAMLConfigFileParser,
                                      description=description)

    # YAML configuration file
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='yaml config file path')

    return parser

def add_base_args(parser):
    #ddp
    parser.add_argument("--local-rank", default=-1, type=int)

    # wandb (optional)
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Name of project to use for wandb. Only used if not None. Note WANDB_API_KEY env variable must also be set properly.')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Display name for this run on wandb')
    parser.add_argument('--wandb_offline', dest='wandb_offline', action='store_true',
                        help="If given, only save wandb data offline.")
    parser.set_defaults(wandb_offline=False)

    # logging
    parser.add_argument('--out', type=str, default='./out/traffic_out',
                        help='Directory to save model weights and logs to.')

    # dataset options
    parser.add_argument('--data_dir', type=str, default='./data/nuscenes',
                        help='Directory to load nuScenes data from')
    parser.add_argument('--data_version', type=str, default='trainval',
                        choices=['trainval', 'mini'], help='Whether to use full nuscenes or mini.')
    parser.add_argument('--use_challenge_splits', dest='use_challenge_splits', action='store_true',
                        help="If given, loads nuScenes prediction challenge splits for data")
    parser.set_defaults(use_challenge_splits=False)

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers to use')

    parser.add_argument('--past_len', type=int, default=4, help='Number of past (input) timesteps to use.')
    parser.add_argument('--future_len', type=int, default=12, help='Number of future (output prediction) timesteps to use.')
    parser.add_argument('--agent_types', type=str, nargs='+', default=['car', 'truck'],
                        help='Which kinds of agents to include in the nuscenes dataset')
    parser.add_argument('--reduce_cats', dest='reduce_cats', action='store_true',
                        help="If given, reduce categories to be only one of car, truck, cyclist, or pedestrian")
    parser.set_defaults(reduce_cats=False)

    # map env options
    parser.add_argument('--map_obs_size_pix', type=int, default=256, help='width of the map observation around each agent (in pixels)')
    parser.add_argument('--map_obs_bounds', type=float, nargs=4, default=[-17.0, -38.5, 60.0, 38.5],
                         help='Bounds (in meters) for observations crop. In order [low_l, low_w, high_l, high_w]')
    parser.add_argument('--map_layers', type=str, nargs='+', default=['drivable_area', 'carpark_area', 'road_divider', 'lane_divider'],
                         help='which nuscenes map layers to return from the environment.')  

    # model options
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model weights to load in.')
    parser.add_argument('--map_feat_size', type=int, default=64, help='Feature size for map crop encoding.')
    parser.add_argument('--past_feat_size', type=int, default=64, help='Feature size for past trajectory encoding.')
    parser.add_argument('--future_feat_size', type=int, default=64, help='Feature size for future traj encoding.')
    parser.add_argument('--latent_size', type=int, default=32, help='CVAE latent space dim.')
    parser.add_argument('--no_output_bicycle', dest='model_output_bicycle', action='store_false',
                        help="If given, does not use kinematic bicycle model as output parameterization, instead directly predicts waypoints.")
    parser.set_defaults(model_output_bicycle=True)

    parser.add_argument('--conv_kernel_list', type=int, nargs='+', default=[7, 5, 5, 3, 3, 3],
                         help='Kernel size to use in each layer of map encoder')
    parser.add_argument('--conv_stride_list', type=int, nargs='+', default=[2, 2, 2, 2, 2, 2],
                         help='Stride size to use in each layer of map encoder')
    parser.add_argument('--conv_filter_list', type=int, nargs='+', default=[16, 32, 64, 64, 128, 128],
                         help='Num filters to output from each layer of map encoder')
    
    parser.add_argument('--dmode', type=str, default='x2x-cond-z', choices= dmode_list, 
                        help='diffusion mode, choose different inputs and conditions')

    return parser
