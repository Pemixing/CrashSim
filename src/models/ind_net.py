# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import itertools

import numpy as np

import torch
from torch import nn
from torch.distributions import Normal

from models.interaction_net import SceneInteractionNet
from models.common import MLP, car_dynamics

from datasets.utils import normalize_scene_graph
from utils.transforms import transform2frame, kinematics2angle, kinematics2vec
from utils.torch import calc_conv_out
from utils.logger import throw_err, Logger

TRAJ_ENCODER_CHOICES = ['mlp', 'gru']

class IndNet(nn.Module):
    def __init__(self, npast, nfuture, map_obs_size_pix, nclasses,
                 map_feat_size=64,
                 past_feat_size=64,
                 output_bicycle=True,
                 traj_encoder='mlp',
                 conv_channel_in=4,
                 conv_kernel_list=[7, 5, 5, 3, 3, 3],
                 conv_stride_list=[2, 2, 2, 2, 2, 2],
                 conv_filter_list=[16, 32, 64, 64, 128, 128]
                 ):
        '''
        :param npast: number of past steps to take as input
        :param nfuture: number of future steps to predict as output
        :param map_obs_size_pix: width (in pixels) of map crop
        :param nclasses: number of different semantic classes
        '''
        super().__init__()

        self.normalizer = self.att_normalizer = None # normalizer for state and vehicle attributes
        self.PT = npast
        self.FT = nfuture
        self.dt = 0.5 # for nusc dataset
        self.NC = nclasses
        self.output_bicycle = output_bicycle
        if self.output_bicycle:
            self.bicycle_params = None
            Logger.log('Using bicycle model as output parameterization of model...')

        self.state_size = 6 #(x,y,hx,hy,s,hdot)
        self.att_feat_size = 2 #(l,w)

        self.traj_encoder_type = traj_encoder
        if self.traj_encoder_type not in TRAJ_ENCODER_CHOICES:
            throw_err('Trajectory encoder type %s not recognized!' % (self.traj_encoder_type))
        else:
            Logger.log('Using %s past/future encoder...' % (self.traj_encoder_type))

        # Map encoding
        self.mapH = map_obs_size_pix
        self.mapW = map_obs_size_pix
        self.map_obs_size_pix = map_obs_size_pix

        conv_layer_list = []
        final_conv_out = map_obs_size_pix
        assert len(conv_kernel_list) == len(conv_stride_list)
        assert len(conv_kernel_list) == len(conv_filter_list)
        conv_filter_list = [conv_channel_in] + conv_filter_list
        for lidx in range(len(conv_kernel_list)):
            cur_conv = nn.Conv2d(conv_filter_list[lidx],
                                 conv_filter_list[lidx+1],
                                 kernel_size=conv_kernel_list[lidx],
                                 stride=conv_stride_list[lidx],
                                 padding=0)
            cur_gn = nn.GroupNorm(1, conv_filter_list[lidx+1])
            conv_layer_list.extend([cur_conv, cur_gn, nn.ReLU()])
            final_conv_out = calc_conv_out(final_conv_out, conv_kernel_list[lidx], conv_stride_list[lidx])

        self.map_conv = nn.Sequential(*conv_layer_list)
        self.map_feat_in_size = conv_filter_list[-1] * final_conv_out * final_conv_out
        self.map_feat_out_size = map_feat_size
        self.map_feature = nn.Linear(self.map_feat_in_size, self.map_feat_out_size)

        # past encoder
        self.past_feat_size = past_feat_size
        if self.traj_encoder_type == 'mlp':
            self.past_in_size = self.NC + self.PT*(self.state_size + self.att_feat_size + 1) # +1 from visibility flag
            self.past_encoder = MLP([self.past_in_size, 128, 128, 128, self.past_feat_size])
        elif self.traj_encoder_type == 'gru':
            self.past_in_size = self.NC + self.state_size + self.att_feat_size + 1 # +1 from visibility flag
            self.past_encoder = nn.GRU(self.past_in_size,
                                        128, # hidden size
                                        4, # num stacked GRU layers
                                        batch_first=True
                                        )
            self.past_out_layer = nn.Linear(128, self.past_feat_size)

        # traj decoder
        if self.output_bicycle:
            self.traj_out_size = 2 # (a,hdot) for a single step
        else:
            self.traj_out_size = 4 # (x,y,hx,hy) for a single step
        decode_in_size = self.past_feat_size + self.map_feat_out_size + self.NC + self.att_feat_size
        # self.decoder_net =  MLP([decode_in_size, 128, 128, 128, self.traj_out_size])
        self.decoder_net =  MLP([decode_in_size, 256, 512, 256, self.traj_out_size])
        # self.decoder_net =  MLP([decode_in_size, 64, 256, 64, self.traj_out_size])

        # GRU for state memory
        self.num_memory_layers = 3
        self.decoder_memory = nn.GRU(4, # input size (x,y,hx,hy)
                                    self.past_feat_size, # hidden size
                                    self.num_memory_layers, # num stacked GRU layers
                                    batch_first=True, # batch_first (B, T, D) inputs outputs
                                    )

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def get_normalizer(self):
        return self.normalizer

    def set_att_normalizer(self, normalizer):
        self.att_normalizer = normalizer

    def get_att_normalizer(self):
        return self.att_normalizer

    def set_bicycle_params(self, bicycle_params):
        '''
        Set bicycle model params including: max speed, max hdot, dt, etc.. used during rollout
        '''
        self.bicycle_params = bicycle_params

    def dyna_to_traj(self, scene_graph, dynamics):
        """_transform action to trajectories

        Args:
            dynamics (_type_): NA x FT x 4
        """
        NA, FT, _ = dynamics.size()
        cur_veh_len = self.att_normalizer.unnormalize(scene_graph.lw)[:,0].unsqueeze(1)
        prev_state = scene_graph.past[:, -1, :] 

        traj_out = []

        for t in range(FT):
            cur_dyna = dynamics[:, t, :].view(NA, 1, 1, 2)

            cur_bike_state = None

            a_out = cur_dyna[:,:,:,0]*self.bicycle_params['a_stats'][1] + self.bicycle_params['a_stats'][0]
            ddh_out = cur_dyna[:,:,:,1]*self.bicycle_params['ddh_stats'][1] + self.bicycle_params['ddh_stats'][0]
            # simulate forward
            init_state = self.normalizer.unnormalize(prev_state)
            cur_bike_state = self.sim_traj(init_state.unsqueeze(1), a_out, ddh_out, cur_veh_len)[:,0,0]
            cur_bike_state = self.normalizer.normalize(cur_bike_state)

            cur_state_global = cur_bike_state[:, :4]
            traj_out.append(cur_state_global)

            prev_state = cur_bike_state

        traj_out = torch.stack(traj_out, dim=1)

        return traj_out


    def forward(self, scene_graph, map_idx, map_env, gen_z=False):
        '''
        Forward pass to be used during training (samples from posterior).
        :param scene_graph: must contain past (NA x PT x 6), past_vis (NA x PT), future (NA x FT x 6), future_vis (NA x FT),
                                 edge_index (2 x num_edges), lw (NA x 2), sem (NA x num_classes), batch (NA)
        :param map_idx: size (B,) and indexes into the maps contained in map_env
        :param map_env: map environment to query for map crop
        '''
        # # solve warning "RNN module weights are not part of single contiguous chunk of memory"
        # if self.traj_encoder_type == 'gru' and not hasattr(self, '_flattened'):
        #     self.future_encoder.flatten_parameters()
        #     self.past_encoder.flatten_parameters()
        #     self.decoder_memory.flatten_parameters()
        #     setattr(self, '_flattened', True)

        NA = scene_graph.past.size(0)
        FT = self.FT 
        cur_veh_len = self.att_normalizer.unnormalize(scene_graph.lw)[:,0].unsqueeze(1)

        prev_state = scene_graph.past[:, -1, :] 
        traj_out = []

        scene_graph.pos = scene_graph.past[:, -1, :4]
        map_feat = self.encode_map(scene_graph, map_idx, map_env) # NA x map_feat
        past_feat = self.encode_past(scene_graph)

        cur_map_feat = map_feat
        cur_past_feat = past_feat
        cur_mem_state = past_feat.unsqueeze(0).expand(self.num_memory_layers, NA, self.past_feat_size).contiguous()

        cur_sem = scene_graph.sem
        cur_lw = scene_graph.lw

        for t in range(FT):
            dynamics_out = self.decoder_net(torch.cat([cur_map_feat, cur_past_feat, cur_sem, cur_lw], dim=-1))# input to sim must be (B x _ x T x 2)
            dynamics_out = dynamics_out.view(NA, 1, 1, 2)

            cur_state_local = None # in the frame of t-1 (NA, 4)
            cur_state_global = None # in the global frame (NA, 4)
            cur_bike_state = None

            a_out = dynamics_out[:,:,:,0]*self.bicycle_params['a_stats'][1] + self.bicycle_params['a_stats'][0]
            ddh_out = dynamics_out[:,:,:,1]*self.bicycle_params['ddh_stats'][1] + self.bicycle_params['ddh_stats'][0]
            # simulate forward
            init_state = self.normalizer.unnormalize(prev_state)
            cur_bike_state = self.sim_traj(init_state.unsqueeze(1), a_out, ddh_out, cur_veh_len)[:,0,0]
            cur_bike_state = self.normalizer.normalize(cur_bike_state)

            cur_state_global = cur_bike_state[:, :4]
            cur_state_local = transform2frame(prev_state[:,:4], cur_state_global.unsqueeze(1))[:,0] #local frame for mem unit

            if not gen_z:
                traj_out.append(cur_state_global)
            else:
                traj_out.append(dynamics_out.view(NA,2))

            prev_state = cur_bike_state

            if t < FT - 1:
                # update past feat using memory
                cur_past_feat, cur_mem_state = self.decoder_memory(cur_state_local.unsqueeze(1), # input
                                                                    cur_mem_state) # init hidden state
                cur_past_feat = cur_past_feat[:,0]

                # crop and encode map around new position
                scene_graph.pos = cur_state_global.detach()
                cur_map_feat = self.encode_map(scene_graph, map_idx, map_env)

        # return all outputs in global frame
        traj_out = torch.stack(traj_out, dim=1)

        return traj_out

    def encode_map(self, scene_graph, map_idx, map_env):
        '''
        Encodes local map patch around each agent based on the .pos attribute.
        NOTE: assumes the scene graph is NORMALIZED, so will unnormalize before doing the crop.

        :param scene_graph: makes use of .pos attribute (NA x 4) or (NA x NS x 4) with (x,y,hx,hy)
        :param map_idx: index of the map to use (B, )
        :param map_env: map environment to get crop with

        :return: NA x feat_dim
        '''
        NA = scene_graph.pos.size(0)
        NS = None if len(scene_graph.pos.size()) != 3 else scene_graph.pos.size(1)
        # first must unnormalize to get true world space state
        normalize_scene_graph(scene_graph,
                                self.normalizer,
                                self.att_normalizer,
                                unnorm=True)
        # get local crops based on .pos
        map_obs = map_env.get_map_crop(scene_graph, map_idx).to(torch.float) # NA x C x mapH x mapW
        # encode
        map_feat = self.map_conv(map_obs)
        # print(map_feat.size())
        bsize = NA if NS is None else NA*NS
        map_feat = self.map_feature(map_feat.view(bsize, self.map_feat_in_size))
        # print(map_feat.size())

        if NS is not None:
            map_feat = map_feat.reshape(NA, NS, -1)

        # re-normalize scene graph
        normalize_scene_graph(scene_graph,
                                self.normalizer,
                                self.att_normalizer,
                                unnorm=False)
        return map_feat

    def encode_past(self, scene_graph):
        '''
        Extract per-agent feature based on past trajectories (in local reference frame of last past step).
        Scene graph is assumed to be NORMALIZED.

        :param scene_graph: must have .past (NA x PT x 6), .past_vis (NA x PT),
                            .sem (NA x NC) and .lw (NA x 2) to encode

        :return: NA x feat_dim
        '''
        NA, PT, _ = scene_graph.past.size()
        # transform to local frame of last step of past
        local_past_kin = transform2frame(scene_graph.past[:, -1, :4], scene_graph.past[:, :, :4])
        local_past_traj = torch.cat([local_past_kin, scene_graph.past[:, :, 4:]], dim=2)
        # zero out any frames that were not observed
        local_past_traj[scene_graph.past_vis == 0.0] = 0.0
        # and append the visibility to the state as input
        local_past_traj = torch.cat([local_past_traj, scene_graph.past_vis.unsqueeze(-1)], dim=-1)
        # also add vehicle attributes
        veh_in_att = scene_graph.lw.unsqueeze(1).expand(NA, PT, self.att_feat_size) 
        encoder_in = torch.cat([local_past_traj, veh_in_att], dim=-1)
        if self.traj_encoder_type == 'mlp':
            # append semantic class too
            encoder_in = torch.cat([encoder_in.view(NA, -1), scene_graph.sem], dim=1)
        elif self.traj_encoder_type == 'gru':
            encoder_in = torch.cat([encoder_in, scene_graph.sem.view(NA, 1, self.NC).expand(NA, PT, self.NC)], dim=2)
        # then encode
        past_feat = self.past_encoder(encoder_in)

        if self.traj_encoder_type == 'gru':
            past_feat = past_feat[0][:, -1, :] # only want output of last step
            past_feat = self.past_out_layer(past_feat)

        return past_feat


    def sim_traj(self, init_state, a, ddh, vehicle_len):
        '''
        Everything is assumed to be UNNORMALIZED.
        :param init_state: (B, NA, 6)
        :param a: acceleration profile (B, NA, FT)
        :param ddh: yaw accel profile (B, NA, FT)
        :param vehicle_len: length of vehicles (B, NA)
        '''
        cur_kinematics = kinematics2angle(init_state)
        sim_steps = a.size(-1)
        kin_seq = []
        for t in range(sim_steps):
            cur_kinematics = car_dynamics(cur_kinematics, a[:,:,t], ddh[:,:,t],
                                        self.bicycle_params['dt'], 0, 1, 2, 3,
                                        4, vehicle_len, self.bicycle_params['maxhdot'],
                                        self.bicycle_params['maxs'])
            kin_seq.append(kinematics2vec(cur_kinematics))
        
        traj_out = torch.stack(kin_seq, dim=2)
        return traj_out


