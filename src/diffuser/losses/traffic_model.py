# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import time

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from losses.common import kl_normal, log_normal
from utils.transforms import transform2frame
from utils.torch import c2c
import datasets.nuscenes_utils as nutils

from losses.adv_gen_nusc import check_pairwise_veh_coll, check_single_veh_coll

ENV_COLL_THRESH = 0.05 # up to 5% of vehicle can be off the road
VEH_COLL_THRESH = 0.02 # IoU must be over this to count as a collision for metric (not loss)

class TrafficModelLoss(nn.Module):
    def __init__(self, loss_weights,
                    state_normalizer=None,
                    att_normalizer=None):
        '''
        :param loss_weights: dict of weightings for loss terms
        :param state_normalizer: normalization object for kinematic state
        :param att_normalizer: normalization object for length/width
        '''
        super(TrafficModelLoss, self).__init__()
        self.loss_weights = loss_weights
        self.state_normalizer = state_normalizer
        self.att_normalizer = att_normalizer

    def forward(self, scene_graph, pred,
                 map_idx=None,
                 map_env=None):
        '''
        Computes loss.

        :param scene_graph: containing input and GT data
        :param pred: dict of model predictions.
        :param map_idx, map_env: only needed for env collision losses
        '''        

        # we have a different number of agents for every sequence, and 
        #       a different number of timesteps for each agent even within
        #       the same sequence, so we mean over all timesteps so that
        #       the weights can be reliably balanced

        # reconstruction loss
        gt_future = scene_graph.future_gt # NA x FT x 6
        pred_future = pred['future_pred'] # NA x FT x 4

        # only want to compute loss for timesteps we have GT data
        gt_future = gt_future[scene_graph.future_vis == 1.0]
        pred_future = pred_future[scene_graph.future_vis == 1.0]

        # assume variance is 1 (i.e. MSE loss with extra constant)
        recon_loss = -log_normal(pred_future, gt_future[:, :4], torch.ones_like(pred_future))

        # KL divergence loss
        pm, pv = pred['prior_out'] # NA x z_size
        qm, qv = pred['posterior_out']
        # kl_normal(qm, qv, pm, pv)
        kl_loss = kl_normal(qm, qv, pm, pv)

        # total weighted loss
        loss = self.loss_weights['recon']*recon_loss.mean() + self.loss_weights['kl']*kl_loss.mean()

        prior_coll_loss = None
        if self.loss_weights['coll_veh_prior'] > 0.0:
            # compute veh2veh collision
            if self.state_normalizer is None or self.att_normalizer is None:
                print('Must have normalizers to compute collisison loss!')
                exit()
            # unnormalize
            veh_att = self.att_normalizer.unnormalize(scene_graph.lw)
            # build the loss function
            veh_coll_loss = VehCollLoss(veh_att, scene_graph.batch, scene_graph.ptr)
            # compute for each desired
            if self.loss_weights['coll_veh_prior'] > 0.0 and 'future_samp' in pred:
                prior_traj = self.state_normalizer.unnormalize(pred['future_samp'])
                prior_coll_pens, na_sqr = veh_coll_loss(prior_traj)
                prior_coll_loss = torch.sum(prior_coll_pens) / na_sqr
                loss = loss + self.loss_weights['coll_veh_prior']*prior_coll_loss

        prior_coll_env_loss = None
        if self.loss_weights['coll_env_prior'] > 0.0:
            assert(map_idx is not None and map_env is not None)
            # compute veh2env collision
            if self.state_normalizer is None or self.att_normalizer is None:
                print('Must have normalizers to compute collisison loss!')
                exit()

            # only compute these losses on ego vehicles since guaranteed should have no collisions
            ego_inds = scene_graph.ptr[:-1]
            # unnormalize
            veh_att = self.att_normalizer.unnormalize(scene_graph.lw[ego_inds])
            # build the loss function
            env_coll_loss = EnvCollLoss(veh_att, map_idx, map_env, pred['future_pred'].size(1))
            # compute for each desired
            if self.loss_weights['coll_env_prior'] > 0.0 and 'future_samp' in pred:
                prior_traj = self.state_normalizer.unnormalize(pred['future_samp'][ego_inds])
                prior_coll_env_loss, _ = env_coll_loss(prior_traj)
                loss = loss + self.loss_weights['coll_env_prior']*prior_coll_env_loss.mean()

        loss_out = {
            'loss' : loss.view((1,)),
            'recon_loss' : recon_loss, # (num_valid_frames, )
            'kl_loss' : kl_loss # (NA, )
        }

        if prior_coll_loss is not None:
            loss_out['coll_veh_prior'] = prior_coll_loss.view((1,))
        if prior_coll_env_loss is not None:
            loss_out['coll_env_prior'] = prior_coll_env_loss.view(-1) # (B, T)

        return loss_out

    def compute_err(self, scene_graph, pred, normalizer):
        '''
        Computes interpretable position and angle errors.

        pos_err is dist from GT averaged over all all timesteps for each future pred
        ang_err is angle diff (absolute degrees) from GT averaged over all timesteps for each future pred
        '''
        gt_future = scene_graph.future_gt # NA x FT x 6
        pred_future = pred['future_pred'] # NA x FT x 4

        NA, FT, _ = gt_future.size()

        gt_future = normalizer.unnormalize(gt_future)
        pred_future = normalizer.unnormalize(pred_future)

        # only want to compute errors for timesteps we have GT data
        gt_future = gt_future[scene_graph.future_vis == 1.0]
        pred_future = pred_future[scene_graph.future_vis == 1.0]

        # positional distance error
        gt_pos = gt_future[:,:2]
        pred_pos = pred_future[:,:2]
        pos_err = torch.norm(gt_pos - pred_pos, dim=-1)
        # angle distance error
        gt_h = gt_future[:,2:4]
        gt_h = gt_h / torch.norm(gt_h, dim=-1, keepdim=True)
        pred_h = pred_future[:,2:4]
        pred_h = pred_h / torch.norm(pred_h, dim=-1, keepdim=True)
        dotprod = torch.sum(gt_h * pred_h, dim=-1).clamp(-1, 1)
        ang_diff = torch.acos(dotprod)
        ang_err = torch.rad2deg(ang_diff)

        # NLL of posterior mean under the prior
        post_mean = pred['posterior_out'][0]
        z_logprob = log_normal(post_mean, pred['prior_out'][0], pred['prior_out'][1])
        z_mdist =  torch.norm((post_mean - pred['prior_out'][0]) / torch.sqrt(pred['prior_out'][1]), dim=-1)

        err_out = {
            'pos_err' : pos_err, # (num_valid_frames, )
            'ang_err' : ang_err, # (num_valid_frames, )
            'z_logprob' : z_logprob, # NA
            'z_mdist' : z_mdist
        }

        return err_out

    def eval_planner(self, planner_fut, bv_fut, veh_atts, dt, human_rollout, env_coll_loss_fn):
        """eval safety-critical indexes for planner rollouts

        Args:
            planner_fut (NA*T*4): unnormalized
            bv_future (NB*T*4): unnormalized
            veh_atts ((NA+NB)*2): _description_
            normalizer (_type_): _description_

        Returns:
            dict: loss_dict
                  keys: 'num_av': 
                        'coll_speeds' (num_coll_av,): [v1, v2,..]
                        'av_l2_err' (1,):  
                        'bv_coll' (2,): [num_bv, num_coll_bv]
                        'av_comfort_hist' (2,): [acc_hist, jerk_hist]
                        'bv_reality_hist' (4,): [v_hist, acc_hist,| leading_dist_hist, nearest_dist_hist|] 
        """

        def compute_deriv(x, dt):
            diff = torch.diff(x, dim=1)
            deriv = diff / dt
            deriv = torch.cat([deriv, torch.zeros_like(deriv[:, :1])], dim=1)
            return deriv

        def cal_ttc(traj, vel):
            """_summary_

            Args:
                traj (N*T*2): _description_
                vel (N*T*2): _description_
            """
            traj = traj.permute(1, 0, 2)
            vel = vel.permute(1, 0, 2)

            dist = torch.norm(traj[:, :, None, :] - traj[:, None, :, :], dim=-1) #T*N*N #检查对角元素
            rel_vel = torch.norm(vel[:, :, None, :] - vel[:, None, :, :], dim=-1) #T*N*N

            epsilon = torch.tensor(1e-5, dtype=rel_vel.dtype, device=rel_vel.device)
            ttc = dist / torch.where(rel_vel < epsilon, epsilon, rel_vel) #T*N*N

            return ttc   

        NA = planner_fut.shape[0]

        av_velocity = compute_deriv(planner_fut[:, :, :2], dt)
        av_acceleration = compute_deriv(av_velocity, dt)       
        av_jerk = compute_deriv(av_acceleration, dt)

        bv_velocity = compute_deriv(bv_fut[:, :, :2], dt)
        bv_acceleration = compute_deriv(bv_velocity, dt)   

        #是否只保留距离最近、ttc最小的
        # bv_ttc = cal_ttc(bv_fut[:, :, :2], bv_velocity) #T*N*N
        # all_ttc = cal_ttc(torch.cat([planner_fut, bv_fut], dim=0)[:, :, :2],
        #                   torch.cat([av_velocity, bv_velocity], dim=0))
        # bv_ttc = bv_ttc.cpu().numpy()
        # all_ttc = all_ttc.cpu().numpy()

        # flattend_bv_ttc = bv_ttc.flatten()
        # bin_edges = np.arange(start=min(flattend_bv_ttc), stop=max(flattend_bv_ttc), step=dt)
        # bv_ttc_hist = np.histogram(bv_ttc, bins=bin_edges)

        # flattend_all_ttc = all_ttc.flatten()
        # bin_edges = np.arange(start=min(flattend_all_ttc), stop=max(flattend_all_ttc), step=dt)
        # all_ttc_hist = np.histogram(all_ttc, bins=bin_edges)

        # av_acceleration = av_acceleration.cpu().numpy()
        # av_jerk = av_jerk.cpu().numpy()
        # bv_velocity = bv_velocity.cpu().numpy()
        # bv_acceleration = bv_acceleration.cpu().numpy()

        # av_acc_norm = np.linalg.norm(av_acceleration, axis=-1)
        # bin_edges = np.arange(start=min(av_acc_norm), stop=max(av_acc_norm), step=0.1)
        # av_acc_hist = np.histogram(av_acc_norm, bins=bin_edges)

        # av_jerk_norm = np.linalg.norm(av_jerk, axis=-1)
        # bin_edges = np.arange(start=min(av_jerk_norm), stop=max(av_jerk_norm), step=0.02)
        # av_jerk_hist = np.histogram(av_jerk_norm, bins=bin_edges)

        # bv_velocity_norm = np.linalg.norm(bv_velocity, axis=-1)
        # bin_edges = np.arange(start=min(bv_velocity_norm), stop=max(bv_velocity_norm), step=0.5)
        # bv_velocity_hist = np.histogram(bv_velocity_norm, bins=bin_edges)

        # bv_acceleration_norm = np.linalg.norm(bv_acceleration, axis=-1)
        # bin_edges = np.arange(start=min(bv_acceleration_norm), stop=max(bv_acceleration_norm), step=0.1)
        # bv_acceleration_hist = np.histogram(bv_acceleration_norm, bins=bin_edges)

        print('planner_fut', planner_fut.shape)
        print('human_rollout', human_rollout.shape)
        l2_pos_err = torch.norm(planner_fut[:, :, :2] - human_rollout[:, :, :2], dim=-1)

        # env_coll_loss = EnvCollLoss(veh_atts[:NA], map_idx, map_env, planner_fut.size(1))
        env_coll = env_coll_loss_fn(torch.cat([planner_fut, bv_fut], dim=0))#((NA+NB)*T)

        bv_veh_coll = check_pairwise_veh_coll(bv_fut, veh_atts[NA:])['did_collide'] #NB
        bv_env_coll = env_coll[NA:].sum(dim=-1) #NB

        av_coll_time = [] #NA
        
        # for av_i in range(NA): 
        #     _, veh_coll_time = check_single_veh_coll(planner_fut[av_i, :, :], veh_atts[av_i, :, :], bv_fut, veh_atts[NA:, :, :])
        #     env_coll_time = env_coll[av_i].nonzero()
        #     if env_coll_time.numel() > 0:          
        #         av_coll_time.append(min(veh_coll_time.min().values, env_coll_time[0]))
        #     else:
        #         av_coll_time.append(veh_coll_time.min().values)   
        # av_coll_vel = av_velocity[av_coll_time < planner_fut.shape[1]] #T-1时刻撞怎么办

        err_out = {
            'num_av': NA,
            # 'coll_speeds': av_coll_vel,
            'av_l2_err':  l2_pos_err.mean(),
            # 'bv_coll': [bv_veh_coll.size(0), (bv_veh_coll+bv_env_coll).nonzero().size(0)],
            # 'av_comfort_hist': [av_acc_hist, av_jerk_hist],
            # 'bv_reality_hist': [bv_velocity_hist, bv_acceleration_hist, bv_ttc_hist, all_ttc_hist],
        }

        return err_out

    def compute_err1(self, scene_graph, pred_future, normalizer, prefix='diff-'):
        '''
        Computes interpretable position and angle errors.

        pos_err is dist from GT averaged over all all timesteps for each future pred
        ang_err is angle diff (absolute degrees) from GT averaged over all timesteps for each future pred
        '''
        gt_future = scene_graph.future_gt # NA x FT x 6
        # pred_future = pred['future_pred'] # NA x FT x 4
        if pred_future.size(2) == 2:
            pred_future = torch.cat((pred_future, gt_future[:,:,2:4]), 2) 

        NA, FT, _ = gt_future.size()

        gt_future = normalizer.unnormalize(gt_future)
        pred_future = normalizer.unnormalize(pred_future)

        valid_agent = scene_graph.future_vis.min(-1).values #(NA)

        # final displacement
        gt_final_pos = gt_future[:,-1,:2][valid_agent == 1.0] #NA*2
        pred_final_pos = pred_future[:,-1,:2][valid_agent == 1.0]
        final_dis = torch.norm(gt_final_pos - pred_final_pos, dim=-1) # (NA)

        # positional distance error
        gt_pos = gt_future[scene_graph.future_vis == 1.0][:,:2]
        pred_pos = pred_future[scene_graph.future_vis == 1.0][:,:2]
        pos_err = torch.norm(gt_pos - pred_pos, dim=-1)

        coll_dict = check_pairwise_veh_coll(pred_future, scene_graph.lw)

        err_out = {
            prefix+'recon-pos_err' : pos_err.mean().view((1,)), # (num_valid_frames, )
            prefix+'recon-minSFDE': final_dis.mean().view((1,)), #mean
            prefix+'recon-num_coll_veh': torch.tensor(coll_dict['num_coll_veh']).view((1,)),
            prefix+'recon-num_traj_veh': torch.tensor(coll_dict['num_traj_veh']).view((1,)),
        }

        return err_out
    
    def compute_err2(self, scene_graph, future_samples, normalizer, prefix='diff-'):
        '''
        Computes interpretable position and angle errors.

        para: future_samples NA*NS*FT*4
        '''
        gt_future = scene_graph.future_gt # NA x FT x 6
        # pred_future = pred['future_pred'] # NA x FT x 4

        NA, FT, _ = gt_future.size()
        num_samples = future_samples.size(1)

        gt_future = normalizer.unnormalize(gt_future)
        gt_final_pos = gt_future[:,-1,:2][scene_graph.future_vis[:,-1] == 1.0]

        sfde = []
        fde = []
        num_coll_veh = 0
        num_traj_veh = 0
        for idx in range(num_samples):
            pred_future = future_samples[:,idx,:,:]
            pred_future = normalizer.unnormalize(pred_future)
            
            pred_final_pos = pred_future[:,-1,:2][scene_graph.future_vis[:,-1] == 1.0]
            final_dis = torch.norm(gt_final_pos - pred_final_pos, dim=-1) # (NA)
            
            sfde.append(final_dis.mean())
            fde.append(final_dis)

            coll_dict = check_pairwise_veh_coll(pred_future, scene_graph.lw) 
            num_coll_veh += coll_dict['num_coll_veh']
            num_traj_veh += coll_dict['num_traj_veh']

        minSFDE = (torch.stack(sfde, 0)).min()
        FDD = torch.sum(torch.max(torch.stack(fde, 1), 1).values)

        # positional distance error
        gt_pos = gt_future[scene_graph.future_vis == 1.0][:,:2]
        pred_pos = pred_future[scene_graph.future_vis == 1.0][:,:2]
        pos_err = torch.norm(gt_pos - pred_pos, dim=-1)

        err_out = {
            prefix+'sample-pos_err' : pos_err.mean().view((1,)), # (num_valid_frames, )
            prefix+'sample-FDD': FDD.view((1,)),
            prefix+'sample-minSFDE': minSFDE.view((1,)),
            prefix+'sample-num_coll_veh': torch.tensor(num_coll_veh / num_samples).view((1,)),
            prefix+'sample-num_traj_veh': torch.tensor(num_traj_veh / num_samples).view((1,)),
            # 'hist':, 
        }

        return err_out
    
    def compute_err3(self, scene_graph, pred_future, normalizer, map_idx, map_env):
        '''
        Computes interpretable position and angle errors.

        pos_err is dist from GT averaged over all all timesteps for each future pred
        ang_err is angle diff (absolute degrees) from GT averaged over all timesteps for each future pred
        '''
        gt_future = scene_graph.future_gt # NA x FT x 6
        # pred_future = pred['future_pred'] # NA x FT x 4
        if pred_future.size(2) == 2:
            pred_future = torch.cat((pred_future, gt_future[:,:,2:4]), 2) 

        NA, FT, _ = gt_future.size()

        gt_future = normalizer.unnormalize(gt_future)
        pred_future = normalizer.unnormalize(pred_future)

        valid_agent = scene_graph.future_vis.min(-1).values #(NA)

        # final displacement
        gt_final_pos = gt_future[:,-1,:2][valid_agent == 1.0] #NA*2
        pred_final_pos = pred_future[:,-1,:2][valid_agent == 1.0]
        final_dis = torch.norm(gt_final_pos - pred_final_pos, dim=-1) # (NA)

        # positional distance error
        gt_pos = gt_future[scene_graph.future_vis == 1.0][:,:2]
        pred_pos = pred_future[scene_graph.future_vis == 1.0][:,:2]
        pos_err = torch.norm(gt_pos - pred_pos, dim=-1)
        
        prior_coll_loss = None
        if self.loss_weights['coll_veh_prior'] > 0.0:
            # compute veh2veh collision
            if self.state_normalizer is None or self.att_normalizer is None:
                print('Must have normalizers to compute collisison loss!')
                exit()
            # unnormalize
            veh_att = self.att_normalizer.unnormalize(scene_graph.lw)
            # build the loss function
            veh_coll_loss = VehCollLoss(veh_att, scene_graph.batch, scene_graph.ptr)
            # compute for each desired
            if self.loss_weights['coll_veh_prior'] > 0.0:
                prior_coll_pens, na_sqr, _ = veh_coll_loss(pred_future, is_adv=False)
                prior_coll_loss = torch.sum(prior_coll_pens) / na_sqr

        prior_coll_env_loss = None
        if self.loss_weights['coll_env_prior'] > 0.0:
            assert(map_idx is not None and map_env is not None)
            # compute veh2env collision
            if self.state_normalizer is None or self.att_normalizer is None:
                print('Must have normalizers to compute collisison loss!')
                exit()

            # unnormalize
            veh_att = self.att_normalizer.unnormalize(scene_graph.lw)
            unique_values, counts = torch.unique(scene_graph.batch, return_counts=True)
            map_idx = torch.repeat_interleave(map_idx, counts)
            # build the loss function
            env_coll_loss= EnvCollLoss(veh_att, map_idx, map_env, pred_future.size(1))
            # compute for each desired
            if self.loss_weights['coll_env_prior'] > 0.0:
                prior_coll_env_loss, _ = env_coll_loss(pred_future)
                env_loss = self.loss_weights['coll_env_prior']*prior_coll_env_loss.mean()

        err_out = {
            'pos_err' : pos_err.mean().view((1,)), # (num_valid_frames, )
            'SFDE': final_dis.mean().view((1,)), #mean
        }
        
        if prior_coll_loss is not None:
            err_out['coll_veh'] = prior_coll_loss.view((1,))
        if prior_coll_env_loss is not None:
            err_out['coll_env'] = prior_coll_env_loss.view(-1) # (B, T)

        return err_out

class VehCollLoss(nn.Module):
    '''
    Penalizes collision between vehicles with circle approximation.
    '''
    def __init__(self, veh_att, batch, ptr,
                       num_circ=5,
                       buffer_dist=0.05):
        '''
        :param veh_att: UNNORMALIZED lw for the vehicles that will be computing loss for (NA x 2)
        :param batch: from the scene graph
        :param ptr: from the scene_graph
        :param num_circ: number of circles used to approximate each vehicle.
        :param buffer: extra buffer distance that circles must be apart to avoid being penalized
        '''
        super(VehCollLoss, self).__init__()
        self.veh_att = veh_att
        self.buffer_dist = buffer_dist
        self.batch = batch
        self.ptr = ptr

        self.graph_sizes = self.ptr[1:] - self.ptr[:-1]
        self.num_pairs = torch.sum(self.graph_sizes*self.graph_sizes - self.graph_sizes)

        NA = self.veh_att.size(0)
        # construct centroids circles of each agent
        self.veh_rad = self.veh_att[:, 1] / 2. # radius of the discs for each vehicle assuming length > width
        cent_min = -(self.veh_att[:, 0] / 2.) + self.veh_rad
        cent_max = (self.veh_att[:, 0] / 2.) - self.veh_rad
        cent_x = torch.stack([torch.linspace(cent_min[vidx].item(), cent_max[vidx].item(), num_circ) for vidx in range(NA)], dim=0).to(veh_att.device)
        # create dummy states for centroids with y=0 and hx,hy=1,0 so can transform later
        self.centroids = torch.stack([cent_x, torch.zeros_like(cent_x), torch.ones_like(cent_x), torch.zeros_like(cent_x)], dim=2)
        self.num_circ = num_circ
        # minimum distance that two vehicle circle centers can be apart without collision
        self.penalty_dists = self.veh_rad.view(NA, 1).expand(NA, NA) + self.veh_rad.view(1, NA).expand(NA, NA) + self.buffer_dist
        # need a mask to ignore "self" collisions and "collisions" from other scene graphs in the batch
        off_diag_mask = ~torch.eye(NA, dtype=torch.bool).to(self.veh_att.device)
        batch_mask = torch.zeros((NA, NA), dtype=torch.bool).to(self.veh_att.device)
        for b in range(1, len(self.ptr)):
            # only the block corresponding to pairs of vehicles in the same scene graph matter
            batch_mask[self.ptr[b-1]:self.ptr[b], self.ptr[b-1]:self.ptr[b]] = True

        self.valid_mask = torch.logical_and(off_diag_mask, batch_mask)

    def forward(self, traj, keep_shape=False, is_adv=True):
        '''
        :param traj: (NA x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :return: loss, number of "interactions" that could have caused a collision, i.e. number of valid vehicle pairs
        '''
        NA, T, _ = traj.size()
        cur_valid_mask = self.valid_mask.view(1, NA, NA).expand(T, NA, NA)

        traj = traj[:, :, :4].view(NA*T, 4)
        cur_cent = self.centroids.view(NA, 1, self.num_circ, 4).expand(NA, T, self.num_circ, 4).reshape(NA*T, self.num_circ, 4)
        # centroids are in local, need to transform to global based on current traj
        world_cent = transform2frame(traj, cur_cent, inverse=True).view(NA, T, self.num_circ, 4)[:, :, :, :2] # only need centers
        world_cent = world_cent.transpose(0, 1) # T x NA X C x 2
        # distances between all pairs of circles between all pairs of agents
        cur_cent1 = world_cent.view(T, NA, 1, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        cur_cent2 = world_cent.view(T, 1, NA, self.num_circ, 2).expand(T, NA, NA, self.num_circ, 2).reshape(T*NA*NA, self.num_circ, 2)
        pair_dists = torch.cdist(cur_cent1, cur_cent2).view(T*NA*NA, self.num_circ*self.num_circ)

        # get minimum distance overall all circle pairs between each pair
        min_pair_dists = torch.min(pair_dists, 1)[0].view(T, NA, NA)
        cur_penalty_dists = self.penalty_dists.view(1, NA, NA)
        is_colliding_mask = min_pair_dists <= cur_penalty_dists
        # diagonals are self collisions so ignore them
        is_colliding_mask = torch.logical_and(is_colliding_mask,cur_valid_mask)
        # compute penalties
        if is_adv:
            # cur_penalties = torch.where(is_colliding_mask, torch.tensor(10.0, device=is_colliding_mask.device), 1.0 - (min_pair_dists / cur_penalty_dists))
            cur_penalties = torch.where(is_colliding_mask, torch.zeros_like(cur_penalty_dists), cur_penalty_dists - min_pair_dists)
        else:
            cur_penalties = torch.where(is_colliding_mask, 1.0 - (min_pair_dists / cur_penalty_dists), torch.zeros_like(cur_penalty_dists))
        
        if not keep_shape:
            cur_penalties = cur_penalties[cur_valid_mask]
        else:
            cur_penalties = cur_penalties * cur_valid_mask
        # print("cur_penalties", cur_penalties)

        return cur_penalties, self.num_pairs, is_colliding_mask
    

def _drivable_frac_per_step(traj_flat, veh_att, mapixes, map_env):
    '''
    Per (agent, timestep) fraction of vehicle bbox on drivable map pixels.
    Matches ``compute_coll_rate_env`` / ``check_on_layer`` geometry.
    NaN trajectory frames default to 1.0 (fully on-road, no collision).
    '''
    n_steps = traj_flat.size(0)
    drivable_frac = torch.ones(n_steps, device=traj_flat.device, dtype=traj_flat.dtype)
    valid_frames = ~torch.isnan(traj_flat.sum(-1))
    if not valid_frames.any():
        return drivable_frac

    drivable_raster = map_env.nusc_raster[:, 0]
    drivable_frac[valid_frames] = nutils.check_on_layer(
        drivable_raster,
        map_env.nusc_dx,
        traj_flat[valid_frames].detach(),
        veh_att[valid_frames],
        mapixes[valid_frames],
    )
    return drivable_frac


def _bilinear_sample_raster(raster, xys_pix):
    '''
    Memory-efficient differentiable bilinear interpolation at arbitrary pixel coords.

    Does NOT expand the raster per agent—only the (B, L, W, 4) corner-value
    lookups are needed, keeping peak memory at O(B * L * W).

    raster  : (H, W) float tensor (the drivable-area raster for one map)
    xys_pix : (B, L, W, 2) floating-point pixel coordinates (x, y)

    Returns : (B,) mean drivable fraction per agent, with gradients flowing
              through the bilinear weights back to xys_pix.
    '''
    H, W_map = raster.shape

    # Integer floor pixel coords, clamped to valid range
    x_f = xys_pix[..., 0]                                   # (B, L, W)
    y_f = xys_pix[..., 1]
    x0  = x_f.detach().floor().long().clamp(0, W_map - 1)
    y0  = y_f.detach().floor().long().clamp(0, H - 1)
    x1  = (x0 + 1).clamp(0, W_map - 1)
    y1  = (y0 + 1).clamp(0, H - 1)

    # Fractional offsets — differentiable w.r.t. xys_pix because
    #   d(wx)/d(x_f) = 1   (floor has zero gradient, subtracted constant)
    wx = x_f - x_f.detach().floor()                         # (B, L, W) in [0, 1)
    wy = y_f - y_f.detach().floor()

    # Out-of-bounds mask: treat pixels outside map as off-road (value = 0)
    oob = (x_f.detach() < 0) | (x_f.detach() >= W_map) | \
          (y_f.detach() < 0) | (y_f.detach() >= H)

    # Sample 4 corners from the raster (constant integer lookup, no autograd needed)
    q00 = raster[y0, x0].to(dtype=x_f.dtype)                # top-left
    q10 = raster[y0, x1].to(dtype=x_f.dtype)                # top-right
    q01 = raster[y1, x0].to(dtype=x_f.dtype)                # bottom-left
    q11 = raster[y1, x1].to(dtype=x_f.dtype)                # bottom-right

    # Bilinear combination — gradient flows through wx, wy → xys_pix
    sampled = (q00 * (1 - wx) * (1 - wy)
             + q10 * wx       * (1 - wy)
             + q01 * (1 - wx) * wy
             + q11 * wx       * wy)                          # (B, L, W)
    sampled = sampled.masked_fill(oob, 0.0)                  # off-road outside map

    return sampled.mean(dim=[1, 2])                          # (B,)


def _drivable_frac_per_step_diff(traj_flat, veh_att, mapixes, map_env):
    '''
    Differentiable version of ``_drivable_frac_per_step``.

    Uses manual bilinear interpolation at the vehicle footprint sample points
    instead of ``check_on_layer``'s integer raster lookup.  This keeps memory
    at O(B * L * W_car) and avoids expanding the full map raster per agent
    (which would cause OOM for large NuScenes maps).

    Gradients flow through the bilinear weights (wx, wy) back to the pixel
    coordinates → world coordinates → traj_flat positions and headings.

    NaN trajectory frames default to 1.0 (fully on-road, no collision).
    '''
    n_steps = traj_flat.size(0)
    drivable_frac = torch.ones(n_steps, device=traj_flat.device, dtype=traj_flat.dtype)
    valid_frames = ~torch.isnan(traj_flat.sum(-1))
    if not valid_frames.any():
        return drivable_frac

    traj_dtype = traj_flat.dtype
    drivable_raster = map_env.nusc_raster[:, 0]  # (M, H, W) binary; keep original dtype for indexing
    dx = map_env.nusc_dx.to(dtype=traj_dtype)    # (M, 2) metres/pixel

    for m in mapixes[valid_frames].unique():
        m_int = m.item()
        mask  = valid_frames & (mapixes == m)
        cars_m = traj_flat[mask]   # (B_m, 4)  [x, y, hx, hy]
        lw_m   = veh_att[mask]     # (B_m, 2)  [l, w]

        # Footprint grid resolution (same heuristic as check_on_layer)
        mdx   = dx[m_int].mean()
        mlw   = lw_m.mean(0)
        L     = max(1, int((mlw[0] / mdx).round().item()))
        W_car = max(1, int((mlw[1] / mdx).round().item()))

        # gen_car_coords is fully differentiable (rotation + linspace + stack).
        # Returns (B_m, 1, L, W_car, 2) world coords of bbox sample grid.
        xys_world = nutils.gen_car_coords(
            cars_m[:, :2], cars_m[:, 2:], 1, L, W_car,
            ls=lw_m[:, 0], ws=lw_m[:, 1],
        )[:, 0]  # (B_m, L, W_car, 2)

        # Pixel coords (differentiable divide)
        xys_pix = xys_world / dx[m_int].view(1, 1, 1, 2)   # (B_m, L, W_car, 2)

        # Bilinear sample — memory: O(B_m * L * W_car), no map expansion
        raster_m = drivable_raster[m_int].float()            # (H, W) float for interp
        drivable_frac[mask] = _bilinear_sample_raster(raster_m, xys_pix)

    return drivable_frac


class EnvCollLoss(nn.Module):
    '''
    Penalizes overlap with non-drivable area.
    Collision mask and penalty use the same rule as ``compute_coll_rate_env``:
    off-road fraction > ENV_COLL_THRESH (default 5%).
    '''
    def __init__(self, veh_att, mapixes, map_env, T):
        '''
        :param veh_att: (NA, 2) UNNORMALIZED
        :param mapixes: (NA, )
        :param map_env: 
        :param T: number of steps in trajectories that loss will be computed on
        '''
        super(EnvCollLoss, self).__init__()
        self.map_env = map_env
        NA = veh_att.size(0)
        assert(NA == mapixes.size(0))
        self.mapixes = mapixes.view(NA, 1).expand(NA, T).reshape(NA*T)
        self.veh_att = veh_att.view(NA, 1, 2).expand(NA, T, 2).reshape(NA*T, 2)
        self.T = T

    def forward(self, traj, differentiable=False):
        '''
        :param traj: (NA x T x 4) trajectories (x,y,hx,hy) for each agent to determine collision penalty.
                                should be UNNORMALIZED.
        :param differentiable: if True, use bilinear grid_sample so that gradients flow through
                               trajectory positions/headings (needed during guidance optimisation).
                               If False (default), use the faster integer-indexed raster lookup
                               with .detach() (no gradient, matches training / eval behaviour).
        :return: (penalties, is_collision_mask) each (NA, T)
                 penalties > 0 iff is_collision_mask is True (aligned with compute_coll_rate_env).
        '''
        NA = traj.size(0)
        T = self.T
        assert(T == traj.size(1))
        assert(NA*T == self.veh_att.size(0))
        traj_flat = traj.reshape(NA * T, 4)

        _frac_fn = _drivable_frac_per_step_diff if differentiable else _drivable_frac_per_step
        drivable_frac = _frac_fn(
            traj_flat, self.veh_att, self.mapixes, self.map_env,
        ).view(NA, T)

        offroad_frac = 1.0 - drivable_frac
        is_collision_mask = offroad_frac > ENV_COLL_THRESH
        all_penalties = torch.relu(offroad_frac - ENV_COLL_THRESH)

        return all_penalties, is_collision_mask
    
def compute_disp_err(scene_graph, pred, normalizer):
    '''
    Computes sample-based displacement errors.

    ONLY computes for ego vehicle since have guaranteed full past/future motion so the statistics
    will be correct.
    '''
    gt_future = scene_graph.future_gt # NA x FT x 6
    pred_future = pred['future_pred'] # NA x NS x FT x 4

    NA, FT, _ = gt_future.size()
    NS = pred_future.size(1)

    # make sure same length
    FT = pred_future.size(2) if pred_future.size(2) < FT else FT
    pred_future = pred_future[:, :, :FT] # if prediction is longer, make sure only compare the steps we have
    gt_future = gt_future[:, :FT]

    gt_future = normalizer.unnormalize(gt_future).view(NA, 1, FT, 6)
    pred_future = normalizer.unnormalize(pred_future)

    # find index of first agent in each batch
    ego_inds = scene_graph.ptr[:-1]

    # ego-only data
    gt_future = gt_future[ego_inds] # B x 1 x FT x 6
    pred_future = pred_future[ego_inds] # B x NS x FT x 4
    B = gt_future.size(0)

    # positional ADE
    gt_pos = gt_future[:,:,:,:2]
    pred_pos = pred_future[:,:,:,:2]
    diff = torch.norm(gt_pos - pred_pos, dim=-1) # B x NS x FT
    ade = diff.mean(dim=-1) # B x NS
    min_ade = torch.min(ade, dim=1)[0] # B

    # positional APD
    pred_pairwise_pos = pred_pos.view(B, NS, 1, FT, 2).expand(B, NS, NS, FT, 2)
    pairwise_diff = torch.norm(pred_pairwise_pos - pred_pairwise_pos.transpose(1, 2), dim=-1) # B x NS x NS x FT
    all_sum = torch.sum(pairwise_diff, dim=[1, 2]).sum(dim=-1)
    apd = all_sum / (NS*(NS-1)*FT) # don't want to include diagonal elmnts

    # positional FDE
    fde = diff[:,:,-1]
    min_fde = torch.min(fde, dim=1)[0] # B 

    # angular ADE
    gt_h = gt_future[:,:,:,2:4]
    gt_h = gt_h / torch.norm(gt_h, dim=-1, keepdim=True)
    pred_h = pred_future[:,:,:,2:4]
    pred_h = pred_h / torch.norm(pred_h, dim=-1, keepdim=True)
    dotprod = torch.sum(gt_h * pred_h, dim=-1).clamp(-1, 1)
    ang_diff = torch.rad2deg(torch.acos(dotprod)) # B x NS x FT
    ang_ade = ang_diff.mean(dim=-1)
    ang_min_ade = torch.min(ang_ade, dim=1)[0] # B

    # angular FDE
    ang_fde = ang_diff[:,:,-1]
    ang_min_fde = torch.min(ang_fde, dim=1)[0]

    disp_err_dict = {
        'pos_minADE' : min_ade,
        'pos_minFDE' : min_fde,
        'ang_minADE' : ang_min_ade,
        'ang_minFDE' : ang_min_fde,
        'APD' : apd
    }
    return disp_err_dict

def compute_coll_rate_env(scene_graph, map_idx, pred, map_env, state_normalizer, att_normalizer,
                        ego_only=False):
    '''
    Computes number of rollouts that collided with map for sampled pred data.
    If a pred is nan, it counts as a NOT collision.

    returns: NA x NS with a 1 if collided
    '''
    from datasets.utils import get_ego_inds

    if isinstance(pred, torch.Tensor):
        pred_future = pred
    else:
        pred_future = pred['future_pred'] # NA x NS x FT x 4
    NA, NS, FT, _ = pred_future.size()

    veh_att = scene_graph.lw
    mapixes = map_idx[scene_graph.batch]

    if ego_only:
        ego_inds = get_ego_inds(scene_graph)
        pred_future = pred_future[ego_inds]
        veh_att = veh_att[ego_inds]
        mapixes = mapixes[ego_inds]
        NA = pred_future.size(0)

    # unnorm preds and attribs
    pred_future = state_normalizer.unnormalize(pred_future).view(NA*NS*FT, 4)
    veh_att = att_normalizer.unnormalize(veh_att).view(NA, 1, 1, 2).expand(NA, NS, FT, 2).reshape(NA*NS*FT, 2)

    mapixes = mapixes.view(NA, 1, 1).expand(NA, NS, FT).reshape(NA*NS*FT)
    final_drivable_frac = _drivable_frac_per_step(
        pred_future, veh_att, mapixes, map_env,
    ).view(NA, NS, FT)
    coll_frame = (final_drivable_frac < (1.0 - ENV_COLL_THRESH))
    map_coll = torch.sum(coll_frame, dim=2) >= 1 # (NA, NS)

    coll_dict = {
        'num_coll_map' : float(c2c(torch.sum(map_coll))),
        'num_traj_map' : float(NS*NA),
        'did_collide' : map_coll
    }

    return coll_dict

def compute_coll_rate_env_from_traj(pred_future, veh_att, mapixes, map_env):
    '''
    Computes number of rollouts that collided with map for sampled pred data.
    If a pred is nan, it counts as a NOT collision.

    :param pred_future: NA x NS x FT x 4 UNNORMALIZED
    :param veh_att:
    :param mapixes: (NA) index of the map for each agent
    :param map_env:

    returns: NA x NS with a 1 if collided
    '''
    NA, NS, FT, _ = pred_future.size()

    # unnorm preds and attribs
    pred_future = pred_future.reshape(NA*NS*FT, 4)
    veh_att = veh_att.reshape(NA, 1, 1, 2).expand(NA, NS, FT, 2).reshape(NA*NS*FT, 2)

    mapixes = mapixes.reshape(NA, 1, 1).expand(NA, NS, FT).reshape(NA*NS*FT)
    final_drivable_frac = _drivable_frac_per_step(
        pred_future, veh_att, mapixes, map_env,
    ).view(NA, NS, FT)
    coll_frame = (final_drivable_frac < (1.0 - ENV_COLL_THRESH))
    map_coll = torch.sum(coll_frame, dim=2) >= 1 # (NA, NS)

    coll_dict = {
        'num_coll_map' : float(c2c(torch.sum(map_coll))),
        'num_traj_map' : float(NS*NA),
        'did_collide' : map_coll
    }

    return coll_dict

def compute_coll_rate_veh(scene_graph, pred, state_normalizer, att_normalizer):
    '''
    Computes number of rollouts that collided with other agents for sampled pred data.
    If a pred is nan, it counts as a NOT collision.

    WARNING: this function assumes the scene graph edges connect all vehicle pairs that
    need to be checked for collisions. Also assumes all edges are bidirectional, i.e.
    if (3, 5) is a pair then (5, 3) is also one. We ONLY check one of the two.

    returns: NA x NS with a 1 if collided
    '''
    import datasets.nuscenes_utils as nutils
    from shapely.geometry import Polygon

    if isinstance(pred, torch.Tensor):
        pred_future = pred
    else:
        pred_future = pred['future_pred'] # NA x NS x FT x 4
    NA, NS, FT, _ = pred_future.size()

    veh_att = scene_graph.lw

    # unnorm preds and attribs
    pred_future = state_normalizer.unnormalize(pred_future)
    veh_att = att_normalizer.unnormalize(veh_att)

    # all the vehicle pairs to go over
    pred_future = pred_future.cpu().numpy()
    veh_att = veh_att.cpu().numpy()
    pairs = scene_graph.edge_index.cpu().numpy().T

    veh_coll = np.zeros((NA, NS), dtype=np.bool_)
    poly_cache = dict()    
    # loop over every timestep in every sample for this combination
    coll_count = 0
    for s in range(NS):
        for veh_pair in pairs:
            aj, ai = veh_pair
            if aj <= ai:
                continue # don't double count
            if veh_coll[ai, s]:
                continue # already determined there has been a collision for this agent at this sample, move on
            for t in range(FT):
                # compute iou
                if (ai, s, t) not in poly_cache:
                    ai_state = pred_future[ai, s, t, :]
                    if np.sum(np.isnan(ai_state)) > 0:
                        poly_cache[(ai, s, t)] = None
                        continue # don't have data for this step
                    ai_corners = nutils.get_corners(ai_state, veh_att[ai])
                    ai_poly = Polygon(ai_corners)
                    poly_cache[(ai, s, t)] = ai_poly
                else:
                    ai_poly = poly_cache[(ai, s, t)]
                    if ai_poly is None:
                        continue
                if (aj, s, t) not in poly_cache:
                    aj_state = pred_future[aj, s, t, :]
                    if np.sum(np.isnan(aj_state)) > 0:
                        poly_cache[(aj, s, t)] = None
                        continue # don't have data for this step
                    aj_corners = nutils.get_corners(aj_state, veh_att[aj])
                    aj_poly = Polygon(aj_corners)
                    poly_cache[(aj, s, t)] = aj_poly
                else:
                    aj_poly = poly_cache[(aj, s, t)]
                    if aj_poly is None:
                        continue
                cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
                if cur_iou > VEH_COLL_THRESH:
                    coll_count += 1
                    veh_coll[ai, s] = True
                    break # don't need to check rest of sequence

    coll_dict = {
        'num_coll_veh' : float(coll_count),
        'num_traj_veh' : float(NS*NA),
        'did_collide' : veh_coll
    }

    return coll_dict