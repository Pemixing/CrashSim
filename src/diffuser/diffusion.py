from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
import os

import torch.optim as optim

import diffuser.utils as utils
from utils.torch import load_model_state
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains finish_flag')


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values, False

@torch.no_grad()
def ddim_sample_fn(model, x, cond, s, step):
    k = s + int(step) -1
    model_mean, _, model_log_variance = model.p_mean_variance_ddim(x=x, cond=cond, k=k, s=s)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x) if not model.zero_var else torch.zeros_like(x)
    noise[s == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values, False

def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
        zero_var=False, use_ddim=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        self.use_ddim = True if zero_var else use_ddim
        self.zero_var = zero_var

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        # print(np.sqrt(alphas_cumprod)[-2:-1])
        # self.register_buffer('posterior_mean_coef3',
        #     torch.cat([torch.ones(1), np.sqrt(alphas_cumprod)[:-1]]))
        # print(self.posterior_mean_coef3[-1])
        # self.register_buffer('posterior_mean_coef4',
        #     torch.cat([torch.zeros(1), np.sqrt(1. - alphas_cumprod - posterior_variance)[:-1]]))
        self.register_buffer('posterior_mean_coef3',
            np.sqrt(alphas_cumprod_prev))
        self.register_buffer('posterior_mean_coef4',
            np.sqrt(1. - alphas_cumprod_prev - posterior_variance))
        
        alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.zeros(1)])
        self.register_buffer('encoder_coef1', np.sqrt(alphas_cumprod_next))
        self.register_buffer('encoder_coef2', np.sqrt(1. - alphas_cumprod_next))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        self.err_fn = Losses['value_log_normal']()

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
        
    def predict_noise_from_start(self, x, t, x_start):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - x_start) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) 
        )
        
    def ddim_reverse_sample(self, x, cond, t):
        if self.predict_epsilon:
            noise = self.model(x, cond, t) 
            # print("=============noise============", noise)
            # noise = torch.zeros_like(x)
            x_start = self.predict_start_from_noise(x, t, noise)
            # print("=============x_start============", x_start)
            # temp = extract(self.encoder_coef1, t, x_t.shape) * x0_t + extract(self.encoder_coef2, t, x_t.shape) * noise_t
            # temp1 = self.predict_start_from_noise(temp, t, noise_t)
        else:
            x_start = self.model(x, cond, t)
            noise = (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - x_start) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        return (
            extract(self.encoder_coef1, t, x.shape) * x_start +
            extract(self.encoder_coef2, t, x.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def q_posterior_ddim(self, x_start, noise_k, k, s):
        posterior_mean = (
            extract(self.posterior_mean_coef3, s, noise_k.shape) * x_start +
            extract(self.posterior_mean_coef4, s, noise_k.shape) * noise_k
        )
        posterior_variance = extract(self.posterior_variance, k, noise_k.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, k, noise_k.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance_ddim(self, x, cond, k, s):
        if self.predict_epsilon:
            noise = self.model(x, cond, k)
            # noise = torch.zeros_like(noise)
            x_recon = self.predict_start_from_noise(x, t=k, noise=noise)
        else: 
            x_recon = self.model(x, cond, k)
            noise = self.predict_noise_from_start(x, t=k, x_start=x_recon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_ddim(
                x_start=x_recon, noise_k=noise, k=k, s=s)
        # temp = self.predict_start_from_noise(model_mean, t=s-1, noise=torch.zeros_like(model_mean))
        # temp1 = self.predict_start_from_noise(model_mean, t=s, noise=torch.zeros_like(model_mean))
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, noise=None, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise
        # x = apply_conditioning(x, cond, self.action_dim)##

        chain = [x] if return_chain else None

        step = 1 
        sample_fn = ddim_sample_fn if self.use_ddim and sample_fn==default_sample_fn else sample_fn
        if 'step' in sample_kwargs and self.use_ddim:
            step = int(sample_kwargs['step'])
            assert(self.n_timesteps // step)

        # temp = self.predict_start_from_noise(x, make_timesteps(batch_size, 99, device), torch.zeros_like(x))
        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps, step)):
            t = make_timesteps(batch_size, i, device)
            x, values, finish_flag = sample_fn(self, x, cond, t, **sample_kwargs)
            # x = apply_conditioning(x, cond, self.action_dim)##

            # progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)
            if finish_flag: break

        # progress.stamp()

        # if not torch.equal(values, torch.zeros_like(values)):
        #     x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain, finish_flag)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        # batch_size = len(cond[0])
        batch_size = cond.shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, vis, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)##

        x_recon = self.model(x_noisy, cond, t)
        # x_recon = apply_conditioning(x_recon, cond, self.action_dim)##

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise, vis)
            x_recon = self.predict_start_from_noise(x_noisy, t, x_recon)
        else:
            loss, info = self.loss_fn(x_recon, x_start, vis)

        info = {'x_recon': x_recon} 
        return loss, info
    
    def compute_err(self, x_recon, x_gt, vis):
        loss = self.err_fn(x_recon, x_gt, vis)
        return loss

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)
    
    def reset_var(self, zero_var):
        if zero_var == self.zero_var: return

        self.zero_var = zero_var
        # if zero_var:
        #     assert(self.use_ddim)
        #     self.posterior_mean_coef4 = torch.cat([torch.zeros(1, device=self.betas.device), torch.sqrt(1. - self.alphas_cumprod)[:-1]])
        # else:
        #     self.posterior_mean_coef4 = torch.cat([torch.zeros(1, device=self.betas.device), torch.sqrt(1. - self.alphas_cumprod - self.posterior_variance)[:-1]])
        if zero_var:
            assert(self.use_ddim)
            self.posterior_mean_coef4 = torch.sqrt(1. - self.alphas_cumprod_prev)
        else:
            self.posterior_mean_coef4 = torch.sqrt(1. - self.alphas_cumprod_prev - self.posterior_variance)

    def encoder(self, x_start, cond):
        batch_size = x_start.shape[0]
        device = self.betas.device

        # x = x_start * torch.sqrt(1. - extract(self.betas, make_timesteps(batch_size, 0, device), x_start.shape))
        x = x_start
        for i in range(0, self.n_timesteps):
            t = make_timesteps(batch_size, i, device)
            x = self.ddim_reverse_sample(x, cond, t)
        return x

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

class TrafficDiffusion(GaussianDiffusion):
    def __init__(self, ae_model, *args, lr=[1e-5, 1e-5], weight_decay=[0.0, 0.0], **kwargs):
        super().__init__(*args, **kwargs)
        self.ae = ae_model

        self.ae_opt = optim.Adam(self.ae.parameters(),
                                    lr=lr[0],
                                    weight_decay=weight_decay[0])

        self.model_opt = optim.Adam(self.model.parameters(),
                                    lr=lr[1],
                                    weight_decay=weight_decay[1])
                

    def embed(self, *args, **kwargs):
        return self.ae.embed(*args, **kwargs)

    def decode_embedding(self, *args, **kwargs):
        return self.ae.decode_embedding(*args, **kwargs)
    
    def reconstruct(self, *args, **kwargs):
        return self.ae.reconstruct(*args, **kwargs)
    
    def sample_batched(self, *args, **kwargs):
        return self.ae.sample_batched(*args, **kwargs)

    def get_normalizer(self, *args, **kwargs):
        return self.ae.get_normalizer(*args, **kwargs)

    def get_att_normalizer(self, *args, **kwargs):
        return self.ae.get_att_normalizer(*args, **kwargs)
    
    def set_att_normalizer(self, *args, **kwargs):
        return self.ae.set_att_normalizer(*args, **kwargs)
    
    def set_normalizer(self, *args, **kwargs):
        return self.ae.set_normalizer(*args, **kwargs)

    def zero_grad(self):
        self.model_opt.zero_grad()
        self.ae_opt.zero_grad()

    def step(self):
        self.model_opt.step()
        self.ae_opt.step()

    def save_state(self, file_outs, cur_epoch=0, min_val_loss=float('Inf'), ignore_keys=None):
        model_state_dict = self.model.state_dict()
        ae_state_dict = self.ae.state_dict()

        if ignore_keys is not None:
            model_state_dict = {k: v for k, v in model_state_dict.items() if k.split('.')[0] not in ignore_keys}
            ae_state_dict = {k: v for k, v in ae_state_dict.items() if k.split('.')[0] not in ignore_keys}

        full_checkpoint_dict = {
            'model' : model_state_dict,
            'ae': ae_state_dict,
            'model_opt' : self.model_opt.state_dict(),
            'ae_opt' : self.ae_opt.state_dict(),
            'epoch' : cur_epoch,
            'min_val_loss' : min_val_loss,
        }

        for _, file_out in enumerate(file_outs):
            torch.save(full_checkpoint_dict, file_out)

    def load_state(self, load_path, map_location=None, ignore_keys=None):
        if not os.path.exists(load_path):
            print('Could not find checkpoint at path ' + load_path)

        full_checkpoint_dict = torch.load(load_path, map_location=map_location)

        # load model weights        
        load_model_state(self.model, full_checkpoint_dict['model'], ignore_keys)
        load_model_state(self.ae, full_checkpoint_dict['ae'], ignore_keys)

        # load optimizer weights
        if self.model_opt is not None:
            self.model_opt.load_state_dict(full_checkpoint_dict['model_opt'])
        if self.ae_opt is not None:
            self.ae_opt.load_state_dict(full_checkpoint_dict['ae_opt'])

        return full_checkpoint_dict['epoch'], full_checkpoint_dict['min_val_loss']
    
    def train(self, both=False):
        self.ae.train()
        self.model.train()

    def eval(self):
        self.ae.eval()
        self.model.eval()

class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info
    
    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)

