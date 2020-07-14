import torch
import random
import numpy as np
import math
from torch.nn import functional as F
from torch import nn
from torch.distributions import Normal
from .module import anneal
from .arch import ARCH
from .bg import BgModule
from .fg import FgModule
from .fg_deter import FgModuleDeter


class GSWM(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
        self.T = ARCH.T[0]
        if ARCH.DETER:
            self.fg_module = FgModuleDeter()
        else:
            self.fg_module = FgModule()
        self.bg_module = BgModule()
        self.sigma = ARCH.SIGMA_START_VALUE if ARCH.SIGMA_ANNEAL else ARCH.SIGMA
    
    def anneal(self, global_step):
        self.fg_module.anneal(global_step)
        self.bg_module.anneal(global_step)
        if ARCH.SIGMA_ANNEAL:
            self.sigma = anneal(global_step, ARCH.SIGMA_START_STEP, ARCH.SIGMA_END_STEP,
                                ARCH.SIGMA_START_VALUE, ARCH.SIGMA_END_VALUE, type='linear')
        
        # Change curriculum
        assert len(ARCH.T) == len(ARCH.T_MILESTONES) + 1, 'len(T) != len(T_MILESTONES) + 1'
        i = 0
        while i < len(ARCH.T_MILESTONES) and global_step > ARCH.T_MILESTONES[i]:
            i += 1
        self.T = ARCH.T[i]
    
    def random_crop(self, seq, T):
        """
        Sample a subsequence of length T
        Args:
            seq: (B, Told, 3, H, W)
            T: length

        Returns:
            seq: (B, T, 3, H, W)
        """
        t_seq = seq.size(1)
        assert t_seq >= T, f't_seq: {t_seq}, T: {T}'
        start = random.randint(0, t_seq - T)
        return seq[:, start:start + T]
    
    

    def forward(self, seq, global_step):
        """

        Args:
            seq: (B, T, 3, H, W)
            global_step: global training step

        Returns:

        """
        self.anneal(global_step)
        seq = self.random_crop(seq, self.T)
        B, T, C, H, W = seq.size()
    
        # Doing tracking
        log = self.track(seq, discovery_dropout=ARCH.DISCOVERY_DROPOUT)
    
        # (B, T, 1, H, W)
        alpha_map = log['alpha_map']
        fg = log['fg']
        bg = log['bg']
        # (B, T)
        kl_fg = log['kl_fg'].sum(-1)
        kl_bg = log['kl_bg'].sum(-1)
        assert kl_fg.size() == kl_bg.size() == (B,)
    
        # Compute total likelihood
        loglikelihood = self.gaussian_likelihood(seq, fg, bg, alpha_map, global_step)
    
        kl = kl_bg if (ARCH.BG_ON and global_step < ARCH.BG_ONLY_STEP) else kl_bg + kl_fg
    
        elbo = loglikelihood - kl
    
        # Visualization
        assert elbo.size() == (B,)
    
    
        log.update(
            elbo=elbo,
            mse=(seq - log['recon']) ** 2,
            loglikelihood=loglikelihood,
            kl=kl
        )
    
        return -elbo, log
    
    def track(self, seq, discovery_dropout):
        B, T, C, H, W = seq.size()
        # Process background
        if ARCH.BG_ON:
            bg_things = self.bg_module.encode(seq)
        else:
            bg_things = dict(bg=torch.zeros_like(seq), kl_bg=torch.zeros(B, T, device=seq.device))
    
        # Process foreground
        seq_diff = seq - bg_things['bg']
        # (B, T, >C<, H, W)
        inpt = torch.cat([seq, seq_diff], dim=2)
        fg_things = self.fg_module.track(inpt, bg_things['bg'], discovery_dropout=discovery_dropout)
    
        # Prepares things to compute reconstruction
        # (B, T, 1, H, W)
        alpha_map = fg_things['alpha_map']
        fg = fg_things['fg']
        bg = bg_things['bg']
        
        log = fg_things.copy()
        log.update(bg_things.copy())
        log.update(
            imgs=seq,
            recon=fg + (1 - alpha_map) * bg,
        )
    
        return log
    
    def generate(self, seq, cond_steps, fg_sample, bg_sample):
    
        if ARCH.BG_ON:
            bg_things = self.bg_module.generate(seq, cond_steps, bg_sample)
        else:
            bg_things = dict(bg=torch.zeros_like(seq))
            # Process foreground
        seq_diff = seq - bg_things['bg']
        # (B, T, >C<, H, W)
        inpt = torch.cat([seq, seq_diff], dim=2)
        fg_things = self.fg_module.generate(inpt, bg_things['bg'], cond_steps, fg_sample)
        
        alpha_map = fg_things['alpha_map']
        fg = fg_things['fg']
        bg = bg_things['bg']
        log = fg_things.copy()
        log.update(bg_things.copy())
        log.update(imgs=seq, recon=fg + (1 - alpha_map) * bg)
        
        return log
    
    
    def gaussian_likelihood(self, x, fg, bg, alpha_map, global_step):
        if ARCH.BG_ON and global_step < ARCH.BG_ONLY_STEP:
            recon = bg
        else:
            recon = fg + (1. - alpha_map) * bg
            
        dist = Normal(recon, self.sigma)
        # (B, T, 3, H, W)
        loglikelihood = dist.log_prob(x)
        # (B,)
        loglikelihood = loglikelihood.flatten(start_dim=1).sum(-1)
        
        return loglikelihood
        
