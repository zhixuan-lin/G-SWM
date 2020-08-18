from numpy.core.getlimits import _discovered_machar
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
from attrdict import AttrDict

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
        
        # Change curriculum #! Milestone strategy of curriculum learning.
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
    
    

    def forward(self, seq, ee_poses, global_step):
        """

        Args:
            seq: (B, T, 3, H, W)
            ee_poses: (B, T, 7) - shape of end-effector poses (x, y, z & quaternions)
            global_step: global training step
            # TODO (cheolhui): test with various priors
        Returns:

        """
        self.anneal(global_step)
        seq = self.random_crop(seq, self.T) #! crop the image sequence 
        ee_poses = self.random_crop(ee_poses, self.T) #! crop the image sequence 
        B, T, C, H, W = seq.size()
        P = ee_poses.size(-1) # dim of ee pose

        # Doing tracking, conditioned on ee_pose
        # log = self.track(seq, discovery_dropout=ARCH.DISCOVERY_DROPOUT)
        #! 1) track_rob_fg: robot is embedded in fg., equal hierarchy w/ objects
        if ARCH.ACTION_COND == 'bg':
            log = self.track_rob_bg(seq, ee_poses, discovery_dropout=ARCH.DISCOVERY_DROPOUT)
        elif ARCH.ACTION_COND == 'fg':
            log = self.track_rob_fg(seq, ee_poses, discovery_dropout=ARCH.DISCOVERY_DROPOUT)
        elif ARCH.ACTION_COND == 'agent': # seperate agent layer in scene space
            log = self.track_agent(seq, ee_poses, _discovery_dropout=ARCH.DISCOVERY_DROPOUT)
        else:
            raise ValueError("Currently Only Either BG or FG is Dupported.")

        #! 2) track_bg: robot is embedded in bg.
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
        # maximize elbo: maximize loglikelihood and minimize kl divergence
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
        if ARCH.BG_ON: # TODO (cheolhui): checkout the backgrounds
            bg_things = self.bg_module.encode(seq) # TODO (cheolhui): check how we can infer background masks
        else:
            bg_things = dict(bg=torch.zeros_like(seq), kl_bg=torch.zeros(B, T, device=seq.device))
    
        # Process foreground
        seq_diff = seq - bg_things['bg'] # [B, T, C, H, W] - [B, T, C, H, W] #? masked foreground?
        # (B, T, >C<, H, W)
        inpt = torch.cat([seq, seq_diff], dim=2) #! Input of Eq.(31) of supl. concat along channel axis ; [B, T, 6, H, W]
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
        ) #! Eq.(50) of supl.
    
        return log

    def track_rob_fg(self, seq, ee_poses, discovery_dropout):
        """ conditioned sequence on ee_poses; the actions or joint values condition 
            ee_poses: [B, T, P] where P is the dimension of end-effector pose
        """
        B, T, C, H, W = seq.size()
        P = ee_poses.size(-1) # 7-dim vector of ee pose
        # Process background
        if ARCH.BG_ON: # TODO (cheolhui): checkout the backgrounds
            bg_things = self.bg_module.encode(seq) # TODO (cheolhui): check how we can infer background masks
        else:
            bg_things = dict(bg=torch.zeros_like(seq), kl_bg=torch.zeros(B, T, device=seq.device))
    
        # Process foreground
        seq_diff = seq - bg_things['bg'] # [B, T, C, H, W] - [B, T, C, H, W] #? masked foreground?
        # (B, T, >C<, H, W)
        inpt = torch.cat([seq, seq_diff], dim=2) #! Input of Eq.(31) of supl.along channel axis ; [B, T, 6, H, W]
        # fg_things = self.fg_module.track(inpt, bg_things['bg'], discovery_dropout=discovery_dropout)
        fg_things = self.fg_module.track_rob_fg(inpt, ee_poses, bg_things['bg'], discovery_dropout=discovery_dropout)
    
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
        ) #! Eq.(50) of supl.
    
        return log

    def track_rob_bg(self, seq, ee_poses, discovery_dropout):
        """ conditioned sequence on ee_poses in the backgrounds; the actions or joint values condition 
            ee_poses: [B, T, P] where P is the dimension of end-effector pose
        """
        B, T, C, H, W = seq.size()
        P = ee_poses.size(-1) # 7-dim vector of ee pose
        # Process background
        if ARCH.BG_ON: # TODO (cheolhui): condition robot on backgrounds
            # bg_things = self.bg_module.encode(seq) # T
            bg_things = self.bg_module.encode_rob_bg(seq, ee_poses) # T
        else:
            bg_things = dict(bg=torch.zeros_like(seq), kl_bg=torch.zeros(B, T, device=seq.device))
    
        # Process foreground
        seq_diff = seq - bg_things['bg'] # [B, T, C, H, W] - [B, T, C, H, W] #? masked foreground?
        # (B, T, >C<, H, W)
        inpt = torch.cat([seq, seq_diff], dim=2) #! Input of Eq.(31) of supl.along channel axis ; [B, T, 6, H, W]
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
        ) #! Eq.(50) of supl.
    
        return log

    def track_agent(self, seq, ee_poses, discovery_dropout):
        """ conditioned sequence on ee_poses in the backgrounds; the actions or joint values condition 
            The agent layer is segmented throug GENESIS.
        """
        B, T, C, H, W = seq.size()
        P = ee_poses.size(-1) # 7-dim vector of ee pose
        # Process background
        if ARCH.BG_ON: # TODO (cheolhui): condition robot on backgrounds
            # bg_things = self.bg_module.encode(seq) # T
            bg_things = self.bg_module.encode_rob_bg(seq, ee_poses) # T
        else:
            bg_things = dict(bg=torch.zeros_like(seq), kl_bg=torch.zeros(B, T, device=seq.device))
    
        # Process foreground
        seq_diff = seq - bg_things['bg'] # [B, T, C, H, W] - [B, T, C, H, W] #? masked foreground?
        # (B, T, >C<, H, W)
        inpt = torch.cat([seq, seq_diff], dim=2) #! Input of Eq.(31) of supl.along channel axis ; [B, T, 6, H, W]
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
        ) #! Eq.(50) of supl.
    
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

    def generate_rob_fg(self, seq, ees, cond_steps, fg_sample, bg_sample):
        # generate video sequence conditioned on robot actions into foreground 
        if ARCH.BG_ON:
            bg_things = self.bg_module.generate(seq, cond_steps, bg_sample) # we don't generate bg conditioned on action
        else:
            bg_things = dict(bg=torch.zeros_like(seq))
            # Process foreground
        seq_diff = seq - bg_things['bg']
        # (B, T, >C<, H, W); ees - [B, T , P], where P is the action-dim
        inpt = torch.cat([seq, seq_diff], dim=2) # TODO (cheolhui): check the shape
        # fg_things = self.fg_module.generate(inpt, bg_things['bg'], cond_steps, fg_sample)
        # fg_things = self.fg_module.generate(inpt, bg_things['bg'], cond_steps, fg_sample)
        fg_things = self.fg_module.generate_rob_fg(inpt, ees, bg_things['bg'], cond_steps, fg_sample)
        
        alpha_map = fg_things['alpha_map']
        fg = fg_things['fg']
        bg = bg_things['bg']
        log = fg_things.copy()
        log.update(bg_things.copy())
        log.update(imgs=seq, recon=fg + (1 - alpha_map) * bg)
        
        return log

    def generate_rob_bg(self, seq, ees, cond_steps, fg_sample, bg_sample):
        # generate video sequence conditioned on robot actions into background 
        if ARCH.BG_ON:
            # bg_things = self.bg_module.generate(seq, cond_steps, bg_sample) # we don't generate bg conditioned on action
            bg_things = self.bg_module.generate_rob_bg(seq, ees, cond_steps, bg_sample) # we don't generate bg conditioned on action
        else:
            bg_things = dict(bg=torch.zeros_like(seq))
            # Process foreground
        seq_diff = seq - bg_things['bg']
        # (B, T, >C<, H, W); ees - [B, T , P], where P is the action-dim
        inpt = torch.cat([seq, seq_diff], dim=2) # TODO (cheolhui): check the shape
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
            
        dist = Normal(recon, self.sigma) #! Eq.(50) of supl.
        # (B, T, 3, H, W)
        loglikelihood = dist.log_prob(x)
        # (B,)
        loglikelihood = loglikelihood.flatten(start_dim=1).sum(-1)
        
        return loglikelihood
        
