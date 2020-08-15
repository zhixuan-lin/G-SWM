import torch
import random
import numpy as np
import math
from torch.nn import functional as F
from torch import nn
from torch.distributions import RelaxedBernoulli, Normal, kl_divergence
from torchvision.models import resnet18

from .module import MLP, gaussian_kernel_2d, kl_divergence_bern_bern, \
    BatchApply, transform_tensors, anneal, MultiLayerSubpixel, MultiLayerConv
from utils import spatial_transform
from .arch import ARCH
from collections import defaultdict


class FgModule(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.Z_DIM = ARCH.Z_PRES_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_WHAT_DIM + ARCH.Z_DYNA_DIM
        self.T = ARCH.T[0]
        self.tau = ARCH.TAU_START_VALUE
        self.z_pres_prior_prob = ARCH.Z_PRES_PROB_START_VALUE
        self.z_scale_prior_loc = ARCH.Z_SCALE_MEAN_START_VALUE
        self.z_scale_prior_scale = ARCH.Z_SCALE_STD
        self.z_shift_prior_loc = ARCH.Z_SHIFT_MEAN
        self.z_shift_prior_scale = ARCH.Z_SHIFT_STD
        
        self.img_encoder = ImgEncoder()
        self.proposal_encoder = ProposalEncoder()
        self.pred_proposal = PredProposal()
        self.glimpse_decoder = GlimpseDecoder()
        self.latent_post_disc = LatentPostDisc()  
        self.latent_post_disc_action = LatentPostDisc() #! action-conditioned
        # self.latent_post_prop = LatentPostProp()
        self.latent_post_prop_action = LatentPostProp() #! action-conditioned
        # self.latent_prior_prop = LatentPriorProp()
        self.latent_prior_prop_action = LatentPriorProp()
        self.pres_depth_where_what_prior = PresDepthWhereWhatPrior()
        #TODO: the latent part is not used. I put it here so we can load old checkpoints
        self.pres_depth_where_what_latent_post_disc = PresDepthWhereWhatPostLatentDisc()
        self.bg_attention_encoder = BgAttentionEncoder()
        
        # For compute propagation map for discovery conditioning
        self.prop_map_mlp = MLP((
            # ARCH.Z_DEPTH_DIM + ARCH.Z_WHAT_DIM,
            self.Z_DIM,
            *ARCH.PROP_MAP_MLP_LAYERS,
            ARCH.PROP_MAP_DIM,
        ),
            act=nn.CELU(),
        )
        # Compute propagation conditioning
        self.prop_cond_self_prior = MLP(
            (
                self.Z_DIM + ARCH.RNN_HIDDEN_DIM,
                *ARCH.PROP_COND_MLP_LAYERS,
                ARCH.PROP_COND_DIM,
            ),
            act=nn.CELU(),
        )
        self.prop_cond_relational_prior = MLP(
            (
                (self.Z_DIM + ARCH.RNN_HIDDEN_DIM) * 2,
                # self.Z_DIM + ARCH.RNN_HIDDEN_DIM + ARCH.PROP_COND_DIM,
                *ARCH.PROP_COND_MLP_LAYERS,
                ARCH.PROP_COND_DIM,
            ),
            act=nn.CELU(),
        )
        
        self.prop_cond_self_post = MLP(
            (
                self.Z_DIM + ARCH.RNN_HIDDEN_DIM,
                *ARCH.PROP_COND_MLP_LAYERS,
                ARCH.PROP_COND_DIM,
            ),
            act=nn.CELU(),
        )
        self.prop_cond_relational_post = MLP(
            (
                (self.Z_DIM + ARCH.RNN_HIDDEN_DIM) * 2,
                # self.Z_DIM + ARCH.RNN_HIDDEN_DIM + ARCH.PROP_COND_DIM,
                *ARCH.PROP_COND_MLP_LAYERS,
                ARCH.PROP_COND_DIM,
            ),
            act=nn.CELU(),
        )
        self.prop_cond_weights_prior = MLP(
            (
                (self.Z_DIM + ARCH.RNN_HIDDEN_DIM) * 2,
                # self.Z_DIM + ARCH.RNN_HIDDEN_DIM + ARCH.PROP_COND_DIM,
                *ARCH.PROP_COND_MLP_LAYERS,
                1,
            ),
            act=nn.CELU(),
        )
        self.prop_cond_weights_post = MLP(
            (
                (self.Z_DIM + ARCH.RNN_HIDDEN_DIM) * 2,
                # self.Z_DIM + ARCH.RNN_HIDDEN_DIM + ARCH.PROP_COND_DIM,
                *ARCH.PROP_COND_MLP_LAYERS,
                1,
            ),
            act=nn.CELU(),
        )
        
        # Propagation RNN initial states
        self.h_init_post = nn.Parameter(torch.randn(1, 1, ARCH.RNN_HIDDEN_DIM))
        self.c_init_post = nn.Parameter(torch.randn(1, 1, ARCH.RNN_HIDDEN_DIM))
        
        self.h_init_prior = nn.Parameter(torch.randn(1, 1, ARCH.RNN_HIDDEN_DIM))
        self.c_init_prior = nn.Parameter(torch.randn(1, 1, ARCH.RNN_HIDDEN_DIM))
        
        # Propagation RNN initial states
        # TODO: not used, remove this
        self.prop_cond_init_post = nn.Parameter(torch.randn(1, 1, ARCH.PROP_COND_DIM))
        self.prop_cond_init_prior = nn.Parameter(torch.randn(1, 1, ARCH.PROP_COND_DIM))
        
        # Temporal object state rnn, used to encode history of z
        self.temporal_rnn_post_input = BatchApply(nn.Linear(
            self.Z_DIM + ARCH.PROP_COND_DIM + ARCH.BG_PROPOSAL_DIM,
            ARCH.RNN_INPUT_DIM
        ))
        self.temporal_rnn_post = BatchApply(nn.LSTMCell(
            ARCH.RNN_INPUT_DIM,
            ARCH.RNN_HIDDEN_DIM
        ))
        
        self.temporal_rnn_prior_input = BatchApply(nn.Linear(
            self.Z_DIM + ARCH.PROP_COND_DIM + ARCH.BG_PROPOSAL_DIM,
            ARCH.RNN_INPUT_DIM
        ))
        self.temporal_rnn_prior = BatchApply(nn.LSTMCell(
            ARCH.RNN_INPUT_DIM,
            ARCH.RNN_HIDDEN_DIM))
    
    def anneal(self, global_step):
        self.tau = anneal(global_step, ARCH.TAU_START_STEP, ARCH.TAU_END_STEP,
                          ARCH.TAU_START_VALUE, ARCH.TAU_END_VALUE, 'linear')
        self.z_scale_prior_loc = anneal(global_step, ARCH.Z_SCALE_MEAN_START_STEP, ARCH.Z_SCALE_MEAN_END_STEP,
                                        ARCH.Z_SCALE_MEAN_START_VALUE, ARCH.Z_SCALE_MEAN_END_VALUE, 'linear')
        self.z_pres_prior_prob = anneal(global_step, ARCH.Z_PRES_PROB_START_STEP, ARCH.Z_PRES_PROB_END_STEP,
                                        ARCH.Z_PRES_PROB_START_VALUE, ARCH.Z_PRES_PROB_END_VALUE, 'exp')
    
    def get_discovery_priors(self, device):
        """
        Returns:
            z_depth_prior
            z_where_prior
            z_what_prior
            z_dyna_prior
        """
        return (
            Normal(0, 1),
            Normal(torch.tensor([self.z_scale_prior_loc] * 2 + [self.z_shift_prior_loc] * 2, device=device),
                   torch.tensor([self.z_scale_prior_scale] * 2 + [self.z_shift_prior_scale] * 2, device=device)
                   ),
            Normal(0, 1),
            Normal(0, 1)
        )
    
    def get_state_init(self, B, prior_or_post):
        assert prior_or_post in ['post', 'prior']
        if prior_or_post == 'prior':
            h = self.h_init_prior.expand(B, ARCH.G ** 2, ARCH.RNN_HIDDEN_DIM)
            c = self.c_init_prior.expand(B, ARCH.G ** 2, ARCH.RNN_HIDDEN_DIM)
        else:
            h = self.h_init_post.expand(B, ARCH.G ** 2, ARCH.RNN_HIDDEN_DIM)
            c = self.c_init_post.expand(B, ARCH.G ** 2, ARCH.RNN_HIDDEN_DIM)
        
        return h, c
    
    def get_dummy_things(self, B, device):
        # Empty previous time step
        h_post = torch.zeros(B, 0, ARCH.RNN_HIDDEN_DIM, device=device)
        c_post = torch.zeros(B, 0, ARCH.RNN_HIDDEN_DIM, device=device)
        h_prior = torch.zeros(B, 0, ARCH.RNN_HIDDEN_DIM, device=device)
        c_prior = torch.zeros(B, 0, ARCH.RNN_HIDDEN_DIM, device=device)
        z_pres = torch.zeros(B, 0, ARCH.Z_PRES_DIM, device=device)
        z_depth = torch.zeros(B, 0, ARCH.Z_DEPTH_DIM, device=device)
        z_where = torch.zeros(B, 0, ARCH.Z_WHERE_DIM, device=device)
        z_what = torch.zeros(B, 0, ARCH.Z_WHAT_DIM, device=device)
        z_dyna = torch.zeros(B, 0, ARCH.Z_DYNA_DIM, device=device)
        ids = torch.zeros(B, 0, device=device).long()
        return (h_post, c_post), (h_prior, c_prior), (z_pres, z_depth, z_where, z_what, z_dyna), ids
    
    def forward(self, *args, **kargs):
        return self.track(*args, **kargs)
    
    def track(self, seq, bg, discovery_dropout):
        """
        Doing tracking
        Args:
            seq: (B, T, C, H, W)
            bg: (B, T, C, H, W)
            discovery_dropout: (0, 1)

        Returns:
            A dictionary. Everything will be (B, T, ...). Refer to things_t.
        """
        B, T, C, H, W = seq.size() 
        #! Empty, N=0, clean states
        state_post, state_prior, z, ids = self.get_dummy_things(B, seq.device) # all params are zeros.
        start_id = torch.zeros(B, device=seq.device).long()
        
        things = defaultdict(list)
        first = True
        for t in range(T): # track over time seq., #! It seems like there's no dependency along the timestep.
            # (B, 3, H, W)
            x = seq[:, t] # seq is concat of image of fg masks
            # Update object states: propagate from prev step to current step
            state_post_prop, state_prior_prop, z_prop, kl_prop, proposal = self.propagate(x, state_post, state_prior, z,
                                                                                          bg[:, t])
            ids_prop = ids
            if first or torch.rand(1) > discovery_dropout: # discovery for intervention of new object
                state_post_disc, state_prior_disc, z_disc, ids_disc, kl_disc = self.discover(x, z_prop, bg[:, t], start_id)
                first = False
            else: # Do not conduct discovery
                state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, seq.device)
                kl_disc = (0.0, 0.0, 0.0, 0.0, 0.0)
            
            # TODO: for proposal of discovery, we are just using z_where
            #! Combine discovered and propagated things, and sort by p(z_pres)
            state_post, state_prior, z, ids, proposal = self.combine(
                state_post_disc, state_prior_disc, z_disc, ids_disc, z_disc[2],
                state_post_prop, state_prior_prop, z_prop, ids_prop, proposal
            )
            kl = [x + y for (x, y) in zip(kl_prop, kl_disc)] # make a paired list of kl-divergences
            fg, alpha_map = self.render(z)
            start_id = ids.max(dim=1)[0] + 1
            
            things_t = dict(
                z_pres=z[0],  # (B, N, 1)
                z_depth=z[1],  # (B, N, 1)
                z_where=z[2],  # (B, N, 4)
                z_what=z[3],  # (B, N, D)
                z_dyna=z[4],  # (B, N, D)
                kl_pres=kl[0],  # (B,)
                kl_depth=kl[1],  # (B,)
                kl_where=kl[2],  # (B,)
                kl_what=kl[3],  # (B,)
                kl_dyna=kl[4],  # (B,)
                kl_fg=kl[0] + kl[1] + kl[2] + kl[3] + kl[4],  # (B,)
                ids=ids,  # (B, N)
                fg=fg,  # (B, C, H, W)
                proposal=proposal,  # (B, N, 4)
                alpha_map=alpha_map  # (B, 1, H, W)
            )
            for key in things_t:
                things[key].append(things_t[key])
        
        things = {k: torch.stack(v, dim=1) for k, v in things.items()}
        return things

    def track_rob_fg(self, seq, ee_poses, bg, discovery_dropout):
            """
            Doing tracking robot as a foreground, requires conditioning robot variables
            Args:
                seq: (B, T, C, H, W)
                ee_poses: [B, T, P]
                bg: (B, T, C, H, W)
                discovery_dropout: (0, 1)

            Returns:
                A dictionary. Everything will be (B, T, ...). Refer to things_t.
            """
            B, T, C, H, W = seq.size() 
            #! Empty, N=0, clean states
            state_post, state_prior, z, ids = self.get_dummy_things(B, seq.device) # all params are zeros.
            start_id = torch.zeros(B, device=seq.device).long() # TODO (cheolhui): Figure out what 'id' does.
            
            things = defaultdict(list)
            first = True
            for t in range(T): # track over time seq., #! It seems like there's no dependency along the timestep.
                # (B, 3, H, W)
                x = seq[:, t] # seq is concat of image of fg masks
                ee = ee_poses[:, t]
                # Update object states: propagate from prev step to current step
                # state_post_prop, state_prior_prop, z_prop, kl_prop, proposal = self.propagate(x, state_post, state_prior, z,
                #                                                                             bg[:, t])
                state_post_prop, state_prior_prop, z_prop, kl_prop, proposal = self.propagate_rob_fg(x, ee, state_post, state_prior, z,
                                                                                            bg[:, t])
                ids_prop = ids
                if first or torch.rand(1) > discovery_dropout: # discovery for intervention of new object
                    state_post_disc, state_prior_disc, z_disc, ids_disc, kl_disc = self.discover(x, z_prop, bg[:, t], start_id)
                    # state_post_disc, state_prior_disc, z_disc, ids_disc, kl_disc = self.discover_rob_fg(x, ee, z_prop, bg[:, t], start_id)
                    first = False
                else: # Do not conduct discovery
                    state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, seq.device)
                    kl_disc = (0.0, 0.0, 0.0, 0.0, 0.0)
                
                # TODO: for proposal of discovery, we are just using z_where
                #! Combine discovered and propagated things, and sort by p(z_pres)
                state_post, state_prior, z, ids, proposal = self.combine(
                    state_post_disc, state_prior_disc, z_disc, ids_disc, z_disc[2],
                    state_post_prop, state_prior_prop, z_prop, ids_prop, proposal
                )
                kl = [x + y for (x, y) in zip(kl_prop, kl_disc)] # make a paired list of kl-divergences
                fg, alpha_map = self.render(z)
                start_id = ids.max(dim=1)[0] + 1
                
                things_t = dict(
                    z_pres=z[0],  # (B, N, 1)
                    z_depth=z[1],  # (B, N, 1)
                    z_where=z[2],  # (B, N, 4)
                    z_what=z[3],  # (B, N, D)
                    z_dyna=z[4],  # (B, N, D)
                    kl_pres=kl[0],  # (B,)
                    kl_depth=kl[1],  # (B,)
                    kl_where=kl[2],  # (B,)
                    kl_what=kl[3],  # (B,)
                    kl_dyna=kl[4],  # (B,)
                    kl_fg=kl[0] + kl[1] + kl[2] + kl[3] + kl[4],  # (B,)
                    ids=ids,  # (B, N)
                    fg=fg,  # (B, C, H, W)
                    proposal=proposal,  # (B, N, 4)
                    alpha_map=alpha_map  # (B, 1, H, W)
                )
                for key in things_t:
                    things[key].append(things_t[key])
            
            things = {k: torch.stack(v, dim=1) for k, v in things.items()}
            return things
    
    def generate(self, seq, bg, cond_steps, sample):
        """
        Generate new frames, given a set of input frames
        Args:
            seq: (B, T, 3, H, W)
            bg: (B, T, 3, H, W), generated bg images
            cond_steps: number of input steps
            sample: bool, sample or take mean

        Returns:
            log
        """
        B, T, *_ = seq.size()
        
        start_id = torch.zeros(B, device=seq.device).long()
        state_post, state_prior, z, ids = self.get_dummy_things(B, seq.device)
        
        things = defaultdict(list)
        for t in range(T):
            
            if t < cond_steps: # inference?
                # Input, use posterior
                x = seq[:, t]
                # Tracking
                state_post_prop, state_prior_prop, z_prop, kl_prop, proposal = self.propagate(x, state_post,
                                                                                              state_prior, z, bg[:, t])
                ids_prop = ids
                if t == 0:
                    state_post_disc, state_prior_disc, z_disc, ids_disc, kl_disc = self.discover(x, z_prop, bg[:, t],
                                                                                                 start_id)
                else:
                    state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, seq.device)
            else:
                # Generation, use prior
                state_prior_prop, z_prop = self.propagate_gen(state_prior, z, bg[:, t], sample)
                state_post_prop = state_prior_prop
                ids_prop = ids
                state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, seq.device)
            
            state_post, state_prior, z, ids, proposal = self.combine(
                state_post_disc, state_prior_disc, z_disc, ids_disc, z_disc[2],
                state_post_prop, state_prior_prop, z_prop, ids_prop, proposal
            )
            
            fg, alpha_map = self.render(z)
            start_id = ids.max(dim=1)[0] + 1
            things_t = dict(
                z_pres=z[0],  # (B, N, 1)
                z_depth=z[1],  # (B, N, 1)
                z_where=z[2],  # (B, N, 4)
                z_what=z[3],  # (B, N, D)
                z_dyna=z[4],  # (B, N, D)
                ids=ids,  # (B, N)
                fg=fg,  # (B, C, H, W)
                alpha_map=alpha_map  # (B, 1, H, W)
            )
            for key in things_t:
                things[key].append(things_t[key])
        
        things = {k: torch.stack(v, dim=1) for k, v in things.items()}
        return things
    
    def generate_rob_fg(self, seq, ees, bg, cond_steps, sample):
        """
        Generate new frames, given a set of input frames, and condition oln
        Args:
            seq: (B, T, 3, H, W)
            ees: (B, T, P)
            bg: (B, T, 3, H, W), generated bg images
            cond_steps: number of input steps
            sample: bool, sample or take mean

        Returns:
            log
        """
        B, T, *_ = seq.size()
        
        start_id = torch.zeros(B, device=seq.device).long()
        state_post, state_prior, z, ids = self.get_dummy_things(B, seq.device) #! get state tensors to be updated
        
        things = defaultdict(list)
        for t in range(T):
            
            if t < cond_steps: # inference?
                # Input, use posterior
                x = seq[:, t]
                ee = ees[:, t]
                # Tracking
                # state_post_prop, state_prior_prop, z_prop, kl_prop, proposal = self.propagate(x, state_post,
                #                                                                               state_prior, z, bg[:, t])
                state_post_prop, state_prior_prop, z_prop, kl_prop, proposal = self.propagate_rob_fg(x, ee, state_post,
                                                                                              state_prior, z, bg[:, t])
                ids_prop = ids
                if t == 0:
                    state_post_disc, state_prior_disc, z_disc, ids_disc, kl_disc = self.discover(x, z_prop, bg[:, t],
                                                                                                 start_id)
                else:
                    state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, seq.device)
            else:
                # Generation, use prior
                # state_prior_prop, z_prop = self.propagate_gen(state_prior, z, bg[:, t], sample)
                state_prior_prop, z_prop = self.propagate_gen_rob_fg(state_prior, z, ee, bg[:, t], sample) #? should I condition action on prior?
                state_post_prop = state_prior_prop
                ids_prop = ids
                state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, seq.device)
            
            state_post, state_prior, z, ids, proposal = self.combine(
                state_post_disc, state_prior_disc, z_disc, ids_disc, z_disc[2],
                state_post_prop, state_prior_prop, z_prop, ids_prop, proposal
            )
            
            fg, alpha_map = self.render(z)
            start_id = ids.max(dim=1)[0] + 1
            things_t = dict(
                z_pres=z[0],  # (B, N, 1)
                z_depth=z[1],  # (B, N, 1)
                z_where=z[2],  # (B, N, 4)
                z_what=z[3],  # (B, N, D)
                z_dyna=z[4],  # (B, N, D)
                ids=ids,  # (B, N)
                fg=fg,  # (B, C, H, W)
                alpha_map=alpha_map  # (B, 1, H, W)
            )
            for key in things_t:
                things[key].append(things_t[key])
        
        things = {k: torch.stack(v, dim=1) for k, v in things.items()}
        return things
    
    def discover(self, x, z_prop, bg, start_id=0):
        """ #! Sec A.3. of supl.
        Given current image and propagated objects, discover new objects
        Args:
            x: (B, D, H, W), current input image
            z_prop:
                z_pres_prop: (B, N, 1)
                z_depth_prop: (B, N, 1)
                z_where_prop: (B, N, 4)
                z_what_prop: (B, N, D)
                z_dyna_prop: (B, N, D)
            start_id: the id to start indexing
        #! authors assume independent prior for each object
        Returns:
            (h_post, c_post): (B, N, D)
            (h_prior, c_prior): (B, N, D)
            z:
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
                z_dyna: (B, N, D)
            ids: (B, N)
            kl:
                kl_pres: (B,)
                kl_depth: (B,)
                kl_where: (B,)
                kl_what: (B,)
                kl_dyna (B,)
        )
        """
        B, *_ = x.size() #! Wow, this kinda operation was possible
        
        # (B, D, G, G)
        x_enc = self.img_encoder(x) #! Conv_{disc} is implemented w/ ResNet
        # For each discovery cell, we combine propagated objects weighted by distances
        # (B, D, G, G)
        prop_map = self.compute_prop_map(z_prop) # construct 2D Gaussian kernel #! Eq.(32) of supl?
        # (B, D, G, G)
        enc = torch.cat([x_enc, prop_map], dim=1)
        #! get posteriors of latents from discovery
        (z_pres_post_prob, z_depth_post_loc, z_depth_post_scale, z_where_post_loc,
         z_where_post_scale, z_what_post_loc, z_what_post_scale, z_dyna_loc,
         z_dyna_scale) = self.pres_depth_where_what_latent_post_disc(enc) #! Eq.(33) of supl
        
        z_dyna_loc, z_dyna_scale = self.latent_post_disc(enc) #! Eq.(34) of supl
        # z_dyna_loc, z_dyna_scale = self.latent_post_disc_action.forward_act(enc, ee) #! Eq.(34) of supl
        z_dyna_post = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_post.rsample()
        
        # Compute posteriors. All (B, G*G, D)
        z_pres_post = RelaxedBernoulli(temperature=self.tau, probs=z_pres_post_prob)
        z_pres = z_pres_post.rsample() #! Eq.(35) of supl
        
        z_depth_post = Normal(z_depth_post_loc, z_depth_post_scale)
        z_depth = z_depth_post.rsample() #! Eq.(36) of supl
        
        z_where_post = Normal(z_where_post_loc, z_where_post_scale)
        z_where = z_where_post.rsample() #! Eq.(37) of supl
        z_where = self.z_where_relative_to_absolute(z_where) #! Eq.(39), (40)
        
        z_what_post = Normal(z_what_post_loc, z_what_post_scale)
        z_what = z_what_post.rsample() #! Eq.(38) of supl
        
        # Combine the posterior samples z_{***}
        z = (z_pres, z_depth, z_where, z_what, z_dyna)

        # Rejection
        if ARCH.REJECTION: # Rejection adopted from SCALOR
            z = self.rejection(z, z_prop, ARCH.REJECTION_THRESHOLD)
        
        # Compute object ids; start_id -> id to start indexing
        # tensor([0, 1, ..., G*G-1]) (B, G*G) + (B, 1) = (B, G*G)
        ids = torch.arange(ARCH.G ** 2, device=x_enc.device).expand(B, ARCH.G ** 2) + start_id[:, None]
        
        # Update temporal states
        state_post_prev = self.get_state_init(B, 'post')
        state_post = self.temporal_encode(state_post_prev, z, bg, prior_or_post='post') # get random hidden & cell state
        
        state_prior_prev = self.get_state_init(B, 'prior')
        state_prior = self.temporal_encode(state_prior_prev, z, bg, prior_or_post='prior')
        
        # All (B, G*G, D)
        # Conditional kl divergences
        kl_pres = kl_divergence_bern_bern(z_pres_post_prob, torch.full_like(z_pres_post_prob, self.z_pres_prior_prob))
        
        z_depth_prior, z_where_prior, z_what_prior, z_dyna_prior = self.get_discovery_priors(x.device)
        # where prior, (B, G*G, 4)
        kl_where = kl_divergence(z_where_post, z_where_prior)
        kl_where = kl_where * z_pres
        
        # what prior (B, G*G, D)
        kl_what = kl_divergence(z_what_post, z_what_prior)
        kl_what = kl_what * z_pres
        
        # what prior (B, G*G, D)
        kl_depth = kl_divergence(z_depth_post, z_depth_prior)
        kl_depth = kl_depth * z_pres
        
        # latent prior (B, G*G, D)
        kl_dyna = kl_divergence(z_dyna_post, z_dyna_prior)
        kl_dyna = kl_dyna
        
        #! Sum over non-batch dimensions 
        kl_pres = kl_pres.flatten(start_dim=1).sum(1)
        kl_where = kl_where.flatten(start_dim=1).sum(1)
        kl_what = kl_what.flatten(start_dim=1).sum(1)
        kl_depth = kl_depth.flatten(start_dim=1).sum(1)
        kl_dyna = kl_dyna.flatten(start_dim=1).sum(1)
        # kl_dyna = torch.zeros_like(kl_dyna)
        kl = (kl_pres, kl_depth, kl_where, kl_what, kl_dyna)
        
        return state_post, state_prior, z, ids, kl
    
    def discover_rob_fg(self, x, ee, z_prop, bg, start_id=0):
        """ #! Sec A.3. of supl.
        Given current image and propagated objects, discover new objects
        Args:
            x: (B, D, H, W), current input image
            ee: (B, P)
            z_prop:
                z_pres_prop: (B, N, 1)
                z_depth_prop: (B, N, 1)
                z_where_prop: (B, N, 4)
                z_what_prop: (B, N, D)
                z_dyna_prop: (B, N, D)
            start_id: the id to start indexing
        #! authors assume independent prior for each object
        Returns:
            (h_post, c_post): (B, N, D)
            (h_prior, c_prior): (B, N, D)
            z:
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
                z_dyna: (B, N, D)
            ids: (B, N)
            kl:
                kl_pres: (B,)
                kl_depth: (B,)
                kl_where: (B,)
                kl_what: (B,)
                kl_dyna (B,)
        )
        """
        B, *_ = x.size() #! Wow, this kinda operation was possible
        
        # (B, D, G, G)
        x_enc = self.img_encoder(x) #! Conv_{disc} is implemented w/ ResNet
        # For each discovery cell, we combine propagated objects weighted by distances
        # (B, D, G, G)
        prop_map = self.compute_prop_map(z_prop) # construct 2D Gaussian kernel #! Eq.(32) of supl?
        # (B, D, G, G)
        enc = torch.cat([x_enc, prop_map], dim=1)
        #! get posteriors of latents from discovery
        (z_pres_post_prob, z_depth_post_loc, z_depth_post_scale, z_where_post_loc,
         z_where_post_scale, z_what_post_loc, z_what_post_scale, z_dyna_loc,
         z_dyna_scale) = self.pres_depth_where_what_latent_post_disc(enc) #! Eq.(33) of supl
        # NOTE that the shap is different!
        z_dyna_loc, z_dyna_scale = self.latent_post_disc(enc) #! Eq.(34) of supl
        # z_dyna_loc, z_dyna_scale = self.latent_post_disc_action.forward_act(enc, ee) # TODO (cheolhui): condition action here.
        z_dyna_post = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_post.rsample()
        
        # Compute posteriors. All (B, G*G, D)
        z_pres_post = RelaxedBernoulli(temperature=self.tau, probs=z_pres_post_prob)
        z_pres = z_pres_post.rsample() #! Eq.(35) of supl
        
        z_depth_post = Normal(z_depth_post_loc, z_depth_post_scale)
        z_depth = z_depth_post.rsample() #! Eq.(36) of supl
        
        z_where_post = Normal(z_where_post_loc, z_where_post_scale)
        z_where = z_where_post.rsample() #! Eq.(37) of supl
        z_where = self.z_where_relative_to_absolute(z_where) #! Eq.(39), (40)
        
        z_what_post = Normal(z_what_post_loc, z_what_post_scale)
        z_what = z_what_post.rsample() #! Eq.(38) of supl
        
        # Combine the posterior samples z_{***}
        z = (z_pres, z_depth, z_where, z_what, z_dyna)

        # Rejection # TODO (cheolhui): check how to deal with incompatible shapes btwn prop and disc.
        if ARCH.REJECTION: # Rejection adopted from SCALOR
            z = self.rejection(z, z_prop, ARCH.REJECTION_THRESHOLD)
        
        # Compute object ids; start_id -> id to start indexing
        # tensor([0, 1, ..., G*G-1]) (B, G*G) + (B, 1) = (B, G*G)
        ids = torch.arange(ARCH.G ** 2, device=x_enc.device).expand(B, ARCH.G ** 2) + start_id[:, None]
        
        # Update temporal states
        state_post_prev = self.get_state_init(B, 'post')
        state_post = self.temporal_encode(state_post_prev, z, bg, prior_or_post='post') # get random hidden & cell state
        
        state_prior_prev = self.get_state_init(B, 'prior')
        state_prior = self.temporal_encode(state_prior_prev, z, bg, prior_or_post='prior')
        
        # All (B, G*G, D)
        # Conditional kl divergences
        kl_pres = kl_divergence_bern_bern(z_pres_post_prob, torch.full_like(z_pres_post_prob, self.z_pres_prior_prob))
        
        z_depth_prior, z_where_prior, z_what_prior, z_dyna_prior = self.get_discovery_priors(x.device)
        # where prior, (B, G*G, 4)
        kl_where = kl_divergence(z_where_post, z_where_prior)
        kl_where = kl_where * z_pres
        
        # what prior (B, G*G, D)
        kl_what = kl_divergence(z_what_post, z_what_prior)
        kl_what = kl_what * z_pres
        
        # what prior (B, G*G, D)
        kl_depth = kl_divergence(z_depth_post, z_depth_prior)
        kl_depth = kl_depth * z_pres
        
        # latent prior (B, G*G, D)
        kl_dyna = kl_divergence(z_dyna_post, z_dyna_prior)
        kl_dyna = kl_dyna
        
        #! Sum over non-batch dimensions 
        kl_pres = kl_pres.flatten(start_dim=1).sum(1)
        kl_where = kl_where.flatten(start_dim=1).sum(1)
        kl_what = kl_what.flatten(start_dim=1).sum(1)
        kl_depth = kl_depth.flatten(start_dim=1).sum(1)
        kl_dyna = kl_dyna.flatten(start_dim=1).sum(1)
        # kl_dyna = torch.zeros_like(kl_dyna)
        kl = (kl_pres, kl_depth, kl_where, kl_what, kl_dyna)
        
        return state_post, state_prior, z, ids, kl
    
    def propagate_gen(self, state_prev, z_prev, bg, sample=False):
        """
        One step of propagation generation
        Args:
            h_prev, c_prev: (B, N, D)
            z_prev:
                z_pres_prev: (B, N, 1)
                z_depth_prev: (B, N, 1)
                z_where_prev: (B, N, 4)
                z_what_prev: (B, N, D)
        Returns:
            h, c: (B, N, D)
            z:
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
        """
        h_prev, c_prev = state_prev
        z_pres_prev, z_depth_prev, z_where_prev, z_what_prev, z_dyna_prev = z_prev
        
        # (B, N, D)
        #! latent size of z_dyna is 128 (default)
        z_dyna_loc, z_dyna_scale = self.latent_prior_prop(state_prev[0])
        z_dyna_prior = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_prior.sample() if sample else z_dyna_loc
        
        # TODO: z_pres_prior is not learned
        # All (B, N, D)
        (z_pres_prob, z_depth_offset_loc, z_depth_offset_scale, z_where_offset_loc, z_where_offset_scale,
         z_what_offset_loc,
         z_what_offset_scale, z_depth_gate, z_where_gate, z_what_gate) = self.pres_depth_where_what_prior(z_dyna)
        
        # Always set them to one during generation
        z_pres_prior = RelaxedBernoulli(temperature=self.tau, probs=z_pres_prob)
        z_pres = z_pres_prior.sample()
        z_pres = (z_pres > 0.5).float()
        z_pres = torch.ones_like(z_pres)
        z_pres = z_pres_prev * z_pres
        
        z_where_prior = Normal(z_where_offset_loc, z_where_offset_scale)
        z_where_offset = z_where_prior.rsample() if sample else z_where_offset_loc
        z_where = torch.zeros_like(z_where_prev)
        # Scale
        z_where[..., :2] = z_where_prev[..., :2] + ARCH.Z_SCALE_UPDATE_SCALE * z_where_gate[..., :2] * torch.tanh(
            z_where_offset[..., :2])
        # Shift
        z_where[..., 2:] = z_where_prev[..., 2:] + ARCH.Z_SHIFT_UPDATE_SCALE * z_where_gate[..., 2:] * torch.tanh(
            z_where_offset[..., 2:])
        
        z_depth_prior = Normal(z_depth_offset_loc, z_depth_offset_scale)
        z_depth_offset = z_depth_prior.rsample() if sample else z_depth_offset_loc
        z_depth = z_depth_prev + ARCH.Z_DEPTH_UPDATE_SCALE * z_depth_gate * z_depth_offset
        
        z_what_prior = Normal(z_what_offset_loc, z_what_offset_scale)
        z_what_offset = z_what_prior.rsample() if sample else z_what_offset_loc
        z_what = z_what_prev + ARCH.Z_WHAT_UPDATE_SCALE * z_what_gate * torch.tanh(z_what_offset)
        
        z = (z_pres, z_depth, z_where, z_what, z_dyna)
        
        state = self.temporal_encode(state_prev, z, bg, prior_or_post='prior')
        
        return state, z

    def propagate_gen_rob_fg(self, state_prev, z_prev, ee, bg, sample=False):
        """
        One step of propagation generation
        Args:
            h_prev: (state_prev[0]), c_prev: (B, N, D)
            z_prev:
                z_pres_prev: (B, N, 1)
                z_depth_prev: (B, N, 1)
                z_where_prev: (B, N, 4)
                z_what_prev: (B, N, D)
            ee: (B, P) - there's only one robot, so axis of N is missing
        Returns:
            h, c: (B, N, D)
            z:
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
        """
        h_prev, c_prev = state_prev
        z_pres_prev, z_depth_prev, z_where_prev, z_what_prev, z_dyna_prev = z_prev
        
        # (B, N, D)
        #! latent size of z_dyna is 128 (default)
        # z_dyna_loc, z_dyna_scale = self.latent_prior_prop_action(state_prev[0], ee)
        z_dyna_loc, z_dyna_scale = self.latent_prior_prop_action.forward_action(state_prev[0], ee)
        z_dyna_prior = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_prior.sample() if sample else z_dyna_loc
        
        # TODO: z_pres_prior is not learned
        # All (B, N, D)
        (z_pres_prob, z_depth_offset_loc, z_depth_offset_scale, z_where_offset_loc, z_where_offset_scale,
         z_what_offset_loc,
         z_what_offset_scale, z_depth_gate, z_where_gate, z_what_gate) = self.pres_depth_where_what_prior(z_dyna)
        
        # Always set them to one during generation
        z_pres_prior = RelaxedBernoulli(temperature=self.tau, probs=z_pres_prob)
        z_pres = z_pres_prior.sample()
        z_pres = (z_pres > 0.5).float()
        z_pres = torch.ones_like(z_pres)
        z_pres = z_pres_prev * z_pres
        
        z_where_prior = Normal(z_where_offset_loc, z_where_offset_scale)
        z_where_offset = z_where_prior.rsample() if sample else z_where_offset_loc
        z_where = torch.zeros_like(z_where_prev)
        # Scale
        z_where[..., :2] = z_where_prev[..., :2] + ARCH.Z_SCALE_UPDATE_SCALE * z_where_gate[..., :2] * torch.tanh(
            z_where_offset[..., :2])
        # Shift
        z_where[..., 2:] = z_where_prev[..., 2:] + ARCH.Z_SHIFT_UPDATE_SCALE * z_where_gate[..., 2:] * torch.tanh(
            z_where_offset[..., 2:])
        
        z_depth_prior = Normal(z_depth_offset_loc, z_depth_offset_scale)
        z_depth_offset = z_depth_prior.rsample() if sample else z_depth_offset_loc
        z_depth = z_depth_prev + ARCH.Z_DEPTH_UPDATE_SCALE * z_depth_gate * z_depth_offset
        
        z_what_prior = Normal(z_what_offset_loc, z_what_offset_scale)
        z_what_offset = z_what_prior.rsample() if sample else z_what_offset_loc
        z_what = z_what_prev + ARCH.Z_WHAT_UPDATE_SCALE * z_what_gate * torch.tanh(z_what_offset)
        
        z = (z_pres, z_depth, z_where, z_what, z_dyna)
        
        state = self.temporal_encode(state_prev, z, bg, prior_or_post='prior')
        
        return state, z
    
    def propagate(self, x, state_post_prev, state_prior_prev, z_prev, bg):
        """
        #! One step of propagation posterior
        Args:
            x: (B, 3, H, W), img
            (h, c), (h, c): each (B, N, D)
            z_prev: #! what's inside the latent
                z_pres: (B, N, 1) -> Bernoulli distrib.
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
                z_dyna: (B, N, D)

        Returns:
            h_post, c_post: (B, N, D)
            h_prior, c_prior: (B, N, D)
            z:
                z_pres: (B, N, 1) #! explicit
                z_depth: (B, N, 1) #! explicit
                z_where: (B, N, 4) #! explicit
                z_what: (B, N, D)
                z_dyna: (B, N, D)
            kl:
                kl_pres: (B,)
                kl_what: (B,)
                kl_where: (B,)
                kl_depth: (B,)
                kl_dyna: (B,)
            proposal_region: (B, N, 4)

        """
        z_pres_prev, z_depth_prev, z_where_prev, z_what_prev, z_dyna_prev = z_prev
        B, N, _ = z_pres_prev.size()
        
        if N == 0: # num of entities. zero if nothing has been discovered before
            # No object is propagated -> then do discovery
            return state_post_prev, state_prior_prev, z_prev, (0.0, 0.0, 0.0, 0.0, 0.0), z_prev[2]
        
        h_post, c_post = state_post_prev # [B, N, 128], [B, N, 128] # NOTE that h_post is inferred by Eq.(18)
        h_prior, c_prior = state_prior_prev # [B, N, 128], [B, N, 128]
        #! 1) Inference procedure of "propagate": Sec A.2 of supl.
        # Predict proposal locations, (B, N, 2)
        proposal_offset = self.pred_proposal(h_post) #! Eq.(19) of supl., by posterior OS-RNN; h_{hat}_{t-1}^k
        proposal = torch.zeros_like(z_where_prev) # [B, N, 4] -> size of proposal area
        # Update size only; extract proposal region of the image, centered @ the prev. obj. location o_{t-1}^{xy,k} 
        proposal[..., 2:] = z_where_prev[..., 2:] # o^{xy} - (x,y)
        proposal[..., :2] = z_where_prev[..., :2] + ARCH.PROPOSAL_UPDATE_MIN + ( #! o^{hw}; Eq.(19) of supl., (h,w)
            ARCH.PROPOSAL_UPDATE_MAX - ARCH.PROPOSAL_UPDATE_MIN) * torch.sigmoid(proposal_offset)
        # proposal_offset -> (0,1) ->  relocate & rescale to (0.1, 0.3)
        # Get proposal glimpses
        # x_repeat: (B*N, 3, H, W)
        x_repeat = torch.repeat_interleave(x[:, :3], N, dim=0) # crop-out the first three channel.
        # if N = 5 & batch is a, b, ..., : aaaaabbbbbccccc ... # if torch.repeat ab...ab...ab...ab...ab...
        #! arch.glimpse_shape = [16, 16]
        # (B*N, 3, H, W) #! Eq.(4) of paper
        proposal_glimpses = spatial_transform(image=x_repeat, z_where=proposal.view(B * N, 4), #! Eq.(20) of supl.
                                              out_dims=(B * N, 3, *ARCH.GLIMPSE_SHAPE))
        # (B, N, 3, H, W)
        proposal_glimpses = proposal_glimpses.view(B, N, 3, *ARCH.GLIMPSE_SHAPE)
        # (B, N, D)
        proposal_enc = self.proposal_encoder(proposal_glimpses) #! Eq.(21) of supl.
        # (B, N, D)
        # This will be used to condition everything: # TODO (cheolhui): condition action here!
        enc = torch.cat([proposal_enc, h_post], dim=-1) # [B, N, D] + [B, N, D] = [B, N, 2D] 
        
        z_dyna_loc, z_dyna_scale = self.latent_post_prop(enc) #! Eq.(22) of supl.
        z_dyna_post = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_post.rsample() #! Eq.(23) of supl.
        # given dynamics latent (z_{dyna}) from posterior (z_dyna_post), the attribute latents are computed as follows
        #! {pres, depth, where, what} are inferred from z_dyna  (B, N, D)
        (z_pres_prob, z_depth_offset_loc, z_depth_offset_scale, z_where_offset_loc, #! Eq.(13) of supl.
         z_where_offset_scale, z_what_offset_loc, z_what_offset_scale, 
         z_depth_gate, z_where_gate, z_what_gate) = self.pres_depth_where_what_prior(z_dyna)
        
        # Sampling
        z_pres_post = RelaxedBernoulli(self.tau, probs=z_pres_prob)  #! Eq.(14) of supl.
        z_pres = z_pres_post.rsample()
        z_pres = z_pres_prev * z_pres # -> refer to Sec 2.5 of PRML
        
        z_where_post = Normal(z_where_offset_loc, z_where_offset_scale)
        z_where_offset = z_where_post.rsample()  #! Eq.(16) of supl.
        z_where = torch.zeros_like(z_where_prev)
        # Scale; [B, N, 4] -> [B, N, 2] (mean), [B, N, 2] (shift). 
        z_where[..., :2] = z_where_prev[..., :2] + ARCH.Z_SCALE_UPDATE_SCALE * z_where_gate[..., :2] * torch.tanh(
            z_where_offset[..., :2])
        # Shift
        z_where[..., 2:] = z_where_prev[..., 2:] + ARCH.Z_SHIFT_UPDATE_SCALE * z_where_gate[..., 2:] * torch.tanh(
            z_where_offset[..., 2:])
        
        z_depth_post = Normal(z_depth_offset_loc, z_depth_offset_scale)
        z_depth_offset = z_depth_post.rsample()  #! Eq.(15) of supl.
        z_depth = z_depth_prev + ARCH.Z_DEPTH_UPDATE_SCALE * z_depth_gate * z_depth_offset # [B, D, 1]
        
        z_what_post = Normal(z_what_offset_loc, z_what_offset_scale)
        z_what_offset = z_what_post.rsample()  #! Eq.(17) of supl.
        z_what = z_what_prev + ARCH.Z_WHAT_UPDATE_SCALE * z_what_gate * torch.tanh(z_what_offset) # [B, D, 64]
        z = (z_pres, z_depth, z_where, z_what, z_dyna)
        
        
        # Update states via RNN; In: state_*_prev - hidden & cell state of RNN, z - tuple of z_att and z_dyna 
        state_post = self.temporal_encode(state_post_prev, z, bg, prior_or_post='post')
        state_prior = self.temporal_encode(state_prior_prev, z, bg, prior_or_post='prior')
        
        z_dyna_loc, z_dyna_scale = self.latent_prior_prop(h_prior) #! Eq(23) of supl.
        z_dyna_prior = Normal(z_dyna_loc, z_dyna_scale)
        kl_dyna = kl_divergence(z_dyna_post, z_dyna_prior) #! we get post & prior z_{dyna} from each Eq.(23) and Eq.(12)
        
        #! This is not kl divergence. This is an auxialiary loss q(z_pres|) is fit to fixed prior
        kl_pres = kl_divergence_bern_bern(z_pres_prob, torch.full_like(z_pres_prob, self.z_pres_prior_prob))
        # If we don't want this auxiliary loss
        if not ARCH.AUX_PRES_KL:
            kl_pres = torch.zeros_like(kl_pres)
        
        # Reduced to (B,)
        
        #! Sec. 3.4.2 of paper; Again, this is not really kl
        kl_pres = kl_pres.flatten(start_dim=1).sum(-1)
        kl_dyna = kl_dyna.flatten(start_dim=1).sum(-1)
        
        # We are not using q, so these will be zero: ->  Only prior exist?
        kl_where = torch.zeros_like(kl_pres)
        kl_what = torch.zeros_like(kl_pres)
        kl_depth = torch.zeros_like(kl_pres)
        assert kl_pres.size(0) == B
        kl = (kl_pres, kl_depth, kl_where, kl_what, kl_dyna)
        
        return state_post, state_prior, z, kl, proposal


    def propagate_rob_fg(self, x, ee, state_post_prev, state_prior_prev, z_prev, bg):
        """
        # propagation conditioned on robot action (ee)
        #! One step of propagation posterior
        Args:
            x: (B, 6, H, W), img -> concat of bg & {bg-fg}
            (h, c), (h, c): each (B, N, D)
            z_prev: #! what's inside the latent
                z_pres: (B, N, 1) -> Bernoulli distrib.
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
                z_dyna: (B, N, D)

        Returns:
            h_post, c_post: (B, N, D)
            h_prior, c_prior: (B, N, D)
            z:
                z_pres: (B, N, 1) #! explicit
                z_depth: (B, N, 1) #! explicit
                z_where: (B, N, 4) #! explicit
                z_what: (B, N, D)
                z_dyna: (B, N, D)
            kl:
                kl_pres: (B,)
                kl_what: (B,)
                kl_where: (B,)
                kl_depth: (B,)
                kl_dyna: (B,)
            proposal_region: (B, N, 4)

        """
        z_pres_prev, z_depth_prev, z_where_prev, z_what_prev, z_dyna_prev = z_prev
        B, N, _ = z_pres_prev.size()
        
        if N == 0: # num of entities. zero if nothing has been discovered before
            # No object is propagated -> then do discovery
            return state_post_prev, state_prior_prev, z_prev, (0.0, 0.0, 0.0, 0.0, 0.0), z_prev[2]
        
        h_post, c_post = state_post_prev # [B, N, 128], [B, N, 128] # NOTE that h_post is inferred by Eq.(18)
        h_prior, c_prior = state_prior_prev # [B, N, 128], [B, N, 128]
        #! 1) Inference procedure of "propagate": Sec A.2 of supl.
        # Predict proposal locations, (B, N, 2)
        proposal_offset = self.pred_proposal(h_post) #! Eq.(19) of supl., by posterior OS-RNN; h_{hat}_{t-1}^k
        proposal = torch.zeros_like(z_where_prev) # [B, N, 4] -> size of proposal area
        # Update size only; extract proposal region of the image, centered @ the prev. obj. location o_{t-1}^{xy,k} 
        proposal[..., 2:] = z_where_prev[..., 2:] # o^{xy} - (x,y)
        proposal[..., :2] = z_where_prev[..., :2] + ARCH.PROPOSAL_UPDATE_MIN + ( #! o^{hw}; Eq.(19) of supl., (h,w)
            ARCH.PROPOSAL_UPDATE_MAX - ARCH.PROPOSAL_UPDATE_MIN) * torch.sigmoid(proposal_offset)
        # proposal_offset -> (0,1) ->  relocate & rescale to (0.1, 0.3)
        # Get proposal glimpses
        # x_repeat: (B*N, 3, H, W)
        x_repeat = torch.repeat_interleave(x[:, :3], N, dim=0) # crop-out the first three channel.
        # if N = 5 & batch is a, b, ..., : aaaaabbbbbccccc ... # if torch.repeat ab...ab...ab...ab...ab...
        #! arch.glimpse_shape = [16, 16]
        # (B*N, 3, H, W) #! Eq.(4) of paper
        proposal_glimpses = spatial_transform(image=x_repeat, z_where=proposal.view(B * N, 4), #! Eq.(20) of supl.
                                              out_dims=(B * N, 3, *ARCH.GLIMPSE_SHAPE))
        # (B, N, 3, H, W)
        proposal_glimpses = proposal_glimpses.view(B, N, 3, *ARCH.GLIMPSE_SHAPE)
        # (B, N, D) #
        proposal_enc = self.proposal_encoder(proposal_glimpses) #! Eq.(21) of supl.
        # proposal_enc = self.proposal_encoder.forward_action(proposal_glimpses, ee) #! Eq.(21) of supl.
        # (B, N, D)
        # This will be used to condition everything: # TODO (cheolhui): condition action here!
        enc = torch.cat([proposal_enc, h_post], dim=-1) # [B, N, D] + [B, N, D] = [B, N, 2D] 
        # TODO (chmin): or concat action here?
        # z_dyna_loc, z_dyna_scale = self.latent_post_prop(enc) #! Eq.(22) of supl.
        z_dyna_loc, z_dyna_scale = self.latent_post_prop_action.forward_action(enc, ee) #! Eq.(22) of supl.
        z_dyna_post = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_post.rsample() #! Eq.(23) of supl.
        # given dynamics latent (z_{dyna}) from posterior (z_dyna_post), the attribute latents are computed as follows
        #! {pres, depth, where, what} are inferred from z_dyna  (B, N, D)
        (z_pres_prob, z_depth_offset_loc, z_depth_offset_scale, z_where_offset_loc, #! Eq.(13) of supl.
         z_where_offset_scale, z_what_offset_loc, z_what_offset_scale, 
         z_depth_gate, z_where_gate, z_what_gate) = self.pres_depth_where_what_prior(z_dyna)
        
        # Sampling
        z_pres_post = RelaxedBernoulli(self.tau, probs=z_pres_prob)  #! Eq.(14) of supl.
        z_pres = z_pres_post.rsample()
        z_pres = z_pres_prev * z_pres # -> refer to Sec 2.5 of PRML
        
        z_where_post = Normal(z_where_offset_loc, z_where_offset_scale)
        z_where_offset = z_where_post.rsample()  #! Eq.(16) of supl.
        z_where = torch.zeros_like(z_where_prev)
        # Scale; [B, N, 4] -> [B, N, 2] (mean), [B, N, 2] (shift). 
        z_where[..., :2] = z_where_prev[..., :2] + ARCH.Z_SCALE_UPDATE_SCALE * z_where_gate[..., :2] * torch.tanh(
            z_where_offset[..., :2])
        # Shift
        z_where[..., 2:] = z_where_prev[..., 2:] + ARCH.Z_SHIFT_UPDATE_SCALE * z_where_gate[..., 2:] * torch.tanh(
            z_where_offset[..., 2:])
        
        z_depth_post = Normal(z_depth_offset_loc, z_depth_offset_scale)
        z_depth_offset = z_depth_post.rsample()  #! Eq.(15) of supl.
        z_depth = z_depth_prev + ARCH.Z_DEPTH_UPDATE_SCALE * z_depth_gate * z_depth_offset # [B, D, 1]
        
        z_what_post = Normal(z_what_offset_loc, z_what_offset_scale)
        z_what_offset = z_what_post.rsample()  #! Eq.(17) of supl.
        z_what = z_what_prev + ARCH.Z_WHAT_UPDATE_SCALE * z_what_gate * torch.tanh(z_what_offset) # [B, D, 64]
        z = (z_pres, z_depth, z_where, z_what, z_dyna)
        
        
        # Update states via RNN; In: state_*_prev - hidden & cell state of RNN, z - tuple of z_att and z_dyna 
        state_post = self.temporal_encode(state_post_prev, z, bg, prior_or_post='post')
        state_prior = self.temporal_encode(state_prior_prev, z, bg, prior_or_post='prior')
        # TODO (cheolhui): condition action for the prior.        
        # z_dyna_loc, z_dyna_scale = self.latent_prior_prop(h_prior) #! Eq(23) of supl.
        z_dyna_loc, z_dyna_scale = self.latent_prior_prop_action.forward_action(h_prior, ee) #! Eq(23) of supl.
        z_dyna_prior = Normal(z_dyna_loc, z_dyna_scale)
        kl_dyna = kl_divergence(z_dyna_post, z_dyna_prior) #! we get post & prior z_{dyna} from each Eq.(23) and Eq.(12)
        
        #! This is not kl divergence. This is an auxialiary loss q(z_pres|) is fit to fixed prior
        kl_pres = kl_divergence_bern_bern(z_pres_prob, torch.full_like(z_pres_prob, self.z_pres_prior_prob))
        # If we don't want this auxiliary loss
        if not ARCH.AUX_PRES_KL:
            kl_pres = torch.zeros_like(kl_pres)
        
        # Reduced to (B,)
        
        #! Sec. 3.4.2 of paper; Again, this is not really kl
        kl_pres = kl_pres.flatten(start_dim=1).sum(-1)
        kl_dyna = kl_dyna.flatten(start_dim=1).sum(-1)
        
        # We are not using q, so these will be zero: ->  Only prior exist?
        kl_where = torch.zeros_like(kl_pres)
        kl_what = torch.zeros_like(kl_pres)
        kl_depth = torch.zeros_like(kl_pres)
        assert kl_pres.size(0) == B
        kl = (kl_pres, kl_depth, kl_where, kl_what, kl_dyna)
        
        return state_post, state_prior, z, kl, proposal
    
    def compute_prop_map(self, z_prop):
        """ #! please refer to paragraph below Eq.(31) of supl.
        Compute a feature volume to condition discovery. The purpose is not to rediscover objects
        Args:
            z_prop:
                z_pres_prop: (B, N, D)
                z_depth_prop: (B, N, D)
                z_where_prop: (B, N, 4)
                z_what_prop: (B, N, D)

        Returns:
            map: (B, D, G, G). This will be concatenated with the image feature
        """
        # get the latents (pres, depth, where, what, dyna) from propagation
        z_pres_prop, z_depth_prop, z_where_prop, z_what_prop, z_dyna_prop = z_prop
        B, N, _ = z_pres_prop.size()
        
        if z_pres_prop.size(1) == 0:
            # First frame, empty prop map
            return torch.zeros(B, ARCH.PROP_MAP_DIM, ARCH.G, ARCH.G, device=z_pres_prop.device)
        
        assert N == ARCH.MAX # check the max # of objects in the scene
        # Use only z_what and z_depth here
        # (B, N, D)
        # TODO: I could have used relative z_where as SILOT here. But that will induce many computations. So I won't do it here.
        z_prev = torch.cat(z_prop, dim=-1) # concat (pres, depth, where, what, dyna)
        #! consider propagated objects to prevent the rediscovery.
        # (B, N, D) -> (B, N, D)
        z_prev_enc = self.prop_map_mlp(z_prev) # MLP^{cond} of Eq.(32)? 
        # (B, N, D), masked out objects with z_pres == 0
        z_prev_enc = z_prev_enc * z_pres_prop # Update the Bernoulli probability
        
        # Compute a weight matrix of size (B, G*G, N)
        
        # (2, G, G) -> (G, G, 2) -> (G*G, 2)
        offset = self.get_offset_grid(z_prev.device) # grid that represents the center of z_shift of each cell
        offset = offset.permute(1, 2, 0).view(-1, 2) 
        
        # crop out (B, N, 2) from (B, N, 4) 
        z_shift_prop = z_where_prop[..., 2:]
        
        # Distance matrix; distance btwn the prop. object and corresponding cell center.
        # (1, G*G, 1, 2)
        offset = offset[None, :, None, :]
        # (B, 1, N, 2)
        z_shift_prop = z_shift_prop[:, None, :, :]
        # (B, G*G, N, 2)
        matrix = offset - z_shift_prop
        # (B, G*G, N)
        weights = gaussian_kernel_2d(matrix, ARCH.PROP_MAP_SIGMA, dim=-1)
        # (B, G*G, N) -> (B, G*G, N, 1)
        weights = weights[..., None]
        # (B, N, D) -> (B, 1, N, D)
        z_prev_enc = z_prev_enc[:, None, ]
        
        # (B, G*G, N, 1) * (B, 1, N, D) -> (B, G*G, N, D) -> sum -> (B, G*G, D=ARCH.PROP_MAP_DIM)
        prop_map = torch.sum(weights * z_prev_enc, dim=-2) #! Eq.(32) of supl.
        assert prop_map.size() == (B, ARCH.G ** 2, ARCH.PROP_MAP_DIM)
        # (B, G, G, D)
        prop_map = prop_map.view(B, ARCH.G, ARCH.G, ARCH.PROP_MAP_DIM)
        # (B, D, G, G)
        prop_map = prop_map.permute(0, 3, 1, 2)
        
        return prop_map
    
    def compute_prop_cond(self, z, state_prev, prior_or_post):
        """
        Object interaction in vprop
        Args:
            z: z_t
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
            state_prev: h_t
                h: (B, N, D)
                c: (B, N, D)
        Returns:
            cond: (B, N, D)
        """
        assert prior_or_post in ['prior', 'post']
        if prior_or_post == 'prior':
            prop_cond_self = self.prop_cond_self_prior
            prop_cond_relational = self.prop_cond_relational_prior
            prop_cond_weights = self.prop_cond_weights_prior
        else:
            prop_cond_self = self.prop_cond_self_post
            prop_cond_relational = self.prop_cond_relational_post
            prop_cond_weights = self.prop_cond_weights_post
        
        z_pres, z_depth, z_where, z_what, z_dyna = z
        B, N, _ = z_pres.size()
        h_post_prev, c_post_prev = state_prev
        
        # The feature of one object include the following
        # (B, N, D); 
        feat = torch.cat(z + (h_post_prev,), dim=-1) # described in Sec.3.3.1
        # (B, N, D)
        enc_self = prop_cond_self(feat)
        
        # Compute weights based on gaussian
        # (B, N, 2)
        z_shift_prop = z_where[:, :, 2:]
        # (B, N, 1, 2)
        z_shift_self = z_shift_prop[:, :, None]
        # (B, 1, N, 2)
        z_shift_other = z_shift_prop[:, None]
        # (B, N, N, 2)
        dist_matrix = z_shift_self - z_shift_other
        # (B, N, 1, D) -> (B, N, N, D)
        # feat_self = enc_self[:, :, None]
        feat_self = feat[:, :, None] # (B, N, 1, D)
        feat_matrix_self = feat_self.expand(B, N, N, self.Z_DIM + ARCH.RNN_HIDDEN_DIM)
        # (B, 1, N, D) -> (B, N, N, D) #? what is the difference?
        feat_other = feat[:, None]
        feat_matrix_other = feat_other.expand(B, N, N, self.Z_DIM + ARCH.RNN_HIDDEN_DIM)
        
        # TODO: Detail here, replace absolute positions with relative ones
        offset = ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_SCALE_DIM # 1+1+2
        # Must clone. Otherwise there will be multiple write
        feat_matrix_other = feat_matrix_other.clone()
        feat_matrix_other[..., offset:offset + ARCH.Z_SHIFT_DIM] = dist_matrix
        # (B, N, N, D) + (B, N, N, D) = (B, N, N, 2D); self + others
        feat_matrix = torch.cat([feat_matrix_self, feat_matrix_other], dim=-1)
        # (B, N, N, D) # NOTE that dim Ds may differ among variables
        relational_matrix = prop_cond_relational(feat_matrix)
        
        # COMPUTE WEIGHTS
        # (B, N, N, 1)
        weight_matrix = prop_cond_weights(feat_matrix)
        # (B, N, >N, 1)
        weight_matrix = weight_matrix.softmax(dim=2)
        # Times z_pres (B, N, 1)-> (B, 1, N, 1)
        weight_matrix = weight_matrix * z_pres[:, None] # shape is kept same.
        # Self mask, set diagonal elements to zero. (B, >N, >N, 1)
        # weights.diagonal: (B, 1, N)
        diag = weight_matrix.diagonal(dim1=1, dim2=2) # get the diagonal matrix
        diag *= 0.0 # remove the self interaction
        # Renormalize (B, N, >N, 1)
        weight_matrix = weight_matrix / (weight_matrix.sum(dim=2, keepdim=True) + 1e-4)
        
        # (B, N1, N2, D) -> (B, N1, D)
        enc_relational = torch.sum(weight_matrix * relational_matrix, dim=2)
        
        # (B, N, D)
        prop_cond = enc_self + enc_relational #! Eq.(3) of paper.
        
        # (B, N, D)
        return prop_cond
    
    def get_offset_grid(self, device):
        """
        Get a grid that represents the center of z_shift of each cell
        Args:
            device: device
        #! called by Gaussian kernel construction and shifting o^{where}
        Returns:
            (2, G, G), where 2 is (x, y)
        """
        
        # (G, G), (G, G)
        offset_y, offset_x = torch.meshgrid(
            [torch.arange(ARCH.G), torch.arange(ARCH.G)])
        # (2, G, G)
        offset = torch.stack((offset_x, offset_y), dim=0).float().to(device)
        #! Scale: (0, G-1) -> (0.5, G-0.5) -> (0, 2) -> (-1, 1)
        offset = (2.0 / ARCH.G) * (offset + 0.5) - 1.0
        return offset
    
    def z_where_relative_to_absolute(self, z_where):
        """
        Convert z_where relative value to absolute value. The result is usable
        in spatial transform
        Args:
            z_where: (B, G, G, 4)

        Returns:
            z_where: (B, G, G, 4)
        """
        B, GG, D = z_where.size()
        # (B, G*G, 2) * 2
        z_scale, z_shift = z_where.chunk(2, dim=2)
        # (2, G, G), in range (-1, 1)
        offset = self.get_offset_grid(z_where.device)
        # scale: (-1, 1) -> (-2 / G, 2 / G). As opposed to the full range (-1, 1),
        # The maximum shift if (2 / G) / 2 = 1 / G, which is one cell
        # (2, G, G) -> (G, G, 2) -> (G*G, 2)
        offset = offset.permute(1, 2, 0).view(GG, 2)
        z_shift = (2.0 / ARCH.G) * torch.tanh(z_shift) + offset #! Eq.(38) of supl.
        z_scale = torch.sigmoid(z_scale) #! Eq.(39) of supl.
        
        # (B, G*G, 4)
        z_where = torch.cat([z_scale, z_shift], dim=2)
        
        return z_where
    
    def temporal_encode(self, state, z, bg, prior_or_post):
        """
        Encode history into rnn states
        Args:
            state: t-1
                h: (B, N, D)
                c: (B, N, D)
            z: t-1
                z_pres: (B, N, D)
                z_depth: (B, N, D)
                z_where: (B, N, D)
                z_what: (B, N, D)
            prop_cond: (B, N, D), t-1
            bg: [B, 3, dim, dim], t
            prior_or_post: either 'prior' or 'post

        Returns:
            state:
                h: (B, N, D)
                c: (B, N, D)
        """
        assert prior_or_post in ['prior', 'post']
        
        B, N, _ = state[0].size()
        prop_cond = self.compute_prop_cond(z, state, prior_or_post) # Sec. 3.3.1 - Interaction & Occlusion 
        bg_enc = self.bg_attention(bg, z) # Sec. 3.3.2 - Situation Awareness
        # bg_enc - (B, N, D)
        # Also encode interaction here
        z = torch.cat(z + (prop_cond, bg_enc), dim=-1)
        
        # BatchApply is a cool thing and it works here.
        if prior_or_post == 'post':
            inpt = self.temporal_rnn_post_input(z)
            state = self.temporal_rnn_post(inpt, state)
        else:
            inpt = self.temporal_rnn_prior_input(z)
            state = self.temporal_rnn_prior(inpt, state)
        
        return state
    
    def render(self, z):
        """
        Render z into an image
        Args:
            z:
                z_pres: (B, N, D)
                z_depth: (B, N, D)
                z_where: (B, N, D)
                z_what: (B, N, D)
                z_dyna: (B, N, D)
        Returns:
            fg: (B, 3, H, W)
            alpha_map: (B, 1, H, W)
        """
        z_pres, z_depth, z_where, z_what, z_dyna = z
        #! Pg.3 of supl.
        B, N, _ = z_pres.size()
        # Reshape to make things easier
        z_pres = z_pres.view(B * N, -1)
        z_where = z_where.view(B * N, -1)
        z_what = z_what.view(B * N, -1)
        
        # Decoder z_what
        # (B*N, 3, H, W), (B*N, 1, H, W)
        o_att, alpha_att = self.glimpse_decoder(z_what) #! Eq.(42) of supl.
        
        # (B*N, 1, H, W)
        alpha_att_hat = alpha_att * z_pres[..., None, None] #! Eq.(43) of supl.
        # (B*N, 3, H, W)
        y_att = alpha_att_hat * o_att #! Eq.(44) of supl.
        # y_att_(hat) and alpha_att_hat are of small glimpse size (H_g * W_g) 
        # To full resolution, apply inverse spatial transform (ST) (B*N, 1, H, W)
        y_att = spatial_transform(y_att, z_where, (B * N, 3, *ARCH.IMG_SHAPE), inverse=True) #! Eq.(45) of supl.
        
        # To full resolution, (B*N, 1, H, W)
        alpha_att_hat = spatial_transform(alpha_att_hat, z_where, (B * N, 1, *ARCH.IMG_SHAPE), inverse=True) #! Eq.(46) of supl.
        # Reshape back to original shape
        y_att = y_att.view(B, N, 3, *ARCH.IMG_SHAPE)
        alpha_att_hat = alpha_att_hat.view(B, N, 1, *ARCH.IMG_SHAPE)
        #! why logit is "-z_depth" ...?: depth increase -> less weight, thus -z_depth
        # (B, N, 1, H, W). H, W are glimpse size. #? why logit is "-z_depth" ...?
        importance_map = alpha_att_hat * torch.sigmoid(-z_depth[..., None, None])
        importance_map = importance_map / (torch.sum(importance_map, dim=1, keepdim=True) + 1e-5) #! Eq.(47) of supl.
        
        # Final fg (B, N, 3, H, W)
        fg = (y_att * importance_map).sum(dim=1) #! Eq.(48) of supl.+
        # Fg mask (B, N, 1, H, W)
        alpha_map = (importance_map * alpha_att_hat).sum(dim=1) #! Eq.(49) of supl.
        
        return fg, alpha_map
    
    def select(self, z_pres, *args):
        """
        
        Args:
            z_pres: (B, N, 1)
            *kargs: each (B, N, *) -> state_post, state_prior, z, ids, proposal

        Returns:ㅡ
            each (B, N, *)
        """
        # Take index
        # 1. sort by z_pres
        # 2. truncate
        # (B, N)
        indices = torch.argsort(z_pres, dim=1, descending=True)[..., 0] # sort along G*G dim
        # Truncate
        indices = indices[:, :ARCH.MAX]
        
        # Now use this thing to index every other thing
        def gather(x, indices):
            if len(x.size()) > len(indices.size()):
                indices = indices[..., None].expand(*indices.size()[:2], x.size(-1))
            return torch.gather(x, dim=1, index=indices)
        
        args = transform_tensors(args, func=lambda x: gather(x, indices))
        # return the sorted latents w.r.t. z_{pres}
        return args
    
    def combine(self,
                state_post_disc, state_prior_disc, z_disc, ids_disc, proposal_disc,
                state_post_prop, state_prior_prop, z_prop, ids_prop, proposal_prop):
        """
        Args:
            state_post_disc:
                h, c: (B, N, D)
            state_prior_disc:
                h, c: (B, N, D)
            z_disc: all (B, N, D)
                z_pres
                z_depth
                z_where
                z_what
            ids_prop: (B, N)

        Returns:
            state_post:
                h, c: (B, N, D)
            state_prior:
                h, c: (B, N, D)
            z:
                z_pres
                z_depth
                z_where
                z_what
        """
        
        def _combine(x, y):
            if isinstance(x, torch.Tensor):
                return torch.cat([x, y], dim=1)
            else: # combine list of torch.Tensors
                return [_combine(*pq) for pq in zip(x, y)]
        
        state_post, state_prior, z, ids, proposal = _combine(
            [state_post_disc, state_prior_disc, z_disc, ids_disc, proposal_disc],
            [state_post_prop, state_prior_prop, z_prop, ids_prop, proposal_prop],
        )
        z_pres = z[0] # z is concatenated Tensor of z_disc and z_prop
        state_post, state_prior, z, ids, proposal = self.select(z_pres, state_post, state_prior, z, ids, proposal)
        
        return state_post, state_prior, z, ids, proposal
    
    def bg_attention(self, bg, z):
        """
        AOE (Attention On Environment)
        
        Args:
            bg: (B, C, H, W)
            z:

        Returns:
            (B, N, D)

        """
        
        # (B, N, D)
        z_pres, z_depth, z_where, z_what, z_dyna = z
        B, N, _ = z_pres.size()
        
        if not ARCH.BG_CONDITIONED:
            return torch.zeros((B, N, ARCH.PROPOSAL_ENC_DIM), device=z_pres.device)
            
        if ARCH.BG_ATTENTION:
            # (G, G), (G, G)
            proposal = z_where.clone() # [B, T, 4 ]
            proposal[..., :2] += ARCH.BG_PROPOSAL_SIZE # add 0.25 for each [B, T, 2]
            
            # Get proposal glimpses
            # (B*N, 3, H, W)
            x_repeat = torch.repeat_interleave(bg, N, dim=0)
            
            # (B*N, 3, H, W) #! Eq.(4) & (5)
            proposal_glimpses = spatial_transform(x_repeat, proposal.view(B * N, 4),
                                                  out_dims=(B * N, 3, *ARCH.GLIMPSE_SHAPE))
            # (B, N, 3, H, W)
            proposal_glimpses = proposal_glimpses.view(B, N, 3, *ARCH.GLIMPSE_SHAPE)
            # (B, N, D)
            proposal_enc = self.bg_attention_encoder(proposal_glimpses)
        else:
            # (B, 3, H, W) -> (B, 1, 3, H, W)
            bg_tmp = bg[:, None]
            # (B, 1, D)
            proposal_enc = self.bg_attention_encoder(bg_tmp)
            proposal_enc = proposal_enc.expand(B, N, ARCH.PROPOSAL_ENC_DIM)
        
        return proposal_enc
    
    def rejection(self, z_disc, z_prop, threshold):
        """ # please refer to SCALOR paper
        If the bbox of an object overlaps too much with a propagated object, we remove it (z_disc)
        Args:
            z_disc: discovery; [B, N1 = G*G, D]
            z_prop: propagation; [B,N2, D]; where N2 is the # of propagated objects
            threshold: iou threshold

        Returns:
            z_disc
        """
        z_pres_disc, z_depth_disc, z_where_disc, z_what_disc, z_dyna_disc = z_disc
        z_pres_prop, z_depth_prop, z_where_prop, z_what_prop, z_dyna_prop = z_prop
        # (B, N1, N2, 1) #? what is N1 and is N2 each?
        iou = self.iou(z_where_disc, z_where_prop) # In: [B, G*G, 4]
        assert torch.all((iou >= 0) & (iou <= 1))
        iou_too_high = iou > threshold # torch.bool Tensors
        # Only for those that exist (B, N1, N2, 1) (B, 1, N2, 1)
        iou_too_high = iou_too_high & (z_pres_prop[:, None, :] > 0.5) # update z_pres
        # (B, N1, 1)
        iou_too_high = torch.any(iou_too_high, dim=-2) # check if there's high IOU along N1 axis
        z_pres_disc_new = z_pres_disc.clone() # rejection mask
        z_pres_disc_new[iou_too_high] = 0.0 # if IOU > 0.8 and p(z_pres) > 0.5, reject z_disc
        
        return (z_pres_disc_new, z_depth_disc, z_where_disc, z_what_disc, z_dyna_disc)
        
    def iou(self, z_where_disc, z_where_prop):
        """
        #! refer to SCALOR paper
        Args:
            z_where_disc: (B, N1, 4) -> N1 : G*G
            z_where_prop: (B, N2, 4)
        #! NOTE that z_{where} is split into z^{hw} and z^{xy}
        Returns:
            (B, N1, N2)
        """
        B, N1, _ = z_where_disc.size() #? N1: What does this mean?
        B, N2, _ = z_where_prop.size() #? N2: 
        def _get_edges(z_where): # z_where_disc: [B, N1 (G*G), 1, 4] , z_where_prop: [B, 1, N2 (#obj), 4]
            z_where = z_where.detach().clone() # we dont' change the original variable
            z_where[..., 2:] = (z_where[..., 2:] + 1) / 2 # relocate X & Y [-1, 1] -> [0, 1]
            # (B, N, 1), (B, N, 1), (B, N, 1), (B, N, 1)
            sx, sy, cx, cy = torch.split(z_where, [1, 1, 1, 1], dim=-1)# h, w, x, y
            left = cx - sx / 2 # x - 2/w
            right = cx + sx / 2 # x + 2/w
            top = cy - sy / 2 # x - 2/h 
            bottom = cy + sy / 2 # x + 2/h
            return left, right, top, bottom # z_w_disc - 4 * [B, N1, 1, 1] : z_w_prop - 4 * [B, N1, 1, 1]
            
        def _area(left, right, top, bottom):
            valid = (bottom >= top) & (right >= left)
            area = (bottom - top) * (right - left)
            area *= valid
            return area
        def _iou(left1, right1, top1, bottom1, left2, right2, top2, bottom2):
            #! make 1-1 comparison; 1st_input: where_disc [B, N1, 1, 1] / 2nd_input: where_prop [B, 1, N2, 1] || out: 
            left = torch.max(left1, left2)
            right = torch.min(right1, right2)
            top = torch.max(top1, top2)
            bottom = torch.min(bottom1, bottom2)
            area1 = _area(left1, right1, top1, bottom1)
            area2 = _area(left2, right2, top2, bottom2)
            area_intersect = _area(left, right, top, bottom)
            iou = area_intersect / (area1 + area2 - area_intersect + 1e-5) #! Intersection / Union
            # If any of these areas are zero, we ignore this iou
            iou = iou * (area_intersect != 0) * (area1 != 0) * (area2 != 0) # If any of area is False, IoU == False
            return iou
        # In: (B, N1 (G*G), 1, D) - Out: (B, N1, 1, 1)
        left1, right1, top1, bottom1 = _get_edges(z_where_disc[:, :, None]) #? why expand rank 3?
        # (B, 1, N2, 1)
        left2, right2, top2, bottom2 = _get_edges(z_where_prop[:, None]) #? why expand rank 2?
        iou = _iou(left1, right1, top1, bottom1, left2, right2, top2, bottom2)
        assert iou.size() == (B, N1, N2, 1)
        return iou # (B, N1, N2, 1)


  
class ImgEncoder(nn.Module):
    """
    Used in discovery. Input is image plus image - bg
    """
    
    def __init__(self):
        super(ImgEncoder, self).__init__()
        
        assert ARCH.G in [8, 4]
        assert ARCH.IMG_SIZE in [64, 128]
        last_stride = ARCH.IMG_SIZE // (8 * ARCH.G)
        # last_stride = 1 if ARCH.G == 8 else 2


        self.last = nn.Conv2d(256, ARCH.IMG_ENC_DIM, 3, last_stride, 1)
        # 3 + 3 = 6
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        resnet = resnet18()
        self.enc = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            # resnet.layer4,
        )
    
    def forward(self, x):
        """
        Get image feature
        Args:
            x: (B, 3, H, W) -> shouldn't this be [B, 6, H , W]?

        Returns:
            enc: (B, 128, G, G)

        """
        B = x.size(0)
        x = self.enc(x)
        x = self.last(x)
        return x


class LatentPostDisc(nn.Module):
    """q(z_dyna|x) in discovery"""
    
    def __init__(self):
        nn.Module.__init__(self)
        self.enc = nn.Sequential(
            nn.Conv2d(ARCH.IMG_ENC_DIM + ARCH.PROP_MAP_DIM, 128, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, ARCH.Z_DYNA_DIM * 2, 1),
        )
        
        self.act_enc = nn.Sequential(
            nn.Conv2d(ARCH.IMG_ENC_DIM + ARCH.PROP_MAP_DIM + ARCH.ACTION_DIM, 128, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, ARCH.Z_DYNA_DIM * 2, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, D, G, G)

        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B = x.size(0)
        # (B, D, G, G)
        params = self.enc(x)
        # (B, G, G, D) -> (B, G*G, D)
        params = params.permute(0, 2, 3, 1).view(B, ARCH.G ** 2, -1)
        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1)
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4
        
        return z_dyna_loc, z_dyna_scale

    def forward_act(self, x, ee):
        """
        Args:
            x: (B, D, G, G) #! NOTE that discovery is based on glimpse
        # TODO (cheolhui) : how can I model condition action here?
        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B, N, *_ = x.size()

        # (B, D, G, G)
        params = self.enc(x)
        # (B, G*G, D)

        _params = [] 
        for n in range(N):
            if n == 0:
                param = self.act_enc(torch.cat([params[:, n, ...], ee], dim=-1)) # In: [B, 2D], Out: [B, D]
            else:
                param = self.enc(params[:, n, ...]) # [B, 2D], Out: [B, D]
            _params.append(param)
        params = torch.stack(_params, dim=1)# [B, N, D]

        params = params.permute(0, 2, 3, 1).view(B, ARCH.G ** 2, -1)
        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1)
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4
        
        return z_dyna_loc, z_dyna_scale


class LatentPostProp(nn.Module):
    """q(z_dyna|x, z[:t]) in propagation"""
    
    def __init__(self):
        nn.Module.__init__(self)
        self.enc = MLP(
            [ARCH.PROPOSAL_ENC_DIM + ARCH.RNN_HIDDEN_DIM, 128, 128,
             ARCH.Z_DYNA_DIM * 2
             ], act=nn.CELU())
        self.act_enc = MLP(
                [ARCH.PROPOSAL_ENC_DIM + ARCH.RNN_HIDDEN_DIM + ARCH.ACTION_DIM, 128, 128,
                ARCH.Z_DYNA_DIM * 2
                ], act=nn.CELU())


    def forward(self, x):
        """
        Args:
            x: (B, N, 2D)

        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B, N, *_ = x.size()
        params = self.enc(x) # [B, 2D], Out: [B, D]
        # (B, G*G, D)
        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1) # [B, N, D]
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4
        
        return z_dyna_loc, z_dyna_scale

    def forward_action(self, x, ee):
        """
        Args:
            x: (B, N, 2D)
            ee: (B, P)
        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B, N, *_ = x.size()
        params = []
        for n in range(N):
            if n == 0:
                param = self.act_enc(torch.cat([x[:, n, ...], ee], dim=-1)) # In: [B, 2D], Out: [B, D]
            else:
                param = self.enc(x[:, n, ...]) # [B, 2D], Out: [B, D]
            params.append(param)
        params = torch.stack(params, dim=1)# [B, N, D]

        # (B, G*G, D)
        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1) # [B, N, D]
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4
        
        return z_dyna_loc, z_dyna_scale


class LatentPriorProp(nn.Module):
    """p(z_dyna|z[:t]) in propagation"""
    
    def __init__(self):
        nn.Module.__init__(self)
        self.enc = MLP(
            [ARCH.RNN_HIDDEN_DIM, 128, 128,
             ARCH.Z_DYNA_DIM * 2
             ], act=nn.CELU())

        self.act_enc = MLP(
            [ARCH.RNN_HIDDEN_DIM + ARCH.ACTION_DIM, 128, 128,
             ARCH.Z_DYNA_DIM * 2
             ], act=nn.CELU())


    
    def forward(self, x):
        """
        Args:
            x: (B, D, G, G)

        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B = x.size(0)
        params = self.enc(x)
        # (B, G*G, D)
        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1)
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4 #!Eq.(12) of supl
        
        return z_dyna_loc, z_dyna_scale

    def forward_action(self, x, ee):
        """
        Args:
            x: (B, N, D)
            ee: (B, P)
        Returns:
            z_dyna_loc, z_dyna_scale: (B, 1, G, G)
        """
        B, N, *_ = x.size()
        params = []
        for n in range(N):
            if n == 0:
                param = self.act_enc(torch.cat([x[:, n, ...], ee], dim=-1)) # In: [B, 2D], Out: [B, D]
            else:
                param = self.enc(x[:, n, ...]) # [B, 2D], Out: [B, D]
            params.append(param)        # (B, G*G, D)
        params = torch.stack(params, dim=1) # [B, N, D]

        (z_dyna_loc, z_dyna_scale) = torch.chunk(params, chunks=2, dim=-1)
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4 #!Eq.(12) of supl
        
        return z_dyna_loc, z_dyna_scale


class PresDepthWhereWhatPrior(nn.Module):
    """p(z_att|z_dyna), where z_att = [z_pres, z_depth, z_where, z_what]"""
    
    def __init__(self):
        nn.Module.__init__(self)
        self.enc = MLP( #! encodes z_dyna into z_pres, z_depth, z_what, z_where
            [ARCH.Z_DYNA_DIM, 128, 128,
             ARCH.Z_PRES_DIM + (ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM) * 2 + (
                     ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM
             )
             ], act=nn.CELU())
    
    def forward(self, enc):
        """
        Args:
            enc: (B, N, D)

        Returns:
            z_pres_prob: (B, N, 1)
            z_depth_loc, z_depth_scale: (B, N, 1)
            z_where_loc, z_where_scale: (B, N, 4)
            z_what_loc, z_what_scale: (B, N, D)
            z_depth_gate: (B, N, 1)
            z_where_gate: (B, N, 4)
            z_what_gate: (B, N, D)
        """
        # (B, N, D)
        params = self.enc(enc) # 
        # split into [Z_PRES(1), Z_DEPTH(1), Z_DEPTH, Z_WHERE(4), Z_WHERE, Z_WHAT(64), Z_WHAT, Z_DEPTH, Z_WHERE, Z_WHAT] = 208
        (z_pres_prob, z_depth_loc, z_depth_scale, z_where_loc, z_where_scale,
         z_what_loc, z_what_scale,
         z_depth_gate, z_where_gate, z_what_gate) = torch.split(params,
                                                                [ARCH.Z_PRES_DIM] + [ARCH.Z_DEPTH_DIM] * 2 + [
                                                                    ARCH.Z_WHERE_DIM] * 2 + [
                                                                    ARCH.Z_WHAT_DIM] * 2 + [ARCH.Z_DEPTH_DIM,
                                                                                            ARCH.Z_WHERE_DIM,
                                                                                            ARCH.Z_WHAT_DIM]
                                                                , dim=-1)
        z_pres_prob = torch.sigmoid(z_pres_prob) # [B, N, 1]
        z_depth_scale = F.softplus(z_depth_scale) + 1e-4 # [B, N, 1]
        z_where_scale = F.softplus(z_where_scale) + 1e-4 # [B, N, 4]
        z_what_scale = F.softplus(z_what_scale) + 1e-4 # [B, N, 64]
        
        z_depth_gate = torch.sigmoid(z_depth_gate) # [B, N, 1]
        z_where_gate = torch.sigmoid(z_where_gate) # [B, N, 4]
        z_what_gate = torch.sigmoid(z_what_gate) # [B, N, 64]
        
        return z_pres_prob, z_depth_loc, z_depth_scale, z_where_loc, z_where_scale, z_what_loc, z_what_scale, z_depth_gate, z_where_gate, z_what_gate


class PredProposal(nn.Module):
    """Given states encoding previous history, compute a proposal location."""
    
    def __init__(self):
        nn.Module.__init__(self)
        self.net = MLP(
            sizes=[ARCH.RNN_HIDDEN_DIM, 128, 128, 2],
            act=nn.CELU(),
        )
        self.net = BatchApply(self.net)
    
    def forward(self, enc):
        """
        Args:
            enc: (B, N, D)

        Returns:
            proposal: (B, N, 2). This is offset from previous where.
        """
        return self.net(enc)


class ProposalEncoder(nn.Module):
    """Same as glimpse encoder, but for encoding the proposal area"""
    
    def __init__(self):
        nn.Module.__init__(self)
        embed_size = ARCH.GLIMPSE_SIZE // 16
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(1, 16),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(2, 32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
        )
        # TODO (cheolhui): robot action should be conditioned independently
        self.enc_what = nn.Linear(128 * embed_size ** 2, ARCH.PROPOSAL_ENC_DIM)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3, H, W)

        Returns:
            enc: (B, N, D)
        """
        B, N, C, H, W = x.size()
        x = x.view(B * N, 3, H, W)
        x = self.enc(x) # [B*N, D, 1, 1]
        x = x.flatten(start_dim=1) # [B*N, D]
        return self.enc_what(x).view(B, N, ARCH.PROPOSAL_ENC_DIM)

    def forward_action(self, x, ee):
        """
        Infer action-conditioned proposal
        Args:
            x: (B, N, 3, H, W)
            ee: (B, P), P: end-effector dim
        Returns:
            enc: (B, N, D)
        """
        B, N, C, H, W = x.size()
        x = x.view(B * N, 3, H, W)
        x = self.enc(x) # [B*N, D, 1, 1] # TODO: seperate first entity to be conditioned on action
        x = x.flatten(start_dim=1) # [B*N, D]
        return self.enc_what(x).view(B, N, ARCH.PROPOSAL_ENC_DIM)


class GlimpseDecoder(nn.Module):
    """Decode z_what into glimpse."""
    
    def __init__(self):
        nn.Module.__init__(self)
        # Everything here is symmetric to encoder, but with subpixel upsampling
        
        self.embed_size = ARCH.GLIMPSE_SIZE // 16 # 1
        self.fc = nn.Linear(ARCH.Z_WHAT_DIM, self.embed_size ** 2 * 128)
        self.net = nn.Sequential(
            nn.Conv2d(128, 64 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(4, 64),
            
            nn.Conv2d(64, 32 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(2, 32),
            
            nn.Conv2d(32, 16 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(1, 16),
            
            # Mask and appearance
            nn.Conv2d(16, 4 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
        )
    
    def forward(self, z_what):
        """
        
        Args:
            z_what: (B, D), where B = batch * (#objects)

        Returns:
            glimpse: (B, 3, H, W)
        """
        B, D = z_what.size()
        x = F.celu(self.fc(z_what))
        #  -> (B, 128, E, E)
        x = x.view(B, 128, self.embed_size, self.embed_size)
        x = self.net(x)
        x = torch.sigmoid(x)
        # (B, 3, H, W), (B, 1, H, W), where H = W = GLIMPSE_SIZE
        o_att, alpha_att = x.split([3, 1], dim=1)
        
        return o_att, alpha_att


class PresDepthWhereWhatPostLatentDisc(nn.Module):
    """Predict attributes posterior given image encoding and propagation map in discovery"""
    
    def __init__(self):
        nn.Module.__init__(self)
        self.enc = nn.Sequential(
            nn.Conv2d(ARCH.IMG_ENC_DIM + ARCH.PROP_MAP_DIM, 128, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128,
                      ARCH.Z_PRES_DIM + (ARCH.Z_DEPTH_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_WHAT_DIM + ARCH.Z_DYNA_DIM) * 2,
                      1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, D, G, G)

        Returns:
            z_pres_prob: (B, G*G, 1)
            z_depth_loc, z_depth_scale: (B, G*G, 1)
            z_where_loc, z_where_scale: (B, G*G, D)
            z_what_loc, z_what_scale: (B, G*G, D)
            z_dyna_loc, z_dyna_scale: (B, G*G, D)
        """
        B = x.size(0)
        # (B, D, G, G)
        params = self.enc(x)
        # (B, G*G, D)
        params = params.permute(0, 2, 3, 1).view(B, ARCH.G ** 2, -1)
        (z_pres_prob, z_depth_loc, z_depth_scale, z_where_loc, z_where_scale,
         z_what_loc, z_what_scale, z_dyna_loc, z_dyna_scale) = torch.split(params,
                                                                               [ARCH.Z_PRES_DIM] + [
                                                                                   ARCH.Z_DEPTH_DIM] * 2 + [
                                                                                   ARCH.Z_WHERE_DIM] * 2 + [
                                                                                   ARCH.Z_WHAT_DIM] * 2 + [
                                                                                   ARCH.Z_DYNA_DIM] * 2, dim=-1)
        z_pres_prob = torch.sigmoid(z_pres_prob)
        z_where_scale = F.softplus(z_where_scale) + 1e-4
        z_depth_scale = F.softplus(z_depth_scale) + 1e-4
        z_what_scale = F.softplus(z_what_scale) + 1e-4
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4
        
        return z_pres_prob, z_depth_loc, z_depth_scale, z_where_loc, z_where_scale, z_what_loc, z_what_scale, z_dyna_loc, z_dyna_scale


class BgAttentionEncoder(nn.Module):
    """Encoding (attended) background region"""
    
    def __init__(self):
        nn.Module.__init__(self)
        if ARCH.BG_ATTENTION:
            embed_size = ARCH.GLIMPSE_SIZE // 16
        else:
            embed_size = ARCH.IMG_SIZE // 16
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(1, 16),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(2, 32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
        )
        
        self.enc_what = nn.Linear(128 * embed_size ** 2, ARCH.BG_PROPOSAL_DIM)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3, H, W)

        Returns:
            enc: (B, N, D)
        """
        B, N, C, H, W = x.size()
        x = x.view(B * N, 3, H, W)
        x = self.enc(x)
        x = x.flatten(start_dim=1)
        return self.enc_what(x).view(B, N, ARCH.BG_PROPOSAL_DIM)
    
