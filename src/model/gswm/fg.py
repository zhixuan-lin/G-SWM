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
        self.latent_post_prop = LatentPostProp()
        self.latent_prior_prop = LatentPriorProp()
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
        # Empty, N=0
        state_post, state_prior, z, ids = self.get_dummy_things(B, seq.device)
        start_id = torch.zeros(B, device=seq.device).long()
        
        things = defaultdict(list)
        first = True
        for t in range(T):
            # (B, 3, H, W)
            x = seq[:, t]
            # Update object states
            state_post_prop, state_prior_prop, z_prop, kl_prop, proposal = self.propagate(x, state_post, state_prior, z,
                                                                                          bg[:, t])
            ids_prop = ids
            if first or torch.rand(1) > discovery_dropout:
                state_post_disc, state_prior_disc, z_disc, ids_disc, kl_disc = self.discover(x, z_prop, bg[:, t], start_id)
                first = False
            else:
                state_post_disc, state_prior_disc, z_disc, ids_disc = self.get_dummy_things(B, seq.device)
                kl_disc = (0.0, 0.0, 0.0, 0.0, 0.0)
            
            # TODO: for proposal of discovery, we are just using z_where
            # Combine discovered and propagated things
            state_post, state_prior, z, ids, proposal = self.combine(
                state_post_disc, state_prior_disc, z_disc, ids_disc, z_disc[2],
                state_post_prop, state_prior_prop, z_prop, ids_prop, proposal
            )
            kl = [x + y for (x, y) in zip(kl_prop, kl_disc)]
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
            
            if t < cond_steps:
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
    
    def discover(self, x, z_prop, bg, start_id=0):
        """
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
        B, *_ = x.size()
        
        # (B, D, G, G)
        x_enc = self.img_encoder(x)
        # For each discovery cell, we combine propagated objects weighted by distances
        # (B, D, G, G)
        prop_map = self.compute_prop_map(z_prop)
        # (B, D, G, G)
        enc = torch.cat([x_enc, prop_map], dim=1)
        
        (z_pres_post_prob, z_depth_post_loc, z_depth_post_scale, z_where_post_loc,
         z_where_post_scale, z_what_post_loc, z_what_post_scale, z_dyna_loc,
         z_dyna_scale) = self.pres_depth_where_what_latent_post_disc(enc)
        
        z_dyna_loc, z_dyna_scale = self.latent_post_disc(enc)
        z_dyna_post = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_post.rsample()
        
        # Compute posteriors. All (B, G*G, D)
        z_pres_post = RelaxedBernoulli(temperature=self.tau, probs=z_pres_post_prob)
        z_pres = z_pres_post.rsample()
        
        z_depth_post = Normal(z_depth_post_loc, z_depth_post_scale)
        z_depth = z_depth_post.rsample()
        
        z_where_post = Normal(z_where_post_loc, z_where_post_scale)
        z_where = z_where_post.rsample()
        z_where = self.z_where_relative_to_absolute(z_where)
        
        z_what_post = Normal(z_what_post_loc, z_what_post_scale)
        z_what = z_what_post.rsample()
        
        # Combine
        z = (z_pres, z_depth, z_where, z_what, z_dyna)

        # Rejection
        if ARCH.REJECTION:
            z = self.rejection(z, z_prop, ARCH.REJECTION_THRESHOLD)
        
        # Compute object ids
        # (B, G*G) + (B, 1)
        ids = torch.arange(ARCH.G ** 2, device=x_enc.device).expand(B, ARCH.G ** 2) + start_id[:, None]
        
        # Update temporal states
        state_post_prev = self.get_state_init(B, 'post')
        state_post = self.temporal_encode(state_post_prev, z, bg, prior_or_post='post')
        
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
        
        # Sum over non-batch dimensions
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
    
    def propagate(self, x, state_post_prev, state_prior_prev, z_prev, bg):
        """
        One step of propagation posterior
        Args:
            x: (B, 3, H, W), img
            (h, c), (h, c): each (B, N, D)
            z_prev:
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
                z_what: (B, N, D)
                z_dyna: (B, N, D)

        Returns:
            h_post, c_post: (B, N, D)
            h_prior, c_prior: (B, N, D)
            z:
                z_pres: (B, N, 1)
                z_depth: (B, N, 1)
                z_where: (B, N, 4)
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
        
        if N == 0:
            # No object is propagated
            return state_post_prev, state_prior_prev, z_prev, (0.0, 0.0, 0.0, 0.0, 0.0), z_prev[2]
        
        h_post, c_post = state_post_prev
        h_prior, c_prior = state_prior_prev
        
        # Predict proposal locations, (B, N, 2)
        proposal_offset = self.pred_proposal(h_post)
        proposal = torch.zeros_like(z_where_prev)
        # Update size only
        proposal[..., 2:] = z_where_prev[..., 2:]
        proposal[..., :2] = z_where_prev[..., :2] + ARCH.PROPOSAL_UPDATE_MIN + (
                ARCH.PROPOSAL_UPDATE_MAX - ARCH.PROPOSAL_UPDATE_MIN) * torch.sigmoid(proposal_offset)
        
        # Get proposal glimpses
        # (B*N, 3, H, W)
        x_repeat = torch.repeat_interleave(x[:, :3], N, dim=0)
        
        # (B*N, 3, H, W)
        proposal_glimpses = spatial_transform(x_repeat, proposal.view(B * N, 4),
                                              out_dims=(B * N, 3, *ARCH.GLIMPSE_SHAPE))
        # (B, N, 3, H, W)
        proposal_glimpses = proposal_glimpses.view(B, N, 3, *ARCH.GLIMPSE_SHAPE)
        # (B, N, D)
        proposal_enc = self.proposal_encoder(proposal_glimpses)
        # (B, N, D)
        # This will be used to condition everything
        enc = torch.cat([proposal_enc, h_post], dim=-1)
        
        z_dyna_loc, z_dyna_scale = self.latent_post_prop(enc)
        z_dyna_post = Normal(z_dyna_loc, z_dyna_scale)
        z_dyna = z_dyna_post.rsample()
        
        # (B, N, D)
        (z_pres_prob, z_depth_offset_loc, z_depth_offset_scale, z_where_offset_loc,
         z_where_offset_scale, z_what_offset_loc, z_what_offset_scale,
         z_depth_gate, z_where_gate, z_what_gate) = self.pres_depth_where_what_prior(z_dyna)
        
        # Sampling
        z_pres_post = RelaxedBernoulli(self.tau, probs=z_pres_prob)
        z_pres = z_pres_post.rsample()
        z_pres = z_pres_prev * z_pres
        
        z_where_post = Normal(z_where_offset_loc, z_where_offset_scale)
        z_where_offset = z_where_post.rsample()
        z_where = torch.zeros_like(z_where_prev)
        # Scale
        z_where[..., :2] = z_where_prev[..., :2] + ARCH.Z_SCALE_UPDATE_SCALE * z_where_gate[..., :2] * torch.tanh(
            z_where_offset[..., :2])
        # Shift
        z_where[..., 2:] = z_where_prev[..., 2:] + ARCH.Z_SHIFT_UPDATE_SCALE * z_where_gate[..., 2:] * torch.tanh(
            z_where_offset[..., 2:])
        
        z_depth_post = Normal(z_depth_offset_loc, z_depth_offset_scale)
        z_depth_offset = z_depth_post.rsample()
        z_depth = z_depth_prev + ARCH.Z_DEPTH_UPDATE_SCALE * z_depth_gate * z_depth_offset
        
        z_what_post = Normal(z_what_offset_loc, z_what_offset_scale)
        z_what_offset = z_what_post.rsample()
        z_what = z_what_prev + ARCH.Z_WHAT_UPDATE_SCALE * z_what_gate * torch.tanh(z_what_offset)
        z = (z_pres, z_depth, z_where, z_what, z_dyna)
        
        
        # Update states
        state_post = self.temporal_encode(state_post_prev, z, bg, prior_or_post='post')
        state_prior = self.temporal_encode(state_prior_prev, z, bg, prior_or_post='prior')
        
        z_dyna_loc, z_dyna_scale = self.latent_prior_prop(h_prior)
        z_dyna_prior = Normal(z_dyna_loc, z_dyna_scale)
        kl_dyna = kl_divergence(z_dyna_post, z_dyna_prior)
        
        # This is not kl divergence. This is an auxialiary loss
        kl_pres = kl_divergence_bern_bern(z_pres_prob, torch.full_like(z_pres_prob, self.z_pres_prior_prob))
        # If we don't want this auxiliary loss
        if not ARCH.AUX_PRES_KL:
            kl_pres = torch.zeros_like(kl_pres)
        
        # Reduced to (B,)
        
        # Again, this is not really kl
        kl_pres = kl_pres.flatten(start_dim=1).sum(-1)
        kl_dyna = kl_dyna.flatten(start_dim=1).sum(-1)
        
        # We are not using q, so these will be zero
        kl_where = torch.zeros_like(kl_pres)
        kl_what = torch.zeros_like(kl_pres)
        kl_depth = torch.zeros_like(kl_pres)
        assert kl_pres.size(0) == B
        kl = (kl_pres, kl_depth, kl_where, kl_what, kl_dyna)
        
        return state_post, state_prior, z, kl, proposal
    
    def compute_prop_map(self, z_prop):
        """
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
        
        z_pres_prop, z_depth_prop, z_where_prop, z_what_prop, z_dyna_prop = z_prop
        B, N, _ = z_pres_prop.size()
        
        if z_pres_prop.size(1) == 0:
            # First frame, empty prop map
            return torch.zeros(B, ARCH.PROP_MAP_DIM, ARCH.G, ARCH.G, device=z_pres_prop.device)
        
        assert N == ARCH.MAX
        # Use only z_what and z_depth here
        # (B, N, D)
        # TODO: I could have used relative z_where as SILOT here. But that will induce many computations. So I won't do it here.
        z_prev = torch.cat(z_prop, dim=-1)
        
        # (B, N, D) -> (B, N, D)
        z_prev_enc = self.prop_map_mlp(z_prev)
        # (B, N, D), masked out objects with z_pres == 0
        z_prev_enc = z_prev_enc * z_pres_prop
        
        # Compute a weight matrix of size (B, G*G, N)
        
        # (2, G, G) -> (G*G, 2)
        offset = self.get_offset_grid(z_prev.device)
        offset = offset.permute(1, 2, 0).view(-1, 2)
        
        # (B, N, 2)
        z_shift_prop = z_where_prop[..., 2:]
        
        # Distance matrix
        # (1, G*G, 1, 2)
        offset = offset[None, :, None, :]
        # (B, 1, N, 2)
        z_shift_prop = z_shift_prop[:, None, :, :]
        # (B, G*G, N, 2)
        matrix = offset - z_shift_prop
        # (B, G*G, N)
        weights = gaussian_kernel_2d(matrix, ARCH.PROP_MAP_SIGMA, dim=-1)
        # (B, G*G, N, 1)
        weights = weights[..., None]
        # (B, 1, N, D)
        z_prev_enc = z_prev_enc[:, None, ]
        
        # (B, G*G, D)
        prop_map = torch.sum(weights * z_prev_enc, dim=-2)
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
        # (B, N, D)
        feat = torch.cat(z + (h_post_prev,), dim=-1)
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
        feat_self = feat[:, :, None]
        feat_matrix_self = feat_self.expand(B, N, N, self.Z_DIM + ARCH.RNN_HIDDEN_DIM)
        # (B, 1, N, D) -> (B, N, N, D)
        feat_other = feat[:, None]
        feat_matrix_other = feat_other.expand(B, N, N, self.Z_DIM + ARCH.RNN_HIDDEN_DIM)
        
        # TODO: Detail here, replace absolute positions with relative ones
        offset = ARCH.Z_PRES_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_SCALE_DIM
        # Must clone. Otherwise there will be multiple write
        feat_matrix_other = feat_matrix_other.clone()
        feat_matrix_other[..., offset:offset + ARCH.Z_SHIFT_DIM] = dist_matrix
        # (B, N, N, 2D)
        feat_matrix = torch.cat([feat_matrix_self, feat_matrix_other], dim=-1)
        # (B, N, N, D)
        relational_matrix = prop_cond_relational(feat_matrix)
        
        # COMPUTE WEIGHTS
        # (B, N, N, 1)
        weight_matrix = prop_cond_weights(feat_matrix)
        # (B, N, >N, 1)
        weight_matrix = weight_matrix.softmax(dim=2)
        # Times z_pres (B, N, 1)-> (B, 1, N, 1)
        weight_matrix = weight_matrix * z_pres[:, None]
        # Self mask, set diagonal elements to zero. (B, >N, >N, 1)
        # weights.diagonal: (B, 1, N)
        diag = weight_matrix.diagonal(dim1=1, dim2=2)
        diag *= 0.0
        # Renormalize (B, N, >N, 1)
        weight_matrix = weight_matrix / (weight_matrix.sum(dim=2, keepdim=True) + 1e-4)
        
        # (B, N1, N2, D) -> (B, N1, D)
        enc_relational = torch.sum(weight_matrix * relational_matrix, dim=2)
        
        # (B, N, D)
        prop_cond = enc_self + enc_relational
        
        # (B, N, D)
        return prop_cond
    
    def get_offset_grid(self, device):
        """
        Get a grid that represents the center of z_shift of each cell
        Args:
            device: device

        Returns:
            (2, G, G), where 2 is (x, y)
        """
        
        # (G, G), (G, G)
        offset_y, offset_x = torch.meshgrid(
            [torch.arange(ARCH.G), torch.arange(ARCH.G)])
        # (2, G, G)
        offset = torch.stack((offset_x, offset_y), dim=0).float().to(device)
        # Scale: (0, G-1) -> (0.5, G-0.5) -> (0, 2) -> (-1, 1)
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
        # (G*G, 2)
        offset = offset.permute(1, 2, 0).view(GG, 2)
        z_shift = (2.0 / ARCH.G) * torch.tanh(z_shift) + offset
        z_scale = torch.sigmoid(z_scale)
        
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
            bg: (B, D), t
            prior_or_post: either 'prior' or 'post

        Returns:
            state:
                h: (B, N, D)
                c: (B, N, D)
        """
        assert prior_or_post in ['prior', 'post']
        
        B, N, _ = state[0].size()
        prop_cond = self.compute_prop_cond(z, state, prior_or_post)
        bg_enc = self.bg_attention(bg, z)
        # (B, N, D)
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
        
        B, N, _ = z_pres.size()
        # Reshape to make things easier
        z_pres = z_pres.view(B * N, -1)
        z_where = z_where.view(B * N, -1)
        z_what = z_what.view(B * N, -1)
        
        # Decoder z_what
        # (B*N, 3, H, W), (B*N, 1, H, W)
        o_att, alpha_att = self.glimpse_decoder(z_what)
        
        # (B*N, 1, H, W)
        alpha_att_hat = alpha_att * z_pres[..., None, None]
        # (B*N, 3, H, W)
        y_att = alpha_att_hat * o_att
        
        # To full resolution, (B*N, 1, H, W)
        y_att = spatial_transform(y_att, z_where, (B * N, 3, *ARCH.IMG_SHAPE), inverse=True)
        
        # To full resolution, (B*N, 1, H, W)
        alpha_att_hat = spatial_transform(alpha_att_hat, z_where, (B * N, 1, *ARCH.IMG_SHAPE), inverse=True)
        
        y_att = y_att.view(B, N, 3, *ARCH.IMG_SHAPE)
        alpha_att_hat = alpha_att_hat.view(B, N, 1, *ARCH.IMG_SHAPE)
        
        # (B, N, 1, H, W). H, W are glimpse size.
        importance_map = alpha_att_hat * torch.sigmoid(-z_depth[..., None, None])
        importance_map = importance_map / (torch.sum(importance_map, dim=1, keepdim=True) + 1e-5)
        
        # Final fg (B, N, 3, H, W)
        fg = (y_att * importance_map).sum(dim=1)
        # Fg mask (B, N, 1, H, W)
        alpha_map = (importance_map * alpha_att_hat).sum(dim=1)
        
        return fg, alpha_map
    
    def select(self, z_pres, *args):
        """
        
        Args:
            z_pres: (B, N, 1)
            *kargs: each (B, N, *)

        Returns:
            each (B, N, *)
        """
        # Take index
        # 1. sort by z_pres
        # 2. truncate
        # (B, N)
        indices = torch.argsort(z_pres, dim=1, descending=True)[..., 0]
        # Truncate
        indices = indices[:, :ARCH.MAX]
        
        # Now use this thing to index every other thing
        def gather(x, indices):
            if len(x.size()) > len(indices.size()):
                indices = indices[..., None].expand(*indices.size()[:2], x.size(-1))
            return torch.gather(x, dim=1, index=indices)
        
        args = transform_tensors(args, func=lambda x: gather(x, indices))
        
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
            else:
                return [_combine(*pq) for pq in zip(x, y)]
        
        state_post, state_prior, z, ids, proposal = _combine(
            [state_post_disc, state_prior_disc, z_disc, ids_disc, proposal_disc],
            [state_post_prop, state_prior_prop, z_prop, ids_prop, proposal_prop],
        )
        z_pres = z[0]
        state_post, state_prior, z, ids, proposal = self.select(z_pres, state_post, state_prior, z, ids, proposal)
        
        return state_post, state_prior, z, ids, proposal
    
    def bg_attention(self, bg, z):
        """
        AOE
        
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
            proposal = z_where.clone()
            proposal[..., :2] += ARCH.BG_PROPOSAL_SIZE
            
            # Get proposal glimpses
            # (B*N, 3, H, W)
            x_repeat = torch.repeat_interleave(bg, N, dim=0)
            
            # (B*N, 3, H, W)
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
        """
        If the bbox of an object overlaps too much with a propagated object, we remove it
        Args:
            z_disc: discovery
            z_prop: propagation
            threshold: iou threshold

        Returns:
            z_disc
        """
        z_pres_disc, z_depth_disc, z_where_disc, z_what_disc, z_dyna_disc = z_disc
        z_pres_prop, z_depth_prop, z_where_prop, z_what_prop, z_dyna_prop = z_prop
        # (B, N1, N2, 1)
        iou = self.iou(z_where_disc, z_where_prop)
        assert torch.all((iou >= 0) & (iou <= 1))
        iou_too_high = iou > threshold
        # Only for those that exist (B, N1, N2, 1) (B, 1, N2, 1)
        iou_too_high = iou_too_high & (z_pres_prop[:, None, :] > 0.5)
        # (B, N1, 1)
        iou_too_high = torch.any(iou_too_high, dim=-2)
        z_pres_disc_new = z_pres_disc.clone()
        z_pres_disc_new[iou_too_high] = 0.0
        
        return (z_pres_disc_new, z_depth_disc, z_where_disc, z_what_disc, z_dyna_disc)
        
    def iou(self, z_where_disc, z_where_prop):
        """
        
        Args:
            z_where_disc: (B, N1, 4)
            z_where_prop: (B, N2, 4)

        Returns:
            (B, N1, N2)
        """
        B, N1, _ = z_where_disc.size()
        B, N2, _ = z_where_prop.size()
        def _get_edges(z_where):
            z_where = z_where.detach().clone()
            z_where[..., 2:] = (z_where[..., 2:] + 1) / 2
            # (B, N, 1), (B, N, 1), (B, N, 1), (B, N, 1)
            sx, sy, cx, cy = torch.split(z_where, [1, 1, 1, 1], dim=-1)
            left = cx - sx / 2
            right = cx + sx / 2
            top = cy - sy / 2
            bottom = cy + sy / 2
            return left, right, top, bottom
            
        def _area(left, right, top, bottom):
            valid = (bottom >= top) & (right >= left)
            area = (bottom - top) * (right - left)
            area *= valid
            return area
        def _iou(left1, right1, top1, bottom1, left2, right2, top2, bottom2):
            # Note (0, 0) is top-left corner
            left = torch.max(left1, left2)
            right = torch.min(right1, right2)
            top = torch.max(top1, top2)
            bottom = torch.min(bottom1, bottom2)
            area1 = _area(left1, right1, top1, bottom1)
            area2 = _area(left2, right2, top2, bottom2)
            area_intersect = _area(left, right, top, bottom)
            iou = area_intersect / (area1 + area2 - area_intersect + 1e-5)
            # If any of these areas are zero, we ignore this iou
            iou = iou * (area_intersect != 0) * (area1 != 0) * (area2 != 0)
            return iou
        # (B, N1, 1, 1)
        left1, right1, top1, bottom1 = _get_edges(z_where_disc[:, :, None])
        # (B, 1, N2, 1)
        left2, right2, top2, bottom2 = _get_edges(z_where_prop[:, None])
        iou = _iou(left1, right1, top1, bottom1, left2, right2, top2, bottom2)
        assert iou.size() == (B, N1, N2, 1)
        return iou


  
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
            x: (B, 3, H, W)

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
        # (B, G*G, D)
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
        z_dyna_scale = F.softplus(z_dyna_scale) + 1e-4
        
        return z_dyna_loc, z_dyna_scale


class PresDepthWhereWhatPrior(nn.Module):
    """p(z_att|z_dyna)"""
    
    def __init__(self):
        nn.Module.__init__(self)
        self.enc = MLP(
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
        params = self.enc(enc)
        
        (z_pres_prob, z_depth_loc, z_depth_scale, z_where_loc, z_where_scale,
         z_what_loc, z_what_scale,
         z_depth_gate, z_where_gate, z_what_gate) = torch.split(params,
                                                                [ARCH.Z_PRES_DIM] + [ARCH.Z_DEPTH_DIM] * 2 + [
                                                                    ARCH.Z_WHERE_DIM] * 2 + [
                                                                    ARCH.Z_WHAT_DIM] * 2 + [ARCH.Z_DEPTH_DIM,
                                                                                            ARCH.Z_WHERE_DIM,
                                                                                            ARCH.Z_WHAT_DIM]
                                                                , dim=-1)
        z_pres_prob = torch.sigmoid(z_pres_prob)
        z_depth_scale = F.softplus(z_depth_scale) + 1e-4
        z_where_scale = F.softplus(z_where_scale) + 1e-4
        z_what_scale = F.softplus(z_what_scale) + 1e-4
        
        z_depth_gate = torch.sigmoid(z_depth_gate)
        z_where_gate = torch.sigmoid(z_where_gate)
        z_what_gate = torch.sigmoid(z_what_gate)
        
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
        x = self.enc(x)
        x = x.flatten(start_dim=1)
        return self.enc_what(x).view(B, N, ARCH.PROPOSAL_ENC_DIM)


class GlimpseDecoder(nn.Module):
    """Decode z_what into glimpse."""
    
    def __init__(self):
        nn.Module.__init__(self)
        # Everything here is symmetric to encoder, but with subpixel upsampling
        
        self.embed_size = ARCH.GLIMPSE_SIZE // 16
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
            z_what: (B, D)

        Returns:
            glimpse: (B, 3, H, W)
        """
        B, D = z_what.size()
        x = F.celu(self.fc(z_what))
        # (B, 128, E, E)
        x = x.view(B, 128, self.embed_size, self.embed_size)
        x = self.net(x)
        x = torch.sigmoid(x)
        # (B, 3, H, W), (B, 1, H, W)
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
    
