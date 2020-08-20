import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence
from .arch import ARCH
from .module import Flatten, MLP


class BgModule(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.embed_size = ARCH.IMG_SIZE // 16
        # Image encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.CELU(),
            nn.GroupNorm(4, 64),
            
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 128),
            
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.CELU(),
            nn.GroupNorm(32, 512),
        )
        
        self.enc_fc = nn.Linear(self.embed_size ** 2 * 512, ARCH.IMG_ENC_DIM)
        self.enc_fc_rob_action = nn.Linear(self.embed_size ** 2 * 512 + ARCH.ACTION_DIM, ARCH.IMG_ENC_DIM)
        self.dec_fc = nn.Linear(ARCH.Z_CTX_DIM, self.embed_size ** 2 * 128)
        # Decoder latent into background
        self.dec = nn.Sequential(
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
            
            nn.Conv2d(16, 3 * 2 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Sigmoid()
        )
        # self.dec = BgDecoder()
        
        self.rnn_post = nn.LSTMCell(ARCH.Z_CTX_DIM, ARCH.RNN_CTX_HIDDEN_DIM)
        self.rnn_prior = nn.LSTMCell(ARCH.Z_CTX_DIM, ARCH.RNN_CTX_HIDDEN_DIM)
        self.h_init_post = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM)) # TODO (cheolhui): figure out what each of h and c is.
        self.c_init_post = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.h_init_prior = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.c_init_prior = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_HIDDEN_DIM))
        self.prior_net = MLP([ARCH.RNN_CTX_HIDDEN_DIM, 128, 128, ARCH.Z_CTX_DIM * 2], act=nn.CELU())
        self.post_net = MLP([ARCH.RNN_CTX_HIDDEN_DIM + ARCH.IMG_ENC_DIM, 128, 128, ARCH.Z_CTX_DIM * 2], act=nn.CELU())
    
    def forward(self, seq): 
        return self.encode(seq) #? Maybe it is not being explicitly used...
    
    def anneal(self, global_step):
        pass
    
    def encode(self, seq):
        """
        Encode input frames into context latents
        Args:
            seq: (B, T, 3, H, W)

        Returns:
            things:
                bg: (B, T, 3, H, W)
                kl: (B, T)
        """
        # (B, T, 3, H, W)
        B, T, C, H, W = seq.size()
        
        # Encode images
        # (B*T, C, H, W)
        enc = self.enc(seq.reshape(B * T, 3, H, W)) #! Eq.(5) of supl.
        # I deliberately do this ugly thing because for future version we may need enc to do bg interaction
        # (B*T, D) #! flatten, no convolution?
        enc = enc.flatten(start_dim=1)
        # (B*T, D)
        enc = self.enc_fc(enc) #! Eq.(5) of supl.
        # (B, T, D)
        enc = enc.view(B, T, ARCH.IMG_ENC_DIM) # 
        
        h_post = self.h_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM) # random init nn_params: shape - [B, H]
        c_post = self.c_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM) # random init nn_params: shape - [B, H]
        h_prior = self.h_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM) # random init nn_params: shape - [B, H]
        c_prior = self.c_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM) # random init nn_params: shape - [B, H]
        
        # (B,)
        kl_list = []
        z_ctx_list = []
        for t in range(T): # first start with random noise of parameters (hidden state + cell_state)
            # Compute posterior
            # (B, D)
            post_input = torch.cat([h_post, enc[:, t]], dim=-1) 
            # (B, D), (B, D)
            params = self.post_net(post_input) #! Eq.(6) of supl.
            # (B, D), (B, D)
            loc, scale = torch.chunk(params, 2, dim=-1) #! Eq.(6) of supl.
            scale = F.softplus(scale) + 1e-4
            # (B, D)
            z_ctx_post = Normal(loc, scale) #! Eq.(7 ) of supl.
            # (B*T, D)
            z_ctx = z_ctx_post.rsample() #! Eq.(7) of supl.
            
            # Compute prior
            params = self.prior_net(h_prior) #!. Eq(2) of supl.
            loc, scale = torch.chunk(params, 2, dim=-1)
            scale = F.softplus(scale) + 1e-4
            z_ctx_prior = Normal(loc, scale) #! Eq.(3) of supl.
            
            # Temporal encode #! encode along the time axes1 -> generate context prior and posterior
            h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post)) #! Eq.(4) of supl.
            h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior)) #! Eq.(1) of supl.
            
            # Compute KL Divergence
            # (B, D)
            kl = kl_divergence(z_ctx_post, z_ctx_prior) # measure the dist btwn ctx prior and posterior
            assert kl.size()[-1] == ARCH.Z_CTX_DIM
            
            # Accumulate things
            z_ctx_list.append(z_ctx)
            kl_list.append(kl.sum(-1))
        
        # (B, T, D) -> (B*T, D)
        z_ctx = torch.stack(z_ctx_list, dim=1) # (B, T, D)
        z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM) # (B*T, D)
        # Before that, let's render our background # TODO (cheolhui): figure out what forces the seperation of fg and bg.
        # (B*T, 3, H, W)
        bg = self.dec( #! Eq.(41) of supl. decode from z_ctx_{post}
            # z_ctx
            self.dec_fc(z_ctx).
                view(B * T, 128, self.embed_size, self.embed_size)
        )
        # Reshape
        bg = bg.view(B, T, 3, H, W) # (B, T, 3, H, W)

        z_ctx = z_ctx.view(B, T, ARCH.Z_CTX_DIM) # (B*T, Z_CTX_DIM) -> (B, T, Z_CTX_DIM)
        # (B, T)
        kl_bg = torch.stack(kl_list, dim=1) # len T ist of [B,] -> [B, T]
        assert kl_bg.size() == (B, T)
        
        things = dict(
            bg=bg,  # (B, T, 3, H, W)
            z_ctx=z_ctx,  # (B, T, D), where D = Z_CTX_DIM
            kl_bg=kl_bg,  # (B, T)
        )
        
        return things

    def encode_rob_bg(self, seq, ee):
        """
        Encode input frames into context latents, conditioned on end-effector / joint values
        Args:
            seq: (B, T, 3, H, W)
            ee: (B, T, P)
        Returns:
            things:
                bg: (B, T, 3, H, W)
                kl: (B, T)
        """
        # (B, T, 3, H, W)
        B, T, C, H, W = seq.size()
        P = ee.size(-1)
        
        # Encode images
        # (B*T, C, H, W) # TODO check the structure of self.enc
        enc = self.enc(seq.reshape(B * T, 3, H, W)) #! [B*T, C, h, w]  Eq.(5) of supl.
        ee = ee.reshape(B * T, P) #! Eq.(5) of supl.
        # I deliberately do this ugly thing because for future version we may need enc to do bg interaction
        # TODO: BG interaction here! ->  
        # (B*T, D) #! flatten, no convolution?
        enc = enc.flatten(start_dim=1)
        enc = torch.cat([enc, ee], dim=-1)
        # (B*T, D) enc_fc_rob_action
        # enc = self.enc_fc(enc) #! Eq.(5) of supl.
        enc = self.enc_fc_rob_action(enc) #! Eq.(5) of supl.
        # (B, T, D)
        enc = enc.view(B, T, ARCH.IMG_ENC_DIM) # 
        
        h_post = self.h_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM) # random init nn_params: shape - [B, H]
        c_post = self.c_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM) # random init nn_params: shape - [B, H]
        h_prior = self.h_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM) # random init nn_params: shape - [B, H]
        c_prior = self.c_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM) # random init nn_params: shape - [B, H]
        
        # (B,)
        kl_list = []
        z_ctx_list = []
        for t in range(T): # first start with random noise of parameters (hidden state + cell_state)
            # Compute posterior
            # (B, D)
            post_input = torch.cat([h_post, enc[:, t]], dim=-1) 
            # (B, D), (B, D)
            params = self.post_net(post_input) #! Eq.(6) of supl.
            # (B, D), (B, D)
            loc, scale = torch.chunk(params, 2, dim=-1) #! Eq.(6) of supl.
            scale = F.softplus(scale) + 1e-4
            # (B, D)
            z_ctx_post = Normal(loc, scale) #! Eq.(7 ) of supl.
            # (B*T, D)
            z_ctx = z_ctx_post.rsample() #! Eq.(7) of supl.
            
            # Compute prior
            params = self.prior_net(h_prior)
            loc, scale = torch.chunk(params, 2, dim=-1) #! Eq.(6) of supl.
            scale = F.softplus(scale) + 1e-4
            z_ctx_prior = Normal(loc, scale) #! Eq.(7) of supl.
            
            # Temporal encode #! encode along the time axes1 -> generate context prior and posterior
            h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post)) #! Eq.(4) of supl.
            h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior)) #! Eq.(1) of supl.
            
            # Compute KL Divergence
            # (B, D)
            kl = kl_divergence(z_ctx_post, z_ctx_prior) # measure the dist btwn ctx prior and posterior
            assert kl.size()[-1] == ARCH.Z_CTX_DIM
            
            # Accumulate things
            z_ctx_list.append(z_ctx)
            kl_list.append(kl.sum(-1))
        
        # (B, T, D) -> (B*T, D)
        z_ctx = torch.stack(z_ctx_list, dim=1) # (B, T, D)
        z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM) # (B*T, D)
        # Before that, let's render our background # TODO (cheolhui): figure out what forces the seperation of fg and bg.
        # (B*T, 3, H, W)
        bg = self.dec(
            # z_ctx
            self.dec_fc(z_ctx).
                view(B * T, 128, self.embed_size, self.embed_size)
        )
        # Reshape
        bg = bg.view(B, T, 3, H, W) # (B, T, 3, H, W)

        z_ctx = z_ctx.view(B, T, ARCH.Z_CTX_DIM) # (B*T, Z_CTX_DIM) -> (B, T, Z_CTX_DIM)
        # (B, T)
        kl_bg = torch.stack(kl_list, dim=1) # len T ist of [B,] -> [B, T]
        assert kl_bg.size() == (B, T)
        
        things = dict(
            bg=bg,  # (B, T, 3, H, W)
            z_ctx=z_ctx,  # (B, T, D), where D = Z_CTX_DIM
            kl_bg=kl_bg,  # (B, T)
        )
        
        return things
    
    def generate(self, seq, cond_steps, sample):
        """
        Generate new frames given a set of input frames
        Args:
            seq: (B, T, 3, H, W)

        Returns:
            things:
                bg: (B, T, 3, H, W)
                kl: (B, T)
        """
        # (B, T, 3, H, W)
        B, T, C, H, W = seq.size()
        
        # Encode images. Only needed for the first few steps
        # (B*T, C, H, W)
        enc = self.enc(seq[:, :cond_steps].reshape(B * cond_steps, 3, H, W))
        # (B*T, D)
        enc = enc.flatten(start_dim=1)
        # (B*T, D)
        enc = self.enc_fc(enc)
        # (B, T, D)
        enc = enc.view(B, cond_steps, ARCH.IMG_ENC_DIM)
        
        h_post = self.h_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_post = self.c_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        h_prior = self.h_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_prior = self.c_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        # (B,)
        z_ctx_list = []
        for t in range(T):
            
            if t < cond_steps:
                # Compute posterior
                # (B, D)
                post_input = torch.cat([h_post, enc[:, t]], dim=-1)
                # (B, D), (B, D)
                params = self.post_net(post_input)
                # (B, D), (B, D)
                loc, scale = torch.chunk(params, 2, dim=-1)
                scale = F.softplus(scale) + 1e-4
                # (B, D)
                z_ctx_post = Normal(loc, scale)
                # (B*T, D)
                z_ctx = z_ctx_post.sample()
                
                # Temporal encode
                h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post))
                h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
            else:
                # Compute prior
                params = self.prior_net(h_prior)
                loc, scale = torch.chunk(params, 2, dim=-1)
                scale = F.softplus(scale) + 1e-4
                z_ctx_prior = Normal(loc, scale)
                z_ctx = z_ctx_prior.sample() if sample else loc
                h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
            
            # Accumulate things
            z_ctx_list.append(z_ctx)
        
        # (B, T, D) -> (B*T, D)
        z_ctx = torch.stack(z_ctx_list, dim=1)
        z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM)
        # Before that, let's render our background
        # (B*T, 3, H, W)
        bg = self.dec(
            # z_ctx
            self.dec_fc(z_ctx).
                view(B * T, 128, self.embed_size, self.embed_size)
        )
        bg = bg.view(B, T, 3, H, W)
        # Split into lists of length t
        z_ctx = z_ctx.view(B, T, ARCH.Z_CTX_DIM)
        things = dict(
            bg=bg,  # (B, T, 3, H, W)
            z_ctx=z_ctx  # (B, T, D)
        )
        
        return things

    def generate_rob_bg(self, seq, ees, cond_steps, sample):
        """
        Generate new frames given a set of input frames
        Args:
            seq: (B, T, 3, H, W)
            ees: (B, T, P); where P is the end-effector pose dimension
        Returns:
            things:
                bg: (B, T, 3, H, W)
                kl: (B, T)
        """
        # (B, T, 3, H, W)
        B, T, C, H, W = seq.size()
        P = ees.size(-1)
        # Encode images. Only needed for the first few steps
        # (B*T, C, H, W)
        enc = self.enc(seq[:, :cond_steps].reshape(B * cond_steps, 3, H, W))
        ees = ees[:, :cond_steps].reshape(B * cond_steps, P)
        # (B*T, D)
        enc = enc.flatten(start_dim=1)
        # (B*T, D)
        enc = torch.cat([enc, ees], dim=-1)
        enc= self.enc_fc_rob_action(enc)
        # enc = self.enc_fc(enc)
        # (B, T, D)
        enc = enc.view(B, cond_steps, ARCH.IMG_ENC_DIM)
        
        h_post = self.h_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_post = self.c_init_post.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        h_prior = self.h_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        c_prior = self.c_init_prior.expand(B, ARCH.RNN_CTX_HIDDEN_DIM)
        # (B,)
        z_ctx_list = []
        for t in range(T):
            
            if t < cond_steps:
                # Compute posterior
                # (B, D)
                post_input = torch.cat([h_post, enc[:, t]], dim=-1)
                # (B, D), (B, D)
                params = self.post_net(post_input)
                # (B, D), (B, D)
                loc, scale = torch.chunk(params, 2, dim=-1)
                scale = F.softplus(scale) + 1e-4
                # (B, D)
                z_ctx_post = Normal(loc, scale)
                # (B*T, D)
                z_ctx = z_ctx_post.sample()
                
                # Temporal encode
                h_post, c_post = self.rnn_post(z_ctx, (h_post, c_post))
                h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
            else:
                # Compute prior
                params = self.prior_net(h_prior)
                loc, scale = torch.chunk(params, 2, dim=-1)
                scale = F.softplus(scale) + 1e-4
                z_ctx_prior = Normal(loc, scale)
                z_ctx = z_ctx_prior.sample() if sample else loc
                h_prior, c_prior = self.rnn_prior(z_ctx, (h_prior, c_prior))
            
            # Accumulate things
            z_ctx_list.append(z_ctx)
        
        # (B, T, D) -> (B*T, D)
        z_ctx = torch.stack(z_ctx_list, dim=1)
        z_ctx = z_ctx.view(B * T, ARCH.Z_CTX_DIM)
        # Before that, let's render our background
        # (B*T, 3, H, W)
        bg = self.dec(
            # z_ctx
            self.dec_fc(z_ctx).
                view(B * T, 128, self.embed_size, self.embed_size)
        )
        bg = bg.view(B, T, 3, H, W)
        # Split into lists of length t
        z_ctx = z_ctx.view(B, T, ARCH.Z_CTX_DIM)
        things = dict(
            bg=bg,  # (B, T, 3, H, W)
            z_ctx=z_ctx  # (B, T, D)
        )
        
        return things
