import torch
from torch import nn
from attrdict import AttrDict
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from .arch import ARCH
from .module import Flatten, MLP

class AgentBgModule(nn.Module):
    "background layer that decomposed into agent & non-agent"
    # TODO (cheolhui): make it compatible with sequential data
    def __init__(self):
        nn.Module.__init__(self)
        
        self.embed_size = ARCH.IMG_SIZE // 16

        self.image_enc = ImageEncoderBg() # for the mask inference
        # Embeds sequential images
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                stride=2, padding=3),
            nn.CELU(),
            nn.GroupNorm(num_groups=4, num_channels=64),

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

        #! temporal relationship ---------------------------------------
        # temporal encoding of mask latents
        self.rnn_mask_post_t = nn.LSTMCell(ARCH.Z_CTX_DIM, ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        self.rnn_mask_prior_t = nn.LSTMCell(ARCH.Z_CTX_DIM, ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        # temporal encoding of comp latents
        self.rnn_comp_post_t = nn.LSTMCell(ARCH.Z_CTX_DIM, ARCH.RNN_CTX_COMP_HIDDEN_DIM)
        self.rnn_comp_prior_t = nn.LSTMCell(ARCH.Z_CTX_DIM, ARCH.RNN_CTX_COMP_HIDDEN_DIM)
        
        self.h_mask_post_t = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_MASK_HIDDEN_DIM))
        self.c_mask_post_t = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_MASK_HIDDEN_DIM))
        self.h_mask_prior_t = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_MASK_HIDDEN_DIM))
        self.c_mask_prior_t = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_MASK_HIDDEN_DIM))

        self.h_comp_post_t = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_COMP_HIDDEN_DIM))
        self.c_comp_post_t = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_COMP_HIDDEN_DIM))
        self.h_comp_prior_t = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_COMP_HIDDEN_DIM))
        self.c_comp_prior_t = nn.Parameter(torch.randn(1, ARCH.RNN_CTX_COMP_HIDDEN_DIM))
        # temporal consistency, inputs for the mask network 
        self.prior_net_t = MLP([ARCH.RNN_CTX_MASK_HIDDEN_DIM, 128, 128, ARCH.Z_CTX_DIM * 2], act=nn.CELU())
        self.post_net_t = MLP([ARCH.RNN_CTX_MASK_HIDDEN_DIM + ARCH.IMG_ENC_DIM, 128, 128, ARCH.Z_CTX_DIM * 2], act=nn.CELU())
    

        #! spatial relationship for ONLY mask latents-------------------
        self.rnn_mask_post_k = nn.LSTMCell(ARCH.Z_MASK_DIM + ARCH.Z_CTX_DIM * 2, ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        self.h_mask_post_k = nn.Parameter(torch.zeros(ARCH.RNN_CTX_MASK_HIDDEN_DIM))
        self.c_mask_post_k = nn.Parameter(torch.zeros(ARCH.RNN_CTX_MASK_HIDDEN_DIM))

        #! posteriors ---------------------------------------
        # Dummy z_mask for first step of rnn_mask
        self.z_mask_0 = nn.Parameter(torch.zeros(ARCH.Z_MASK_DIM))
        # Predict mask latent given h
        self.predict_mask = PredictMask()
        # Compute masks given mask latents
        self.mask_decoder = MaskDecoder()
        # Encode mask and image into component latents
        self.comp_encoder = CompEncoder()
        # Component decoder
        if ARCH.K > 1:
            self.comp_decoder = CompDecoder()
        else:
            self.comp_decoder = CompDecoderStrong()
        #! priors ---------------------------------------------
        self.rnn_mask_prior_k = nn.LSTMCell(ARCH.Z_MASK_DIM, ARCH.RNN_CTX_MASK_PRIOR_HIDDEN_DIM)
        # Initial h and c
        self.h_mask_prior_k = nn.Parameter(torch.zeros(ARCH.RNN_CTX_MASK_PRIOR_HIDDEN_DIM))
        self.c_mask_prior_k = nn.Parameter(torch.zeros(ARCH.RNN_CTX_MASK_PRIOR_HIDDEN_DIM))
        # Compute mask latents
        self.predict_mask_prior = PredictMask()
        # Compute component latents
        self.predict_comp_prior = PredictComp()
        # ==== Prior related ====

        self.bg_sigma = ARCH.BG_SIGMA
    
    def anneal(self, global_step):
        pass
    
    def forward(self, seq):
        return self.encode(seq)

    def encode(self, seq, ee):
        """
        Background inference backward pass
        # TODO (cheolhui): add robot priors
        # TODO (cheolhui): we modify the process to be spatio-temporal
        :param seq: shape (B, T, C, H, W)
        :param ee: shape (B, T, P)
        :return:
            bg_likelihood: (B, 3, H, W)
            bg: (B, 3, H, W)
            kl_bg: (B,)
            log: a dictionary containing things for visualization
        """
        B, T, C, H, W = seq.size()
        P = ee.size(-1)

        # TODO 1) reshape the sequence input
        seq = seq.reshape(B * T, 3, H, W)
        ee = ee.reshape(B * T, P)
        #! enc = self.image(enc) -> [B*T, 64]; SPACE
        enc = self.enc(seq) # TODO check shape
        seq = seq.view(B, T, 3, H, W)
        enc = enc.flatten(start_dim=1) # (B*T, D)

        enc = self.enc_fc(enc) # [B, T, 128]
        enc = enc.view(B, T, ARCH.IMG_ENC_DIM) # [B, T, 128]
        
        # Mask and component latents over the K slots

        # Generate autoregressive masks

        # spatial initial RNN paramters  encode x and dummy z_mask_0

        #? h_prior_k, c_prior_k?
        #! temporal initial RNN parameters (posts & priors)
        h_mask_post_t = self.h_mask_post_t.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) # random init nn_params: shape - [B, Hm]
        c_mask_post_t = self.c_mask_post_t.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) # random init nn_params: shape - [B, Hm]
        h_comp_post_t = self.h_comp_post_t.expand(B, ARCH.RNN_CTX_COMP_HIDDEN_DIM) # random init nn_params: shape - [B, Hc]
        c_comp_post_t = self.c_comp_post_t.expand(B, ARCH.RNN_CTX_COMP_HIDDEN_DIM) # random init nn_params: shape - [B, Hc]

        h_mask_prior_t = self.h_mask_prior_t.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) # random init nn_params: shape - [B, Hm]
        c_mask_prior_t = self.c_mask_prior_t.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) # random init nn_params: shape - [B, Hm]
        h_comp_prior_t = self.h_comp_prior_t.expand(B, ARCH.RNN_CTX_COMP_HIDDEN_DIM) # random init nn_params: shape - [B, Hc]
        c_comp_prior_t = self.c_comp_prior_t.expand(B, ARCH.RNN_CTX_COMP_HIDDEN_DIM) # random init nn_params: shape - [B, Hc]
        
        
        kl_list = [] #? remove this?
        z_ctx_list = []
        # interate over T
        
        masks_t = []
        comps_t = []
        # These two are Normal instances
        bg_likelihoods = [] # list of Tensors of background s
        bgs = [] # list of background reconstruction Tensors
        z_mask_total_kl_k_t = 0.0 # sum of mask kls over K and T axes
        z_comp_total_kl_k_t = 0.0 # sum of comp kls over K and T axes
        # iterate over truncated time steps
        for t in range(T):
            #! initialize spatial latents of dim K for every time T
            z_mask = self.z_mask_0.expand(B, ARCH.Z_MASK_DIM) # spatial
            h_mask_post_k = self.h_mask_post_k.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) # spatial
            c_mask_post_k = self.c_mask_post_k.expand(B, ARCH.RNN_CTX_MASK_HIDDEN_DIM) # spatial
            #! 1) Compute posteriors
            masks = []
            z_masks = []
            # These two are Normal instances
            z_mask_posteriors = []
            z_comp_posteriors = []

            # Compute posterior
            # [B, Z_CTX_DIM + IMG_ENC_DIM]
            post_input = torch.cat([h_mask_post_t, enc[:, t]], dim=-1) # 'enc' should span along K
            # [B, D]
            enc = self.post_net_t(post_input) # TODO: which one of 'post_input' / 'params' should be given as input?
            # iterate over K entities, infer autoregressive mask latents
            for k in range(ARCH.K):
                # Encode x and z_{mask, 1:k}, (b, D)
                rnn_input = torch.cat((z_mask, enc), dim=1)
                (h_mask_post_k, c_mask_post_k) = self.rnn_mask_post_k(rnn_input, (h_mask_post_k, c_mask_post_k))
                # Predict next mask from x and z_{mask, 1:k-1}
                z_mask_loc, z_mask_scale = self.predict_mask(h_mask_post_k) # [B, Zm]
                z_mask_post = Normal(z_mask_loc, z_mask_scale) # [B, Zm]
                z_mask = z_mask_post.rsample() # [B, Zm]
                z_masks.append(z_mask)
                z_mask_posteriors.append(z_mask_post)
                # Decode masks
                mask = self.mask_decoder(z_mask) # In - [B, Zm], Out - [B, 1, H, W]
                masks.append(mask)
            
            # (B, K, 1, H, W) in range (0, 1)
            masks = torch.stack(masks, dim=1)
            # SBP to ensure to be summed up to 1.
            masks = self.SBP(masks) # or use SoftMax: masks = F.softmax(masks, dim=1)

            B, K, _, H, W = masks.size()
            
            # Reshape (B, K, 1, H, W) -> (B*K, 1, H, W) 
            masks = masks.view(B*K, 1, H, W)
            # add 1e-5 to mask to avoid infinity
            # Concatenate images along channel axis (B*K, 4, H, W); torch.repeat == tf.tile
            comp_vae_input = torch.cat([(masks + 1e-5).log(), 
                seq[:, t, None].repeat(1, K, 1, 1, 1).view(B*K, 3, H, W)], dim=1)
            # Component latents, each [B*K, L]
            z_comp_loc, z_comp_scale = self.comp_encoder(comp_vae_input)
            z_comp_post = Normal(z_comp_loc, z_comp_scale)
            z_comp = z_comp_post.r_sample() # TODO: check its shape

            #! 1-1) acquire component posteriors here, for computing KL divergences
            z_comp_loc_reshape = z_comp_loc.view(B, K, -1)
            z_comp_scale_reshape = z_comp_scale.view(B, K, -1)
            for i in range(ARCH.K):
                z_comp_post_this = Normal(z_comp_loc_reshape[:, i], z_comp_scale_reshape[:, i])
                z_comp_posteriors.append(z_comp_post_this)
            comps = self.comp_decoder(z_comp)


            # Decode into component images, [B*K, 3, H, W]
            comps = comps.view(B, K, 3, H, W)
            masks = masks.view(B, K, 1, H, W)

            # Compute background likelihoods
            # (B, K, 3, H, W)
            comp_dist = Normal(comps, torch.full_like(comps, self.bg_sigma))
            log_likelihoods = comp_dist.log_prob(seq[:, t, None].expand_as(comps)) # TODO: check where to expand dims

            # (B, K, 3, H, W) -> (B, 3, H, W), mixture likelihood
            log_sum = log_likelihoods + (masks + 1e-5).log()
            bg_likelihood = torch.logsumexp(log_sum, dim=1)
            bg_likelihoods.append(bg_likelihood) #! final return

            bg = (comps * masks).sum(dim=1)
            bgs.append(bg) #! final return

            masks_t.append(masks) #! final return
            comps_t.append(comps) #! final return

            #! 2) Compute priors and KL divergences, for both mask z_m and component z_c

            z_mask_total_kl_k = 0.0
            z_comp_total_kl_k = 0.0
            h_mask_prior_k = self.rnn_mask_h_prior.expand(B, ARCH.RNN_MASK_HIDDEN_DIM) # spatial
            c_mask_prior_k = self.rnn_mask_c_prior.expand(B, ARCH.RNN_MASK_HIDDEN_DIM) # spatial

            # TODO (cheolhui): add temporal consistency of prior
            enc = self.prior_net_t(h_mask_prior_t) # update temporal params of prior

            for k in range(ARCH.K):
                #! 2-1) Compute prior distribution over z_masks
                rnn_input = enc
                (h_mask_prior_k, c_mask_prior_k) = self.rnn_mask_prior_k(rnn_input, (h_mask_prior_k, c_mask_prior_k))
                z_mask_loc_prior, z_mask_scale_prior = self.predict_mask_prior(h_mask_prior_k)
                z_mask_prior = Normal(z_mask_loc_prior, z_mask_scale_prior)
                #! 2-2) Compute component prior, using posterior samples
                z_comp_loc_prior, z_comp_scale_prior = self.predict_comp_prior(z_masks[k])
                z_comp_prior = Normal(z_comp_loc_prior, z_comp_scale_prior)
                # Compute KL for each entity
                z_mask_kl = kl_divergence(z_mask_posteriors[k], z_mask_prior).sum(dim=1)
                z_comp_kl = kl_divergence(z_comp_posteriors[k], z_comp_prior).sim(dim=1)
                # [B,]
                z_mask_total_kl_k += z_mask_kl
                z_comp_total_kl_k += z_comp_kl                

            #! 3) temporal encode of z_mask & z_comp into h_post & h_prior -> update temporal latents along T-axis
            z_masks = torch.cat(z_masks, dim=-1) # TODO: check shapes
            h_mask_post_t, c_mask_post_t = self.rnn_mask_post_t(z_masks, (h_mask_post_t, c_mask_post_t))
            h_mask_prior_t, c_mask_prior_t = self.rnn_mask_prior_t(z_masks, (h_mask_prior_t, c_mask_prior_t))
            
            z_comps = z_comp.view(B, K, -1)
            h_comp_post_t, c_comp_post_t = self.rnn_comp_post_t(z_comps, (h_comp_post_t, c_comp_post_t))
            h_comp_prior_t, c_comp_prior_t = self.rnn_comp_prior_t(z_comps, (h_comp_prior_t, c_comp_prior_t))

            #! 4) accumulate losses for each time T
            z_mask_total_kl_k_t += z_mask_total_kl_k
            z_comp_total_kl_k_t += z_comp_total_kl_k

        #! ------------------------- For visualization ------------------------------
        kl_bg = z_mask_total_kl_k_t + z_comp_total_kl_k_t

        # TODO: check if returning these logs is necessary
        log = {
            # (B, K, 3, H, W)
            'comps': comps_t,
            # (B, 1, 3, H, W)
            'masks': masks_t,
            # (B, 3, H, W)
            'bgs': bgs,
            'kl_bg': kl_bg
        }
        return bg_likelihoods, bgs, kl_bg, log

    @staticmethod
    def SBP(masks):
        """
        Stick breaking process to produce masks
        :param: masks (B, K, 1, H, W). In range (0, 1)
        :return: (B, K, 1, H, W)
        """
        B, K, _, H, W = masks.size()
    
        # (B, 1, H, W)
        remained = torch.ones_like(masks[:, 0])
        # remained = torch.ones_like(masks[:, 0]) - fg_mask
        new_masks = []
        for k in range(K):
            if k < K - 1: # k = 0, 1, ... K-1
                mask = masks[:, k] * remained
            else:
                mask = remained
            remained = remained - mask
            new_masks.append(mask)
    
        new_masks = torch.stack(new_masks, dim=1) # (B, K, 1, H, W)
    
        return new_masks
            


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ImageEncoderBg(nn.Module):
    """Background image encoder"""
    
    def __init__(self):
        embed_size = ARCH.IMG_SHAPE[0] // 16 # ARCH.IMG_SIZE // 4
        nn.Module.__init__(self)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # 16x downsampled: (64, H/16, W/16)
            Flatten(),
            nn.Linear(64 * embed_size ** 2, ARCH.IMG_ENC_DIM),
            nn.ELU(),
        )
    
    def forward(self, x):
        """
        Encoder image into a feature vector
        Args:
            x: (B, 3, H, W)
        Returns:
            enc: (B, D)
        """
        return self.enc(x)


class PredictMask(nn.Module):
    """
    Predict z_mask given states from rnn. Used in inference
    """
    
    def __init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Linear(ARCH.RNN_CTX_MASK_HIDDEN_DIM, ARCH.Z_MASK_DIM * 2)
    
    def forward(self, h):
        """
        Predict z_mask given states from rnn. Used in inference
    
        :param h: hidden state from rnn_mask - (B, Hm = ARCH.RNN_CTX_MASK_HIDDEN_DIM)
        :return:
            z_mask_loc: (B, D)
            z_mask_scale: (B, D)
        
        """
        x = self.fc(h) # In: [B, Hm], Out: [B, 2 * Zm]
        z_mask_loc = x[:, :ARCH.Z_MASK_DIM] # [B, Zm]
        z_mask_scale = F.softplus(x[:, ARCH.Z_MASK_DIM:]) + 1e-4 # [B, Zm]
        
        return z_mask_loc, z_mask_scale


class MaskDecoder(nn.Module):
    """Decode z_mask into mask"""
    
    def __init__(self):
        super(MaskDecoder, self).__init__()
        # NOTE: all channels of Conv2d are 1d
        # self.c1 = nn.Conv2d(in_channels=ARCH.Z_MASK_DIM, out_channels=256, kernel_size=1)

        self.dec = nn.Sequential(
            nn.Conv2d(in_channels=ARCH.Z_MASK_DIM, out_channels=256, kernel_size=1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            
            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4), # (256 * 4 * 4) -> (256, 4, 4)
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            
            nn.Conv2d(256, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            
            nn.Conv2d(128, 64 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            
            nn.Conv2d(64, 16 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            
            # nn.Conv2d(16, 16, 3, 1, 1),
            # nn.CELU(),
            # nn.GroupNorm(4, 16),
            nn.Conv2d(16, 1, 3, 1, 1)
        
        )
    
    def forward(self, z_mask):
        """
        Decode z_mask into mask
        
        :param z_mask: (B, D)
        :return: mask: (B, 1, H, W)
        """
        B = z_mask.size(0)
        # 1d -> 3d, (B, D, 1, 1)
        z_mask = z_mask.view(B, -1, 1, 1)

        mask = torch.sigmoid(self.dec(z_mask)) # [B, 1, H, W]
        return mask


class CompEncoder(nn.Module):
    """
    Predict component latent parameters given image and predicted mask concatenated
    """
    
    def __init__(self):
        nn.Module.__init__(self)
        
        embed_size = ARCH.IMG_SHAPE[0] // 16
        self.enc = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            Flatten(),
            # 16x downsampled: (64, 4, 4)
            nn.Linear(64 * embed_size ** 2, ARCH.Z_COMP_DIM * 2),
        )
    
    def forward(self, x):
        """
        Predict component latent parameters given image and predicted mask concatenated
        
        :param x: (B, 3+1, H, W). Image and mask concatenated
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.enc(x)
        z_comp_loc = x[:, :ARCH.Z_COMP_DIM]
        z_comp_scale = F.softplus(x[:, ARCH.Z_COMP_DIM:]) + 1e-4
        
        return z_comp_loc, z_comp_scale


class SpatialBroadcast(nn.Module):
    """
    Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
    """
    
    def __init__(self):
        nn.Module.__init__(self)
    
    def forward(self, x, width, height):
        """
        Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
        
        :param x: (B, L)
        :param width: W
        :param height: H
        :return: (B, L + 2, W, H)
        """
        B, L = x.size()
        # (B, L, 1, 1)
        x = x[:, :, None, None]
        # (B, L, W, H)
        x = x.expand(B, L, width, height)
        xx = torch.linspace(-1, 1, width, device=x.device)
        yy = torch.linspace(-1, 1, height, device=x.device)
        yy, xx = torch.meshgrid((yy, xx))
        # (2, H, W)
        coords = torch.stack((xx, yy), dim=0)
        # (B, 2, H, W)
        coords = coords[None].expand(B, 2, height, width)
        
        # (B, L + 2, W, H)
        x = torch.cat((x, coords), dim=1)
        
        return x


class CompDecoder(nn.Module):
    """
    Decoder z_comp into component image
    """
    
    def __init__(self):
        nn.Module.__init__(self)
        self.spatial_broadcast = SpatialBroadcast()
        # Input will be (B, L+2, H, W)
        self.decoder = nn.Sequential(
            nn.Conv2d(ARCH.Z_COMP_DIM + 2, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # 16x downsampled: (32, 4, 4)
            nn.Conv2d(32, 3, 1, 1),
        )
    
    def forward(self, z_comp):
        """
        :param z_comp: (B, L)
        :return: component image (B, 3, H, W)
        """
        h, w = ARCH.IMG_SHAPE 
        # (B, L) -> (B, L+2, H, W)
        z_comp = self.spatial_broadcast(z_comp, h + 8, w + 8)
        # -> (B, 3, H, W)
        comp = self.decoder(z_comp)
        comp = torch.sigmoid(comp)
        return comp


class CompDecoderStrong(nn.Module):
    
    def __init__(self):
        super(CompDecoderStrong, self).__init__()
        
        self.dec = nn.Sequential(
            nn.Conv2d(ARCH.Z_COMP_DIM, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            
            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            
            nn.Conv2d(256, 128 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            
            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            
            nn.Conv2d(64, 16 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 3, 3, 1, 1)
        
        )
    
    def forward(self, x):
        """

        :param x: (B, L)
        :return:
        """
        x = x.view(*x.size(), 1, 1)
        comp = torch.sigmoid(self.dec(x))
        return comp


class PredictComp(nn.Module):
    """
    Predict component latents given mask latent
    """
    
    def __init__(self):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(
            nn.Linear(ARCH.Z_MASK_DIM, ARCH.PREDICT_COMP_HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(ARCH.PREDICT_COMP_HIDDEN_DIM, ARCH.PREDICT_COMP_HIDDEN_DIM),
            nn.ELU(),
            nn.Linear(ARCH.PREDICT_COMP_HIDDEN_DIM, ARCH.Z_COMP_DIM * 2),
        )
    
    def forward(self, h):
        """
        :param h: (B, D) hidden state from rnn_mask
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.mlp(h)
        z_comp_loc = x[:, :ARCH.Z_COMP_DIM]
        z_comp_scale = F.softplus(x[:, ARCH.Z_COMP_DIM:]) + 1e-4
        
        return z_comp_loc, z_comp_scale
