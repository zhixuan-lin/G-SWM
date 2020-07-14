from yacs.config import CfgNode

ARCH = CfgNode()

# Whether to use deterministic version
ARCH.DETER = False

ARCH.G = 8

# T: sequence length
ARCH.T = [2, 4, 6, 8, 10]
# When to change T, in terms of global steps
ARCH.T_MILESTONES = [10000, 12500, 15000, 17500]

# Maximum number of objects
ARCH.MAX = 16

# Enable background modeling
ARCH.BG_ON = False
# This will be useful only when BG_ON is True. Before this step, we learn background only.
ARCH.BG_ONLY_STEP = 1000


# For gaussian
# If we use gaussian
ARCH.SIGMA = 0.2
ARCH.SIGMA_ANNEAL = False
ARCH.SIGMA_START_VALUE = 0.2
ARCH.SIGMA_END_VALUE = 0.2
ARCH.SIGMA_START_STEP = 0
ARCH.SIGMA_END_STEP = 10



# Latent dimensions
ARCH.Z_DYNA_DIM = 128
ARCH.Z_PRES_DIM = 1
ARCH.Z_SCALE_DIM = 2
ARCH.Z_SHIFT_DIM = 2
ARCH.Z_WHERE_DIM = 4
ARCH.Z_DEPTH_DIM = 1
ARCH.Z_WHAT_DIM = 64
ARCH.Z_CTX_DIM = 128
# 1 + 1 + 4 + 128 + 128
# ARCH.Z_DIM = ARCH.Z_PRES_DIM + ARCH.Z_WHERE_DIM + ARCH.Z_DEPTH_DIM + ARCH.Z_WHAT_DIM + ARCH.Z_DYNA_DIM

# Image shape and size
ARCH.IMG_SIZE = 64
ARCH.IMG_SHAPE = (ARCH.IMG_SIZE,) * 2

# Glimpse shape and size
ARCH.GLIMPSE_SIZE = 16
ARCH.GLIMPSE_SHAPE = (ARCH.GLIMPSE_SIZE,) * 2

# Img encoding dimension
ARCH.IMG_ENC_DIM = 128
# Glimpse encoding dimension
ARCH.GLIMPSE_ENC_DIM = 128
# Proposal encoding dimension
ARCH.PROPOSAL_ENC_DIM = 128
# Temporal rnn input dimension
ARCH.RNN_INPUT_DIM = 128
# Temporal rnn latent dimensions
ARCH.RNN_HIDDEN_DIM = 128
# Background rnn hidden dimension
ARCH.RNN_CTX_HIDDEN_DIM = 128

# Temperature for gumbel-softmax
ARCH.TAU_START_STEP = 0
ARCH.TAU_END_STEP = 10000
ARCH.TAU_START_VALUE = 1.0
ARCH.TAU_END_VALUE = 1.0

# Prior for scale in discovery
ARCH.Z_SCALE_MEAN_START_STEP = 0
ARCH.Z_SCALE_MEAN_END_STEP = 10000
ARCH.Z_SCALE_MEAN_START_VALUE = -1.5
ARCH.Z_SCALE_MEAN_END_VALUE = -1.5
ARCH.Z_SCALE_STD = 0.3

# Prior for z_shift
ARCH.Z_SHIFT_MEAN = 0.0
ARCH.Z_SHIFT_STD = 1.0

# Prior for presence in discovery
ARCH.Z_PRES_PROB_START_STEP = 0
ARCH.Z_PRES_PROB_END_STEP = 1500
ARCH.Z_PRES_PROB_START_VALUE = 1e-10
ARCH.Z_PRES_PROB_END_VALUE = 1e-10

# Update z_where and z_what
ARCH.PROPOSAL_UPDATE_MIN = 0.0
ARCH.PROPOSAL_UPDATE_MAX = 0.3
ARCH.Z_SHIFT_UPDATE_SCALE = 0.1
ARCH.Z_SCALE_UPDATE_SCALE = 0.3
ARCH.Z_WHAT_UPDATE_SCALE = 0.2
ARCH.Z_DEPTH_UPDATE_SCALE = 1.0

# Mlp layer sizesa in the mlp that compute the propagation map
ARCH.PROP_MAP_MLP_LAYERS = [128, 128]
# Propagation map depth/channels/dimensions
ARCH.PROP_MAP_DIM = 128
# This is used for the gaussian kernel
ARCH.PROP_MAP_SIGMA = 0.1

# Mlp layer sizesa in the mlp that compute the propagation map
ARCH.PROP_COND_MLP_LAYERS = [128, 128]
# Propagation conditioning vector depth/channels/dimensions
ARCH.PROP_COND_DIM = 128
# This is used in the gaussian kernel
ARCH.PROP_COND_SIGMA = 0.1


# Rejection
ARCH.REJECTION = True
ARCH.REJECTION_THRESHOLD = 0.8

# AOE
ARCH.BG_CONDITIONED = True
ARCH.BG_ATTENTION = True
ARCH.BG_PROPOSAL_DIM = 128
ARCH.BG_PROPOSAL_SIZE = 0.25

# Discovery Dropout
ARCH.DISCOVERY_DROPOUT = 0.5

# Auxiliary presence loss in propagation
ARCH.AUX_PRES_KL = True

