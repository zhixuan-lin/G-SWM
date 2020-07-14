from attrdict import AttrDict
import os
import sys
import yacs
from yacs.config import CfgNode
from model.gswm.arch import ARCH

dirname = os.path.basename(os.path.dirname(os.path.abspath(__file__)))


cfg = CfgNode({
    # By default, we use the folder name
    'seed': 0,
    'run_num': 1,
    'exp_name': '',
    'model': 'SCALOR',
    
    # Resume training or not
    'resume': True,
    # If resume is true, then we load this checkpoint. If '', we load the last checkpoint
    'resume_ckpt': '',
    # Whether to use multiple GPUs
    'parallel': True,
    # Device ids to use
    'device_ids': [0, 1],
    # Note if we use parallel, this must be the same as device_ids[0]
    'device': 'cuda:0',
    'logdir': '../output/logs/',
    'checkpointdir': '../output/checkpoints/',
    'evaldir': '../output/eval/',
    'resultdir': '../output/results/',
    
    # Dataset to use
    'dataset': 'OBJ3D',
    
    'dataset_roots': {
        'BALLS_INTERACTION': '../data/BALLS_INTERACTION',
        'BALLS_INTERACTION_ROTATE': '../data/BALLS_INTERACTION_ROTATE',
        'BALLS_OCCLUSION': '../data/BALLS_OCCLUSION',
        'BALLS_TWO_LAYER': '../data/BALLS_TWO_LAYER',
        'BALLS_TWO_LAYER_DENSE': '../data/BALLS_TWO_LAYER_DENSE',
        'OBJ3D': '../data/OBJ3D',
        'SINGLE_BALL': '../data/SINGLE_BALL',
        'MAZE': '../data/MAZE',
    },
    
    'solver': {
        'optim': 'Adam',
        'lr': 1e-4,
    },
    
    
    # TODO: a perfect design will require one item per function in engine.
    'train': {
        'batch_size': 16,
        'max_epochs': 1000,
        'max_steps': 1000000,
        'print_every': 500,
        # Save checkpoint per N iterations
        'save_every': 1000,
        'num_workers': 4,
        # Gradient clipping. If 0.0, not clipping.
        'clip_norm': 0.0,
        # Maximum number checkpoints maintained
        'max_ckpt': 3,
    },
    
    # For eval in engine. Well, not really since ison is also here
    'val': {
        'evaluator': 'ball',
        # Used during training
        'ison': True,
        'val_every': 1000,
        # Dataset related
        'batch_size': 16,
        # Number of batches for validation
        'num_workers': 4,
        
        # Used in engine.eval
        # Use test set or training set in evaluation. Should be in ['val', 'test']
        'mode': 'val',
        # Either tracking or generation
        'metrics': ['mot_iou', 'mot_dist', 'med'],
        'eval_types': ['tracking', 'generation'],
        'intervals': [[k, k+10] for k in range(10, 100+1, 10)],
        'cond_steps': 10,
    },
    
    'vis': {
        'ison': True,
        'cond_steps': 5,
        'vis_every': 1000,
        'indices': [0, 1, 2, 3],
        'fg_sample': True,
        'bg_sample': False,
        'num_gen': 1,
    },
    'test': {
        # Dataset related
        'batch_size': 16,
        # Number of batches for validation
        'num_workers': 4,
    },
})


cfg.ARCH = ARCH
