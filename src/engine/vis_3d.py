import os
import os.path as osp
import numpy as np
import torch
import time
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from utils import MetricLogger, Checkpointer, TensorAccumulator
from dataset import get_dataloader, get_dataset
from model import get_model
from torch import nn
from visualize import get_vislogger


@torch.no_grad()
def vis_3d(cfg, genlen=50, cond_steps=10):
    indices = cfg.vis.indices
    print('\nLoading data...')
    dataset = get_dataset(cfg, cfg.val.mode)
    dataset = Subset(dataset, indices=list(indices))
    # dataloader = get_dataloader(cfg, cfg.val.mode)
    dataloader = DataLoader(dataset, batch_size=cfg.val.batch_size, num_workers=4)
    print('Data loaded.')
    
    print('Initializing model...')
    model = get_model(cfg)
    model = model.to(cfg.device)
    print('Model initialized.')
    
    checkpointer = Checkpointer(os.path.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    
    global_step = 0
    if cfg.resume:
        checkpoint = checkpointer.load(cfg.resume_ckpt, model, None)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step'] + 1
    if cfg.parallel:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)
    vislogger = get_vislogger(cfg)
    
    resultdir = os.path.join(cfg.resultdir, cfg.exp_name)
    os.makedirs(resultdir, exist_ok=True)
    path = osp.join(resultdir, '3d.gif')
    vislogger.show_gif(model, dataset, indices, cfg.device, cond_steps, genlen, path, fps=7)

