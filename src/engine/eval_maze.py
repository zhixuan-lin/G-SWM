import os
import json
import torch
import time
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from utils import MetricLogger, Checkpointer, TensorAccumulator
from dataset import get_dataloader, get_dataset
from model import get_model
from solver import get_optimizer
from torch import nn
import argparse
from argparse import ArgumentParser
from config import cfg
from visualize import get_vislogger
from evaluate import get_evaluator
from attrdict import AttrDict
from tqdm import tqdm
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


def scalor_pred_to_boxes(z_where):
    """
    Args:
        z_where: (B, T, N, 4) where 4 => [width, height, center_x, center_y]
        z_pres: (B, T, N, 1)
        pres_threshold: only include objects over this threshould
    Returns:
        boxes: (B, T, N, 4) where 4 => [center_x, center_y, width, height] normalized between 0 and 1
    """
    pred_boxes = z_where[:, :, :, [2, 3, 0, 1]].clone()
    pred_boxes[:, :, :, 0:2] = (pred_boxes[:, :, :, 0:2] + 1) / 2
    return pred_boxes


@torch.no_grad()
def eval_maze(cfg, cond_steps=5):
    print('\nLoading data...')
    assert cfg.val.mode == 'test', 'Please set cfg.val.mode to "test"'
    dataset = get_dataset(cfg, cfg.val.mode)
    dataloader = get_dataloader(cfg, cfg.val.mode)
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
    
    evaluator = get_evaluator(cfg)
    
    evaldir = os.path.join(cfg.evaldir, cfg.exp_name)
    os.makedirs(evaldir, exist_ok=True)
    start = time.perf_counter()
    model.eval()
    
    evaluator.evaluate(model, dataloader, cond_steps, cfg.device, evaldir, cfg.exp_name, cfg.resume_ckpt)
    file_name = 'maze-{}.json'.format(cfg.exp_name)
    jsonpath = os.path.join(evaldir, file_name)
    with open(jsonpath) as f:
        metrics = json.load(f)
    num_mean = metrics['num_mean']
    f, ax = plt.subplots()
    ax: plt.Axes
    ax.plot(num_mean)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('#Agents')
    ax.set_title(cfg.exp_name)
    plt.savefig(os.path.join(evaldir, 'plot_maze.png'))

if __name__ == '__main__':
    pass

