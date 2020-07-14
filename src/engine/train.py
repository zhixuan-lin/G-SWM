import torch

import sys

if '.' not in sys.path:
    sys.path.insert(0, '.')
import torch
import numpy as np

# Seed

import os
import time
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from utils import MetricLogger, Checkpointer
from dataset import get_dataloader, get_dataset
from model import get_model
from solver import get_optimizer
from torch import nn
import argparse
from argparse import ArgumentParser
from config import cfg
from visualize import get_vislogger
from evaluate import get_evaluator


def train(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Some info
    print('Experiment name:', cfg.exp_name)
    print('Model name:', cfg.model)
    print('Dataset:', cfg.dataset)
    print('Resume:', cfg.resume)
    if cfg.resume:
        print('Checkpoint:', cfg.resume_ckpt if cfg.resume_ckpt else 'last checkpoint')
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)
    
    print('\nLoading data...')
    
    trainloader = get_dataloader(cfg, 'train')
    if cfg.val.ison or cfg.vis.ison:
        valset = get_dataset(cfg, 'val')
        valloader = get_dataloader(cfg, 'val')
    print('Data loaded.')
    
    print('Initializing model...')
    model = get_model(cfg)
    model = model.to(cfg.device)
    print('Model initialized.')
    model.train()
    
    optimizer = get_optimizer(cfg, model)
    
    # Checkpointer will print information.
    checkpointer = Checkpointer(os.path.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    
    start_epoch = 0
    start_iter = 0
    global_step = 0
    if cfg.resume:
        checkpoint = checkpointer.load(cfg.resume_ckpt, model, optimizer)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step'] + 1
    if cfg.parallel:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)
    
    writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, cfg.exp_name), purge_step=global_step, flush_secs=30)
    metric_logger = MetricLogger()
    vis_logger = get_vislogger(cfg)
    evaluator = get_evaluator(cfg)
    
    print('Start training')
    end_flag = False
    for epoch in range(start_epoch, cfg.train.max_epochs):
        if end_flag: break
        start = time.perf_counter()
        for i, data in enumerate(trainloader):
            end = time.perf_counter()
            data_time = end - start
            start = end
            
            imgs, *_ = [d.to(cfg.device) for d in data]
            model.train()
            loss, log = model(imgs, global_step)
            # If you are using DataParallel
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            if cfg.train.clip_norm:
                clip_grad_norm_(model.parameters(), cfg.train.clip_norm)
            optimizer.step()
            
            end = time.perf_counter()
            batch_time = end - start
            
            metric_logger.update(data_time=data_time)
            metric_logger.update(batch_time=batch_time)
            metric_logger.update(loss=loss.item())
            
            if (global_step + 1) % cfg.train.print_every == 0:
                start = time.perf_counter()
                log.update(loss=metric_logger['loss'].median)
                vis_logger.model_log_vis(writer, log, global_step + 1)
                end = time.perf_counter()
                device_text = cfg.device_ids if cfg.parallel else cfg.device
                print(
                    'exp: {}, device: {}, epoch: {}, iter: {}/{}, global_step: {}, loss: {:.2f}, batch time: {:.4f}s, data time: {:.4f}s, log time: {:.4f}s'.format(
                        cfg.exp_name, device_text, epoch + 1, i + 1, len(trainloader), global_step + 1,
                        metric_logger['loss'].median,
                        metric_logger['batch_time'].avg, metric_logger['data_time'].avg, end - start))
            
            if (global_step + 1) % cfg.train.save_every == 0:
                start = time.perf_counter()
                checkpointer.save(model, optimizer, epoch, global_step)
                print('Saving checkpoint takes {:.4f}s.'.format(time.perf_counter() - start))
            
            if (global_step + 1) % cfg.vis.vis_every == 0 and cfg.vis.ison:
                print('Doing visualization...')
                start = time.perf_counter()
                vis_logger.train_vis(model, valset, writer, global_step,
                                     cfg.vis.indices, cfg.device,
                                     cond_steps=cfg.vis.cond_steps,
                                     fg_sample=cfg.vis.fg_sample,
                                     bg_sample=cfg.vis.bg_sample,
                                     num_gen=cfg.vis.num_gen)
                print('Visualization takes {:.4f}s.'.format(time.perf_counter() - start))
                
            if (global_step + 1) % cfg.val.val_every == 0 and cfg.val.ison:
                print('Doing evaluation...')
                start = time.perf_counter()
                evaluator.train_eval(evaluator, os.path.join(cfg.evaldir, cfg.exp_name), cfg.val.metrics, cfg.val.eval_types,
                                     cfg.val.intervals, cfg.val.cond_steps, model, valset, valloader,
                                     cfg.device, writer, global_step,
                                     [model, optimizer, epoch, global_step], checkpointer)
                print('Evaluation takes {:.4f}s.'.format(time.perf_counter() - start))
            
            start = time.perf_counter()
            global_step += 1
            if global_step >= cfg.train.max_steps:
                end_flag = True
                break
