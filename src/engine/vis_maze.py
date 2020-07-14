import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from utils import MetricLogger, Checkpointer, TensorAccumulator
from dataset import get_dataloader, get_dataset
from model import get_model
from solver import get_optimizer
from torch import nn
from visualize.dataset_vis_tools import maze_vis
from evaluate import get_evaluator
from attrdict import AttrDict
from tqdm import tqdm
from utils import transform_tensors
import h5py


@torch.no_grad()
def vis_maze(cfg, genlen=50, num_gen=4, cond_steps=5):
    assert cfg.val.mode == 'val'
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
    
    
    print("Maze...")
    start = time.perf_counter()
    model.eval()
    results = {}

    model_fn = lambda model, imgs: model.generate(imgs, cond_steps=cond_steps, fg_sample=True, bg_sample=False)
    seqs_all = []
    for i, data in enumerate(tqdm(dataloader)):
        data = [d.to(cfg.device) for d in data]
        # (B, T, C, H, W), (B, T, O, 2), (B, T, O)
        imgs, *_ = data
        # (B, T, C, H, W)
        imgs = imgs[:, :genlen]
        B, T, C, H, W = imgs.size()


        seqs = [list() for i in range(B)]
        for j in range(num_gen):
            log = model_fn(model, imgs)
            log = AttrDict(log)
            # (B, T, C, H, W)
            recon = log.recon
            for b in range(B):
                seqs[b].append((recon[b].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8))
        
        # (N, G, T, 3, H, W)
        seqs_all.extend(seqs)
    frames = maze_vis(seqs_all)
    resultdir = os.path.join(cfg.resultdir, cfg.exp_name)
    os.makedirs(resultdir, exist_ok=True)
    path = os.path.join(resultdir, 'maze.gif')
    make_gif(frames, path)
        


def make_gif(frames, path, fps=5):
    """
    Show some gif.

    :param videos: (N, T, H, W, 3)
    :param path: directory
    """
    import moviepy.editor as mpy
    clip = mpy.ImageSequenceClip(list(frames), fps=fps)
    clip.write_gif(path, fps=fps)


if __name__ == '__main__':
    pass

