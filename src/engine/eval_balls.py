import os
import numpy as np
import time
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from utils import MetricLogger, Checkpointer
from dataset import get_dataloader, get_dataset
from model import get_model
from solver import get_optimizer
from torch import nn
from evaluate import get_evaluator
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def eval_balls(cfg):
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
    ###
    
    evaldir = os.path.join(cfg.evaldir, cfg.exp_name)
    os.makedirs(evaldir, exist_ok=True)
    print("Evaluating...")
    start = time.perf_counter()
    model.eval()
    results = {}
    for eval_type in cfg.val.eval_types:
        if eval_type == 'tracking':
            model_fn = lambda model, imgs: model.track(imgs, discovery_dropout=0)
        elif eval_type == 'generation':
            model_fn = lambda model, imgs: model.generate(imgs, cond_steps=cfg.val.cond_steps, fg_sample=False, bg_sample=False)
            
        print(f'Evaluating {eval_type}...')
        skip = cfg.val.cond_steps if eval_type == 'generation' else 0
        (iou_summary, euclidean_summary, med_summary) = evaluator.evaluate(eval_type, model, model_fn, skip, dataset, dataloader, evaldir, cfg.device, cfg.val.metrics)
        # print('iou_summary: {}'.format(iou_summary))
        # print('euclidean_summary: {}'.format(euclidean_summary))
        # print('med_summary: {}'.format(med_summary))
        
        results[eval_type] = [iou_summary, euclidean_summary, med_summary]

    for eval_type in cfg.val.eval_types:
        evaluator.dump_to_json(*results[eval_type], evaldir,
                               'ours', cfg.dataset.lower(), eval_type, cfg.run_num, cfg.resume_ckpt, cfg.exp_name)
    print('Evaluation takes {}s.'.format(time.perf_counter() - start))
    
    
    # Plot figure
    if 'generation' in cfg.val.eval_types and 'med' in cfg.val.metrics:
        med_list = results['generation'][-1]['meds_over_time']
        assert len(med_list) == 90
        steps = np.arange(10, 100)
        f, ax = plt.subplots()
        ax: plt.Axes
        ax.plot(steps, med_list)
        ax.set_xlabel('Time step')
        ax.set_ylim(0.0, 0.6)
        ax.set_ylabel('Position error')
        ax.set_title(cfg.exp_name)
        plt.savefig(os.path.join(evaldir, 'plot_balls.png'))
        print('Plot saved to', os.path.join(evaldir, 'plot_balls.png'))
        print('MED summed over the first 10 prediction steps: ', sum(med_list[:10]))
    


if __name__ == '__main__':
    pass

