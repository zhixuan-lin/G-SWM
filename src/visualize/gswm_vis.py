import torch
import os
import os.path as osp
import numpy as np

import torchvision
from .utils import draw_boxes, figure_to_numpy, make_gif
from utils import transform_tensors
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import Subset, DataLoader
from PIL import Image, ImageDraw
from attrdict import AttrDict
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class GSWMVis:
    def __init__(self):
        pass
    
    def clean_log(self, log, num):
        log = transform_tensors(log, lambda x: x.cpu().detach())
        def take_batch(x):
            if isinstance(x, torch.Tensor) and x.size(0) > 1: return x[:num]
            else: return x
        log = transform_tensors(log, take_batch)
        log = AttrDict(log)
        return log
    
    @torch.no_grad()
    def show_tracking(self, model, dataset, indices, device):
        """
        
        Args:
            imgs: (B, T, 3, H, W)
            recons: (B, T, 3, H, W)
            z_where: [(B, N, 4)] * T
            z_pres: [(B, N, 1)] * T
            ids: [(B, N)] * T

        Returns:
            grid: (3, H, W)
            gif: (B, T, 3, N*H, W)
        """
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.eval()
    
        # data = self.get_batch(dataset, indices, device)
        imgs, ees = self.get_batch(dataset, indices, device) # we get both images and ees to condition on
        # things = model.track(data, discovery_dropout=0.0)
        things = model.track_rob_fg(imgs, ees, discovery_dropout=0.0)
    
        log = self.clean_log(things, len(indices))
    
        B, T, _, H, W = log.fg.size()
    
        fg_boxes = torch.zeros_like(log.fg)
        fg_proposals = torch.zeros_like(log.fg)
        for t in range(T):
            fg_boxes[:, t] = draw_boxes(log.fg[:, t], log.z_where[:, t], log.z_pres[:, t], log.ids[:, t])
            fg_proposals[:, t] = draw_boxes(log.fg[:, t], log.proposal[:, t], log.z_pres[:, t], log.ids[:, t])
    
        # (B, 3T, 3, H, W)
        grid = torch.cat([log.imgs, log.recon, log.fg, fg_boxes, fg_proposals, log.bg], dim=1)
        grid = grid.view(-1, 3, H, W)
        # (3, H, W)
        grid = make_grid(grid, nrow=T, pad_value=1)
        # writer.add_image('tracking/grid', grid, global_step)
    
        # (B, T, 3, 3H, W)
        gif = torch.cat([log.imgs, log.recon, log.fg, fg_boxes, fg_proposals, log.bg], dim=-1)
        # gif = torch.cat([imgs, fg_boxes, fg], dim=-1)
        # for i in range(gif.size(0)):
        #     writer.add_video(f'tracking/video_{i}', gif[i:i+1], global_step)
        model.train()
        return grid, gif
    

    @torch.no_grad()
    def show_generation(self, model, dataset, indices, device, cond_steps, fg_sample, bg_sample, num, action_cond='fg'):
        """
        # TODO (cheolhui): figure out what each bg_sample and fg_sample implies
        Args:
            imgs: (B, T, 3, H, W)
            recons: (B, T, 3, H, W)
            z_where: [(B, N, 4)] * T
            z_pres: [(B, N, 1)] * T
            ids: [(B, N)] * T
            cond_steps
            sample
            num: number of generation samples

        Returns:

        """
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.eval()
        #  data = self.get_batch(dataset, indices, device) 
        imgs, ees = self.get_batch(dataset, indices, device) # TODO (cheolhui): modify the data return
    
        gifs = []
        for i in range(num):
            if i == 0:
                # things = model.generate(data, cond_steps=cond_steps, fg_sample=fg_sample, bg_sample=False)
                if action_cond == 'fg':
                    things = model.generate_rob_fg(imgs, ees, cond_steps=cond_steps, fg_sample=fg_sample, bg_sample=False)
                elif action_cond == 'bg':
                    things = model.generate_rob_bg(imgs, ees, cond_steps=cond_steps, fg_sample=fg_sample, bg_sample=False)
                else:
                    raise ValueError("Currently other than fg/bg is not supported.")
            else:
                # things = model.generate(data, cond_steps=cond_steps, fg_sample=fg_sample, bg_sample=bg_sample)

                if action_cond == 'fg':
                    things = model.generate_rob_fg(imgs, ees, cond_steps=cond_steps, fg_sample=fg_sample, bg_sample=False)
                elif action_cond == 'bg':
                    things = model.generate_rob_bg(imgs, ees, cond_steps=cond_steps, fg_sample=fg_sample, bg_sample=False)
                else:
                    raise ValueError("Currently other than fg/bg is not supported.")
            log = self.clean_log(things, len(indices))
        
            B, T, _, H, W = log.fg.size()
        
            fg_boxes = torch.zeros_like(log.fg)
            for t in range(T):
                fg_boxes[:, t] = draw_boxes(log.fg[:, t], log.z_where[:, t], log.z_pres[:, t], log.ids[:, t])
        
            if i == 0:
                # (B, 3T, 3, H, W)
                grid = torch.cat([log.imgs, log.recon, log.fg, fg_boxes, log.bg], dim=1)
                grid = grid.view(-1, 3, H, W)
                # (3, H, W)
                grid = make_grid(grid, nrow=T, pad_value=1)
                # writer.add_image('generation/grid', grid, global_step)
        
            # (B, T, 3, H, 3W)
            gif = torch.cat([log.imgs, log.recon, log.fg, fg_boxes, log.bg], dim=-1)
            add_boundary(gif[:, cond_steps:])
            gifs.append(gif)
    
        # (B, T, 3, num*H, N*W)
        gif = torch.cat(gifs, dim=-2)
        
        B, T, N, _ = log.z_depth.size()
        # for i in range(gif.size(0)):
        #     writer.add_video(f'generation/video_{i}', gif[i:i+1], global_step)
        model.train()
        return grid, gif
        
    
    @torch.no_grad()
    def model_log_vis(self, writer:SummaryWriter, log, global_step):
        """
        Show whether return in log
        """
        log = self.clean_log(log, 10)

        B, T, *_ = log.imgs.size()
        
        grid = self.make_gswm_grid(
            log.imgs, log.recon, log.fg, log.bg, log.z_where, log.z_pres, log.proposal, log.ids
        )
        
        writer.add_image('train/grid', grid, global_step)
        writer.add_scalar('train/loss', log.loss.mean(), global_step)
        writer.add_scalar('train/elbo', log.elbo.mean(), global_step)
        writer.add_scalar('train/mse', log.mse.mean(), global_step)
        writer.add_scalar('train/loglikelihood', log.loglikelihood.mean(), global_step)
        writer.add_scalar('train/kl', log.kl.mean(), global_step)
        writer.add_scalar('train/kl_pres', log.kl_pres.mean(), global_step)
        writer.add_scalar('train/kl_depth', log.kl_depth.mean(), global_step)
        writer.add_scalar('train/kl_where', log.kl_where.mean(), global_step)
        writer.add_scalar('train/kl_what', log.kl_what.mean(), global_step)
        writer.add_scalar('train/kl_dyna', log.kl_dyna.mean(), global_step)
        writer.add_scalar('train/kl_fg', log.kl_fg.mean(), global_step)
        writer.add_scalar('train/kl_bg', log.kl_bg.mean(), global_step)
        
        
        count = 0
        for t in log.z_pres:
            count += t.sum()
        count /= (B * T)
        writer.add_scalar('train/count', count, global_step)

    def make_gswm_grid(
            self,
            imgs,
            recon,
            fg,
            bg,
            z_where,
            z_pres,
            proposal,
            ids
    ):
        """
        
        Args:
            imgs: (B, T, 3, H, W)
            recons: (B, T, 3, H, W)
            z_where: [(B, N, 4)] * T
            z_pres: [(B, N, 1)] * T
            proposal: [(B, N, 4)] * T
            ids: [(B, N)] * T

        Returns:

        """
        B, T, _, H, W = fg.size()
        
        fg_boxes = torch.zeros_like(fg)
        fg_proposals = torch.zeros_like(fg)
        for t in range(T):
            fg_boxes[:, t] = draw_boxes(fg[:, t], z_where[:, t], z_pres[:, t], ids[:, t])
            fg_proposals[:, t] = draw_boxes(fg[:, t], proposal[:, t], z_pres[:, t], ids[:, t])
            
        # (B, 3T, 3, H, W)
        grid = torch.cat([imgs, recon, fg_boxes, fg_proposals, bg], dim=1)
        grid = grid.view(B * 5 * T, 3, H, W)
        # (3, H, W)
        grid = make_grid(grid, nrow=T, pad_value=1)
        
        return grid

    @torch.no_grad()
    def train_vis(self, model, dataset, writer: SummaryWriter, global_step, indices, device, cond_steps, fg_sample, bg_sample, num_gen):

        
        grid, gif = self.show_tracking(model, dataset, indices, device)
        writer.add_image('tracking/grid', grid, global_step)
        for i in range(len(gif)):
            writer.add_video(f'tracking/video_{i}', gif[i:i+1], global_step)

        # Generation
        grid, gif = self.show_generation(model, dataset, indices, device, cond_steps, fg_sample=fg_sample, bg_sample=bg_sample,
             num=num_gen, action_cond='bg')
        writer.add_image('generation/grid', grid, global_step)
        for i in range(len(gif)):
            writer.add_video(f'generation/video_{i}', gif[i:i+1], global_step)
            
    def get_batch(self, dataset, indices, device):
        # TODO (cheolhui): resolve shape mistmatch errors
        # For some chosen data we show something here
        dataset = Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        # Do not use multiple GPUs
        # (B, T, 3, H, W)
        data = next(iter(dataloader)) #! create data generator
        data = [d.to(device) for d in data]
        # data, *_ = data
        imgs, ees, *_ = data
        # return data
        return imgs, ees
    
    @torch.no_grad()
    def show_gif(self, model, dataset, indices, device, cond_steps, gen_len, path, fps):
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.eval()
        # get data
        # imgs = self.get_batch(dataset, indices, device)
        imgs, ees = self.get_batch(dataset, indices, device)
        imgs = imgs[:, :gen_len]
        ees = ees[:, :gen_len]
        model_fn = lambda model, imgs: model.generate(imgs, cond_steps=cond_steps, fg_sample=False, bg_sample=False)
        log = model_fn(model, imgs)
        log = AttrDict(log)
        recon = log.recon

        ori = []
        gen = []
        def trans(img):
            # (T, 3, H, W)
            img = img.permute(0, 2, 3, 1).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            return img
        add_boundary(recon[:, cond_steps:])
        for b in range(len(indices)):
            ori.append(trans(imgs[b]))
            gen.append(trans(recon[b]))
            
        frames = []
        heights = [1] * 2
        widths = [1] * (len(indices) + 1)
        h, w = sum(heights), sum(widths)
        f, axes = plt.subplots(2, len(indices) + 1, figsize=(w, h), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
        axes[0][0].text(0.95, 0.5, 'GT', fontdict={'fontsize': 12}, verticalalignment='center', horizontalalignment='right')
        axes[1][0].text(0.95, 0.5, 'GSWM', fontdict={'fontsize': 12}, verticalalignment='center', horizontalalignment='right')
        for ax in axes.ravel():
            ax.axis('off')
        for t in range(gen_len):
            im_list = []
            for b in range(len(indices)):
                im1 = axes[0][b + 1].imshow(ori[b][t])
                im2 = axes[1][b + 1].imshow(gen[b][t])
                im_list.extend((im1, im2))
            frames.append(figure_to_numpy(f))
            for x in im_list: x.remove()
        make_gif(images=frames, path=path, fps=fps)
        
        
    
    

def draw_grid(imgs):
    """
    
    Args:
        imgs: (..., 3, H, W)

    Returns:
    """
    # img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    G = 4
    IMG_H = 64
    IMG_W = 64
    SCALE_FACTOR = 0.8
    l = (-SCALE_FACTOR + 1.0) / 2 * (IMG_H - 1)
    l = int(l)
    r = (SCALE_FACTOR + 1.0) / 2 * (IMG_H - 1)
    r = int(r)
    for pos in torch.linspace(-1, 1, G + 1):
        pos *= SCALE_FACTOR
        pos = (pos + 1.0) / 2 * (IMG_H - 1)
        pos = int(pos)
        imgs[..., :, pos, l:r] = 1.0
        imgs[..., :, l:r, pos] = 1.0

def add_boundary(img, width=1, color=(1.0, 1.0, 0)):
    """

    Args:
        img: (..., 3, H, W)
        width:
        color:

    Returns:

    """
    color = torch.tensor(color, device=img.device, dtype=img.dtype)
    assert color.size() == (3,)
    color = color.view(-1, 1, 1)
    img[..., :, :width, :] = color
    img[..., :, -width:, :] = color
    img[..., :, :, :width] = color
    img[..., :, :, -width:] = color
    
def draw_trajectories(z_where, z_pres):
    """
    
    Args:
        z_where: (B, N, D) * T
        z_pres: (B, N, D) * T

    Returns:
        (B, 3, H, W)
    """
    # Transform z_where to image coordinates
    z_where = [((x[..., 2:] + 1.0) / 2 * 64) for x in z_where]
    T = len(z_where)
    B, N, _ = z_where[0].size()
    images = [Image.new('RGB', (64, 64)) for b in range(B)]
    
    for b in range(B):
        for t in range(T-1):
            for n in range(N):
                # (2,)
                start = tuple(int(x) for x in z_where[t][b][n])
                end = tuple(int(x) for x in z_where[t+1][b][n])
                draw = ImageDraw.Draw(images[b])
                color = int(255 * z_pres[t][b][n])
                draw.line([start, end], fill=(color,)*3)
                
    images = [torchvision.transforms.ToTensor()(x) for x in images]
    images = torch.stack(images, dim=0)
    return images
        

