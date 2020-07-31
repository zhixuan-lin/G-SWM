import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import pickle
import os
import numpy as np
import torch
from torch import nn
import numbers
import time

class TensorAccumulator:
    """ concat tensors with optional (right) padding if shapes don't match """
    def __init__(self, pad=False, pad_value=0):
        self.items = {}
        self.pad = pad
        self.pad_value = pad_value

    def add(self, key, value):
        def _get_item_with_padding(item, item_shape, max_shape):
            diff = max_shape - item_shape
            zeros = torch.zeros_like(diff)

            padding = reversed([(x, y) for x, y, in zip(zeros, diff)])
            padding = [y for x in padding for y in x]
            return F.pad(item, padding, value=self.pad_value)

        if key not in self.items:
            self.items[key] = value.clone()
        else:
            prev_items = self.items[key]
            new_item = value.clone()
            if self.pad:
                prev_shape = torch.tensor(self.items[key].shape[1:])
                new_shape = torch.tensor(value.shape[1:])
                max_shape = torch.max(prev_shape, new_shape)

                if not torch.all(prev_shape == max_shape):
                    prev_items = _get_item_with_padding(prev_items, prev_shape, max_shape)

                if not torch.all(new_shape == max_shape):
                    new_item = _get_item_with_padding(new_item, new_shape, max_shape)

            self.items[key] = torch.cat([prev_items, new_item], dim=0)

    def get(self, key, default=None):
        return self.items.get(key, default)
class Timer:
    def __init__(self):
        from collections import OrderedDict
        self.start = time.perf_counter()
        self.times = OrderedDict()
    def check(self, name):
        self.times[name] = time.perf_counter() - self.start
        self.start = time.perf_counter()

class SmoothedValue:
    """
    Record the last several values, and return summaries
    """
    
    def __init__(self, maxsize=20):
        self.values = deque(maxlen=maxsize)
        self.count = 0
        self.sum = 0.0
    
    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.values.append(value)
        self.count += 1
        self.sum += value
    
    @property
    def median(self):
        return np.median(np.array(self.values))
    
    @property
    def avg(self):
        return np.mean(self.values)
    
    @property
    def global_avg(self):
        return self.sum / self.count


class MetricLogger:
    def __init__(self):
        self.values = defaultdict(SmoothedValue)
    
    def update(self, **kargs):
        for key, value in kargs.items():
            self.values[key].update(value)
    
    def __getitem__(self, key):
        return self.values[key]
    
    def __setitem__(self, key, item):
        self.values[key].update(item)

def spatial_transform(image, z_where, out_dims, inverse=False):
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    theta = torch.zeros(2, 3, device=image.device).repeat(image.shape[0], 1, 1) #! like tf.tile
    # set scaling -> [[s, 0], [0, s]]
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-9) # [B*N]
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-9) # [B*N]
    
    # set translation @ the last column [..., -1]
    theta[:, 0, -1] = z_where[:, 2] if not inverse else - z_where[:, 2] / (z_where[:, 0] + 1e-9)
    theta[:, 1, -1] = z_where[:, 3] if not inverse else - z_where[:, 3] / (z_where[:, 1] + 1e-9)
    # 2. construct sampling grid: In: theta of shape [B*N, 2, 3]
    grid = F.affine_grid(theta, torch.Size(out_dims)) # out_dims target_output_image
    # 3. sample image from grid
    return F.grid_sample(image, grid)

def transform_tensors(x, func):
    """
    Transform each tensor in x using func. We preserve the structure of x.
    Args:
        x: some Python objects
        func: function object

    Returns:
        x: transformed version
    """
    
    if isinstance(x, torch.Tensor):
        return func(x)
    elif isinstance(x, numbers.Number):
        return x
    elif isinstance(x, list):
        return [transform_tensors(item, func) for item in x]
    elif isinstance(x, dict):
        return {k: transform_tensors(v, func) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(transform_tensors(item, func) for item in x)
    else:
        raise TypeError('Non tensor or number object must be either tuple, list or dict, '
                        'but {} encountered.'.format(type(x)))


class Checkpointer:
    def __init__(self, path, max_num=3):
        self.max_num = max_num
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.listfile = os.path.join(path, 'model_list.pkl')
        if not os.path.exists(self.listfile):
            with open(self.listfile, 'wb') as f:
                model_list = []
                pickle.dump(model_list, f)
    
    def save(self, model, optimizer, epoch, global_step):
        if isinstance(model, nn.DataParallel):
            model = model.module
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            # 'iteration': iteration,
            'global_step': global_step
        }
        filename = os.path.join(self.path, 'model_{:09}.pth'.format(global_step + 1))
        
        with open(self.listfile, 'rb+') as f:
            model_list = pickle.load(f)
            if len(model_list) >= self.max_num:
                if os.path.exists(model_list[0]):
                    os.remove(model_list[0])
                del model_list[0]
            model_list.append(filename)
        with open(self.listfile, 'rb+') as f:
            pickle.dump(model_list, f)
        
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
            print(f'Checkpoint has been saved to "{filename}".')
            
    def save_to_path(self, model, optimizer, epoch, global_step, path):
        if isinstance(model, nn.DataParallel):
            model = model.module
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            # 'iteration': iteration,
            'global_step': global_step
        }
        with open(path, 'wb') as f:
            torch.save(checkpoint, f)
            # print(f'Checkpoint has been saved to "{filename}".')
    
    def load(self, path, model, optimizer):
        """
        Return starting epoch
        """
        
        if path == '':
            with open(self.listfile, 'rb') as f:
                model_list = pickle.load(f)
                if len(model_list) == 0:
                    print('No checkpoint found. Starting from scratch')
                    return None
                else:
                    path = model_list[-1]
        
        assert os.path.exists(path), f'Checkpoint {path} does not exist.'
        print('Loading checkpoint from {}...'.format(path))
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint.pop('model'))
        if optimizer:
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        print('Checkpoint loaded.')
        return checkpoint
