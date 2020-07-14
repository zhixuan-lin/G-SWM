from torch.optim import Adam, RMSprop
import torch

__all__ = ['get_optimizer']

def get_optimizer(cfg, model: torch.nn.Module):
    optimizer = _get_optimizer(cfg.solver.optim, cfg.solver.lr, model.parameters())
    
    return optimizer
    
def _get_optimizer(name, lr, param):
    optim_class = {
        'Adam': Adam,
        'RMSprop': RMSprop
    }[name]
    
    return optim_class(param, lr=lr)

