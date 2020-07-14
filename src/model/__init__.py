
__all__ = ['get_model']

def get_model(cfg):
    
    model = None
    if cfg.model == 'GSWM':
        from .gswm.gswm import GSWM
        model = GSWM()
    
    return model
