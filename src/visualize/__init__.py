__all__ = ['get_vislogger']

from .gswm_vis import GSWMVis
def get_vislogger(cfg):
    return GSWMVis()
