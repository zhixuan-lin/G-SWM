__all__ = ['get_evaluator']


def get_evaluator(cfg):
    if cfg.val.evaluator == 'ball':
        from .gswm_eval import GSWMEvalBalls
        return GSWMEvalBalls()
    elif cfg.val.evaluator == 'maze':
        from .gswm_eval import GSWMEvalMaze
        return GSWMEvalMaze()
