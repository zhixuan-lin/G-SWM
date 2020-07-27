import os.path as osp
from torch.utils.data import DataLoader

__all__ = ['get_dataset', 'get_dataloader']


def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    if 'BALLS' in cfg.dataset:
        from .balls import Balls
        if cfg.dataset == 'BALLS_INTERACTION':
            return Balls(cfg.dataset_roots.BALLS_INTERACTION, mode)
        if cfg.dataset == 'BALLS_OCCLUSION':
            return Balls(cfg.dataset_roots.BALLS_OCCLUSION, mode)
        if cfg.dataset == 'BALLS_TWO_LAYER':
            return Balls(cfg.dataset_roots.BALLS_TWO_LAYER, mode)
        if cfg.dataset == 'BALLS_TWO_LAYER_DENSE':
            return Balls(cfg.dataset_roots.BALLS_TWO_LAYER_DENSE, mode)
    if cfg.dataset == 'OBJ3D':
        from .obj3d import Obj3D
        return Obj3D(cfg.dataset_roots.OBJ3D, mode)
    if cfg.dataset == 'MAZE':
        from .maze import Maze
        return Maze(root=cfg.dataset_roots.MAZE, mode=mode)
    if cfg.dataset == 'SINGLE_BALL':
        from .single_ball import SingleBall
        root = osp.join(cfg.dataset_roots.SINGLE_BALL)
        return SingleBall(root, mode)
    if cfg.dataset == 'ROBOT_OBJ': # TODO (cheolhui); modify this.
        from .robot_obj import RobotObj
        return RobotObj(cfg.dataset_roots.ROBOT_OBJ, mode)
    else:
        raise ValueError(f'Dataset "{cfg.dataset}" not defined in dataset.__init__.py')


def get_dataloader(cfg, mode):
    assert mode in ['train', 'val', 'test']
    
    batch_size = getattr(cfg, mode).batch_size
    shuffle = True if mode == 'train' else False
    num_workers = getattr(cfg, mode).num_workers
    
    dataset = get_dataset(cfg, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader
