from engine.config import get_config
from engine.train import train
from engine.vis_maze import vis_maze
from engine.vis_3d import vis_3d
from engine.vis_balls import vis_balls
from engine.eval_balls import eval_balls
from engine.eval_maze import eval_maze

if __name__ == '__main__':

    task_dict = {
        'train': train,
        'eval_balls': eval_balls,
        'vis_maze': vis_maze,
        'vis_balls': vis_balls,
        'vis_3d': vis_3d,
        'eval_maze': eval_maze,
    }
    cfg, task = get_config()
    assert task in task_dict
    task_dict[task](cfg)


