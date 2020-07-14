import torch
import math
import numpy as np
import json
from utils import TensorAccumulator
import evaluate.mot as mot
import h5py
import os
import json
from tqdm import tqdm
from attrdict import AttrDict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Evaluator:
    def __init__(self):
        pass

    def save_best(self, evaldir, metric_name, value, checkpoint, checkpointer, min_is_better):
        """
        Save the best checkpoint
        Args:
            metric_name: for file name
            value: scalor
            min_is_better: as the name suggests
        """
        metric_file = os.path.join(evaldir, f'best_{metric_name}.json')
        checkpoint_file = os.path.join(evaldir, f'best_{metric_name}.pth')
        
        now = datetime.now()
        log = {
            'name': metric_name,
            'value': float(value),
            'date': now.strftime("%Y-%m-%d %H:%M:%S"),
            'global_step': checkpoint[-1]
        }
        
        if not os.path.exists(metric_file):
            dump = True
        else:
            with open(metric_file, 'r') as f:
                previous_best = json.load(f)
            # In case of 'nan'
            if not math.isfinite(previous_best['value']):
                dump = True
            elif (min_is_better and log['value'] < previous_best['value']) or (
                    not min_is_better and log['value'] > previous_best['value']):
                dump = True
            else:
                dump = False
        if dump:
            with open(metric_file, 'w') as f:
                json.dump(log, f)
            checkpointer.save_to_path(*checkpoint, checkpoint_file)

    
class GSWMEvalBalls(Evaluator):
    
    @staticmethod
    def dump_to_json(iou_summary, euclidean_summary, med_summary, evaldir,
                     model_name, dataset, task, run_num, path_to_checkpoint, exp_name):
        """
        
        Args:
            iou_summary, euclidean_summary, med_summary: as returned by
                metrics_from_file. Can be None if empty
                for med_summary, if not None, make sure 'meds_over_time' is there
            evaldir: directory to save the json file
            model_name: one of ['ours', 'scalor', 'stove', 'silot']
            dataset: one of ['interaction_fixed_fast', 'occlusion_fixed_fast', 'two_layer_fast', 'two_layer_many_inside']
            task: one of ['tracking', 'generation']
            run_num: one of [0, 1, 2, 3, 4]

        Returns:
            nothing
        """
        assert model_name in ['ours', 'scalor', 'stove', 'silot']
        assert dataset in ['balls_interaction', 'balls_occlusion', 'balls_two_layer', 'balls_two_layer_dense']
        assert task in ['tracking', 'generation']
        assert run_num in [0, 1, 2, 3, 4]
        if task == 'generation':
            assert med_summary is not None and 'meds_over_time' in med_summary
            assert len(med_summary['meds_over_time']) == 90
        if task == 'tracking':
            assert iou_summary is not None and 'mota' in iou_summary
            try:
                float(iou_summary['mota'])
            except ValueError:
                print('mota is not an float number')
        
        import json
        import numpy as np
        from collections import OrderedDict
        from datetime import datetime
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        
        file_name = f'{model_name}-{dataset}-{task}-{run_num}.json'
        os.makedirs(evaldir, exist_ok=True)
        path = os.path.join(evaldir, file_name)
        
        things = OrderedDict([
            ('date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ('exp_name', exp_name),
            ('path_to_checkpoint', path_to_checkpoint),
            ('iou_summary', iou_summary),
            ('euclidean_summary', euclidean_summary),
            ('med_summary', med_summary)
        ])
        # if os.path.exists(path):
        #     # print(f'Warning: overriding {path}. If this is not what you intended, exit by Ctrl-C. You have 10 seconds to make the decision.')
        #     print(f'Warning: overriding {path}. Check your model name, dataset, task, run_num.')
        #     print('If this is what you intended, Ctrl-C to exit. Otherwise type <Enter> to proceed.')
        #     print('')
        #     input('Type <Enter> to proceed:')
            # import time; time.sleep(10)
        print(f'Writing to {path}...')
        with open(path, 'w') as f:
            json.dump(things, f, indent=2, cls=NumpyEncoder)
        print(f'Result file dumped to {path}.')
    
    @torch.no_grad()
    def train_eval(self, evaluator, evaldir, metrics, eval_types, intervals, cond_steps, model, dataset, dataloader, device, writer: SummaryWriter, global_step, checkpoint, checkpointer):
        """
        For evaluation during training
        """
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        for eval_type in eval_types:
            if eval_type == 'tracking':
                model_fn = lambda model, imgs: model.track(imgs, discovery_dropout=0)
            elif eval_type == 'generation':
                model_fn = lambda model, imgs: model.generate(imgs, cond_steps=cond_steps, fg_sample=False,
                                                              bg_sample=False)
            else:
                raise ValueError()
    
            print(f'Evaluating {eval_type}...')
            skip = cond_steps if eval_type == 'generation' else 0
            (iou_summary, euclidean_summary, med_summary) = evaluator.evaluate(eval_type, model, model_fn, skip,
                                                                               dataset, dataloader, evaldir,
                                                                               device, metrics)
            if eval_type == 'tracking':
                if 'mot_iou' in metrics:
                    writer.add_scalar('tracking/mota_iou', iou_summary['mota'], global_step)
                    writer.add_scalar('tracking/motp_iou', iou_summary['motp'], global_step)
                    self.save_best(evaldir, 'mot_iou', iou_summary['mota'], checkpoint, checkpointer, min_is_better=False)
                if 'mot_dist' in metrics:
                    writer.add_scalar('tracking/mota_dist', euclidean_summary['mota'], global_step)
                    writer.add_scalar('tracking/motp_dist', euclidean_summary['motp'], global_step)
            else:
                # (T)
                if 'med' in metrics:
                    med = med_summary['meds_over_time']
                    for (low, high) in intervals:
                        this_med = med[low-cond_steps:high-cond_steps].mean()
                        writer.add_scalar(f'generation/med_{low}_{high}', this_med, global_step)
                    writer.add_scalar('generation/med_overall', med.mean(), global_step)
                    self.save_best(evaldir, 'med_overall', med.mean(), checkpoint, checkpointer, min_is_better=True)
                    self.save_best(evaldir, 'med_fisrt_10', med[:10].mean(), checkpoint, checkpointer, min_is_better=True)
                    
    @torch.no_grad()
    def evaluate(self, eval_type, model, model_fn, num_skip_steps, dataset, dataloader, evaldir, device, metrics):
        """
        Real evaluation
        """
        os.makedirs(evaldir, exist_ok=True)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.eval()

        batch_logs = TensorAccumulator()
        # print(len(dataloader))
        for i, data in enumerate(tqdm(dataloader)):
            data = [d.to(device) for d in data]
            # (B, T, C, H, W), (B, T, O, 2), (B, T, O)
            imgs, gt_positions, gt_sizes, gt_ids, gt_in_camera = data
    
            log = model_fn(model, imgs)
            log = AttrDict(log)
            # log = model.generate(imgs, num_steps=num_steps)
    
            # (B, T, C, H, W)
            # batch_logs.add('imgs', imgs)
            # (B, T, N)
            batch_logs.add('ids', log.ids)
            # (B, T, N, 1)
            batch_logs.add('z_pres', log.z_pres)
            # (B, T, N, 4)
            batch_logs.add('z_where', log.z_where)
            # (B, T, 20, 2)
            batch_logs.add('gt_positions', gt_positions)
            # (B, T, 20)
            batch_logs.add('gt_sizes', gt_sizes)
            # (B, T, 20)
            batch_logs.add('gt_ids', gt_ids)
            # (B, T, 20)
            batch_logs.add('gt_in_camera', gt_in_camera)

        # vis_logger.make_vis(model, dataset, writer, global_step, cfg.vis.indices, cfg.device)

        # imgs = batch_logs.get('imgs')[:, num_skip_steps:]
        ids = batch_logs.get('ids')[:, num_skip_steps:]
        z_pres = batch_logs.get('z_pres')[:, num_skip_steps:]
        z_where = batch_logs.get('z_where')[:, num_skip_steps:]
        gt_positions = batch_logs.get('gt_positions')[:, num_skip_steps:]
        gt_sizes = batch_logs.get('gt_sizes')[:, num_skip_steps:]
        gt_ids = batch_logs.get('gt_ids')[:, num_skip_steps:]
        gt_in_camera = batch_logs.get('gt_in_camera')[:, num_skip_steps:]

        # (B, T, N, 4)
        pred_boxes = mot.scalor_pred_to_boxes(z_where)

        eval_file = os.path.join(evaldir, f'{eval_type}.hdf5')
        with h5py.File(eval_file, 'w') as f:
            f.create_dataset('pred', data=pred_boxes.cpu().numpy())
            f.create_dataset('pred_conf', data=z_pres.cpu().numpy())
            f.create_dataset('pred_ids', data=ids.cpu().numpy())
            f.create_dataset('gt_positions', data=gt_positions.cpu().numpy())
            f.create_dataset('gt_sizes', data=gt_sizes.cpu().numpy())
            f.create_dataset('gt_ids', data=gt_ids.cpu().numpy())
            f.create_dataset('gt_in_camera', data=gt_in_camera.cpu().numpy())

        iou_summary, euclidean_summary, med_summary = mot.metrics_from_file(eval_file, metrics)
        model.train()
        
        return iou_summary, euclidean_summary, med_summary
    
    
class GSWMEvalMaze(Evaluator):
    
    @torch.no_grad()
    def evaluate(self, model, dataloader, cond_steps, device, evaldir, exp_name, path_to_checkpoint, num_gen=10):
        model.eval()
        print('Evaluating Maze...')
        # (T,)
        listoflist = []
        for i in range(num_gen):
            num_list = self.eval_maze(model, dataloader, cond_steps, device)
            listoflist.append(num_list)
        listoflist = np.array(listoflist)
        num_mean = listoflist.mean(axis=0)
        num_std = listoflist.std(axis=0)
        
        file_name = f'maze-{exp_name}.json'
        os.makedirs(evaldir, exist_ok=True)
        path = os.path.join(evaldir, file_name)

        from collections import OrderedDict
        things = OrderedDict([
            ('date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ('exp_name', exp_name),
            ('path_to_checkpoint', path_to_checkpoint),
            ('cond_steps', cond_steps),
            ('num_mean', num_mean.tolist()),
            ('num_std', num_std.tolist()),
        ])
        # print(num_mean)
        # print(num_std)
        with open(path, 'w') as f:
            json.dump(things, f, indent=2)
        print('Results dumped to {0}'.format(path))
        model.train()
        
    @torch.no_grad()
    def train_eval(self, evaluator, evaldir, metrics, eval_types, intervals, cond_steps, model, dataset, dataloader, device, writer: SummaryWriter, global_step, checkpoint, checkpointer, num_gen=10):
        
        model.eval()
        print('Evaluating Maze...')
        # (T,)
        listoflist = []
        for i in range(num_gen):
            num_list = self.eval_maze(model, dataloader, cond_steps, device)
            listoflist.append(num_list)
        listoflist = np.array(listoflist)
        num_mean = listoflist.mean(axis=0)
        num_std = listoflist.std(axis=0)
        
    
        file_name = f'{global_step:010}.json'
        evaldir = os.path.join(evaldir, 'train')
        os.makedirs(evaldir, exist_ok=True)
        path = os.path.join(evaldir, file_name)
    
        from collections import OrderedDict
        things = OrderedDict([
            ('date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ('global_step', global_step),
            ('cond_steps', cond_steps),
            ('num_mean', num_mean.tolist()),
            ('num_std', num_std.tolist()),
        ])
        # print(num_list)
        with open(path, 'w') as f:
            json.dump(things, f, indent=2)
        print('Results dumped to {0}'.format(path))
        intervals = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        for i in range(len(intervals) - 1):
            if intervals[i + 1] <= len(num_mean):
                avg = np.mean(num_mean[intervals[i]:intervals[i+1]])
                std_avg = np.mean(num_std[intervals[i]:intervals[i+1]])
                writer.add_scalar('train/agent_num_mean_{}_{}'.format(intervals[i], intervals[i+1]), avg, global_step)
                writer.add_scalar('train/agent_num_std_{}_{}'.format(intervals[i], intervals[i+1]), std_avg, global_step)
        
        writer.add_scalar('train/agent_num_avg', np.mean(num_mean), global_step)
        
        import matplotlib.pyplot as plt
        f, ax = plt.subplots()
        ax.plot(num_mean)
        ax.fill_between(np.arange(len(num_mean)), num_mean-num_std, num_mean+num_std, alpha=0.3, facecolor='b')

        ax.set_ylim(0, 4)
        writer.add_figure('train/agent_num_plot', f, global_step)
        self.save_best(evaldir, 'num_mean', np.mean(num_mean), checkpoint, checkpointer, min_is_better=False)
        model.train()
        

    @torch.no_grad()
    def eval_maze(self, model, dataloader, cond_steps, device):
        """
        
        Args:
            model: GSWM
            dataset: Maze dataset
            device: device

        Returns:
            num_list, of length (T), where T is the sequence length of the dataset
        """
        num_list = []
        model.eval()
        batch_logs = TensorAccumulator()
        from tqdm import tqdm
        for i, (seq, grid) in enumerate(tqdm(dataloader)):
            seq = seq.to(device)
            logs = model.generate(seq, cond_steps, fg_sample=True, bg_sample=False)
            logs = AttrDict(logs)
            # (B, T, N, 1)
            batch_logs.add('z_pres', logs.z_pres)
            # (B, T, N, 4)
            batch_logs.add('z_where', logs.z_where)
            # (B, H, W)
            batch_logs.add('grid', grid)
            
        z_where = batch_logs.get('z_where')
        z_pres = batch_logs.get('z_pres')
        grid = batch_logs.get('grid')
        B = grid.shape[0]
        for i in range(B):
            # (T,), _
            num, inside = self.compute_num_corridor_scalor(z_where[i].cpu().numpy(), z_pres[i].cpu().numpy(), grid[i].numpy())
            num_list.append(num)
        
        # (T, B)
        num_list = np.stack(num_list, axis=-1)
        assert num_list.shape[1] == B
        # (T,)
        num_list = num_list.mean(axis=-1)
        
        model.train()
        
        return num_list
        
    
    def compute_num_corridor_scalor(self, z_where, z_pres, grid):
        """
        Compute the number of agents that is inside the corridor
        Args:
            z_where: (T, N, 4), normalized to (-1, 1), in (s, s, x, y) order
            z_pres: (T, N, 1)
            grid: (H, W)

        Returns:
            num: (T,), indices
        """
        z_pres = z_pres > 0.5
        # (-1, 1) -> (0, 1)
        z_where = (z_where + 1) / 2
        # (T, N, 2)
        z_where = z_where[..., 2:]
        # (T, N, 2)
        traj = z_where[:, z_pres[0, :, 0]]
        assert traj.shape[1] == np.sum(z_pres[0])
        
        return self.compute_num_corridor(traj, grid)
        
    def compute_num_corridor(self, trajectory, grid):
        """
        
        Args:
            trajectory: (T, N, 2), normalized to (0, 1), in (x, y) order.
            grid: (Hg, Wg), binary
            img_size: (H, W)
            threshold: in pixels
        Returns:
            num: (T,), integers
        """
        T, N, _ = trajectory.shape
        inside_list = []
        last = np.ones(N, dtype=np.bool)
        for t in range(T):
            is_inside = self.is_inside_corridor(trajectory[t], grid)
            # Once the agent leaves the corridor, we no longer count it in following frames
            is_inside = is_inside & last
            inside_list.append(is_inside)
            last = is_inside
            
        num_list = [np.sum(x) for x in inside_list]
        
        return np.array(num_list), np.array(inside_list)
        
    def is_inside_corridor(self, pos, grid):
        """
        
        Args:
            pos: (N, 2), normalized to (0, 1). (x, y)
            grid: (Hg, Wg). binary. 1 denotes corridor
            img_size: H=W
            threshold: in pixels
        Returns:
            is_inside: (N), binary
        """
        Hgrid, Wgrid = grid.shape
        N = pos.shape[0]
        assert Hgrid == Wgrid
        # Use position as grid index
        # (N, 2), in (0, Hgrid)
        pos_index = pos * Hgrid
        # (N, 2), order (x, y) -> (y, x)
        pos_index = pos_index[:, ::-1]
        # Round to integers
        pos_index = np.floor(pos_index).astype(np.int)
        
        # only consider legal indices
        # (N, 2)
        legal = (0 <= pos_index) & (pos_index <= Hgrid - 1)
        # (N, 2) -> (N)
        legal = legal[:, 0] & legal[:, 1]
        # (N, 2)
        pos_index = pos_index[legal]
        # (N,)
        is_inside_legal = grid[pos_index[:, 0], pos_index[:, 1]]
        
        # (N,)
        is_inside = np.zeros(N, dtype=np.bool)
        is_inside[legal] = is_inside_legal
        
        return is_inside
        
        
if __name__ == '__main__':
    a = GSWMEvalMaze()
    T = 10
    N = 10
    pos = np.random.rand(T, N, 2)
    grid = np.array([[1, 0, 1],
                     [0, 0, 0],
                     [1, 1, 1]])
    a = GSWMEvalMaze()
    num_list, inside_list = a.compute_num_corridor(pos, grid)
    print(num_list)
    
    def draw_results(pos, grid, num, is_inside):
        img_size = 64
        img_shape = (img_size,) * 2
        import cv2
        import matplotlib.pyplot as plt
        grid_image = ((1 - grid) * 255).astype(np.uint8)
        # (H, W)
        grid_image = cv2.resize(grid_image, (img_shape), interpolation=cv2.INTER_NEAREST)
        plt.imshow(grid_image, cmap='gray')
        pos = pos * img_size
        for i, xy in enumerate(pos):
            plt.text(xy[0], xy[1], '{}'.format(is_inside[i] == 1), c='red')
        plt.title(f'{num}')
        plt.show()
        
    for t in range(T):
        draw_results(pos[t], grid, num_list[t], inside_list[t])
    
    # draw_results(pos, grid)
    
