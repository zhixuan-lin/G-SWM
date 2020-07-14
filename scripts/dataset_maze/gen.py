import argparse
from mazelib import Maze
import cv2
import os
import os.path as osp
from tqdm import tqdm
import h5py
import numpy as np
import random
from mazelib.generate.Kruskal import Kruskal
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
import imageio

SQUARE = np.full((16, 16, 3), 0, dtype=np.uint8)
parent_dir = osp.dirname(__file__)
CIRCLE = imageio.imread(osp.join(parent_dir, 'shapes/circle16.png'))
SHAPE = SQUARE

COLORS = np.array([
    # [255, 255, 255], # white
    [255, 0, 0], # red
    [0, 255, 0], # lime
    [0, 0, 255], # blue
    [255, 255, 0], # yellow
    [0, 255, 255], # cyan
    [255, 0, 255], # magenta
]) / 255.0

DIRS = {
    'left': np.array((0, -1)),
    'right': np.array((0, 1)),
    'up': np.array((-1, 0)),
    'down': np.array((1, 0)),
}
TURN_LEFT = {
    'left': 'down',
    'down': 'right',
    'right': 'up',
    'up': 'left'
}
TURN_RIGHT = {v: k for k, v in TURN_LEFT.items()}

# CIRCLE = imageio.imread('shapes/circle16.png')

def random_trajectory(grid, length):
    yx = random.choice(list(zip(*grid.nonzero())))
    trajectory = []
    trajectory.append(yx)
    direction = random.choice(get_valid_directions(grid, yx))
    for i in range(length - 1):
        yx = yx + DIRS[direction]
        trajectory.append(yx)
        direction = get_next_direction(grid, direction, yx)
        
    return trajectory

def is_valid(grid, yx):
    y, x = yx
    H, W = grid.shape
    if 0 <= y < H and 0 <= x < W and grid[y, x] == 1:
        return True

def get_valid_directions(grid, yx):
    yx = np.array(yx)
    return [d for d in DIRS if is_valid(grid, yx + DIRS[d])]

def get_next_direction(grid, current_dir, yx):
    valid_dirs = get_valid_directions(grid, yx)
    valid_dirs = {current_dir, TURN_LEFT[current_dir], TURN_RIGHT[current_dir]} & set(valid_dirs)
    return random.choice(list(valid_dirs))

    
def render(grid, trajectories):
    # (H, W, 3)
    grid = np.broadcast_to(grid[..., np.newaxis], grid.shape + (3,)).astype(np.float)
    grid = 1.0 - grid
    # grid = np.zeros(grid.shape)
    grids = []
    colors = COLORS[np.random.choice(len(COLORS), size=len(trajectories), replace=False)]
    # colors = random.choices(COLORS, k=len(trajectories))
    for points in zip(*trajectories):
        this = grid.copy()
        for i, (y, x) in enumerate(points):
            this[y, x] = colors[i]
        grids.append(this)
        
    return grids

def draw_shape(canvas, position, size, shape, color):
    """
    Draw a shape

    :param canvas: (H, W, 3), uint8
    :param position: (x, y), in pixel space
    :param size: integer, in pixel
    :param shape: image, (H, W, 3), uint8
    :return:
    """
    H, W, _ = canvas.shape
    h, w, *_ = shape.shape
    shape_corners = np.float32([[0, 0], [0, h], [w, 0]])
    x_min = position[0] - size
    y_min = position[1] - size
    x_max = position[0] + size
    y_max = position[1] + size
    shape = 255 - shape
    shape_corners_transformed = np.float32([[x_min, y_min], [x_min, y_max], [x_max, y_min]])
    M = cv2.getAffineTransform(shape_corners, shape_corners_transformed)
    # transformed = cv2.warpAffine(shape, M, (W, H), flags=cv2.INTER_NEAREST)
    transformed = cv2.warpAffine(shape, M, (W, H), flags=cv2.INTER_LINEAR)
    # transformed = cv2.warpAffine(shape, M, (W, H))
    # import ipdb;ipdb.set_trace()
    transformed = (transformed / 255.)
    # if len(transformed.shape) == 2:
    #     transformed = transformed[..., None]
    canvas = canvas * (1 - transformed) + color * transformed
    return canvas

def smooth_render(grid, trajectories, render_size, img_size, shape, inter):
    # (H, W, 3)
    grid = np.broadcast_to(grid[..., np.newaxis], grid.shape + (3,)).astype(np.float)
    grid = 1.0 - grid
    
    
    
    corridor_width = render_size / (grid.shape[0])
    size = corridor_width / 2
    h, w = (render_size,) * 2
    grid = resize(grid, (h, w), order=0)
    # grid = cv2.resize(grid, (h, w), interpolation=cv2.INTER_NEAREST)
    # grid = np.zeros(grid.shape)
    grids = []
    colors = COLORS[np.random.choice(len(COLORS), size=len(trajectories), replace=False)]
    # colors = random.choices(COLORS, k=len(trajectories))
    # import ipdb; ipdb.set_trace()
    trajectories = (np.array(trajectories) + 0.5) * corridor_width
    trajectories = interpolate(trajectories, inter)
    for points in zip(*trajectories):
        this = grid.copy()
        for i, (y, x) in enumerate(points):
            # this[y, x] = colors[i]
            # In (0, 1)
            this = draw_shape(this, (x, y), size, shape, colors[i])
        
        # Resize to image size
        this = (this * 255).astype(np.uint8)
        this = cv2.resize(this, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        # this = cv2.resize(this, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        grids.append(this)
    
    return grids

def smooth_render2(grid, trajectories, render_size, img_size, shape, inter):
    # (H, W, 3)
    grid = np.broadcast_to(grid[..., np.newaxis], grid.shape + (3,)).astype(np.float)
    grid = 1.0 - grid
    
    
    
    corridor_width = img_size / (grid.shape[0])
    size = corridor_width / 2
    h, w = (img_size,) * 2
    grid = resize(grid, (h, w), order=0)
    # grid = cv2.resize(grid, (h, w), interpolation=cv2.INTER_NEAREST)
    # grid = np.zeros(grid.shape)
    grids = []
    colors = COLORS[np.random.choice(len(COLORS), size=len(trajectories), replace=False)]
    # colors = random.choices(COLORS, k=len(trajectories))
    # import ipdb; ipdb.set_trace()
    trajectories = (np.array(trajectories) + 0.5) * corridor_width
    trajectories = interpolate(trajectories, inter)
    for points in zip(*trajectories):
        this = grid.copy()
        for i, (y, x) in enumerate(points):
            # this[y, x] = colors[i]
            # In (0, 1)
            this = draw_shape(this, (x, y), size, shape, colors[i])

        # Resize to image size
        this = (this * 255).astype(np.uint8)
        # this = cv2.resize(this, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        # this = cv2.resize(this, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        grids.append(this)
    
    return grids

def interpolate(trajectories, n):
    """
    
    Args:
        trajectories:  (N, T, 2)

    Returns:

    """
    N, T, _ = trajectories.shape
    new = np.zeros((N, T + (T - 1) * n, 2))
    for t in range(T - 1):
        # (N, 2)
        start = trajectories[:, t]
        end = trajectories[:, t + 1]
        diff = (end - start) / (n + 1)
        for i in range(n + 1):
            new[:, (n + 1) * t + i] = start + i * diff
    new[:, -1] = trajectories[:, -1]
    
    return new
    
        

def make_maze(grid_size, num_objs, length, img_size, inter, render_resize_factor):
    """
    
    Args:
        grid_size: G
        num_objs: N
        colors: list of length N, in range (0, 1)

    Returns:
        imgs: (T, H, W, 3), uint8
        grid: (H, W), binary
        trajs: (T, N, 2), 2 indices grid
    """
    m = Maze()
    m.generator = Kruskal(grid_size, grid_size)
    m.generate()
    remove_deadend(m.grid)
    # uint8, binary
    grid = m.grid
    # Generate trajectories
    trajs = []
    # import ipdb; ipdb.set_trace()
    for i in range(num_objs):
        traj = random_trajectory(m.grid, length)
        trajs.append(traj)
        
    # SQUARE = np.full((16, 16, 3), 0, dtype=np.uint8)
    # CIRCLE = imageio.imread('shapes/circle16.png')
    render_size  = (m.grid.shape[0]) * render_resize_factor
    # render_size = img_size
    # render_size = img_size
    if SHAPE is CIRCLE:
        imgs = smooth_render2(grid, trajs, render_size, img_size, SHAPE, inter)
    else:
        imgs = smooth_render(grid, trajs, render_size, img_size, SHAPE, inter)
    
    imgs, grid, trajs = (np.array(x) for x in [imgs, grid, trajs])
    # (N, T, 2) -> (T, N, 2)
    trajs = trajs.transpose(1, 0, 2)
    return imgs, grid, trajs
    

def show_video(videos, figsize=(10, 10), fps=5):
    # plt.figure(figsize=figsize)
    
    for img in videos:
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])
        plt.pause(1. / fps)
        plt.close()
    
def showPNG(grid):
    """Generate a simple image of the maze."""
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.pause(0.1)
    plt.close()
    # plt.show()
    
    

def remove_deadend(grid):
    H, W = grid.shape
    
    for y in range(H):
        for x in range(W):
            if grid[y][x]:
                good, bad = get_neighbors(grid, (y,x))
                # print(len(good), len(bad))
                if len(good) <= 1:
                    y_wall, x_wall = random.choice(bad)
                    grid[y_wall][x_wall] = 1
                    
                
    
def get_neighbors(grid, yx):
    neighbors = _get_neighbors(grid, yx)
    good, bad = [], []
    for (y,x) in neighbors:
        if grid[y][x]:
            good.append((y, x))
        else:
            bad.append((y, x))
            
    return good, bad
    
def _get_neighbors(grid, yx):
    H, W = grid.shape
    y, x = yx
    neighbors = []
    if y > 0: neighbors.append((y - 1, x))
    if y < H - 1: neighbors.append((y + 1, x))
    if x > 0: neighbors.append((y, x - 1))
    if x < W - 1: neighbors.append((y, x + 1))
    return neighbors


def make_data(num_data, grid_size, obj_min, obj_max, length, img_size, inter, render_resize_factor):
    videos = []
    grids = []
    trajs = []
    pres = []
    for i in tqdm(range(num_data)):
        num_objs = np.random.randint(obj_min, obj_max)
        imgs, grid, traj = make_maze(grid_size, num_objs, length, img_size, inter, render_resize_factor)
        
        # (T, N, 2) -> (T, MAX, 2)
        presence = np.zeros((length, obj_max))
        presence[:, :traj.shape[1]] = 1.0
        traj = np.pad(traj, [(0, 0), (0, obj_max - traj.shape[1]), (0, 0)], mode='constant')
        videos.append(imgs)
        grids.append(grid)
        trajs.append(traj)
        pres.append(presence)
    
    # (N, T, H, W, 3)
    videos = np.stack(videos, axis=0)
    # (N, H, W)
    grids = np.stack(grids, axis=0)
    trajs = np.stack(trajs, axis=0)
    pres = np.stack(pres, axis=0)
    
    return videos, grids, trajs, pres

def dump_data(videos, grids, trajs, pres, output_path):
    with h5py.File(output_path, mode='w') as f:
        f.create_dataset('imgs', data=videos)
        f.create_dataset('grids', data=grids)
        f.create_dataset('trajs', data=trajs)
        f.create_dataset('pres', data=pres)
        
        
def show_images(path, indices):
    with h5py.File(path, 'r') as f:
        imgs = f['imgs']
        for i in indices:
            seq = imgs[i]
            show_video(seq)


def make_gif(videos, path='./gif', num=10, fps=5):
    """
    Show some gif.

    :param videos: (N, T, H, W, 3)
    :param path: directory
    """
    import moviepy.editor as mpy
    os.makedirs(path, exist_ok=True)
    assert num <= videos.shape[0]
    for i in range(num):
        clip = mpy.ImageSequenceClip(list(videos[i]), fps=fps)
        clip.write_gif(os.path.join(path, f'{i}.gif'), fps=fps)


if __name__ == '__main__':
    
    # from combine_images import combine_images
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--output_dir', type=str, default='./data', help='output directory')
    parser.add_argument('--img_size', type=int, default=64)
    # parser.add_argument('--img_h', type=int, default=64)
    parser.add_argument('--grid_size', type=int, default=6, help='Actually size will be [grad_size] * 2 + 1')
    parser.add_argument('--obj_min', type=int, default=3, help='minimum number of objects')
    parser.add_argument('--obj_max', type=int, default=5, help='maximum number of objects')
    parser.add_argument('--interpolate', type=int, default=0, help='how many points to interpolate positions')
    parser.add_argument('--seq_len', type=int, default=50, help='sequence length')
    parser.add_argument('--split', type=int, nargs='+', default=[10000, 100, 100], help='train, val, test split')
    parser.add_argument('--render_resize_factor', type=int, default=1)
    parser.add_argument('--shape', type=str, default='square')
    
    args = parser.parse_args()
    
    SHAPE = dict(circle=CIRCLE, square=SQUARE)[args.shape]
    os.makedirs(args.output_dir, exist_ok=True)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    for i, name in enumerate(['train', 'val', 'test']):
        split_num = args.split[i]
        videos, grids, trajs, pres = make_data(
            num_data=split_num,
            grid_size=args.grid_size,
            obj_min=args.obj_min,
            obj_max=args.obj_max,
            length=args.seq_len,
            img_size=args.img_size,
            inter=args.interpolate,
            render_resize_factor=args.render_resize_factor,
        )
        path = osp.join(args.output_dir, f'{name}.hdf5')
        dump_data(videos, grids, trajs, pres, path)
    

