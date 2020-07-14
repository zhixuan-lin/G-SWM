import imageio
import os.path as osp
import os
import sys
import h5py
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm, trange




def generate_single_trajectory(direction, img_h, img_w, size, n_seg_bottom, n_seg_top):
    assert direction in ['left', 'right']
    top = np.array([img_w // 2, size])
    center = np.array([img_w // 2, img_h // 2])
    bottom_left = np.array([size, img_h - size])
    bottom_right = np.array([img_w - size, img_h - size])
    vec = (center - top) / n_seg_top
    curr = top
    points = [curr]
    for i in range(n_seg_top):
        curr = curr + vec
        points.append(curr)
    final = dict(left=bottom_left, right=bottom_right)[direction]
    vec = (final - center) / n_seg_bottom
    for i in range(n_seg_bottom):
        curr = curr + vec
        points.append(curr)
        
    return points

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
    transformed = cv2.warpAffine(shape, M, (W, H))
    # import ipdb;ipdb.set_trace()
    transformed = (transformed / 255.)
    # if len(transformed.shape) == 2:
    #     transformed = transformed[..., None]
    canvas = canvas * (1 - transformed) + color * transformed
    return canvas

def draw_sequence(points, img_h, img_w, size, shape, color):
    imgs = []
    for point in points:
        canvas = np.zeros((img_h, img_w, 3))
        canvas = draw_shape(canvas, position=point, size=size, shape=shape, color=color)
        imgs.append((canvas * 255).astype(np.uint8))
        
    return imgs

def make_single_seq(probability, img_h, img_w, size, color, shape, n_seg_bottom, n_seg_top):
    """
    Returns:
        imgs: (T, H, W, 3)
        points: (T, 2), in pixels, (x, y)
        direction: integer, 0 for left and 1 for right
    """
    i = np.random.choice([0, 1], p=[probability, 1 - probability])
    direction = ['left', 'right'][i]
    # (N, 2)
    points = generate_single_trajectory(direction, img_h, img_w, size, n_seg_bottom, n_seg_top)
    # (N, 3, H, W)
    imgs = draw_sequence(points, img_h, img_w, size, shape, color)
    
    return np.array(imgs), np.array(points), i

def make_dataset(num_data, probability, img_h, img_w, size, color, shape, n_seg_bottom, n_seg_top):
    seqs = []
    positions = []
    directions = []
    for i in trange(num_data):
        seq, points, direction = make_single_seq(probability, img_h, img_w, size, color, shape, n_seg_bottom, n_seg_top)
        seqs.append(seq)
        positions.append(points)
        directions.append(direction)
    seqs = np.array(seqs)
    positions = np.array(positions)
    directions = np.array(directions)
    T = n_seg_bottom + n_seg_top + 1
    assert seqs.shape == (num_data, T, img_h, img_w, 3)
    assert positions.shape == (num_data, T, 2)
    assert  directions.shape == (num_data,)
    return seqs, positions, directions

def dump_data(seqs, positions, directions, path):
    with h5py.File(path, 'w') as f:
        f.create_dataset('imgs', data=seqs)
        f.create_dataset('positions', data=positions)
        f.create_dataset('directions', data=directions)
    print(f'File dumped to {path}')
    
def show_video(imgs, fps=3):
    """
    
    Args:
        imgs: (B, H, W, 3)
    """
    # import ipdb; ipdb.set_trace()
    f = plt.figure()
    plt.draw()
    ax = f.add_subplot(1, 1, 1)
    ax.set_axis_off()
    for img in imgs:
        im = ax.imshow(img)
        plt.pause(1 / fps)
        im.remove()
    plt.close(f)
    # plt.close(f)
    
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


def show_trajectory(points, img_h, img_w, fill, linewidth):
    canvas = Image.new('RGB', (img_w, img_h))
    draw = ImageDraw.Draw(canvas)
    points = [(x, y) for x, y in points]
    # import ipdb; ipdb.set_trace()
    draw.line(points, fill=fill, width=linewidth)
    # for i in range(len(points) - 1):
    #     draw.line((points[i], points[i + 1]), width=linewidth)
    return np.array(canvas)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./data', help='path to put files')
    parser.add_argument('--img_h', type=int, default=64, help='img height')
    parser.add_argument('--img_w', type=int, default=64, help='img width')
    parser.add_argument('--size', type=float, default=7, help='HALF of object size')
    parser.add_argument('--n_seg_top', type=float, default=4, help='ignore this')
    parser.add_argument('--n_seg_bottom', type=float, default=4, help='ignore this')
    parser.add_argument('--color', type=float, nargs=3, default=(1.0, 1.0, 1.0), help='color, a 3 item list')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--split', type=int, nargs=3, default=[10, 10, 10], help='train, val, test split')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    parent_dir = osp.dirname(__file__)
    CIRCLE = imageio.imread(osp.join(parent_dir, 'shapes', 'circle16.png'))
    
    output_dir = os.path.join(args.output_dir, 'SINGLE_BALL')
    os.makedirs(output_dir, exist_ok=True)
    for name, split in zip(['train', 'val', 'test'], args.split):
        print(f'Making split={name}...')
        seqs, positions, directions = make_dataset(split, 0.5, args.img_h, args.img_w, args.size, args.color, CIRCLE, args.n_seg_bottom, args.n_seg_top)
        path = os.path.join(output_dir, f'{name}.hdf5')
        dump_data(seqs, positions, directions, path)
    # sys.exit()
    

