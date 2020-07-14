import imageio
import os
import os.path as osp
import numpy as np
from .utils import combine_images, figure_to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def maze_vis(sequences, cond_steps=5, seq_len=15, num_samples=4):
    """
    
    Args:
        sequences: (N, G, T, H, W, 3), numpy, uint8
        cond_steps: input steps
        seq_len: input steps + generation steps
        num_samples: number of samples to shown

    Returns:
        frames: (T, H, W, 3)
    """


    imgs = []
    for all_images in sequences:
    
        start = all_images[0][:cond_steps]
        start = combine_images(start, factor=0.85, transparency=0.95, interval=1)
        start = start.astype(np.uint8)
    
        allfutures = []
    
        for frame_id in range(len(all_images[0])):
            samples = []
            for sample_id, images in enumerate(all_images):
                sample = combine_images(images[:frame_id + 1], factor=0.85, transparency=0.99, interval=1)
                samples.append(sample)
                if len(samples) >= num_samples:
                    break
            allfuture = combine_images(samples, factor=1.0, transparency=0.5, interval=1)
            allfuture = allfuture.astype(np.uint8)
            allfutures.append(allfuture)
    
        imgs.append([start, samples, allfutures, all_images])

    # start, future + 4 samples
    widths = [1, 1, 0.2] + num_samples * [1]
    heights = [1] * len(sequences)
    gridspec_kw = {'left': 0, 'right': 1, 'top': 0.9, 'bottom': 0, 'wspace': 0.01, 'hspace': 0.01, 'width_ratios': widths, 'height_ratios': heights}
    figsize=(sum(widths), sum(heights))
    # spec = f.add_gridspec(nrows=len(sequences), ncols=2 + num_samples + 1, width_ratios=widths, height_ratios=heights,
    #                       **gridspec_kw)
    # ax = f.subplots(len(seq_indices), 2 + 4 + 1, gridspec_kw=gridspec_kw)
    # for a in ax.ravel():
    #     a.axis('off')
    frames = []
    for frame_id in range(seq_len):
        f, axes = plt.subplots(nrows=len(sequences), ncols=2 + num_samples + 1, gridspec_kw=gridspec_kw, figsize=figsize)
        for x in axes.ravel():
            x.axis('off')
        for i, data in enumerate(imgs):
            # start
            start, samples, futures, all_images = data
        
            ax_start = axes[i, 0]
            ax_start.imshow(start)
            ax_future = axes[i, 1]
            ax_future.imshow(futures[frame_id])
            sample_axes = []
            for j, images in enumerate(all_images):
                # ax = f.add_subplot(spec[i, 3 + j])
                # ax.axis('off')
                axes[i, 3 + j].imshow(images[frame_id])
                sample_axes.append(axes[i, 3 + j])
        
            if i == 0:
                ax_start.set_title('Input')
                ax_future.set_title('Futures')
                for j, sample in enumerate(sample_axes):
                    sample.set_title('Sample {}'.format(j + 1))
        # plt.pause(1.0)
        frames.append(figure_to_numpy(f))
        plt.close(f)
        # f.close()
    return frames
