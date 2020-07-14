import numpy as np
import torch
from utils import spatial_transform

# https://www.rapidtables.com/web/color/RGB_Color.html
# (N, 3)
COLORS = torch.tensor([
    [255, 255, 255], # white
    [255, 0, 0], # red
    [0, 255, 0], # lime
    [0, 0, 255], # blue
    [255, 255, 0], # yellow
    [0, 255, 255], # cyan
    [255, 0, 255], # magenta
]) / 255.0

N_COLORS = COLORS.size(0)

BOX_H = 32
BOX_W = 32
LINE_WIDTH = 2

def figure_to_numpy(fig):
    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    # draw the figure first...
    fig.canvas.draw()
    
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def make_gif(images, path, fps):
    """
    Show some gif.

    :param videos: (N, T, H, W, 3)
    :param path: directory
    """
    import moviepy.editor as mpy
    clip = mpy.ImageSequenceClip(list(images), fps=fps)
    clip.write_gif(path, fps=fps)


def combine_images(imgs, factor, transparency, interval):
    """
    Combine several images into one

    Args:
        imgs: a list or array of images (N, H, W, 3)
        factor: float.

    Returns:
        combined: (H, W, 3)

    """
    imgs = imgs[::interval]
    H, W, C = imgs[0].shape
    canvas = np.zeros((H, W, C))
    
    # img_prev = np.zeros_like(imgs[0])
    for img in imgs:
        canvas = canvas * factor
        mask_prev, mask_this = get_masks(canvas, img, transparency)
        # canvas = (1 - mask) * canvas * factor + mask * img
        canvas = mask_prev * canvas + mask_this * img
    
    return canvas


def is_red(img, up_threshold=3, low_threshold=3):
    mask = (img[..., 0] > up_threshold) & (img[..., 1] < low_threshold) & (img[..., 2] < low_threshold)
    return mask[..., None]


def is_blue(img, up_threshold=3, low_threshold=3):
    mask = (img[..., 0] < up_threshold) & (img[..., 1] < low_threshold) & (img[..., 2] > low_threshold)
    return mask[..., None]


def get_masks(img1, img2, weight):
    """

    Args:
        img: (H, W, 3)

    Returns:
        mask: (H, W, 1)
    """
    # (H, W)
    # mask1 = cv2.cvtColor(img1, code=cv2.COLOR_RGB2GRAY)
    mask1 = img1.max(axis=-1, keepdims=True).astype(np.float)
    mask2 = img2.max(axis=-1, keepdims=True).astype(np.float)
    
    red1 = is_red(img1)
    blue1 = is_blue(img1)
    red2 = is_red(img2)
    blue2 = is_blue(img2)
    
    mask1 *= 1. - weight
    mask2 *= weight
    
    sum = mask1 + mask2
    mask1 = mask1 / (sum + 1e-5)
    mask2 = mask2 / (sum + 1e-5)
    assert np.all(mask1 + mask2 <= 1.0), np.max(mask1 + mask2)
    return mask1, mask2

def get_boxes(ids: torch.Tensor):
    """
    Given object ids, return boxes
    Args:
        id: (...), whatever size

    Returns:
        boxes: (..., 3, H, W), boxes with colors
    """
    # (N, 3) -> (..., 3)
    colors = COLORS[ids.long() % N_COLORS]
    # (..., 3) -> (..., 3, H, W)
    boxes = colors[..., None, None].repeat((1,) * ids.dim() + (1, BOX_H, BOX_W))
    # Only retain the edges
    boxes[..., LINE_WIDTH:-LINE_WIDTH, LINE_WIDTH:-LINE_WIDTH] = 0.0
    
    return boxes

def draw_boxes(images, z_where, z_pres, ids):
    """
    Draw bounding boxes over image.
    Args:
        images: (..., 3, H, W)
        z_where: (..., N, 4)
        z_pres: (..., N, 1). This can be soft
        ids: (..., N). Id of each box

    Returns:
        images: images with boxes drawn
    """
    *ORI, N = ids.size()
    
    
    ORIN = np.prod(ORI + [N])
    
    *_, H, W = images.size()
    # Reshape everything for convenience
    # (ORIN,)
    ids = ids.reshape(-1)
    # (ORIN, 4)
    z_where = z_where.reshape(-1, 4)
    # (ORIN, 1)
    z_pres = z_pres.reshape(-1, 1)
    
    # Get boxes, (ORI*N, 3, H, W)
    boxes = get_boxes(ids)
    
    # (ORIN, 3, H, W)
    boxes = spatial_transform(boxes, z_where, (ORIN, 3, H, W), inverse=True)
    
    # Use z_pres as masks
    # (ORIN, 3, H, W) * (ORIN, 1)
    boxes = boxes * z_pres[..., None, None]
    
    # (*ORI, N, 3, H, W)
    boxes = boxes.reshape(*ORI, N, 3, H, W)
    
    # (*ORI, 3, H, W)
    boxes = boxes.sum(dim=-4).clamp_max(1.0)
    
    # (*ORI, 3, H, W)
    img_box = (images + boxes).clamp_max(1.0)
    
    return img_box
    
    
    
