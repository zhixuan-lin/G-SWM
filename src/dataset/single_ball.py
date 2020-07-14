from torch.utils.data import Dataset
import h5py
from torchvision import transforms
import glob
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SingleBall(Dataset):
    def __init__(self, root, mode):
        assert mode in ['train', 'val', 'test']
        self.root = root
        self.file = os.path.join(self.root, f'{mode}.hdf5')
        assert os.path.exists(self.file), 'Path {} does not exist'.format(self.file)
        
        
    def __getitem__(self, index):
    
    
        with h5py.File(self.file, 'r') as f:
            imgs = f['imgs']
            img = imgs[index]
        assert len(img) == 9
        
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img = img.float() / 255.0
        
        return img, torch.zeros(0)
    
    def __len__(self):
        with h5py.File(self.file, 'r') as f:
            imgs = f['imgs']
            return len(imgs)
    

