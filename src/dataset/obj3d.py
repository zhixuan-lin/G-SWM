from torch.utils.data import Dataset
import h5py
from torchvision import transforms
import glob
from skimage import io
import os
import os.path as osp
import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Obj3D(Dataset):
    def __init__(self, root, mode, ep_len=30, sample_length=20):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'test']
        self.root = os.path.join(root, mode)
        
        self.mode = mode
        self.sample_length = sample_length
        
        
        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()
        
        self.epsisodes = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1
        
        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, 'test_*.png')))
            # if len(paths) != self.EP_LEN:
            #     continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0].partition('_')[-1])
            paths.sort(key=get_num)
            self.epsisodes.append(paths)
        
    
    def __getitem__(self, index):
        
        
        imgs = []
        if self.mode == 'train':
            # Implement continuous indexing
            ep = index // self.seq_per_episode
            offset = index % self.seq_per_episode
            end = offset + self.sample_length
            
            e = self.epsisodes[ep]
            for image_index in range(offset, end):
                img = Image.open(osp.join(e[image_index]))
                img = img.resize((64, 64))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
        else:
            for path in self.epsisodes[index]:
                img = Image.open(path)
                img = img.resize((64, 64))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
                
        
        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)
        
        return img, pos, size, id, in_camera
    
    def __len__(self):
        length = len(self.epsisodes)
        if self.mode == 'train':
            return length * self.seq_per_episode
        else:
            return length

