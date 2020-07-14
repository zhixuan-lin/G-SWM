from torch.utils.data import Dataset
import h5py
import os
import numpy as np
import torch


class Balls(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=20):
        """
        Args:
            root: dataset root
            mode: one of ['train', 'val', 'test']
            ep_len: epsisode length of the dataset file
            sample_length: the actual maximum episode length you want
        """
        assert mode in ['train', 'val', 'test']
        self.root = root
        file = os.path.join(self.root, f'{mode}.hdf5')
        assert os.path.exists(file), 'Path {} does not exist'.format(file)
        self.file = file

        self.mode = mode
        self.sample_length = sample_length
        
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1
        
    def __getitem__(self, index):
        
        
        with h5py.File(self.file, 'r') as f:
            self.imgs = f['imgs']
            self.positions = f['positions']
            self.sizes = f['sizes']
            self.ids = f['ids']
            self.in_camera = f['in_camera']
        
            if self.mode == 'train':
                # Implement continuous indexing
                ep = index // self.seq_per_episode
                offset = index % self.seq_per_episode
                end = offset + self.sample_length
                img = self.imgs[ep][offset:end]
                pos = self.positions[ep][offset:end]
                size = self.sizes[ep][offset:end]
                id = self.ids[ep][offset:end]
                in_camera = self.in_camera[ep][offset:end]
            else:
                img = self.imgs[index]
                pos = self.positions[index]
                size = self.sizes[index]
                id = self.ids[index]
                in_camera = self.in_camera[index]
                assert img.shape[0] == self.EP_LEN
        
        
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img = img.float() / 255.0
        
        return img, pos, size, id, in_camera
    
    def __len__(self):
        with h5py.File(self.file, 'r') as f:
            length = len(f['imgs'])
            if self.mode == 'train':
                return length * self.seq_per_episode
            else:
                return length
    

