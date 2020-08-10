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
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RobotObj(Dataset):
    def __init__(self, root, mode, ep_len=30, sample_length=20):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'test']
        # root: '../data/robot_obj', mod: 'train'/ 'val'/ 'test'
        self.root = os.path.join(root, mode) # '../data/robot_obj/train'
        
        self.mode = mode
        self.sample_length = sample_length # 20
        
        
        # Get all numbers
        # iterate over folders: 'episode1, episode2, ...'

        self.var_idx = 0
        for file in os.listdir(self.root): 
            try:
                self.var_idx += 1
            except ValueError:
                continue
        self.variations = []
        for var in range(self.var_idx):
            self.folders = []
            for file in os.listdir(os.path.join(self.root, 'variation' + str(var), 'episodes')): 
                try:
                    self.folders.append(int(file.replace('episode', '')))
                except ValueError:
                    continue
            self.folders.sort() # sort by number
            self.variations.append((var, self.folders))
        
        self.episodes = []
        self.low_dims = []
        self.EP_LEN = ep_len # -30 # sample_length -20 
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1
        
        for v, folders in self.variations: # var_idx, folder_idx
            ep_len = 0
            for f in folders:
                ep_len += 1
                dir_name = os.path.join(self.root, 'variation' + str(v), 'episodes', 'episode' + str(f)) # '../data/robot_obj/val/0'
                paths = list(glob.glob(osp.join(dir_name, 'left_shoulder_rgb', '*.png')))
                ld_pkl_path = osp.joint(dir_name, 'low_dim_obs.pkl')
                self.get_low_dim_robot_data(ld_pkl_path)
                if len(paths) < 70: #! ignore imperfect demos
                    continue
                # if len(paths) != self.EP_LEN:
                #     continue
                # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
                get_num = lambda x: int(osp.splitext(osp.basename(x))[0].partition('_')[0])

                paths.sort(key=get_num) # soft the file names...
                paths = paths[58:] # crop out first 60 episodes 
                self.episodes.append(paths) # append lists 
                self.low_dims = []
            
    
    def __getitem__(self, index):
        
        # TODO (cheolhui): return the low-dim data here
        imgs = []
        if self.mode == 'train':
            # Implement continuous indexing
            ep = index // self.seq_per_episode
            offset = index % self.seq_per_episode
            end = offset + self.sample_length
            
            e = self.episodes[ep]
            for image_index in range(offset, end):
                img = Image.open(osp.join(e[image_index]))
                img = img.resize((64, 64))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
        else:
            for path in self.episodes[index]:
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

    def get_low_dim_robot_data(self, low_dim_path):
        # return low-dimensional data of robot
        with open(low_dim_path, 'rb') as f:
            obs = pickle.load(f) # AttrDict
        return obs

