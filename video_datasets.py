import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from pathlib import Path

class ClevrerDataset(Dataset):
    def __init__(self, seq_len, is_train, size=64):
        self.is_train = is_train
        self.seq_len = seq_len
        self.size = size

        if is_train:
            paths = list(Path('clevrer/train').resolve().glob('**/*.mp4'))
            self.data = [str(p) for p in paths]
        else:
            paths = list(Path('TODO').resolve().glob('**/*.mp4'))
            self.data = [str(p) for p in paths]

        self.resizer = Resize(size=(size, size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        # path = self.data[46]
        # path = 'clevrer/train/video_00000-01000/video_00099.mp4'
        frames, _, _ = torchvision.io.read_video(path, pts_unit='sec') # fps ~= 25
        frames = frames[0::3]
        # if self.is_train:
        #     rand_i = np.random.randint(low=0, high=frames.shape[0]-self.seq_len)
        #     frames = frames[rand_i:rand_i+self.seq_len]
        # else:
        #     frames = frames[:self.seq_len]
        frames = frames[:self.seq_len]
        frames = frames.permute(0, 3, 1, 2)
        frames = frames.float()
        frames /= 255.
        frames = self.resizer(frames)
        return frames

class SpmotDataset(Dataset):
    def __init__(self, is_train, seq_len=10, size=64):
        self.is_train = is_train
        self.seq_len = seq_len
        self.size = size

        if is_train:
            self.data = np.load('spmot/spmot_train.npy') # (9600, 10, 3, 64, 64)
        else:
            self.data = np.load('spmot/spmot_val.npy')
        self.data = torch.tensor(self.data)

        self.resizer = Resize(size=(size, size))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        frames = self.data[idx] # (10, 3, 64, 64)

        frames = frames[:self.seq_len]
        frames = frames.float()
        frames /= 255.
        frames = self.resizer(frames)
        return frames

if __name__ == '__main__':
    dataset = ClevrerDataset(seq_len=24, is_train=True)
    print(dataset.__getitem__(46).shape)
    dataset = SpmotDataset(is_train=True)
    print(dataset.__getitem__(46).shape)