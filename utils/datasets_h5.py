import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import wfdb
from wfdb import processing
import pdb
import neurokit2 as nk
import pandas as pd
import h5py

class CustomDataset(Dataset):
    def __init__(self, data_path: str = ""):
        self.data_path = data_path
        h5_file = h5py.File(self.data_path , 'r')
        self.data = list(h5_file.keys())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        key = self.data[idx]
        h5_file = h5py.File(self.data_path , 'r')
        return torch.from_numpy(np.array(h5_file[key])), torch.tensor(idx)