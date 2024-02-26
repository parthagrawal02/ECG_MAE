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
import h5pickle

class CustomDataset(Dataset):
    def __init__(self, data_path: str = ""):
        self.file = h5pickle.File(data_path, 'r',skip_cache=False)
        self.data = self.file['signals']
        self.signals = self.data[:2000]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return np.array(self.signals[idx])[None,:,:], np.array(idx)