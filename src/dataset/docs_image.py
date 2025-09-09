import os

import pandas as pd
import numpy as np
from PIL import Image 
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, csv, path, transform = None):
        if isinstance(csv, (str, os.PathLike)):
            self.df = pd.read_csv(csv).values
        elif isinstance(csv, pd.DataFrame):
            self.df = csv.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target