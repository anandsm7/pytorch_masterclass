import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd
from PIL import Image, ImageFile

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_path, root_dir,resize=256, transform=None):
        self.annotations = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        targets = self.annotations.iloc[index, 1]
        image = image.resize((self.resize,self.resize),resample=Image.BILINEAR)
        image = np.array(image)
        if self.transform:
            image = self.transform(image)
            
        return(torch.tensor(image, dtype=torch.float),
               torch.tensor(targets, dtype=torch.long))
        
        
        