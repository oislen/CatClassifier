import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, df, transforms, mode):
        self.mode = mode
        self.filepath = df['filepath'].tolist()
        if mode == 'train':
            self.category = df['category'].tolist()
        self.transform = transforms
        
    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, idx):
        image_filepath = self.filepath[idx]
        image = Image.open(image_filepath)
        
        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            label = torch.tensor(self.category[idx], dtype=torch.int64)
        else:
            label = torch.tensor(np.nan)
        
        return image, label