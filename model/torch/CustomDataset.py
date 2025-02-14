import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, df):
        self.image_tensors = df['image_tensors'].values
        self.category_tensors = df['category_tensors'].values
        
    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        image_tensor = self.image_tensors[idx]
        category_tensor = self.category_tensors[idx]
        return image_tensor, category_tensor
    
    #def __getitems__(self, idx_list):
    #    image_tensors = self.image_tensors[idx_list].tolist()
    #    category_tensors = self.category_tensors[idx_list].tolist()
    #    return image_tensors, category_tensors

def collate_fn(data):
    arrays, categories = data
    return arrays, categories