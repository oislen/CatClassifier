import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):

    def __init__(self, df, transforms):
        self.filepath = df['filepath'].tolist()
        self.category = df['category'].tolist()
        self.source = df['source'].tolist()
        self.transform = transforms
        
    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, idx):
        image_filepath = self.filepath[idx]
        image = Image.open(image_filepath)
        
        label = torch.tensor(self.category[idx], dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label