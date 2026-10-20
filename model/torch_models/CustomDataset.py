import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df):
        self.image_tensors = df["image_tensors"].values
        self.category_tensors = df["category_tensors"].values

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        image_tensor = self.image_tensors[idx]
        category_tensor = self.category_tensors[idx]
        return image_tensor, category_tensor

    @staticmethod
    def collate_fn(data_list):
        arrays, categories = zip(*data_list)
        return torch.stack(list(arrays)), torch.stack(list(categories))
