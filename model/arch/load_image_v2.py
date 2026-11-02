import os
import re
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from PIL import Image


def load_image_v2(image_fpaths):
    """ """
    images = []
    for image_fpath in image_fpaths:
        temp = Image.open(image_fpath)
        keep = deepcopy(temp)
        images.append(keep)
        temp.close()
    return pd.Series(images)


class TorchLoadImages:
    def __init__(self, torch_transforms, n_workers=None):
        self.torch_transforms = torch_transforms
        self.n_workers = n_workers

    def load_image(self, filepath):
        """ """
        # determine the filename and source
        file_name = os.path.basename(filepath)
        # determine label from image file path
        if ("cat" in file_name) or ("dog" in file_name):
            file_source = (
                "kaggle"
                if len(re.findall(pattern=r"^(cat|dog)(\.\d+\.jpg)$", string=file_name))
                > 0
                else "webscraper"
            )
            label_name = file_name.split(".")[0]
            label = label_name == "dog"
            label_tensor = torch.tensor(label, dtype=torch.int64)
        else:
            file_source = "kaggle"
            label_name = np.nan
            label = np.nan
            label_tensor = torch.tensor(label)
        # load image file and apply torch transforms
        image = Image.open(filepath)
        torch_transform_error = None
        try:
            image_tensor = self.torch_transforms(image)
        except Exception as e:
            image_tensor = None
            torch_transform_error = str(e)
        n_dims = len(np.asarray(image).shape)
        image.close()
        # create an output record — raw pixel array is not stored to reduce memory usage
        record = {
            "filepaths": filepath,
            "filenames": file_name,
            "source": file_source,
            "categoryname": label_name,
            "category": label,
            "ndims": n_dims,
            "category_tensors": label_tensor,
            "image_tensors": image_tensor,
            "torch_transform_error": torch_transform_error,
        }
        return record

    def load_images(self, filepaths):
        """ """
        if self.n_workers is None or self.n_workers <= 1:
            records = [self.load_image(filepath) for filepath in filepaths]
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                records = list(executor.map(self.load_image, filepaths))
        return records
