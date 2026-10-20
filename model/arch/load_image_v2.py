import os
import re
from copy import deepcopy
from multiprocessing import Pool

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
                if len(re.findall(pattern="^(cat|dog)(.[0-9]+.jpg)$", string=file_name))
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
        image_array = np.asarray(image)
        image.close()
        n_dims = len(image_array.shape)
        # create an output record
        record = {
            "filepaths": filepath,
            "filenames": file_name,
            "source": file_source,
            "categoryname": label_name,
            "category": label,
            "images": image_array,
            "ndims": n_dims,
            "category_tensors": label_tensor,
            "image_tensors": image_tensor,
            "torch_transform_error": torch_transform_error,
        }
        # close open image
        return record

    def multi_process(self, func, args):
        """ """
        pool = Pool(self.n_workers)
        results = pool.map(func, args)
        pool.close()
        return results

    def load_images(self, filepaths):
        """ """
        if self.n_workers is None:
            records = [self.load_image(filepath) for filepath in filepaths]
        else:
            records = self.multi_process(self.load_image, filepaths)
        return records
