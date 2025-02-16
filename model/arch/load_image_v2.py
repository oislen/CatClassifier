
import pandas as pd
import numpy as np
import re
import os
import torch
from PIL import Image
from matplotlib.image import imread
from copy import deepcopy
from multiprocessing import Pool
from copy import deepcopy

def load_image_v2(image_fpaths):
    """
    """
    images = []
    for image_fpath in image_fpaths:
        temp = Image.open(image_fpath)
        keep = deepcopy(temp)
        images.append(keep)
        temp.close()
    return pd.Series(images)

class TorchLoadImages():
    
    def __init__(self, torch_transforms, n_workers=None):
        self.torch_transforms = torch_transforms
        self.n_workers = n_workers
    
    def loadImage(self, filepath):
        """
        """
        # determine the filename and source
        fileName = os.path.basename(filepath)
        # determine label from image file path
        if ("cat" in fileName) or ("dog" in fileName):
            fileSource = "kaggle" if len(re.findall(pattern='^(cat|dog)(.[0-9]+.jpg)$', string=fileName)) > 0 else "webscraper"
            labelName = fileName.split(".")[0]
            label = labelName == "dog"
            labelTensor = torch.tensor(label, dtype=torch.int64)
        else:
            fileSource = "kaggle"
            labelName = np.nan
            label = np.nan
            labelTensor = torch.tensor(label)
        # load image file and apply torch transforms
        image = Image.open(filepath)
        imageTensor = self.torch_transforms(image)
        imageArray = np.asarray(image)
        image.close()
        nDims = len(imageArray.shape)
        # create an output record
        record = {
            "filepaths":filepath,
            "filenames":fileName,
            "source":fileSource,
            "categoryname":labelName,
            "category":label,
            "images":imageArray,
            "ndims":nDims,
            "category_tensors":labelTensor,
            "image_tensors":imageTensor
            }
        # close open image
        return record
    
    def multiProcess(self, func, args):
        """
        """
        pool = Pool(self.n_workers)
        results = pool.map(func, args)
        pool.close()
        return results
    
    def loadImages(self, filepaths):
        """
        """
        if self.n_workers == None:
            records = [self.loadImage(filepath) for filepath in filepaths]
        else:
            records = self.multiProcess(self.loadImage, filepaths)
        return records