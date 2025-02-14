
import pandas as pd
import numpy as np
from PIL import Image
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