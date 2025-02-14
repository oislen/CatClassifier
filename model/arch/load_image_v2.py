
import pandas as pd
from PIL import Image

def load_image_v2(image_fpaths):
    """
    """
    images = []
    for image_fpath in image_fpaths:
        temp = Image.open(image_fpath)
        keep = temp.copy()
        images.append(keep)
        temp.close()
    return pd.Series(images)