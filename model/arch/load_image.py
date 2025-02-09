import numpy as np
import cv2
from beartype import beartype

@beartype
def load_image(image_fpath:str) -> np.array:
    """
    Loads an image file as an image array from disk

    Parameters
    ----------
    image_fpath : str
        The file path to the image file to load as an image array
    
    Returns
    -------
    numpy.array
        The loaded image array
    """
    # load image from file path
    image_imread = cv2.imread(image_fpath)
    # convert to numpy array
    rgb_image_array = np.array(image_imread)
    return rgb_image_array