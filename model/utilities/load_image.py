import numpy as np
import cv2

def load_image(image_fpath):
    """"""
    # load image from file path
    image_imread = cv2.imread(image_fpath)
    # convert to numpy array
    rgb_image_array = np.array(image_imread)
    return rgb_image_array