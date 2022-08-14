import numpy as np

def greyscale_image(rgb_image_array):
    """"""
    # apply grey scale transformation
    grey_image_array =  np.dot(rgb_image_array[:, :, :3], [0.2125, 0.7154, 0.0721])
    # floor transformed pixel floats
    gray_img = grey_image_array.astype(np.uint8)
    return gray_img
