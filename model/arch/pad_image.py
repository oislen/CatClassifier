import numpy as np
from beartype import beartype
from typing import Union

@beartype
def pad_image(
    image_array:np.array,
    pad_shape_wh:Union[list,tuple]
    ) -> np.array:
    """
    Pads an image array to a desired width and height

    Parameters
    ----------
    image_array: np.array
        The image array to pad to a specified dimension
    pad_shape_wh : list, tuple
        The desired dimensions to pad the input image array to

    Returns
    -------
    numpy.array
        The padded image array
    """
    image_array_shape = image_array.shape
    (img_h, img_w) = image_array_shape[0:2]
    (pad_img_w, pad_img_h) = pad_shape_wh
    # calculate amount of padding need
    pad_h = pad_img_h - img_h
    pad_w = pad_img_w - img_w
    n_imade_array_dim = len(image_array_shape)
    if n_imade_array_dim == 3:
        pad_image_array = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)))
    else:
        pad_image_array = np.pad(image_array, ((0, pad_h), (0, pad_w)))
    return pad_image_array