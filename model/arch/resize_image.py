import numpy as np
import cv2
from beartype import beartype
from typing import Union

@beartype
def resize_image(
    image_array:np.array,
    reshape_wh:Union[list,tuple],
    interpolation=cv2.INTER_LINEAR,
    keep_3dim:bool=True
    ) -> np.array:
    """
    Resizes a numpy image array to a specified shape width and height

    Parameters
    ----------
    image_array : numpy.array
        The image array to reshape to new dimensions
    reshape_wh : list, tuple
        The dimensions to reshape the image array to
    interpolation : cv2.INTER_LINEAR
        The interpolation function for reshaping the image array, default is cv2.INTER_LINEAR
    keep_3dim : bool
        Whether to maintain the third dimension of the input numpy image array, default is True
    
    Returns
    -------
    numpy.array
        The reshaped numpy image array
    """
    # rescale the image either by shrinking or expanding
    res_image_array = cv2.resize(image_array, dsize = reshape_wh, interpolation=interpolation)
    # keep 3dim; when applying resizing to (:, :, 1) shape images
    if keep_3dim and len(res_image_array.shape) == 2:
        res_image_array = res_image_array[..., np.newaxis]

    return res_image_array