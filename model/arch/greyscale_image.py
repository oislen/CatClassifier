import numpy as np
from beartype import beartype

@beartype
def greyscale_image(
    rgb_image_array:np.array,
    keep_3dim:bool=True
    ) -> np.array:
    """
    Transforms an image array to greyscale

    Parameters
    ----------
    rgb_image_array : numpy.array
        The coloured numpy image array to transform to greyscale
    keep_3dim : bool
        Whether to keep the third dimension of the image array, default is True
    
    Returns
    -------
    numpy.array
        The transformed greyscale numpy image
    """
    # apply grey scale transformation
    grey_image_array =  np.dot(rgb_image_array[:, :, :3], [0.2125, 0.7154, 0.0721])
    # floor transformed pixel floats
    gray_img = grey_image_array.astype(np.uint8)
    # if keeping third dimension
    if keep_3dim:
        gray_img = gray_img[..., np.newaxis]
    return gray_img
