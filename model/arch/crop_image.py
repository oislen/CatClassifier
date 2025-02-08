import numpy as np
from beartype import beartype
from types import UnionType

@beartype
def crop_image(
    image_array:np.array,
    pt1_wh:UnionType[tuple,list],
    pt2_wh:UnionType[tuple,list]
    ) -> np.array:
    """Crops an image array to specified combination of two diagonal points

    Parameters
    ---------
    image : numpy.array
        The numpy image array to crop
    pt1_wh : list, tuple
        Diagonal point 1 coordinates for cropping the image
    pt2_wh : list, tuple
        Diagonal point 2 coordinates for cropping the image
    
    Returns
    -------
    numpy.array
        The cropped image array
    """
    # extract out diagonal cropping points
    (pt1_w, pt1_h) = pt1_wh
    (pt2_w, pt2_h) = pt2_wh
    # group height and width points
    wpts = (pt1_h, pt2_h)
    hpts = (pt1_w, pt2_w)
    # extract out image shape
    n_image_dims = len(image_array.shape)
    if n_image_dims == 3:
        crop_image_array = image_array[min(hpts):max(hpts), min(wpts):max(wpts), :] 
    else:
        crop_image_array = image_array[min(hpts):max(hpts), min(wpts):max(wpts)] 
    return crop_image_array