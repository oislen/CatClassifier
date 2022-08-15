import numpy as np
import cv2

def resize_image(image_array, reshape_wh, interpolation = cv2.INTER_LINEAR, keep_3dim = True):
    """"""
    # rescale the image either by shrinking or expanding
    res_image_array = cv2.resize(image_array, dsize = reshape_wh, interpolation=interpolation)
    # keep 3dim; when applying resizing to (:, :, 1) shape images
    if keep_3dim and len(res_image_array.shape) == 2:
        res_image_array = res_image_array[..., np.newaxis]

    return res_image_array