import numpy as np

def pad_image(image_array, pad_shape_wh):
    """"""
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