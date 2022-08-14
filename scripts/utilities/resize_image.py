import cv2

def resize_image(image_array, reshape_wh, interpolation = cv2.INTER_LINEAR):
    """"""
    # rescale the image either by shrinking or expanding
    res_image_array = cv2.resize(image_array, dsize = reshape_wh, interpolation=interpolation)
    return res_image_array