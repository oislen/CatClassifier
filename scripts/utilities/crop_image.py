def crop_image(image_array, pt1_wh, pt2_wh):
    """"""
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