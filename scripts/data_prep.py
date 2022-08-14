import sys
sys.path.append('E:\\GitHub\\cat_classifier\\scripts')
from utilities.load_image import load_image
from utilities.plot_image import plot_image
from utilities.greyscale_image import greyscale_image
from utilities.crop_image import crop_image
from utilities.resize_image import resize_image

# set file path
image_fpath = 'E:\\GitHub\\cat_classifier\\data\\cats\\cat_domestic_cat_young_2.jpg'

# load image into python
rgb_image_array = load_image(image_fpath)

# convert image to grey scale
grey_image_array = greyscale_image(rgb_image_array)

# plot images
plot_image(rgb_image_array)
plot_image(grey_image_array)

# test cropping function
plot_image(crop_image(rgb_image_array, pt1_wh = (150, 150), pt2_wh = (100, 250)))
plot_image(crop_image(grey_image_array, pt2_wh = (150, 150), pt1_wh = (100, 250)))

# test rescaling function
plot_image(resize_image(rgb_image_array, reshape_wh = (334, 206)))
plot_image(resize_image(grey_image_array, reshape_wh = (334, 206)))