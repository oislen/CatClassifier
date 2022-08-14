# load relevant libraries
import os
import pandas as pd

# load custom modules
import sys
root_fdir = 'E:\\GitHub\\cat_classifier'
sys.path.append(os.path.join(root_fdir, 'scripts'))
from utilities.load_image import load_image
from utilities.plot_image import plot_image
from utilities.greyscale_image import greyscale_image
from utilities.pad_image import pad_image

# set image file directories
data_fdir = os.path.join(root_fdir, 'data')
cat_image_fdir = os.path.join(data_fdir, 'cats')
dog_image_fdir = os.path.join(data_fdir, 'dogs')

# create image file paths and classes
cat_image_fpaths = {os.path.join(cat_image_fdir, cat_image_fname):'cat' for cat_image_fname in os.listdir(cat_image_fdir)}
dog_image_fpaths = {os.path.join(dog_image_fdir, dog_image_fname):'dog' for dog_image_fname in os.listdir(dog_image_fdir)}

# combine image file paths and classes
comb_image_fpaths = {**cat_image_fpaths, **dog_image_fpaths}

# create list to hold image data
image_data = []
# iterate over image fpaths and classes to create image data object
for image_fpath, target_class in comb_image_fpaths.items():
    # load in rgb image array
    rgb_image_array = load_image(image_fpath)
    # convert image to grey scale
    grey_image_array = greyscale_image(rgb_image_array)
    # create target variable
    target = 1 if target_class == 'cat' else 0
    # generate row dict with relevant image info
    row_dict = {'image_fpath':image_fpath, 'image_array':grey_image_array, 'image_shape':grey_image_array.shape, 'target':target}
    # append row dict to image data list
    image_data.append(row_dict)

# convert image data object into a pandas dataframe
image_dataframe = pd.DataFrame(image_data)

# find the largest image dimensions
image_dataframe['image_shape'].apply(lambda x: x[0]).max() # height
image_dataframe['image_shape'].apply(lambda x: x[1]).max() # width

# apply padding to standardize all images shapes
image_dataframe['pad_image_array'] = image_dataframe['image_array'].apply(lambda x: pad_image(x, pad_shape_wh = (515, 260)))

# subset the output image data
output = image_dataframe[['image_fpath', 'pad_image_array', 'target']]

# save processed data as a pickle file
output.to_pickle(os.path.join(data_fdir, 'model_data.pickle'))
