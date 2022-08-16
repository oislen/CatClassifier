# load relevant libraries
import os
import pandas as pd

# load custom modules
import cons
from utilities.load_image import load_image
from utilities.greyscale_image import greyscale_image
from utilities.pad_image import pad_image
from utilities.plot_image import plot_image
from utilities.resize_image import resize_image

def data_prep(cons):

    """"""

    print("Generating image file paths and classes ...")


    if False:

        # set image file directories
        cat_image_fdir = os.path.join(cons.data_fdir, 'cats')
        dog_image_fdir = os.path.join(cons.data_fdir, 'dogs')

        # create image file paths and classes
        cat_image_fpaths = {os.path.join(cat_image_fdir, cat_image_fname):'cat' for cat_image_fname in os.listdir(cat_image_fdir)}
        dog_image_fpaths = {os.path.join(dog_image_fdir, dog_image_fname):'dog' for dog_image_fname in os.listdir(dog_image_fdir)}
        
        # combine image file paths and classes
        train_image_fpaths = {**cat_image_fpaths, **dog_image_fpaths}
    
    else :

        # set image file directories
        train_image_fdir = cons.train_fdir
        test_image_fdir = cons.test_fdir

        # create image file paths and classes
        shuffled_image_fnames = pd.Series(os.listdir(train_image_fdir)).sample(n = cons.sample_size, replace = False)
        train_image_fpaths = {os.path.join(train_image_fdir, image_fname):image_fname.split('.')[0] for image_fname in shuffled_image_fnames}

    print("Creating image dataframe ...")

    # create list to hold image data
    image_data = []
    # iterate over image fpaths and classes to create image data object
    for idx, (image_fpath, target_class) in enumerate(train_image_fpaths.items()):
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

    print("Padding images ...")

    # find the largest image dimensions
    max_height = image_dataframe['image_shape'].apply(lambda x: x[0]).max() # height
    max_width = image_dataframe['image_shape'].apply(lambda x: x[1]).max() # width

    # set padded shape
    pad_shape = (max_width, max_height)

    # apply padding to standardize all images shapes
    image_dataframe['pad_image_array'] = image_dataframe['image_array'].apply(lambda x: pad_image(x, pad_shape_wh = pad_shape))

    print('Down sizing image ...')

    # set down size shape
    downsize_shape = tuple([round(dim * 1/2) for dim in pad_shape])
    
    # apply resizing to downsize image shapes
    image_dataframe['pad_image_array'] = image_dataframe['pad_image_array'].apply(lambda x: resize_image(x, reshape_wh = downsize_shape))

    print("Pickling processed output data ...")

    # subset the output image data
    output = image_dataframe[['image_fpath', 'pad_image_array', 'target']]

    # save processed data as a pickle file
    output.to_pickle(os.path.join(cons.data_fdir, 'model_data.pickle'))

    return 0

# if running as main programme
if __name__ == "__main__":

    # run data prep
    data_prep(cons = cons)